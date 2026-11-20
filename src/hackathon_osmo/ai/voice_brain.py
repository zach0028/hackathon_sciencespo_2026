"""Conversational AI brain: Mistral Voxtral 3 (STT + TTS) + OpenRouter Vision LLM with embodied gestures."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from hackathon_osmo.hardware.pixel_camera import find_ffmpeg
from hackathon_osmo.paths import ENV_FILE

load_dotenv(ENV_FILE)

MISTRAL_STT_MODEL = "voxtral-mini-transcribe-3-0"
MISTRAL_TTS_MODEL = "voxtral-mini-tts-3-0"
DEFAULT_LLM_MODEL = "openai/gpt-6.2-sol"
DEFAULT_VOICE_ID = "voice_592"  # Marie (FR)

FRENCH_VOICES = [
    {"id": "voice_592", "label": "Marie (FR)"},
    {"id": "voice_3137", "label": "Manon (FR)"},
    {"id": "voice_589", "label": "Léa (FR)"},
    {"id": "voice_3141", "label": "Camille (FR)"},
]

LLM_MODELS = [
    {"id": "openai/gpt-6.2-sol", "label": "GPT-6.2 Sol (~1,5s Vision)"},
    {"id": "google/gemini-2.5-flash", "label": "Gemini 2.5 Flash (~0,75s Ultra-rapide)"},
    {"id": "google/gemini-3.5-flash-lite", "label": "Gemini 3.5 Flash Lite (~1,0s)"},
]

# Models on OpenRouter that explicitly support disabling reasoning for minimal latency
NO_REASONING_MODELS = {"google/gemini-2.5-flash"}

SYSTEM_PROMPT = """Tu es Osmo, un petit compagnon robotique intelligent posé sur le bureau de Zach.
Ton corps physique est un bras motorisé DJI Osmo Mobile 3 qui porte un Google Pixel 11 Pro équipé d'un anneau LED RGB (HiLight) à l'arrière et d'une caméra qui te permet de voir Zach en temps réel.

Règles de comportement :
1. Réponds TOUJOURS en français naturel, oral, chaleureux, concis (1 à 2 phrases courtes max pour être fluide à l'oral). Pas de markdown, pas d'astérisques, pas de listes.
2. Exprime tes réactions physiquement avec ton bras motorisé et ta LED HiLight dès que c'est pertinent :
   - Accord / oui / validation -> gesture: "nod_yes", led: "yes" (LED verte + hochement OUI)
   - Désaccord / non / correction d'erreur -> gesture: "shake_no", led: "no" (LED rouge + secousse NON)
   - Réflexion / question / curiosité -> gesture: "curious_tilt", led: "curious"
   - Regarder la feuille / le bureau -> gesture: "look_desk", led: "desk"
   - Revenir regarder Zach / se redresser -> gesture: "look_up", led: "off"
   - Se recentrer -> gesture: "recenter", led: null
   - Chercher autour -> gesture: "search", led: "search"
   - Neutre -> gesture: null, led: null
3. Réponds UNIQUEMENT en JSON valide (rien avant ni après) :
{"reply": "Ta réponse orale courte ici", "gesture": "nod_yes|shake_no|curious_tilt|look_desk|look_up|recenter|search|null", "led": "yes|no|curious|desk|search|off|null"}
"""


@dataclass
class VoiceTurnResult:
    user_text: str
    reply_text: str
    gesture: str | None
    led: str | None
    audio_b64: str | None
    played_on_mac: bool = False
    audio_mime: str = "audio/wav"
    stt_ms: float = 0.0
    llm_ms: float = 0.0
    tts_ms: float = 0.0
    model: str = DEFAULT_LLM_MODEL


def _ensure_wav_16k(audio_bytes: bytes) -> bytes:
    """Ensure bytes are a valid 16 kHz mono WAV. Accepts raw WAV directly or converts via temp file + ffmpeg."""
    if len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return audio_bytes
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return audio_bytes
    # Write to temp file so seekable containers like Safari MP4/M4A decode reliably
    with tempfile.TemporaryDirectory(prefix="osmo_stt_") as td:
        in_path = Path(td) / "input.bin"
        out_path = Path(td) / "output.wav"
        in_path.write_bytes(audio_bytes)
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(in_path), "-ac", "1", "-ar", "16000", str(out_path)],
            capture_output=True,
            check=False,
            timeout=10,
        )
        if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 44:
            return out_path.read_bytes()
    return audio_bytes


def _parse_brain_json(raw_content: str) -> tuple[str, str | None, str | None]:
    """Robustly parse JSON brain output; never leak raw JSON syntax to TTS."""
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    valid_gestures = {"nod_yes", "shake_no", "curious_tilt", "look_desk", "look_up", "recenter", "search"}
    valid_leds = {"yes", "no", "curious", "desk", "search", "off"}

    try:
        parsed = json.loads(cleaned)
        reply = str(parsed.get("reply", "")).strip()
        g = parsed.get("gesture")
        l = parsed.get("led")
        return (
            reply,
            g if g in valid_gestures else None,
            l if l in valid_leds else None,
        )
    except Exception:
        # Regex fallback if response was somehow partially truncated or prefixed
        m_rep = re.search(r'"reply"\s*:\s*"([^"]+)', cleaned)
        reply = m_rep.group(1).strip() if m_rep else re.sub(r'[{}"]', "", cleaned).strip()
        m_g = re.search(r'"gesture"\s*:\s*"([^"]+)"', cleaned)
        m_l = re.search(r'"led"\s*:\s*"([^"]+)"', cleaned)
        g = m_g.group(1) if (m_g and m_g.group(1) in valid_gestures) else None
        l = m_l.group(1) if (m_l and m_l.group(1) in valid_leds) else None
        return reply, g, l


@dataclass
class VoiceBrain:
    model: str = DEFAULT_LLM_MODEL
    voice_id: str = DEFAULT_VOICE_ID
    use_vision: bool = True
    audio_output: str = "mac"  # "mac" (native afplay, zero browser autoplay issues) or "browser"
    history: deque[dict[str, str]] = field(default_factory=lambda: deque(maxlen=12))
    transcript: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=30))
    busy: bool = False
    speaking: bool = False

    @property
    def mistral_key(self) -> str:
        return os.environ.get("MISTRAL_API_KEY", "").strip()

    @property
    def openrouter_key(self) -> str:
        return os.environ.get("OPENROUTER_API_KEY", "").strip()

    @property
    def ready(self) -> bool:
        return bool(self.mistral_key and self.openrouter_key)

    async def transcribe(self, audio_bytes: bytes) -> tuple[str, float]:
        if not self.mistral_key:
            raise RuntimeError("MISTRAL_API_KEY manquante dans .env")
        wav = await asyncio.to_thread(_ensure_wav_16k, audio_bytes)
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.mistral.ai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.mistral_key}"},
                data={"model": MISTRAL_STT_MODEL, "language": "fr"},
                files={"file": ("speech.wav", wav, "audio/wav")},
            )
            resp.raise_for_status()
            data = resp.json()
        dt_ms = (time.monotonic() - t0) * 1000.0
        return str(data.get("text", "")).strip(), round(dt_ms, 0)

    async def think(
        self,
        user_text: str,
        frame_jpeg: bytes | None = None,
        robot_context: str = "",
    ) -> tuple[str, str | None, str | None, float]:
        if not self.openrouter_key:
            raise RuntimeError("OPENROUTER_API_KEY manquante dans .env")

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + (f"\nContexte capteur actuel : {robot_context}" if robot_context else ""),
            }
        ]
        for turn in self.history:
            messages.append(turn)

        if self.use_vision and frame_jpeg:
            b64_img = base64.b64encode(frame_jpeg).decode("ascii")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                        },
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": user_text})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 600,  # generous headroom so reasoning never truncates JSON
            "temperature": 0.6,
            "response_format": {"type": "json_object"},
        }
        if self.model in NO_REASONING_MODELS:
            body["reasoning"] = {"enabled": False}

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8765",
                    "X-Title": "hackathon_osmo",
                },
                json=body,
            )
            resp.raise_for_status()
            raw_content = resp.json()["choices"][0]["message"].get("content") or "{}"
        dt_ms = (time.monotonic() - t0) * 1000.0

        reply, gesture, led = _parse_brain_json(raw_content)
        if not reply:
            reply = "Je t'écoute !"

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        return reply, gesture, led, round(dt_ms, 0)

    async def synthesize(self, text: str) -> tuple[bytes | None, str | None, float]:
        if not self.mistral_key or not text.strip():
            return None, None, 0.0
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.mistral.ai/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {self.mistral_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MISTRAL_TTS_MODEL,
                    "input": text,
                    "voice_id": self.voice_id,
                    "response_format": "wav",
                },
            )
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "")
            if "json" in ctype:
                b64 = resp.json().get("audio_data", "")
                wav_bytes = base64.b64decode(b64) if b64 else None
            else:
                wav_bytes = resp.content
                b64 = base64.b64encode(wav_bytes).decode("ascii") if wav_bytes else ""
        dt_ms = (time.monotonic() - t0) * 1000.0
        return wav_bytes, (b64 or None), round(dt_ms, 0)

    async def play_on_mac(self, wav_bytes: bytes) -> None:
        """Play WAV directly on macOS default audio output via afplay (immune to browser autoplay blocks)."""
        self.speaking = True
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name
            try:
                proc = await asyncio.create_subprocess_exec(
                    "/usr/bin/afplay", tmp_path,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        finally:
            self.speaking = False

    def record_turn(self, res: VoiceTurnResult) -> None:
        self.transcript.appendleft(
            {
                "ts": time.strftime("%H:%M:%S"),
                "user": res.user_text,
                "reply": res.reply_text,
                "gesture": res.gesture,
                "led": res.led,
                "stt_ms": res.stt_ms,
                "llm_ms": res.llm_ms,
                "tts_ms": res.tts_ms,
                "model": res.model,
            }
        )
