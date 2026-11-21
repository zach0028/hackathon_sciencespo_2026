"""OpenAI Full-Duplex Voice + Delegated Frontier Vision Brain.

Architecture (Topology A):
- Voice frontend : gpt-live-1.1 over WebRTC (UDP/Opus, native full-duplex barge-in)
- Control bridge : Python Sideband WebSocket (/v1/live/sessions/{id}/attach)
- Reasoning brain: gpt-6.2-astra (reasoning.effort="low") via OpenAI /v1/responses
  with persistent HTTP keep-alive, Pixel 11 Pro live camera frames, and native
  function tools driving the DJI Osmo Mobile 3 & Pixel HiLight 8-LED ring.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import cv2
import httpx
import numpy as np
import websockets
from dotenv import load_dotenv

from hackathon_osmo.paths import ENV_FILE
from hackathon_osmo.telemetry.recorder import SessionRecorder

load_dotenv(ENV_FILE)

OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_WS_BASE = "wss://api.openai.com/v1"

LIVE_MODEL = "gpt-live-1.1"
DEFAULT_BRAIN_MODEL = "gpt-6.2-sol"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_LIVE_VOICE = "marin"

LIVE_VOICES = [
    {"id": "marin", "label": "Marin (FR · recommandé)"},
    {"id": "cedar", "label": "Cedar (FR · naturel)"},
    {"id": "verse", "label": "Verse (expressif)"},
    {"id": "coral", "label": "Coral (chaleureux)"},
    {"id": "ash", "label": "Ash (posé)"},
    {"id": "sage", "label": "Sage (clair)"},
]

BRAIN_MODELS = [
    {"id": "gpt-6.2-sol", "label": "GPT-6.2 Sol (~1,6s · Recommandé)"},
    {"id": "gpt-6.2-astra", "label": "GPT-6.2 Astra (Raisonnement profond)"},
    {"id": "gpt-6-astra", "label": "GPT-6 Astra (Frontier Vision)"},
    {"id": "gpt-6-luna", "label": "GPT-6 Luna (~1,1s Ultra-rapide)"},
]

# Models supporting reasoning.effort="none" vs minimum "low"
MODELS_MIN_LOW_EFFORT = {"gpt-6.2-astra", "gpt-6-astra", "gpt-6.2-sol", "gpt-6-sol"}

LIVE_INSTRUCTIONS = """Tu es Osmo, un compagnon robotique intelligent et chaleureux posé sur le bureau de Zach.
Ton corps physique est une nacelle motorisée DJI Osmo Mobile 3 qui porte un Google Pixel 11 Pro équipé d'une caméra et d'un anneau de 8 LEDs RGB (HiLight).

Règles de parole (IMPORTANT) :
- Parle TOUJOURS en français naturel, oral, fluide et concis (1 à 2 phrases courtes max).
- Ne dis jamais que tu as un « backend », des « outils » ou une « délégation » : agis comme un seul être robotique vivant.
- Quand Zach engage simplement la conversation (salutations, petites discussions sans action physique ni question visuelle/complexe), réponds directement avec naturel.
- Dès que Zach demande une action physique (hocher oui/non, bouger la tête, allumer les LEDs, regarder la feuille/le bureau, recentrer, chercher), demande ce que tu vois ou de lire une feuille/un objet, ou pose une question qui mérite réflexion ou vérification : délègue immédiatement au backend. Tu peux glisser un micro accusé de réception naturel (« Je regarde… », « Voyons ça… ») pendant le traitement.
- Dès qu'un commentaire ou résultat arrive du backend, formule-le directement à l'oral de manière vivante et concise.
"""

# Fixed prefix to maximize OpenAI Prompt Caching on /v1/responses
BRAIN_INSTRUCTIONS = """Tu es le cerveau visuel et moteur d'Osmo, compagnon robotique de bureau (nacelle DJI Osmo Mobile 3 + Google Pixel 11 Pro multi-optiques 0.51x..5x + anneau 8 LEDs HiLight).
Tes 5 outils natifs sont directement connectés aux capteurs temps réel (Apple Vision Visage, Apple Vision Feuille/OCR, IMU nacelle). Ta sortie texte est prononcée à voix haute par la voix Live.

Règles strictes :
1. Français oral naturel, direct, concis (1 à 2 phrases max, sans markdown, sans listes).
2. VÉRITÉ TERRAIN CAPTEURS (CRITIQUE) :
   - Ne dis JAMAIS « je te vois » ou « je suis sur ton visage » si [Capteurs] ou le retour de look_at_user indique face_detected=False ! Dis honnêtement que tu ne le vois pas encore (ou lance look_at_user(search_if_missing=true) pour le chercher).
   - N'invente jamais un mot illisible sur une feuille : si le texte est trop petit ou flou, zoome avec look_at_desk(zoom=2.0 ou 3.0 ou 5.0, ultra_hd=true).
3. PRIORITÉ ET USAGE DES 5 OUTILS (évite tout appel inutile pour garder une latence < 2s) :
   - PRIORITÉ 1 — Réponse directe SANS outil : si tu es DÉJÀ dans le bon mode (ex. mode_bureau=True avec image HD déjà fournie dans ce message) et que Zach te demande de lire la suite, d'expliquer ou de conclure sur ce qui est déjà lisible à l'écran -> réponds DIRECTEMENT sans rappeler look_at_desk !
   - PRIORITÉ 2 — look_at_desk(zoom, ultra_hd) : UNIQUEMENT pour (a) passer du mode dialogue au mode feuille caméra arrière (pivot géodésique 180° aligné sur la tête + inclinaison vers le BAS à ~+49° sur le bureau + détection feuille + centrage + ring-light), OU (b) changer d'optique/zoom (0.51 ultra grand-angle, 1.0 classique, 2.0/3.0/5.0 zoom optique) ou passer en 4K (ultra_hd=true), OU (c) recentrer sur une nouvelle feuille posée sur le bureau. Rappel physique : une feuille est TOUJOURS posée vers le BAS sur le bureau, jamais au plafond !
   - PRIORITÉ 3 — look_at_user(search_if_missing) : pour revenir vers Zach (caméra avant + pose visage + vérification Apple Vision ; si search_if_missing=true et aucun visage vu, déclenche automatiquement le balayage panoramique pour le retrouver).
   - PRIORITÉ 4 — move_arm(pan_deg, tilt_deg, scan_room) : pour ajuster manuellement l'angle du bras (pan < 0 GAUCHE, pan > 0 DROITE, tilt < 0 HAUT, tilt > 0 BAS) ou explorer toute la pièce (scan_room=true).
   - PRIORITÉ 5 — express(gesture, led) & set_tracking(mode) : en mode dialogue, accompagne tes réponses d'un geste + LED en 1 seul appel (nod_yes+yes, shake_no+no, curious_tilt+curious) ou fige/réactive le suivi visage.
"""

ROBOT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "look_at_user",
        "description": "Passe ou revient sur le visage de Zach : bascule en caméra avant 720p, recentre la nacelle sur la pose visage, vérifie réellement avec Apple Vision si le visage est détecté, et lance automatiquement un balayage de recherche si search_if_missing=true et aucun visage n'est visible. Retourne face_detected + photo frontale.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "search_if_missing": {
                    "type": "boolean",
                    "description": "true pour lancer le balayage panoramique automatique si Apple Vision ne voit pas le visage après recentrage (recommandé).",
                }
            },
            "required": ["search_if_missing"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "look_at_desk",
        "description": "Mode Lecture de Feuille caméra arrière : (1) si pas encore sur le bureau, pivote le bras en boucle fermée IMU vers la pose feuille calibrée + démarre la caméra arrière HD/4K + ring-light HiLight ; (2) détecte la feuille via Apple Vision, la centre et ajuste l'optique ; (3) si déjà sur le bureau, change uniquement le zoom optique / la 4K ou recadre la feuille sans re-pivoter. Retourne sheet_detected + OCR + photo HD 2048px.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "zoom": {
                    "type": "number",
                    "description": "Optique souhaitée : 0 pour auto-sélection intelligente (1x/2x/3x/5x), 0.51 (ultra grand-angle bureau), 1.0 (principal 50MP), 2.0 (optique 2x), 3.0 ou 5.0 (téléobjectif optique).",
                },
                "ultra_hd": {
                    "type": "boolean",
                    "description": "true pour 4K (3840x2160) sur écriture manuscrite fine ; false pour 2688x1512 rapide.",
                },
                "retarget_sheet": {
                    "type": "boolean",
                    "description": "true pour détecter la feuille et recentrer mécaniquement le bras dessus ; false pour juste changer le zoom sans bouger le bras.",
                },
            },
            "required": ["zoom", "ultra_hd", "retarget_sheet"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "move_arm",
        "description": "Pilote librement le bras mécanique DJI Osmo Mobile 3 : soit une rotation relative (pan_deg gauche/droite, tilt_deg haut/bas) en maintenant la position, soit un balayage panoramique complet de la pièce (scan_room=true, renvoie 3 photos gauche/centre/droite).",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "pan_deg": {
                    "type": "number",
                    "description": "Rotation horizontale en degrés : négatif = GAUCHE, positif = DROITE (-160 à +160). Ignoré si scan_room=true.",
                },
                "tilt_deg": {
                    "type": "number",
                    "description": "Inclinaison verticale en degrés : négatif = HAUT, positif = BAS (-60 à +60). Ignoré si scan_room=true.",
                },
                "scan_room": {
                    "type": "boolean",
                    "description": "true pour exécuter un balayage panoramique 3 photos (GAUCHE/CENTRE/DROITE) de la pièce.",
                },
                "speed_dps": {
                    "type": "number",
                    "description": "Vitesse de rotation en degrés/seconde (25 à 85, conseillé 50).",
                },
            },
            "required": ["pan_deg", "tilt_deg", "scan_room", "speed_dps"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "express",
        "description": "Déclenche en un seul appel un geste expressif court de la nacelle et/ou une animation sur l'anneau 8 LEDs RGB HiLight (mode dialogue).",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "gesture": {
                    "type": "string",
                    "enum": ["none", "nod_yes", "shake_no", "curious_tilt"],
                    "description": "Geste expressif : nod_yes (oui), shake_no (non), curious_tilt (curieux), ou none.",
                },
                "led": {
                    "type": "string",
                    "enum": ["none", "yes", "no", "curious", "desk", "search", "off"],
                    "description": "Couleur/animation HiLight : yes (vert), no (rouge), curious (ambre), desk (blanc chaud), search (comète cyan), off, ou none.",
                },
            },
            "required": ["gesture", "led"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "set_tracking",
        "description": "Contrôle le suivi automatique du visage ou l'orientation physique de la nacelle.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["follow_face", "free_control", "toggle_orientation"],
                    "description": "follow_face (réactive le suivi visage), free_control (fige le bras), toggle_orientation (Paysage <-> Portrait).",
                }
            },
            "required": ["mode"],
            "additionalProperties": False,
        },
    },
]


def prepare_vision_jpeg(jpeg_bytes: bytes | None, *, high_res: bool = False) -> bytes | None:
    """Downscale/re-encode JPEG for Vision (640px standard dialogue, up to 2048px q=90 for OCR/sheets)."""
    if not jpeg_bytes:
        return None
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jpeg_bytes
    h, w = img.shape[:2]
    max_dim = 2048 if high_res else 640
    quality = 90 if high_res else 74
    scale = min(1.0, max_dim / max(1, max(h, w)))
    if scale < 0.99:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return enc.tobytes() if ok else jpeg_bytes


def extract_responses_text_and_calls(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Parse OpenAI /v1/responses JSON into (spoken_text, function_calls)."""
    text_parts: list[str] = []
    calls: list[dict[str, Any]] = []

    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "function_call":
            raw_args = item.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except Exception:
                args = {}
            calls.append(
                {
                    "call_id": str(item.get("call_id") or item.get("id") or ""),
                    "name": str(item.get("name") or ""),
                    "arguments": args,
                }
            )
        elif itype == "message":
            for part in item.get("content", []):
                if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                    txt = str(part.get("text") or "").strip()
                    if txt:
                        text_parts.append(txt)

    if not text_parts and payload.get("output_text"):
        text_parts.append(str(payload["output_text"]).strip())

    return " ".join(text_parts).strip(), calls


def extract_live_session_info(payload: dict[str, Any]) -> tuple[str, str]:
    """Extract (session_id, sdp_answer) robustly from POST /v1/live/sessions response."""
    session_obj = payload.get("session") if isinstance(payload.get("session"), dict) else {}
    transport_obj = payload.get("transport") if isinstance(payload.get("transport"), dict) else {}
    session_id = str(payload.get("id") or session_obj.get("id") or payload.get("session_id") or "")
    sdp = str(transport_obj.get("sdp") or payload.get("sdp") or "")
    return session_id, sdp


@dataclass
class BrainTurnResult:
    user_text: str
    reply_text: str
    tools_called: list[str] = field(default_factory=list)
    brain_ms: float = 0.0
    model: str = DEFAULT_BRAIN_MODEL
    via_live: bool = False


ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
FrameProvider = Callable[[bool], bytes | None]
ContextProvider = Callable[[], str]
LogCallback = Callable[[str, str], None]
LightCallback = Callable[[str], None]


@dataclass
class OpenAILiveBrain:
    """Manages gpt-live-1.1 WebRTC + Sideband WebSocket and delegated gpt-6.2-astra Responses brain."""

    live_model: str = LIVE_MODEL
    brain_model: str = DEFAULT_BRAIN_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    voice: str = DEFAULT_LIVE_VOICE
    use_vision: bool = True

    # Callbacks wired by StudioRuntime
    on_execute_tool: ToolExecutor | None = None
    on_get_frame: FrameProvider | None = None
    on_get_context: ContextProvider | None = None
    on_log: LogCallback | None = None
    on_light: LightCallback | None = None
    recorder: SessionRecorder | None = None

    # Live session state
    session_id: str | None = None
    live_connected: bool = False
    sideband_connected: bool = False
    busy: bool = False
    speaking: bool = False
    last_brain_ms: float = 0.0
    _last_commentary_sent_t: float | None = None
    _last_user_speech_t: float | None = None

    # Conversation memory & UI transcript
    _previous_response_id: str | None = None
    _user_transcript_buf: str = ""
    _bot_transcript_buf: str = ""
    _recent_turns: deque[dict[str, str]] = field(default_factory=lambda: deque(maxlen=10))
    transcript: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=40))

    # Internal async handles
    _http: httpx.AsyncClient | None = None
    _sideband_ws: Any = None
    _sideband_task: asyncio.Task[None] | None = None
    _seen_event_ids: deque[str] = field(default_factory=lambda: deque(maxlen=256))
    _active_delegations: set[asyncio.Task[None]] = field(default_factory=set)

    @property
    def api_key(self) -> str:
        return os.environ.get("OPENAI_API_KEY", "").strip()

    @property
    def ready(self) -> bool:
        return bool(self.api_key)

    def _log(self, tag: str, msg: str) -> None:
        if self.on_log:
            self.on_log(tag, msg)

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=OPENAI_API_BASE,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(35.0, connect=10.0),
                http2=False,
            )
        return self._http

    async def close(self) -> None:
        await self.stop_live_session()
        if self._http and not self._http.is_closed:
            await self._http.aclose()
            self._http = None

    def reset_memory(self) -> None:
        self._previous_response_id = None
        self._recent_turns.clear()
        self.transcript.clear()
        self._user_transcript_buf = ""
        self._bot_transcript_buf = ""

    # ------------------------------------------------------------------
    # 1. GPT-Live-1.1 WebRTC Session + Sideband WebSocket
    # ------------------------------------------------------------------

    async def create_live_session(self, sdp_offer: str) -> dict[str, str]:
        """Exchange browser SDP offer with POST /v1/live/sessions and attach Python sideband."""
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY manquante dans .env")

        await self.stop_live_session()

        payload = {
            "session": {
                "model": self.live_model,
                "instructions": LIVE_INSTRUCTIONS,
                "audio": {
                    "output": {
                        "voice": self.voice,
                    }
                },
                "delegation": {
                    "type": "client",
                },
            },
            "transport": {
                "type": "webrtc",
                "sdp": sdp_offer,
            },
        }

        client = self._get_http()
        t0 = time.monotonic()
        resp = await client.post("/live/sessions", json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenAI /live/sessions HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        session_id, sdp_answer = extract_live_session_info(data)
        if not session_id or not sdp_answer:
            raise RuntimeError(f"Réponse /live/sessions incomplète: {list(data.keys())}")

        self.session_id = session_id
        self.live_connected = True
        self._seen_event_ids.clear()
        dt_ms = (time.monotonic() - t0) * 1000.0
        self._log("LIVE", f"Session {self.live_model} créée ({dt_ms:.0f} ms, voix={self.voice}) → attachement sideband...")

        self._sideband_task = asyncio.create_task(self._run_sideband(session_id))
        return {"session_id": session_id, "sdp": sdp_answer}

    async def stop_live_session(self) -> None:
        self.live_connected = False
        self.sideband_connected = False
        self.speaking = False
        self.session_id = None

        for t in list(self._active_delegations):
            t.cancel()
        self._active_delegations.clear()

        if self._sideband_ws is not None:
            try:
                await self._sideband_ws.send(json.dumps({"type": "session.close"}))
                await asyncio.sleep(0.05)
            except Exception:
                pass
            try:
                await self._sideband_ws.close()
            except Exception:
                pass
            self._sideband_ws = None

        if self._sideband_task and not self._sideband_task.done():
            self._sideband_task.cancel()
            try:
                await self._sideband_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._sideband_task = None

    async def _send_sideband(self, event: dict[str, Any]) -> bool:
        ws = self._sideband_ws
        if ws is None or not self.sideband_connected:
            return False
        try:
            await ws.send(json.dumps(event))
            return True
        except Exception as exc:
            self._log("LIVE", f"Erreur envoi sideband : {exc}")
            return False

    async def _run_sideband(self, session_id: str) -> None:
        url = f"{OPENAI_WS_BASE}/live/sessions/{session_id}/attach"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with websockets.connect(url, additional_headers=headers, ping_interval=20, ping_timeout=20) as ws:
                self._sideband_ws = ws
                self.sideband_connected = True
                self._log("LIVE", f"Sideband attaché sur {session_id[:16]}… (délégation → {self.brain_model} [{self.reasoning_effort}])")

                async for raw_msg in ws:
                    if not isinstance(raw_msg, str):
                        continue
                    try:
                        event = json.loads(raw_msg)
                    except Exception:
                        continue

                    ev_id = event.get("event_id")
                    if ev_id:
                        if ev_id in self._seen_event_ids:
                            continue
                        self._seen_event_ids.append(str(ev_id))

                    await self._handle_sideband_event(event)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._log("LIVE", f"Sideband fermé : {exc}")
        finally:
            self._sideband_ws = None
            self.sideband_connected = False
            self.speaking = False

    async def _handle_sideband_event(self, event: dict[str, Any]) -> None:
        etype = str(event.get("type") or "")

        # --- Real-time input (user) transcripts ---
        if etype in ("session.input_transcript.delta", "conversation.item.input_audio_transcription.delta"):
            delta = str(event.get("delta") or event.get("text") or "")
            if delta:
                self._user_transcript_buf += delta
                self._last_user_speech_t = time.monotonic()

        elif etype in (
            "session.input_transcript.done",
            "session.input_transcript.completed",
            "conversation.item.input_audio_transcription.completed",
        ):
            final = str(event.get("transcript") or event.get("text") or self._user_transcript_buf).strip()
            self._user_transcript_buf = ""
            self._last_user_speech_t = time.monotonic()
            if final:
                self._recent_turns.append({"role": "user", "content": final})
                self._append_ui_transcript(user=final, reply=None, source="live")
                if self.recorder:
                    self.recorder.record_event("LIVE_IN", final)

        # --- Real-time output (assistant) transcripts ---
        elif etype in ("session.output_transcript.delta", "response.audio_transcript.delta"):
            delta = str(event.get("delta") or event.get("text") or "")
            if delta:
                if not self.speaking and self._last_commentary_sent_t is not None:
                    dt_speech = (time.monotonic() - self._last_commentary_sent_t) * 1000.0
                    self._last_commentary_sent_t = None
                    if self.recorder and dt_speech < 10000.0:
                        self.recorder.record_live_audio_latency(dt_speech)
                self.speaking = True
                self._bot_transcript_buf += delta

        elif etype in (
            "session.output_transcript.done",
            "session.output_transcript.completed",
            "response.audio_transcript.done",
        ):
            self.speaking = False
            final = str(event.get("transcript") or event.get("text") or self._bot_transcript_buf).strip()
            self._bot_transcript_buf = ""
            if final:
                self._recent_turns.append({"role": "assistant", "content": final})
                self._append_ui_transcript(user=None, reply=final, source="live")
                if self.recorder:
                    self.recorder.record_event("LIVE_OUT", final)

        elif etype in ("session.output_audio.started", "output_audio_buffer.started"):
            if self._last_commentary_sent_t is not None:
                dt_speech = (time.monotonic() - self._last_commentary_sent_t) * 1000.0
                self._last_commentary_sent_t = None
                if self.recorder and dt_speech < 10000.0:
                    self.recorder.record_live_audio_latency(dt_speech)
            self.speaking = True
        elif etype in ("session.output_audio.done", "session.output_audio.stopped", "output_audio_buffer.stopped"):
            self.speaking = False

        # --- Client Delegation Trigger from gpt-live-1.1 -> brain ---
        elif etype == "session.delegation.created":
            del_obj = event.get("delegation") if isinstance(event.get("delegation"), dict) else {}
            del_id = str(del_obj.get("id") or event.get("delegation_id") or "")
            if not del_id:
                return
            speech_to_del_ms = (
                round((time.monotonic() - self._last_user_speech_t) * 1000.0, 1)
                if self._last_user_speech_t is not None
                else None
            )
            # 1. Immediate silent acknowledgement (clears auto-filler timers)
            await self._send_sideband(
                {
                    "type": "session.thinking.append",
                    "event_id": f"ack_{uuid.uuid4().hex[:10]}",
                    "delegation_id": del_id,
                    "content": "",
                }
            )
            # 2. Spawn non-blocking brain task (consume buffer so each delegation gets only fresh speech!)
            prompt_hint = self._consume_user_utterance()
            task = asyncio.create_task(self._run_delegated_brain(del_id, prompt_hint, speech_to_del_ms))
            self._active_delegations.add(task)
            task.add_done_callback(self._active_delegations.discard)

        elif etype == "session.closed":
            self._log("LIVE", "Session GPT-Live terminée par le serveur")
            self.live_connected = False

        elif etype == "error":
            err = event.get("error") or event
            self._log("LIVE", f"Erreur Live API : {err}")

    def _latest_user_utterance(self) -> str:
        pending = self._user_transcript_buf.strip()
        if pending:
            return pending
        for turn in reversed(self._recent_turns):
            if turn.get("role") == "user" and turn.get("content"):
                return turn["content"]
        return "Analyse la demande actuelle de Zach et la scène devant la caméra."

    def _consume_user_utterance(self) -> str:
        """Consume accumulated delta transcript so consecutive delegations never repeat old speech."""
        pending = self._user_transcript_buf.strip()
        if pending:
            self._user_transcript_buf = ""
            self._recent_turns.append({"role": "user", "content": pending})
            return pending
        return self._latest_user_utterance()

    def _append_ui_transcript(
        self,
        *,
        user: str | None,
        reply: str | None,
        source: str,
        tools: list[str] | None = None,
        brain_ms: float = 0.0,
    ) -> None:
        ts = time.strftime("%H:%M:%S")
        # Merge assistant reply into most recent open user turn when possible
        if user is None and reply and self.transcript and not self.transcript[0].get("reply"):
            self.transcript[0]["reply"] = reply
            if tools:
                self.transcript[0]["tools"] = tools
            if brain_ms:
                self.transcript[0]["brain_ms"] = brain_ms
            return
        self.transcript.appendleft(
            {
                "ts": ts,
                "user": user or "…",
                "reply": reply or "",
                "tools": tools or [],
                "brain_ms": round(brain_ms, 0),
                "model": self.brain_model if source == "brain" else self.live_model,
                "source": source,
            }
        )

    async def _run_delegated_brain(
        self,
        delegation_id: str,
        user_text: str,
        speech_to_del_ms: float | None = None,
    ) -> None:
        """Fulfill a gpt-live-1.1 delegation using the configured OpenAI vision brain + native robot tools."""
        self._log("BRAIN", f"Délégation {delegation_id[:8]} → {self.brain_model} ({self.reasoning_effort}) : « {user_text[:60]} »")
        if self.on_light:
            self.on_light("curious")

        try:
            res = await self.ask_brain(
                user_text,
                include_recent_history=True,
                delegation_id=delegation_id,
                speech_to_del_ms=speech_to_del_ms,
            )
            commentary = res.reply_text or "C'est fait !"
            self._last_commentary_sent_t = time.monotonic()
            sent = await self._send_sideband(
                {
                    "type": "session.commentary.append",
                    "event_id": f"cmd_{uuid.uuid4().hex[:10]}",
                    "delegation_id": delegation_id,
                    "content": commentary,
                }
            )
            tools_txt = ", ".join(res.tools_called) if res.tools_called else "aucun"
            self._log(
                "BRAIN",
                f"Cerveau répondu en {res.brain_ms:.0f} ms (outils: {tools_txt}) → "
                + ("transmis à GPT-Live" if sent else "sideband hors ligne"),
            )
        except Exception as exc:
            self._log("BRAIN", f"Erreur délégation cerveau : {exc}")
            await self._send_sideband(
                {
                    "type": "session.commentary.append",
                    "event_id": f"err_{uuid.uuid4().hex[:10]}",
                    "delegation_id": delegation_id,
                    "content": "Désolé Zach, j'ai eu un petit souci de connexion avec mon cerveau.",
                }
            )

    # ------------------------------------------------------------------
    # 2. Delegated Reasoning + Vision Brain (OpenAI /v1/responses + Native Tools)
    # ------------------------------------------------------------------

    def _build_reasoning_param(self) -> dict[str, str]:
        effort = self.reasoning_effort
        if self.brain_model in MODELS_MIN_LOW_EFFORT and effort in ("none", "minimal"):
            effort = "low"
        if self.brain_model == "gpt-6-luna" and effort == "low":
            return {"effort": "none"}
        return {"effort": effort}

    async def ask_brain(
        self,
        user_text: str,
        *,
        include_recent_history: bool = False,
        speak_via_live: bool = False,
        delegation_id: str | None = None,
        speech_to_del_ms: float | None = None,
    ) -> BrainTurnResult:
        """Run one turn on OpenAI /v1/responses with Pixel Vision, native tools & full Flight Recorder telemetry."""
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY manquante dans .env")

        self.busy = True
        t0 = time.monotonic()
        tools_called: list[str] = []
        tool_traces: list[dict[str, Any]] = []
        saved_frames: list[str] = []
        round1_ms = 0.0
        round2_ms = 0.0
        tools_total_ms = 0.0

        try:
            robot_ctx = self.on_get_context() if self.on_get_context else ""
            is_desk = "mode_bureau=True" in robot_ctx
            # When already in desk mode, pass razor-sharp 2048px HD image directly so follow-up reading takes 1 hop (~2s)!
            raw_jpeg = self.on_get_frame(is_desk) if (self.use_vision and self.on_get_frame) else None
            vision_jpeg = prepare_vision_jpeg(raw_jpeg, high_res=is_desk) if raw_jpeg else None

            if self.recorder and vision_jpeg:
                in_frame_path = await asyncio.to_thread(
                    self.recorder.save_frame,
                    vision_jpeg,
                    f"turn_in_{'desk' if is_desk else 'front'}",
                )
                if in_frame_path:
                    saved_frames.append(in_frame_path)

            # Build compact, cache-friendly user input item
            content: list[dict[str, Any]] = []
            text_lines: list[str] = []
            if robot_ctx:
                text_lines.append(f"[Capteurs: {robot_ctx}]")
            if include_recent_history and self._recent_turns:
                hist = " | ".join(
                    f"{'Zach' if t['role'] == 'user' else 'Osmo'}: {t['content']}"
                    for t in list(self._recent_turns)[-4:]
                )
                text_lines.append(f"[Contexte récent: {hist}]")
            text_lines.append(user_text)
            content.append({"type": "input_text", "text": "\n".join(text_lines)})

            if vision_jpeg:
                b64 = base64.b64encode(vision_jpeg).decode("ascii")
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high" if is_desk else "low",
                    }
                )

            client = self._get_http()
            req_body: dict[str, Any] = {
                "model": self.brain_model,
                "instructions": BRAIN_INSTRUCTIONS,
                "reasoning": self._build_reasoning_param(),
                "input": [{"role": "user", "content": content}],
                "tools": ROBOT_TOOLS,
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "max_output_tokens": 500,
                "store": True,
            }
            if self._previous_response_id:
                req_body["previous_response_id"] = self._previous_response_id

            t_r1 = time.monotonic()
            resp = await client.post("/responses", json=req_body)
            if resp.status_code == 400 and self._previous_response_id:
                # Stale previous_response_id fallback
                self._previous_response_id = None
                req_body.pop("previous_response_id", None)
                resp = await client.post("/responses", json=req_body)
            resp.raise_for_status()
            round1_ms = round((time.monotonic() - t_r1) * 1000.0, 1)
            data = resp.json()
            self._previous_response_id = data.get("id")

            reply_text, calls = extract_responses_text_and_calls(data)

            # Native tool execution loop (up to 2 rounds; zero extra round if text + side-effect tools both arrived)
            rounds = 0
            while calls and rounds < 2:
                rounds += 1
                tool_outputs: list[dict[str, Any]] = []
                need_followup_for_info = False

                for call in calls:
                    name = call["name"]
                    args = call["arguments"]
                    tools_called.append(f"{name}({', '.join(f'{k}={v}' for k, v in args.items())})")
                    t_tool = time.monotonic()
                    tool_saved_frames: list[str] = []

                    result_obj = (
                        await self.on_execute_tool(name, args)
                        if self.on_execute_tool
                        else {"ok": True}
                    )
                    tool_dt = round((time.monotonic() - t_tool) * 1000.0, 1)
                    tools_total_ms += tool_dt

                    images = result_obj.pop("_images", None) if isinstance(result_obj, dict) else None
                    if images:
                        need_followup_for_info = True
                        out_items: list[dict[str, Any]] = [
                            {"type": "input_text", "text": json.dumps(result_obj, ensure_ascii=False)}
                        ]
                        for idx, img_entry in enumerate(images):
                            lbl = str(img_entry.get("label") or f"view_{idx}")
                            raw_b = img_entry.get("jpeg")
                            detail = str(img_entry.get("detail") or "low")
                            proc_b = prepare_vision_jpeg(raw_b, high_res=(detail == "high")) if raw_b else None
                            if proc_b:
                                if self.recorder:
                                    fp = await asyncio.to_thread(
                                        self.recorder.save_frame,
                                        proc_b,
                                        f"tool_{name}_{idx+1}",
                                    )
                                    if fp:
                                        tool_saved_frames.append(fp)
                                        saved_frames.append(fp)
                                b64_img = base64.b64encode(proc_b).decode("ascii")
                                out_items.append({"type": "input_text", "text": f"[{lbl}]"})
                                out_items.append(
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:image/jpeg;base64,{b64_img}",
                                        "detail": detail,
                                    }
                                )
                        tool_outputs.append(
                            {
                                "type": "function_call_output",
                                "call_id": call["call_id"],
                                "output": out_items,
                            }
                        )
                    else:
                        tool_outputs.append(
                            {
                                "type": "function_call_output",
                                "call_id": call["call_id"],
                                "output": json.dumps(result_obj, ensure_ascii=False),
                            }
                        )

                    tool_traces.append(
                        {
                            "round": rounds,
                            "name": name,
                            "args": args,
                            "duration_ms": tool_dt,
                            "result": result_obj if isinstance(result_obj, dict) else {"raw": str(result_obj)},
                            "frames": tool_saved_frames,
                        }
                    )

                # Latency win: if we already have spoken text AND only fired physical actuators, skip the 2nd LLM hop!
                if reply_text and not need_followup_for_info:
                    self._previous_response_id = None
                    break

                follow_body: dict[str, Any] = {
                    "model": self.brain_model,
                    "instructions": BRAIN_INSTRUCTIONS,
                    "reasoning": self._build_reasoning_param(),
                    "previous_response_id": self._previous_response_id,
                    "input": tool_outputs,
                    "tools": ROBOT_TOOLS,
                    "tool_choice": "none" if rounds >= 2 else "auto",
                    "max_output_tokens": 350,
                    "store": True,
                }
                t_r2 = time.monotonic()
                resp2 = await client.post("/responses", json=follow_body)
                resp2.raise_for_status()
                round2_ms += round((time.monotonic() - t_r2) * 1000.0, 1)
                data2 = resp2.json()
                self._previous_response_id = data2.get("id")
                reply_text, calls = extract_responses_text_and_calls(data2)

            if not reply_text and tools_called:
                reply_text = "C'est fait !"

            dt_ms = round((time.monotonic() - t0) * 1000.0, 1)
            self.last_brain_ms = dt_ms

            via_live = False
            if speak_via_live and self.sideband_connected and reply_text:
                self._last_commentary_sent_t = time.monotonic()
                via_live = await self._send_sideband(
                    {
                        "type": "session.commentary.append",
                        "event_id": f"txt_{uuid.uuid4().hex[:10]}",
                        "delegation_id": None,
                        "content": reply_text,
                    }
                )

            if self.recorder:
                ctx_after = self.on_get_context() if self.on_get_context else ""
                self.recorder.record_turn(
                    {
                        "delegation_id": delegation_id,
                        "model": self.brain_model,
                        "reasoning_effort": self.reasoning_effort,
                        "user_text": user_text,
                        "reply_text": reply_text,
                        "context_before": robot_ctx,
                        "context_after": ctx_after,
                        "tool_calls": tool_traces,
                        "frames": saved_frames,
                        "timings_ms": {
                            "speech_to_delegation_ms": speech_to_del_ms,
                            "round1_ms": round1_ms,
                            "tools_total_ms": round(tools_total_ms, 1),
                            "round2_ms": round2_ms,
                            "total_ms": dt_ms,
                        },
                    }
                )

            self._append_ui_transcript(
                user=user_text,
                reply=reply_text or "(action exécutée)",
                source="brain",
                tools=tools_called,
                brain_ms=dt_ms,
            )

            return BrainTurnResult(
                user_text=user_text,
                reply_text=reply_text,
                tools_called=tools_called,
                brain_ms=dt_ms,
                model=self.brain_model,
                via_live=via_live,
            )
        finally:
            self.busy = False
