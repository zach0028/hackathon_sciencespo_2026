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

load_dotenv(ENV_FILE)

OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_WS_BASE = "wss://api.openai.com/v1"

LIVE_MODEL = "gpt-live-1.1"
DEFAULT_BRAIN_MODEL = "gpt-6.2-astra"
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
    {"id": "gpt-6.2-astra", "label": "GPT-6.2 Astra (Vision + Raisonnement)"},
    {"id": "gpt-6-astra", "label": "GPT-6 Astra (Frontier Vision)"},
    {"id": "gpt-6.2-sol", "label": "GPT-6.2 Sol (~1,6s Rapide)"},
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
BRAIN_INSTRUCTIONS = """Tu es le cerveau visuel et moteur d'Osmo, un robot compagnon de bureau doté d'un bras mécanique motorisé (DJI Osmo Mobile 3 portant un Google Pixel 11 Pro avec caméras multi-optiques 0.51x..5x et anneau 8 LEDs RGB HiLight).
Tu pilotes RÉELLEMENT ce bras mécanique et ses optiques en temps réel grâce à tes outils natifs. Ta sortie texte sera immédiatement prononcée à voix haute par la voix temps réel d'Osmo.

Règles strictes :
1. Réponds TOUJOURS en français oral naturel, direct et concis (1 à 2 phrases max, sans markdown, sans astérisques, sans listes numérotées).
2. Deux modes majeurs — utilise TOUJOURS l'outil macro unique (jamais plusieurs petits outils à la place !) :
   - MODE LECTURE DE FEUILLE / BUREAU ("regarde la feuille", "lis ce que j'ai écrit", "passe sur le bureau", "regarde mon carnet") :
     -> Appelle UNIQUEMENT enter_desk_reading_mode(zoom=0, auto_target=true, ultra_hd=false).
     Cet outil atomique pivote automatiquement le bras de 180°, incline la caméra arrière sur la feuille (pose calibrée +46°), allume l'anneau HiLight face à la feuille, détecte la feuille sur le bureau (en passant en 0.51x ultra grand-angle si besoin), la centre, choisit le meilleur zoom optique (1x/2x/3x/5x), et te renvoie la photo haute définition + l'OCR pré-détecté !
     Tu peux aussi préciser zoom=1.0 (angle classique), zoom=2.0, zoom=3.0 ou zoom=5.0 (téléobjectif optique), ou ultra_hd=true (4K) pour de très petites écritures.
   - ZOOM OPTIQUE EN MODE FEUILLE ("zoome sur le texte", "passe en 2x / 5x", "dézoome en ultra grand-angle") :
     -> Appelle set_camera_zoom(zoom=..., capture_after=true). Optiques Pixel 11 Pro arrière : 0.51 (ultra grand-angle bureau entier), 1.0 (principal 50MP), 2.0 (zoom optique capteur), 3.0 / 5.0 (téléobjectif optique).
   - RETOUR MODE DIALOGUE ("regarde-moi", "remonte sur mon visage", "reviens en mode dialogue") :
     -> Appelle UNIQUEMENT return_to_face_mode(). Il repasse en caméra avant 720p, recentre la nacelle face à Zach, coupe le HiLight et réactive le face-tracking.
3. Pilotage libre du bras mécanique :
   - Pour tourner/ajuster la caméra en degrés relatifs -> move_gimbal(pan_deg, tilt_deg, speed_dps, hold_position=true, capture_after=true). En caméra arrière comme avant : pan_deg < 0 = GAUCHE, pan_deg > 0 = DROITE, tilt_deg < 0 = HAUT, tilt_deg > 0 = BAS.
   - Pour explorer toute la pièce de gauche à droite -> scan_room(span_deg=150, speed_dps=55, return_to_center=true).
   - Pour activer/couper le suivi visage -> set_tracking_mode(mode="follow_face"|"free_control").
4. Gestes expressifs & LEDs HiLight en dialogue :
   - Accord / oui -> trigger_gesture(gesture="nod_yes") et set_hilight(effect="yes")
   - Désaccord / non / erreur -> trigger_gesture(gesture="shake_no") et set_hilight(effect="no")
   - Curiosité / réflexion -> trigger_gesture(gesture="curious_tilt") et set_hilight(effect="curious")
   - Recentrer / Paysage↔Portrait -> control_gimbal(action="recenter"|"toggle_orientation")
5. Appuie-toi fidèlement sur les images haute définition capturées ; si un texte manuscrit est encore trop petit en 1x ou 2x, appelle set_camera_zoom(zoom=3.0 ou 5.0) pour le lire nettement au lieu d'abandonner !
"""

ROBOT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "enter_desk_reading_mode",
        "description": "Passe en mode Lecture de Feuille complet : pivot 180° du bras + inclinaison calibrée vers le bureau + caméra arrière HD/4K + détection automatique de feuille/bureau (0.51x..5x) + centrage + ring-light HiLight + capture HD immédiate.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "zoom": {
                    "type": "number",
                    "description": "Zoom optique souhaité : 0 pour auto-sélection intelligente selon la taille de la feuille (recommandé), 0.51 (ultra grand-angle), 1.0 (classique), 2.0, 3.0 ou 5.0 (téléobjectif optique).",
                },
                "auto_target": {
                    "type": "boolean",
                    "description": "true pour détecter automatiquement la feuille sur le bureau, la centrer mécaniquement et choisir la meilleure optique.",
                },
                "ultra_hd": {
                    "type": "boolean",
                    "description": "true pour capturer en 4K (3840x2160) pour des notes manuscrites très fines ; false pour 2688x1512 rapide.",
                },
            },
            "required": ["zoom", "auto_target", "ultra_hd"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "set_camera_zoom",
        "description": "Change l'optique / zoom natif de la caméra Pixel 11 Pro (arrière : 0.51x ultra grand-angle, 1.0x principal, 2.0x sans perte, 3.0x/5.0x téléobjectif optique) et capture immédiatement une photo HD.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "zoom": {
                    "type": "number",
                    "description": "Facteur de zoom optique : 0.51, 1.0, 2.0, 3.0, 5.0 (max 10.0).",
                },
                "capture_after": {
                    "type": "boolean",
                    "description": "true pour renvoyer immédiatement la photo HD prise avec cette nouvelle optique.",
                },
            },
            "required": ["zoom", "capture_after"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "return_to_face_mode",
        "description": "Quitte le mode feuille et revient en mode Dialogue : repasse sur la caméra avant 720p, recentre la nacelle face au visage de Zach, coupe le HiLight et réactive le suivi visage.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "move_gimbal",
        "description": "Pilote directement le bras mécanique DJI Osmo Mobile 3 en degrés relatifs (gauche/droite/haut/bas), suspend le suivi visage si hold_position=true, et capture la nouvelle vue caméra.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "pan_deg": {
                    "type": "number",
                    "description": "Rotation horizontale en degrés : négatif = GAUCHE (ex: -70), positif = DROITE (ex: +70). Plage : -160 à +160.",
                },
                "tilt_deg": {
                    "type": "number",
                    "description": "Inclinaison verticale en degrés : négatif = HAUT (ex: -25), positif = BAS (ex: +35). Plage : -60 à +60.",
                },
                "speed_dps": {
                    "type": "number",
                    "description": "Vitesse de rotation en degrés/seconde (entre 20 et 90, conseillé 50).",
                },
                "hold_position": {
                    "type": "boolean",
                    "description": "true pour maintenir le bras orienté sur la nouvelle position (met en pause le suivi visage), false pour laisser le suivi visage reprendre après.",
                },
                "capture_after": {
                    "type": "boolean",
                    "description": "true pour capturer et analyser la nouvelle image caméra juste après le mouvement.",
                },
            },
            "required": ["pan_deg", "tilt_deg", "speed_dps", "hold_position", "capture_after"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "scan_room",
        "description": "Exécute un vrai balayage panoramique physique de la pièce (GAUCHE -> CENTRE -> DROITE) avec le bras mécanique, sans être interrompu par le visage, et renvoie 3 photos panoramiques pour décrire la pièce.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "span_deg": {
                    "type": "number",
                    "description": "Amplitude horizontale totale du balayage en degrés (entre 80 et 220, conseillé 150).",
                },
                "speed_dps": {
                    "type": "number",
                    "description": "Vitesse de balayage en degrés/seconde (entre 35 et 85, conseillé 55).",
                },
                "return_to_center": {
                    "type": "boolean",
                    "description": "true pour revenir au centre et reprendre le suivi de Zach après le balayage.",
                },
            },
            "required": ["span_deg", "speed_dps", "return_to_center"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "set_tracking_mode",
        "description": "Active le suivi automatique du visage (follow_face) ou passe le bras en contrôle libre IA fixe (free_control).",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["follow_face", "free_control"],
                    "description": "follow_face pour suivre le visage de Zach automatiquement, free_control pour laisser le bras immobile là où l'IA le place.",
                }
            },
            "required": ["mode"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "trigger_gesture",
        "description": "Exécute un geste physique expressif court sur la nacelle motorisée DJI Osmo Mobile 3.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "gesture": {
                    "type": "string",
                    "enum": ["nod_yes", "shake_no", "curious_tilt"],
                    "description": "Geste expressif en dialogue : nod_yes (oui), shake_no (non), curious_tilt (curieux).",
                }
            },
            "required": ["gesture"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "set_hilight",
        "description": "Anime l'anneau 8 LEDs RGB HiLight à l'arrière du Pixel 11 Pro.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "effect": {
                    "type": "string",
                    "enum": ["yes", "no", "curious", "desk", "search", "off"],
                    "description": "Effet lumineux : yes (vert), no (rouge), curious (ambre/violet), desk (blanc chaud feuille), search (comète cyan), off (éteint).",
                }
            },
            "required": ["effect"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "control_gimbal",
        "description": "Pilote une commande matérielle de la nacelle DJI Osmo Mobile 3.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["recenter", "toggle_orientation"],
                    "description": "recenter (recentrage matériel Paysage + reprise suivi), toggle_orientation (Paysage <-> Portrait).",
                }
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "inspect_camera_hd",
        "description": "Capture une nouvelle image haute résolution (jusqu'à 2048px) de la caméra Pixel active pour lire précisément du texte manuscrit ou un petit détail.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Ce que tu cherches à lire ou vérifier précisément sur l'image.",
                }
            },
            "required": ["reason"],
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

    # Live session state
    session_id: str | None = None
    live_connected: bool = False
    sideband_connected: bool = False
    busy: bool = False
    speaking: bool = False
    last_brain_ms: float = 0.0

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

        elif etype in (
            "session.input_transcript.done",
            "session.input_transcript.completed",
            "conversation.item.input_audio_transcription.completed",
        ):
            final = str(event.get("transcript") or event.get("text") or self._user_transcript_buf).strip()
            self._user_transcript_buf = ""
            if final:
                self._recent_turns.append({"role": "user", "content": final})
                self._append_ui_transcript(user=final, reply=None, source="live")

        # --- Real-time output (assistant) transcripts ---
        elif etype in ("session.output_transcript.delta", "response.audio_transcript.delta"):
            delta = str(event.get("delta") or event.get("text") or "")
            if delta:
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

        elif etype in ("session.output_audio.started", "output_audio_buffer.started"):
            self.speaking = True
        elif etype in ("session.output_audio.done", "session.output_audio.stopped", "output_audio_buffer.stopped"):
            self.speaking = False

        # --- Client Delegation Trigger from gpt-live-1.1 -> gpt-6.2-astra ---
        elif etype == "session.delegation.created":
            del_obj = event.get("delegation") if isinstance(event.get("delegation"), dict) else {}
            del_id = str(del_obj.get("id") or event.get("delegation_id") or "")
            if not del_id:
                return
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
            task = asyncio.create_task(self._run_delegated_brain(del_id, prompt_hint))
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

    async def _run_delegated_brain(self, delegation_id: str, user_text: str) -> None:
        """Fulfill a gpt-live-1.1 delegation using gpt-6.2-astra + Pixel Vision + native robot tools."""
        self._log("BRAIN", f"Délégation {delegation_id[:8]} → {self.brain_model} ({self.reasoning_effort}) : « {user_text[:60]} »")
        if self.on_light:
            self.on_light("curious")

        try:
            res = await self.ask_brain(user_text, include_recent_history=True)
            commentary = res.reply_text or "C'est fait !"
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
    ) -> BrainTurnResult:
        """Run one turn on OpenAI /v1/responses with Pixel Vision + native robot function calling."""
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY manquante dans .env")

        self.busy = True
        t0 = time.monotonic()
        tools_called: list[str] = []

        try:
            robot_ctx = self.on_get_context() if self.on_get_context else ""
            raw_jpeg = self.on_get_frame(False) if (self.use_vision and self.on_get_frame) else None
            is_desk = "mode_bureau=True" in robot_ctx
            vision_jpeg = prepare_vision_jpeg(raw_jpeg, high_res=is_desk) if raw_jpeg else None

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

            resp = await client.post("/responses", json=req_body)
            if resp.status_code == 400 and self._previous_response_id:
                # Stale previous_response_id fallback
                self._previous_response_id = None
                req_body.pop("previous_response_id", None)
                resp = await client.post("/responses", json=req_body)
            resp.raise_for_status()
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

                    if name == "inspect_camera_hd":
                        need_followup_for_info = True
                        hd_raw = self.on_get_frame(True) if self.on_get_frame else None
                        hd_jpeg = prepare_vision_jpeg(hd_raw, high_res=True) if hd_raw else None
                        if hd_jpeg:
                            b64_hd = base64.b64encode(hd_jpeg).decode("ascii")
                            tool_outputs.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": call["call_id"],
                                    "output": [
                                        {"type": "input_text", "text": "Image HD 1280px capturée à l'instant."},
                                        {
                                            "type": "input_image",
                                            "image_url": f"data:image/jpeg;base64,{b64_hd}",
                                            "detail": "high",
                                        },
                                    ],
                                }
                            )
                        else:
                            tool_outputs.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": call["call_id"],
                                    "output": json.dumps({"ok": False, "error": "Aucune image caméra disponible"}),
                                }
                            )
                    else:
                        result_obj = (
                            await self.on_execute_tool(name, args)
                            if self.on_execute_tool
                            else {"ok": True}
                        )
                        images = result_obj.pop("_images", None) if isinstance(result_obj, dict) else None
                        if images:
                            need_followup_for_info = True
                            out_items: list[dict[str, Any]] = [
                                {"type": "input_text", "text": json.dumps(result_obj, ensure_ascii=False)}
                            ]
                            for img_entry in images:
                                lbl = str(img_entry.get("label") or "Vue caméra")
                                raw_b = img_entry.get("jpeg")
                                detail = str(img_entry.get("detail") or "low")
                                proc_b = prepare_vision_jpeg(raw_b, high_res=(detail == "high")) if raw_b else None
                                if proc_b:
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

                # Latency win: if we already have spoken text AND only fired physical actuators, skip the 2nd LLM hop!
                if reply_text and not need_followup_for_info:
                    # Do not chain next turn off a response with unsubmitted tool outputs
                    self._previous_response_id = None
                    break

                follow_body: dict[str, Any] = {
                    "model": self.brain_model,
                    "instructions": BRAIN_INSTRUCTIONS,
                    "reasoning": self._build_reasoning_param(),
                    "previous_response_id": self._previous_response_id,
                    "input": tool_outputs,
                    "tools": ROBOT_TOOLS,
                    # On the last round, force a spoken text reply (no more tool calls -> never silent!)
                    "tool_choice": "none" if rounds >= 2 else "auto",
                    "max_output_tokens": 350,
                    "store": True,
                }
                resp2 = await client.post("/responses", json=follow_body)
                resp2.raise_for_status()
                data2 = resp2.json()
                self._previous_response_id = data2.get("id")
                reply_text, calls = extract_responses_text_and_calls(data2)

            # Safety net: if a 2nd round still emitted tool calls without spoken text, never leave Osmo silent
            if not reply_text and tools_called:
                reply_text = "C'est fait !"

            dt_ms = round((time.monotonic() - t0) * 1000.0, 0)
            self.last_brain_ms = dt_ms

            via_live = False
            if speak_via_live and self.sideband_connected and reply_text:
                via_live = await self._send_sideband(
                    {
                        "type": "session.commentary.append",
                        "event_id": f"txt_{uuid.uuid4().hex[:10]}",
                        "delegation_id": None,
                        "content": reply_text,
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
