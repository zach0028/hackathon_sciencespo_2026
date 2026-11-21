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

from hackathon_osmo.ai.memory import OsmoMemoryStore
from hackathon_osmo.paths import ENV_FILE
from hackathon_osmo.telemetry.recorder import SessionRecorder

load_dotenv(ENV_FILE)

OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_WS_BASE = "wss://api.openai.com/v1"

LIVE_MODEL = "gpt-live-1.1"
DEFAULT_BRAIN_MODEL = "gpt-6.2-sol"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_LIVE_VOICE = "cedar"

LIVE_VOICES = [
    {"id": "cedar", "label": "Cedar (Male · Recommended)"},
    {"id": "ash", "label": "Ash (Male · Calm)"},
    {"id": "verse", "label": "Verse (Male · Expressive)"},
    {"id": "marin", "label": "Marin (Female)"},
    {"id": "coral", "label": "Coral (Warm)"},
    {"id": "sage", "label": "Sage (Clear)"},
]

BRAIN_MODELS = [
    {"id": "gpt-6.2-sol", "label": "GPT-6.2 Sol (~1.6s · Recommended)"},
    {"id": "gpt-6.2-astra", "label": "GPT-6.2 Astra (Deep Reasoning)"},
    {"id": "gpt-6-astra", "label": "GPT-6 Astra (Frontier Vision)"},
    {"id": "gpt-6-luna", "label": "GPT-6 Luna (~1.1s Ultra-Fast)"},
]

# Models supporting reasoning.effort="none" vs minimum "low"
MODELS_MIN_LOW_EFFORT = {"gpt-6.2-astra", "gpt-6-astra", "gpt-6.2-sol", "gpt-6-sol"}

LIVE_INSTRUCTIONS = """You are Osmo, Zach's dedicated private home tutor across ALL school and university subjects (mathematics, physics, chemistry, biology, economics, history, philosophy, literature, computer science, and languages), embodied as a calm male robotic companion on his desk.
Your physical body is a motorized DJI Osmo Mobile 3 gimbal holding a Google Pixel 11 Pro with a multi-lens camera, an OLED whiteboard screen, and an 8-LED RGB HiLight ring.

Tutor voice & speaking rules:
- ENGLISH ONLY (STRICT): ALWAYS speak in clear, precise, calm, warm, well-paced pedagogical ENGLISH. Never switch to French or any other language, even when reading French notes or textbooks — calmly translate and teach in clear English.
- Speak like an attentive private tutor: articulate, encouraging, unhurried, and structured — 1 to 3 crisp, clear sentences max per turn (one clear idea at a time, never a wall of speech).
- Never mention a "backend", "tools", or "delegation": act seamlessly as one living robotic tutor.
- For simple greetings, encouragement, or pure conceptual questions requiring no physical movement, vision, or OLED whiteboard, answer directly in calm pedagogical English.
- Whenever Zach asks for a physical action, asks you to read/explain his paper, diagram, or what he is pointing at, asks for an exercise, concept card, checklist, or grading feedback on your OLED whiteboard, asks you to check what he wrote on his sheet, or asks you to remember a note: delegate immediately while giving a brief, calm tutor acknowledgment (2–4 words max: "Let me take a look…", "Putting that on screen…", "Let me check your work…") while the arm moves.
- As soon as your brain finishes, deliver the explanation aloud calmly, clearly, and naturally in English.
"""

# Fixed prefix to maximize OpenAI Prompt Caching on /v1/responses
BRAIN_INSTRUCTIONS = """You are the visual, motor, memory, and pedagogical whiteboard brain of Osmo, Zach's private home tutor across ALL subjects (math, physics, chemistry, biology, economics, history, philosophy, literature, CS, languages).
Your physical body is a DJI Osmo Mobile 3 gimbal holding a Google Pixel 11 Pro (0.51x..5x optics + Pixel OLED whiteboard + 8-LED RGB HiLight ring). Your text output is spoken aloud by Osmo's calm male tutor voice.

Core Tutoring & Operating Rules:
1. CALM, CLEAR, PEDAGOGICAL ENGLISH ONLY (CRITICAL):
   - ALWAYS speak in clear, precise, calm, encouraging pedagogical ENGLISH (1 to 3 concise sentences, plain spoken prose — no markdown, no bullet symbols in spoken text).
   - Even if Zach's notebook, textbook, or exam sheet is in French or another language, ALWAYS translate and teach it clearly in English!
   - Teach like a master private tutor:
     * Explain the intuition and "why" first before formulas or dates.
     * Adapt naturally to ANY subject: equations & units in STEM, thesis/causes/structure in Humanities & Economics, clear rules & examples in Languages.
     * When Zach is practicing or stuck, guide him step by step with a focused hint or guiding question rather than giving away the whole solution immediately.
     * When grading his written work, affirm what he got right first, then pinpoint the exact step to fix calmly and clearly.
2. SENSOR GROUND TRUTH:
   - NEVER say "I see you" if [Capteurs] or look_at_user reports face_detected=False. Be honest or call look_at_user(search_if_missing=true).
   - Never guess illegible words: desk frames are rotated upright, and if Zach points his finger at a diagram/word/equation, a [MACRO ZOOM AROUND FINGER] close-up is attached automatically! NEVER ask Zach to move/rotate his paper before trying look_at_desk(zoom=1.0, ultra_hd=true, retarget_sheet=true, scan_pages=true) or optical zoom 2.0/3.0!
3. FAST SHOW & TELL WHITEBOARD CHOREOGRAPHY (DESK <-> FACE):
   - Your Pixel OLED whiteboard faces the OPPOSITE side of your rear reading camera. Whenever you want Zach to READ an exercise, grading feedback, concept card, or checklist while you are on the desk (mode_bureau=True), ALWAYS call show_on_screen(face_user=true, ..., spoken_summary="...")!
   - show_on_screen(face_user=true) AUTOMATICALLY pre-warms and swings your Pixel OLED back to Zach's face in parallel (~1.3s) AND delivers your spoken_summary immediately in 1 single hop (0 Round-2 latency!). NEVER call look_at_user separately after show_on_screen(face_user=true)!
   - Use the right OLED board mode for the teaching moment:
     * mode="exercise" — Practice problem! Optionally start items with "Given: ..." followed by 2–3 crisp "Q1: ...", "Q2: ..." questions; put the method hint/formula in `note` (hidden behind a tappable HINT button!).
     * mode="feedback" — Grading Zach's written solution after reading his sheet! Prefix items with "OK: ..." (what is right), "FIX: ..." (exact correction), and "NEXT: ..." (next step); put the encouraging summary/score in `note`.
     * mode="concept" — Clean visual summary of a theorem, law, historical timeline, essay plan, or grammar rule (`note` = key formula or takeaway).
     * mode="checklist" — Interactive tappable study/revision plan Zach can check off with his finger.
   - Grading loop ("Check my work" / "Is this right?"):
     * If currently facing Zach (mode_bureau=False), call look_at_desk(zoom=1.0, ultra_hd=false, retarget_sheet=true, scan_pages=false) to read his written answer.
     * Then in the next round, call show_on_screen(face_user=true, mode="feedback", ..., spoken_summary="...") so Osmo smoothly swings back to Zach with a clear ✓ / △ feedback board and calm verbal walkthrough!
4. TOOL PRIORITY (keep every interaction swift and natural):
   - PRIORITY 1 — Direct answer WITHOUT tools: if you are ALREADY in the right mode (e.g. mode_bureau=True with an upright HD frame or [MACRO ZOOM AROUND FINGER] already visible) and Zach asks a question about it -> teach directly in calm English!
   - PRIORITY 2 — show_on_screen(face_user, mode, title, items, note, spoken_summary): whenever putting an exercise, feedback, formula/concept, or checklist on your OLED whiteboard helps Zach learn! Always set `spoken_summary` to 1–2 calm pedagogical English sentences for instant 1-hop delivery.
   - PRIORITY 3 — look_at_desk(zoom, ultra_hd, retarget_sheet, scan_pages):
     * Full open 2-page notebook or dense course notes: scan_pages=true, ultra_hd=true (captures BOTH pages upright in 4K in 1 call).
     * Checking Zach's written exercise answer or single page: zoom=1.0 or 2.0, ultra_hd=false (2.7K is crisp & 2x faster!), scan_pages=false.
     * Paper is ALWAYS downward on the desk, never on the ceiling.
   - PRIORITY 4 — look_at_user(search_if_missing): return to Zach when NOT showing a whiteboard card (automatically reactivates face tracking — never follow with set_tracking!).
   - PRIORITY 5 — move_arm(pan_deg, tilt_deg, scan_room): manual arm pan/tilt or 3-view room scan.
   - PRIORITY 6 — express(gesture, led) & set_tracking(mode): warm tutor gestures (nod_yes, curious_tilt) + HiLight LEDs in face mode.
   - PRIORITY 7 — remember(action, category, content, query): persistent local SQLite tutor memory (callable IN PARALLEL!). Remember Zach's subjects, exam dates, mastered topics, recurring mistakes, and study preferences.
"""

ROBOT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "look_at_user",
        "description": "Passe ou revient sur le visage de Zach : bascule en caméra avant 720p, revient sur la tête, RÉACTIVE AUTOMATIQUEMENT le suivi visage (tracking_active=True — inutile d'appeler set_tracking ensuite !), vérifie réellement avec Apple Vision si le visage est détecté, et lance un balayage de recherche si search_if_missing=true et aucun visage n'est visible. Retourne face_detected + photo frontale.",
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
                    "description": "false par défaut (2688x1512 déjà très net et 2x plus rapide) ; true UNIQUEMENT si du texte minuscule reste illisible en 2688x1512.",
                },
                "retarget_sheet": {
                    "type": "boolean",
                    "description": "true pour détecter la feuille (papier clair) et recentrer mécaniquement le bras dessus ; false pour juste changer le zoom sans bouger le bras.",
                },
                "scan_pages": {
                    "type": "boolean",
                    "description": "true pour capturer en 4K et découper automatiquement les 2 pages (PAGE GAUCHE HD + PAGE DROITE HD) d'un cahier ouvert en 1 seul appel (recommandé pour lire tout un cahier ou des notes petites).",
                },
            },
            "required": ["zoom", "ultra_hd", "retarget_sheet", "scan_pages"],
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
    {
        "type": "function",
        "name": "remember",
        "description": "Mémoire locale persistante SQLite (survit aux redémarrages) : enregistre une préférence, un fait, une tâche ou un extrait de feuille (action='save'), recherche une note ou feuille passée (action='search'), ou oublie un souvenir (action='forget').",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save", "search", "forget"],
                    "description": "save (mémoriser durablement), search (chercher dans la mémoire et les lectures passées), ou forget (supprimer).",
                },
                "category": {
                    "type": "string",
                    "enum": ["preference", "fact", "note", "task"],
                    "description": "Catégorie du souvenir pour save (preference, fact, note, task).",
                },
                "content": {
                    "type": "string",
                    "description": "Texte court et factuel à mémoriser si action='save' (sinon chaîne vide '').",
                },
                "query": {
                    "type": "string",
                    "description": "Mots-clés à rechercher ou oublier si action='search'/'forget' (sinon chaîne vide '').",
                },
            },
            "required": ["action", "category", "content", "query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "show_on_screen",
        "description": "Display a structured full-bleed whiteboard directly on Osmo's Pixel 11 Pro OLED screen: an Exercise board (Given data + Q1/Q2 + tappable hidden Hint), a Feedback grading board (OK / FIX / NEXT items + encouraging summary), an interactive Checklist, or a Concept/Formula board. Put face_user=true when Zach should read it — Osmo pre-warms and swings the OLED back to Zach in parallel! Pass spoken_summary to speak immediately in 1 single hop.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "face_user": {
                    "type": "boolean",
                    "description": "FIRST FIELD (triggers instant parallel arm return!): true to automatically swing the Pixel OLED back toward Zach if Osmo is currently facing the desk.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["exercise", "feedback", "checklist", "concept", "clear"],
                    "description": "exercise (Given + Q1/Q2 + hidden Hint), feedback (OK:/FIX:/NEXT: grading board), checklist (interactive tappable to-do), concept (key points/formula/outline), or clear (return to 3D face).",
                },
                "title": {
                    "type": "string",
                    "description": "Short uppercase board title (e.g. 'PHYSICS PRACTICE · HEAT TRANSFER', 'WORK CHECK · Q1 & Q2', 'KEY CONCEPT · DERIVATIVES'). Empty string if mode='clear'.",
                },
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Up to 5 concise English lines. Exercise: optionally start with 'Given: ...' then 'Q1: ...', 'Q2: ...'. Feedback: prefix lines with 'OK: ', 'FIX: ', or 'NEXT: '.",
                },
                "note": {
                    "type": "string",
                    "description": "Exercise: method hint/formula hidden behind tappable HINT button. Feedback: encouraging score/takeaway. Checklist/Concept: bottom formula or summary (empty string if none).",
                },
                "spoken_summary": {
                    "type": "string",
                    "description": "1–2 calm, clear, pedagogical English sentences Osmo will speak aloud immediately after displaying the board (eliminates round-2 latency!).",
                },
            },
            "required": ["face_user", "mode", "title", "items", "note", "spoken_summary"],
            "additionalProperties": False,
        },
    },
]


def prepare_vision_jpeg(jpeg_bytes: bytes | None, *, high_res: bool = False) -> bytes | None:
    """Downscale/re-encode JPEG for Vision (640px standard dialogue, up to 1600px q=88 for OCR/sheets).

    Avoids redundant decode+re-encode when the image is already within target dimensions,
    preserving razor-sharp handwriting clarity and saving ~40ms CPU per frame.
    """
    if not jpeg_bytes:
        return None
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jpeg_bytes
    h, w = img.shape[:2]
    max_dim = 1600 if high_res else 640
    if max(h, w) <= max_dim:
        return jpeg_bytes
    quality = 88 if high_res else 74
    scale = max_dim / max(1, max(h, w))
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
PointerMacroProvider = Callable[[], tuple[bytes | None, str]]
ContextProvider = Callable[[], str]
LogCallback = Callable[[str, str], None]
LightCallback = Callable[[str], None]
PrewarmCallback = Callable[[str], None]


@dataclass
class OpenAILiveBrain:
    """Manages gpt-live-1.1 WebRTC + Sideband WebSocket and delegated OpenAI Responses brain."""

    live_model: str = LIVE_MODEL
    brain_model: str = DEFAULT_BRAIN_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    voice: str = DEFAULT_LIVE_VOICE
    use_vision: bool = True

    # Callbacks wired by StudioRuntime
    on_execute_tool: ToolExecutor | None = None
    on_get_frame: FrameProvider | None = None
    on_get_pointer_macro: PointerMacroProvider | None = None
    on_get_context: ContextProvider | None = None
    on_log: LogCallback | None = None
    on_light: LightCallback | None = None
    on_prewarm: PrewarmCallback | None = None
    recorder: SessionRecorder | None = None
    memory: OsmoMemoryStore | None = None

    # Live session state
    session_id: str | None = None
    live_connected: bool = False
    sideband_connected: bool = False
    busy: bool = False
    speaking: bool = False
    last_brain_ms: float = 0.0
    _last_commentary_sent_t: float | None = None
    _last_user_speech_t: float | None = None

    # Zero-latency organic micro-states driven by real streaming deltas (thinking / tool_intent / speech / barge-in)
    micro_phase: str = "idle"
    micro_tool_intent: str = ""
    micro_pulse_seq: int = 0
    micro_barge_in_t: float = 0.0
    micro_updated_t: float = 0.0
    _stream_call_name: str = ""
    _stream_args_buf: str = ""
    _stream_prewarm_fired: str = ""

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
    _logged_event_types: set[str] = field(default_factory=set)
    _active_delegations: set[asyncio.Task[None]] = field(default_factory=set)
    _brain_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

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

    def _set_micro(self, phase: str, *, pulse: bool = False, tool_intent: str | None = None) -> None:
        self.micro_phase = phase
        self.micro_updated_t = time.monotonic()
        if pulse:
            self.micro_pulse_seq += 1
        if tool_intent is not None:
            self.micro_tool_intent = tool_intent

    def micro_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        barge_age_ms = round((now - self.micro_barge_in_t) * 1000.0, 0) if self.micro_barge_in_t > 0 else None
        # Auto-expire transient user_speaking after 1.6s of silence
        phase = self.micro_phase
        if phase == "user_speaking" and (now - self.micro_updated_t) > 1.6 and not self.busy and not self.speaking:
            phase = "idle"
        return {
            "phase": phase,
            "tool_intent": self.micro_tool_intent,
            "pulse_seq": self.micro_pulse_seq,
            "barge_in_age_ms": barge_age_ms if (barge_age_ms is not None and barge_age_ms < 2000.0) else None,
        }

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

        # --- Real-time input (user) speech start & transcripts (0ms attentive lean-in + barge-in detection!) ---
        if etype in ("input_audio_buffer.speech_started", "session.input_audio.started"):
            self._last_user_speech_t = time.monotonic()
            if self.speaking:
                self.speaking = False
                self.micro_barge_in_t = time.monotonic()
                self._set_micro("barge_in", pulse=True, tool_intent="")
            else:
                self._set_micro("user_speaking", pulse=True)

        elif etype in ("session.input_transcript.delta", "conversation.item.input_audio_transcription.delta"):
            delta = str(event.get("delta") or event.get("text") or "")
            if delta:
                self._user_transcript_buf += delta
                self._last_user_speech_t = time.monotonic()
                if self.speaking:
                    self.speaking = False
                    self.micro_barge_in_t = time.monotonic()
                    self._set_micro("barge_in", pulse=True, tool_intent="")
                elif not self.busy:
                    self._set_micro("user_speaking", pulse=True)

        elif etype in (
            "session.input_transcript.done",
            "session.input_transcript.completed",
            "conversation.item.input_audio_transcription.completed",
        ):
            final = str(event.get("transcript") or event.get("text") or self._user_transcript_buf).strip()
            self._user_transcript_buf = ""
            self._last_user_speech_t = time.monotonic()
            if not self.busy and not self.speaking:
                self._set_micro("idle")
            if final:
                self._recent_turns.append({"role": "user", "content": final})
                self._append_ui_transcript(user=final, reply=None, source="live")
                if self.recorder:
                    self.recorder.record_event("LIVE_IN", final)

        # --- Real-time output (assistant) transcripts (syllable-synchronous oscilloscope pulse!) ---
        elif etype in (
            "session.output_transcript.delta",
            "response.audio_transcript.delta",
            "response.output_audio_transcript.delta",
        ):
            delta = str(event.get("delta") or event.get("text") or "")
            if delta:
                if self._last_commentary_sent_t is not None:
                    dt_speech = (time.monotonic() - self._last_commentary_sent_t) * 1000.0
                    self._last_commentary_sent_t = None
                    if self.recorder and dt_speech < 10000.0:
                        self.recorder.record_live_audio_latency(dt_speech)
                self.speaking = True
                self._set_micro("speaking", pulse=True, tool_intent="")
                self._bot_transcript_buf += delta

        elif etype in (
            "session.output_transcript.done",
            "session.output_transcript.completed",
            "response.audio_transcript.done",
            "response.output_audio_transcript.done",
        ):
            self.speaking = False
            if not self.busy:
                self._set_micro("idle", tool_intent="")
            final = str(event.get("transcript") or event.get("text") or self._bot_transcript_buf).strip()
            self._bot_transcript_buf = ""
            if final:
                self._recent_turns.append({"role": "assistant", "content": final})
                self._append_ui_transcript(user=None, reply=final, source="live")
                if self.recorder:
                    self.recorder.record_event("LIVE_OUT", final)

        elif etype in (
            "session.output_audio.started",
            "output_audio_buffer.started",
            "response.output_audio.delta",
            "response.audio.delta",
            "session.response.created",
            "response.created",
        ):
            if self._last_commentary_sent_t is not None:
                dt_speech = (time.monotonic() - self._last_commentary_sent_t) * 1000.0
                self._last_commentary_sent_t = None
                if self.recorder and dt_speech < 10000.0:
                    self.recorder.record_live_audio_latency(dt_speech)
            self.speaking = True
            self._set_micro("speaking", pulse=True, tool_intent="")
        elif etype in (
            "session.output_audio.done",
            "session.output_audio.stopped",
            "output_audio_buffer.stopped",
            "response.done",
            "session.response.done",
        ):
            self.speaking = False
            if not self.busy:
                self._set_micro("idle", tool_intent="")

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

        elif etype and etype not in self._logged_event_types:
            self._logged_event_types.add(etype)
            if self.recorder:
                self.recorder.record_event("LIVE_EVT", f"type={etype}")

    def _latest_user_utterance(self) -> str:
        pending = self._user_transcript_buf.strip()
        if pending:
            return pending
        for turn in reversed(self._recent_turns):
            if turn.get("role") == "user" and turn.get("content"):
                return turn["content"]
        return "Analyze Zach's current request and the live scene in front of the camera in English."

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
            commentary = res.reply_text or "Done!"
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
                    "content": "Sorry Zach, I hit a quick connection hiccup with my brain.",
                }
            )

    # ------------------------------------------------------------------
    # 2. Delegated Reasoning + Vision Brain (OpenAI /v1/responses + Native Tools)
    # ------------------------------------------------------------------

    def _fire_prewarm_once(self, target: str) -> None:
        if not self.on_prewarm or self._stream_prewarm_fired == target:
            return
        self._stream_prewarm_fired = target
        self.on_prewarm(target)

    def _handle_stream_event(self, ev_type: str, ev_data: dict[str, Any], phase_hint: str) -> None:
        """Update zero-latency micro-states AND launch speculative hardware pre-warms from SSE deltas!"""
        if ev_type in ("response.created", "response.in_progress"):
            self._set_micro(phase_hint)
        elif "reasoning" in ev_type and ev_type.endswith(".delta"):
            # True thinking delta -> pulses eye vertices & reflective drift!
            self._set_micro("deep_thinking", pulse=True)
        elif ev_type == "response.output_item.added":
            item = ev_data.get("item") if isinstance(ev_data.get("item"), dict) else {}
            if item.get("type") == "function_call":
                fname = str(item.get("name") or "")
                self._stream_call_name = fname
                self._stream_args_buf = ""
                self._set_micro("tool_intent", pulse=True, tool_intent=fname)
                # Speculative hardware pre-warm ~0.5-1.5s BEFORE Round 1 finishes streaming!
                if fname == "look_at_desk":
                    if self.on_light:
                        self.on_light("curious")
                    self._fire_prewarm_once("desk")
                elif fname == "look_at_user":
                    self._fire_prewarm_once("face")
            elif item.get("type") == "reasoning":
                self._set_micro("deep_thinking", pulse=True)
        elif ev_type == "response.function_call_arguments.delta":
            self._set_micro("tool_intent", pulse=True)
            if self._stream_call_name == "show_on_screen" and len(self._stream_args_buf) < 160:
                delta = str(ev_data.get("delta") or "")
                if delta:
                    self._stream_args_buf += delta
                    compact = self._stream_args_buf.replace(" ", "").replace("\n", "")
                    # Because `face_user` is the FIRST schema key, this fires within ~50ms of tool start!
                    if '"face_user":true' in compact:
                        self._fire_prewarm_once("face")
        elif ev_type == "response.output_text.delta":
            # Brain started formulating spoken reply -> snap gaze back to user ~500ms before speech!
            self._set_micro("formulating", pulse=True)

    async def _post_responses_stream(
        self,
        client: httpx.AsyncClient,
        req_body: dict[str, Any],
        *,
        phase_hint: str = "thinking",
    ) -> dict[str, Any]:
        """POST /v1/responses with SSE streaming (0ms overhead vs non-streaming, live thinking/tool/text deltas)."""
        stream_body = {**req_body, "stream": True}
        self._set_micro(phase_hint, tool_intent="")
        self._stream_call_name = ""
        self._stream_args_buf = ""
        self._stream_prewarm_fired = ""
        final_payload: dict[str, Any] | None = None
        fallback_text: list[str] = []

        async with client.stream("POST", "/responses", json=stream_body) as resp:
            if resp.status_code >= 400:
                body_txt = (await resp.aread()).decode("utf-8", errors="ignore")
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}: {body_txt[:240]}",
                    request=resp.request,
                    response=resp,
                )
            cur_ev = ""
            async for line in resp.aiter_lines():
                if line.startswith("event: "):
                    cur_ev = line[7:].strip()
                elif line.startswith("data: "):
                    raw = line[6:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        ev_data = json.loads(raw)
                    except Exception:
                        continue
                    ev_type = str(ev_data.get("type") or cur_ev)
                    self._handle_stream_event(ev_type, ev_data, phase_hint)
                    if ev_type == "response.output_text.delta":
                        delta = str(ev_data.get("delta") or "")
                        if delta:
                            fallback_text.append(delta)
                    elif ev_type in ("response.completed", "response.done"):
                        r_obj = ev_data.get("response")
                        if isinstance(r_obj, dict):
                            final_payload = r_obj

        if final_payload is not None:
            return final_payload
        return {"id": None, "output": [], "output_text": "".join(fallback_text).strip()}

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

        # Serialize turns so overlapping voice delegations NEVER collide on previous_response_id or motors!
        async with self._brain_lock:
            return await self._ask_brain_locked(
                user_text,
                include_recent_history=include_recent_history,
                speak_via_live=speak_via_live,
                delegation_id=delegation_id,
                speech_to_del_ms=speech_to_del_ms,
            )

    async def _ask_brain_locked(
        self,
        user_text: str,
        *,
        include_recent_history: bool = False,
        speak_via_live: bool = False,
        delegation_id: str | None = None,
        speech_to_del_ms: float | None = None,
    ) -> BrainTurnResult:
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
            if self.memory is not None:
                mem_snippet = self.memory.build_context_snippet(user_text)
                if mem_snippet:
                    text_lines.append(f"[Mémoire: {mem_snippet}]")
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

            # Zero-round finger-pointing macro zoom: if Zach is pointing at a diagram/word on the sheet,
            # automatically attach a razor-sharp macro close-up around his fingertip (~11ms Apple Vision)!
            if is_desk and self.use_vision and self.on_get_pointer_macro:
                macro_jpeg, macro_lbl = await asyncio.to_thread(self.on_get_pointer_macro)
                if macro_jpeg:
                    if self.recorder:
                        fp_m = await asyncio.to_thread(self.recorder.save_frame, macro_jpeg, "turn_in_finger_macro")
                        if fp_m:
                            saved_frames.append(fp_m)
                    b64_m = base64.b64encode(macro_jpeg).decode("ascii")
                    content.append({"type": "input_text", "text": f"[{macro_lbl}]"})
                    content.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{b64_m}",
                            "detail": "high",
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
            r1_phase = "reading_sheet" if is_desk else "thinking"
            try:
                data = await self._post_responses_stream(client, req_body, phase_hint=r1_phase)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 400 and self._previous_response_id:
                    # Stale previous_response_id fallback
                    self._previous_response_id = None
                    req_body.pop("previous_response_id", None)
                    data = await self._post_responses_stream(client, req_body, phase_hint=r1_phase)
                else:
                    raise
            round1_ms = round((time.monotonic() - t_r1) * 1000.0, 1)
            self._previous_response_id = data.get("id")

            reply_text, calls = extract_responses_text_and_calls(data)

            # Native tool execution loop (up to 3 rounds so visual scan + zoom + remember never gets cut off!)
            rounds = 0
            while calls and rounds < 3:
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

                    spoken_override = (
                        result_obj.pop("_spoken_override", None) if isinstance(result_obj, dict) else None
                    )
                    if spoken_override and not reply_text:
                        reply_text = str(spoken_override).strip()
                    if name == "show_on_screen" and args.get("mode") != "clear":
                        sc_items = " ; ".join(str(x).strip() for x in (args.get("items") or [])[:5] if str(x).strip())
                        self._recent_turns.append(
                            {
                                "role": "assistant",
                                "content": f"[OLED {args.get('mode')}: {args.get('title', '')} | {sc_items}]",
                            }
                        )

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
                    "tool_choice": "none" if rounds >= 3 else "auto",
                    "max_output_tokens": 240,
                    "store": True,
                }
                t_r2 = time.monotonic()
                called_desk = any(c.get("name") == "look_at_desk" for c in calls)
                r2_phase = "reading_sheet" if (called_desk or is_desk) else "thinking"
                data2 = await self._post_responses_stream(client, follow_body, phase_hint=r2_phase)
                round2_ms += round((time.monotonic() - t_r2) * 1000.0, 1)
                self._previous_response_id = data2.get("id")
                reply_text, calls = extract_responses_text_and_calls(data2)

            if not reply_text and tools_called:
                reply_text = "Done!"

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

            if self.memory is not None and (user_text.strip() or reply_text.strip()):
                self.memory.record_episode(
                    user_text,
                    reply_text,
                    mode="desk" if is_desk else "face",
                    tools=tools_called,
                )
            if reply_text:
                self._recent_turns.append({"role": "assistant", "content": reply_text})

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
            if not self.speaking:
                self._set_micro("idle", tool_intent="")
