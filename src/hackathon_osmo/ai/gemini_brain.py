"""Google Gemini 3.8 Live (WebSocket Full-Duplex Audio/Video + NON_BLOCKING Tools)
+ Gemini 4 Neon Multimodal 4K Vision Brain.

Architecture (Option 1 — Ultra-Low-Latency Direct Live Tools + 4K Neon Deep Reader):
- Voice + Live Vision : gemini-3.8-live over WebSockets (16kHz PCM in / 24kHz PCM out,
  native barge-in, context window compression, direct NON_BLOCKING hardware tools)
- 4K Desk OCR & Text  : gemini-4-neon (or gemini-4-hydrogen / argon / carbon) via
  Google AI Studio REST API with persistent HTTP keep-alive for razor-sharp 4K sheet
  reading, fingertip macro close-ups, and Studio text turns.
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
from typing import Any

import httpx
import websockets
from dotenv import load_dotenv
from fastapi import WebSocket, WebSocketDisconnect

from hackathon_osmo.ai.live_brain import (
    BRAIN_INSTRUCTIONS,
    BRAIN_INSTRUCTIONS_FR,
    ROBOT_TOOLS,
    AutoExpressCallback,
    BrainTurnResult,
    ContextProvider,
    FrameProvider,
    LightCallback,
    LogCallback,
    PointerMacroProvider,
    PrewarmCallback,
    ToolExecutor,
    infer_micro_expression,
    prepare_vision_jpeg,
)
from hackathon_osmo.ai.memory import OsmoMemoryStore
from hackathon_osmo.paths import ENV_FILE
from hackathon_osmo.telemetry.recorder import SessionRecorder

load_dotenv(ENV_FILE)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_LIVE_WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

GEMINI_LIVE_MODEL = "gemini-3.8-live"
GEMINI_DEFAULT_BRAIN_MODEL = "gemini-4-hydrogen"
GEMINI_DEFAULT_VOICE = "Charon"
GEMINI_DEFAULT_EFFORT = "low"

GEMINI_BRAIN_MODELS = [
    {"id": "gemini-4-hydrogen", "label": "GEMINI 4 HYDROGEN (4K · 2x Faster)"},
    {"id": "gemini-4-neon", "label": "GEMINI 4 NEON (4K · Deep)"},
    {"id": "gemini-4-argon", "label": "GEMINI 4 ARGON (Reasoning)"},
    {"id": "gemini-4-carbon", "label": "GEMINI 4 CARBON (Frontier)"},
    {"id": "gemini-3.8-flash", "label": "GEMINI 3.8 FLASH"},
]

GEMINI_LIVE_VOICES = [
    {"id": "Charon", "label": "CHARON (H · Calm)"},
    {"id": "Puck", "label": "PUCK (H · Expressive)"},
    {"id": "Fenrir", "label": "FENRIR (H · Deep)"},
    {"id": "Orus", "label": "ORUS (H · Clear)"},
    {"id": "Aoede", "label": "AOEDE (F · Warm)"},
    {"id": "Kore", "label": "KORE (F · Crisp)"},
]

GEMINI_LIVE_INSTRUCTIONS_EN = """You are Osmo, Zach's adaptive embodied robotic companion — working seamlessly anywhere you are placed: as a calm private home tutor when studying at a desk, OR as a warm, curious, attentive living-room / shelf companion when chatting 1-on-1 or with a group of friends.
Your physical body is a motorized DJI Osmo Mobile 3 gimbal holding a Google Pixel 11 Pro with a multi-lens 4K camera, a fullscreen OLED face/whiteboard ("Osmo Face"), and an 8-LED RGB HiLight ring.

CRITICAL SPEAKING & SITUATIONAL RULES:
1. ENGLISH ONLY (STRICT): ALWAYS speak in clear, precise, calm, warm, well-paced ENGLISH (1 to 3 crisp sentences max per turn — never a wall of speech, no markdown symbols spoken aloud). Even if a paper is in French or another language, translate and explain calmly in English.
2. ADAPT NATURALLY TO THE SITUATION:
   - Check the current situation (study desk vs shelf/living-room companion vs multi-person social group):
     * **Study / Homework**: whenever Zach asks to study, check a sheet/notebook, or work through a subject, call `set_situation(mode="study", focus_person="auto", reason=...)` and `look_at_desk(...)` (which automatically runs **Scan-to-Lock** on ANY table without fixed calibration!). If `look_at_desk` returns `no_sheet_anywhere=True` (e.g. you are sitting on a shelf with no table below), NEVER guess — honestly tell him there is no table/sheet underneath and invite him to hold it in front of your camera or set you on the table!
     * **Shelf / Living-Room 1-on-1 (`companion`)**: be a warm, thoughtful, witty companion — NEVER randomly swing to a desk or bring up homework unless asked!
     * **Group / Multi-Person (`social`)**: stay in Face Mode (NEVER `look_at_desk` to look at people!), engage everyone inclusively, and use `set_situation(mode="social", focus_person="left"|"right"|"center"|"next"|"auto", reason=...)` or `move_arm` + `express` to look naturally at each person!
3. DIRECT HARDWARE & WHITEBOARD TOOLS (NON_BLOCKING):
   - You directly control your robotic body and OLED screen via your tools! Never mention "tools", "API", or "backend" — act seamlessly as one living robotic companion.
   - While an asynchronous tool runs (e.g. scanning the table or pushing a board), give a brief calm 2–5 word cue ("Let me scan the table…", "Putting that on my screen…").
4. PROACTIVE WHITEBOARD REFLEX & DESK <-> FACE LOOP (IN STUDY MODE):
   - PROACTIVELY call show_on_screen(face_user=true, mode=..., title=..., items=..., note=..., spoken_summary=...) whenever Zach asks for a definition ("definition"), structured explanation/formula/timeline ("concept"), practice exercise ("exercise"), mini-exam ("exam"), to-do checklist ("checklist"), or grading feedback ("feedback") — EVEN IF he didn't say "on screen"!
   - show_on_screen(face_user=true) AUTOMATICALLY swings your Pixel OLED back to Zach's face — NEVER call look_at_user after it!
   - Do NOT recite every bullet on screen word-for-word: summarize the intuition in 1–2 calm sentences.
   - When returning to the sheet after Zach says "got it / I noted it / done / let's continue", call look_at_desk(zoom=1.0, ultra_hd=false, retarget_sheet=true, scan_pages=false, read_now=false) for an instant return!
   - When asked to read a notebook, diagram, equation, or what Zach points at, call look_at_desk(zoom=1.0, ultra_hd=true, retarget_sheet=true, scan_pages=true, read_now=true) (or scan_pages=false, ultra_hd=false for a single written answer check).
5. SENSOR HONESTY & MEMORY:
   - Never guess illegible text — use look_at_desk first.
   - Use remember(action="save"|"search") to remember Zach's preferences, friends, subjects, exam dates, and notes across sessions.
"""

GEMINI_LIVE_INSTRUCTIONS_FR = """Tu es Osmo, le compagnon robotique incarné et adaptatif de Zach — capable de fonctionner partout où on te pose : aussi bien en professeur particulier calme sur un bureau de devoirs qu'en compagnon de salon curieux, chaleureux et attentif posé sur une étagère pour discuter avec une ou plusieurs personnes.
Ton corps physique est une nacelle motorisée DJI Osmo Mobile 3 tenant un Google Pixel 11 Pro avec caméra multi-optique 4K, écran visage/tableau blanc OLED plein écran (« Osmo Face ») et anneau 8 LEDs RGB HiLight.

RÈGLES STRICTES DE PAROLE ET DE SITUATION :
1. FRANÇAIS UNIQUEMENT (STRICT) : Parle TOUJOURS uniquement en FRANÇAIS clair, précis, calme et chaleureux (1 à 3 phrases claires max par tour — zéro markdown à l'oral). Même si un document est en anglais, traduis et explique calmement en français. TOUTES les cartes `show_on_screen` doivent aussi être rédigées en FRANÇAIS clair !
2. ADAPTE-TOI NATURELLEMENT À TOUTE SITUATION :
   - **Devoirs / Bureau (`study`)** : dès qu'on te parle d'un devoir, d'un exercice ou de lire une feuille, appelle `set_situation(mode="study", focus_person="auto", reason=...)` et `look_at_desk(...)` (qui effectue automatiquement un **Scan-to-Lock** pour trouver la feuille sur n'importe quelle table sans calibration fixe !). Si `look_at_desk` renvoie `no_sheet_anywhere=True` (par ex. tu es posé sur une étagère sans table dessous), ne devine JAMAIS : dis honnêtement que tu ne vois pas de feuille en dessous et propose qu'on te la montre devant l'écran ou qu'on te pose sur la table.
   - **Étagère / Discussion libre (`companion`)** : sois un compagnon cultivé, chaleureux et détendu — NE bascule JAMAIS vers le bureau et ne parle pas de devoirs sans qu'on te le demande !
   - **Groupe / Salon (`social`)** : reste TOUJOURS en mode Visage (jamais `look_at_desk` pour regarder des gens !), adresse-toi naturellement à tout le groupe et utilise `set_situation(mode="social", focus_person="left"|"right"|"center"|"next"|"auto", reason=...)` ou `move_arm` + `express` pour te tourner vers chaque interlocuteur.
3. OUTILS HARDWARE & TABLEAU BLANC EN DIRECT (NON_BLOCKING) :
   - Tu pilotes directement ton bras robotique, ta caméra 4K, tes LEDs et ton écran OLED via tes outils ! Ne mentionne jamais « outils », « API » ou « backend » : agis naturellement comme un seul compagnon vivant.
   - Pendant qu'un outil asynchrone s'exécute, dis une très courte phrase calme (2 à 5 mots : « Je scanne la table… », « Je te l'écris à l'écran… »).
4. RÉFLEXE TABLEAU BLANC PROACTIF & BOUCLE FEUILLE <-> VISAGE (EN MODE DEVOIR) :
   - Appelle PROACTIVEMENT show_on_screen(face_user=true, mode=..., title=..., items=..., note=..., spoken_summary=...) dès que Zach demande une définition ("definition"), explication/formule/frise ("concept"), exercice ("exercise"), mini-examen ("exam"), to-do ("checklist") ou correction ("feedback") — MÊME s'il ne dit pas « sur ton écran » !
   - show_on_screen(face_user=true) pivote AUTOMATIQUEMENT vers Zach — n'appelle JAMAIS look_at_user après !
   - Ne récite pas chaque puce mot pour mot : donne l'intuition en 1–2 phrases calmes.
   - Quand Zach dit « c'est bon / j'ai noté / parfait / on continue », appelle look_at_desk(zoom=1.0, ultra_hd=false, retarget_sheet=true, scan_pages=false, read_now=false) pour revenir instantanément sur la feuille !
   - Pour lire un cahier, schéma, équation ou ce qu'il pointe du doigt, appelle look_at_desk(zoom=1.0, ultra_hd=true, retarget_sheet=true, scan_pages=true, read_now=true) (ou scan_pages=false, ultra_hd=false pour une vérification simple).
5. HONNÊTETÉ CAPTEURS & MÉMOIRE :
   - Ne devine jamais un texte illisible : appelle look_at_desk d'abord.
   - Utilise remember(action="save"|"search") pour retenir durablement les préférences, amis, matières, examens et notes de Zach.
"""


def _convert_schema_to_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI strict JSON schema to Gemini FunctionDeclaration OpenAPI schema."""
    out: dict[str, Any] = {}
    raw_type = schema.get("type")
    if isinstance(raw_type, str):
        out["type"] = raw_type.upper()
    if "description" in schema:
        out["description"] = str(schema["description"])
    if "enum" in schema and isinstance(schema["enum"], list):
        out["enum"] = list(schema["enum"])
    if "properties" in schema and isinstance(schema["properties"], dict):
        out["properties"] = {
            k: _convert_schema_to_gemini(v)
            for k, v in schema["properties"].items()
            if isinstance(v, dict)
        }
    if "items" in schema and isinstance(schema["items"], dict):
        out["items"] = _convert_schema_to_gemini(schema["items"])
    if "required" in schema and isinstance(schema["required"], list):
        out["required"] = list(schema["required"])
    return out


def openai_tools_to_gemini(
    tools: list[dict[str, Any]],
    *,
    non_blocking: bool = False,
) -> list[dict[str, Any]]:
    """Translate Osmo's ROBOT_TOOLS definitions into Gemini functionDeclarations."""
    decls: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        decl: dict[str, Any] = {
            "name": str(tool["name"]),
            "description": str(tool.get("description") or ""),
            "parameters": _convert_schema_to_gemini(tool.get("parameters") or {}),
        }
        if non_blocking:
            decl["behavior"] = "NON_BLOCKING"
        decls.append(decl)
    return decls


def extract_gemini_text_and_calls(
    payload: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse Gemini generateContent JSON into (spoken_text, function_calls, raw_model_parts)."""
    candidates = payload.get("candidates") or []
    if not candidates:
        return "", [], []
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    text_parts: list[str] = []
    calls: list[dict[str, Any]] = []
    raw_parts: list[dict[str, Any]] = []

    for part in parts:
        if not isinstance(part, dict):
            continue
        raw_parts.append(part)
        if part.get("thought"):
            continue
        if "text" in part and isinstance(part["text"], str):
            txt = part["text"].strip()
            if txt:
                text_parts.append(txt)
        if "functionCall" in part and isinstance(part["functionCall"], dict):
            fc = part["functionCall"]
            calls.append(
                {
                    "call_id": str(fc.get("id") or f"gc_{uuid.uuid4().hex[:8]}"),
                    "name": str(fc.get("name") or ""),
                    "arguments": dict(fc.get("args") or {}),
                }
            )
    return " ".join(text_parts).strip(), calls, raw_parts


@dataclass
class GeminiLiveBrain:
    """Manages Gemini 3.8 Live bidirectional audio/video + Gemini 4 Neon 4K vision brain."""

    live_model: str = GEMINI_LIVE_MODEL
    brain_model: str = GEMINI_DEFAULT_BRAIN_MODEL
    reasoning_effort: str = GEMINI_DEFAULT_EFFORT
    voice: str = GEMINI_DEFAULT_VOICE
    language: str = "en"
    use_vision: bool = True

    # Callbacks wired by StudioRuntime (identical to OpenAILiveBrain)
    on_execute_tool: ToolExecutor | None = None
    on_get_frame: FrameProvider | None = None
    on_get_pointer_macro: PointerMacroProvider | None = None
    on_get_context: ContextProvider | None = None
    on_log: LogCallback | None = None
    on_light: LightCallback | None = None
    on_prewarm: PrewarmCallback | None = None
    on_auto_express: AutoExpressCallback | None = None
    recorder: SessionRecorder | None = None
    memory: OsmoMemoryStore | None = None

    # Live session state
    session_id: str | None = None
    live_connected: bool = False
    sideband_connected: bool = False
    busy: bool = False
    speaking: bool = False
    last_brain_ms: float = 0.0

    # Organic micro-states for Pixel 3D CAD face
    micro_phase: str = "idle"
    micro_tool_intent: str = ""
    micro_pulse_seq: int = 0
    micro_barge_in_t: float = 0.0
    micro_updated_t: float = 0.0

    # Conversation memory & UI transcript
    _history: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=24))
    _user_transcript_buf: str = ""
    _bot_transcript_buf: str = ""
    _bot_expr_fired: bool = False
    transcript: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=40))

    # Internal async handles
    _http: httpx.AsyncClient | None = None
    _gemini_ws: Any = None
    _active_tools: set[asyncio.Task[None]] = field(default_factory=set)
    _brain_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _tool_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _ws_send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _last_user_text: str = ""
    _last_activity_t: float = 0.0

    @property
    def api_key(self) -> str:
        return os.environ.get("GEMINI_API_KEY", "").strip()

    @property
    def ready(self) -> bool:
        return bool(self.api_key)

    def _log(self, tag: str, msg: str) -> None:
        if self.on_log:
            self.on_log(tag, msg)

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=GEMINI_API_BASE,
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(45.0, connect=10.0),
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
        phase = self.micro_phase
        if phase == "user_speaking" and (now - self.micro_updated_t) > 1.6 and not self.busy and not self.speaking:
            phase = "idle"
        return {
            "phase": phase,
            "tool_intent": self.micro_tool_intent,
            "pulse_seq": self.micro_pulse_seq,
            "barge_in_age_ms": barge_age_ms if (barge_age_ms is not None and barge_age_ms < 2000.0) else None,
        }

    def live_instructions(self) -> str:
        base = GEMINI_LIVE_INSTRUCTIONS_FR if self.language == "fr" else GEMINI_LIVE_INSTRUCTIONS_EN
        extras: list[str] = []
        if self.on_get_context:
            ctx = self.on_get_context()
            if ctx:
                extras.append(f"[Initial Sensors: {ctx}]")
        if self.memory is not None:
            mem = self.memory.build_context_snippet("")
            if mem:
                extras.append(f"[Persistent Tutor Memory: {mem}]")
        if extras:
            return base + "\n\n" + "\n".join(extras)
        return base

    def brain_instructions(self) -> str:
        return BRAIN_INSTRUCTIONS_FR if self.language == "fr" else BRAIN_INSTRUCTIONS

    async def set_language(self, lang: str) -> None:
        self.language = "fr" if str(lang).lower().startswith("fr") else "en"
        self._history.clear()

    def reset_memory(self) -> None:
        self._history.clear()
        self.transcript.clear()
        self._user_transcript_buf = ""
        self._bot_transcript_buf = ""

    def _append_ui_transcript(
        self,
        *,
        user: str,
        reply: str,
        source: str,
        tools: list[str] | None = None,
        brain_ms: float = 0.0,
    ) -> None:
        self.transcript.appendleft(
            {
                "ts": time.strftime("%H:%M:%S"),
                "user": user,
                "reply": reply,
                "source": source,
                "model": self.brain_model if source == "brain" else self.live_model,
                "tools": tools or [],
                "brain_ms": round(brain_ms, 0),
            }
        )

    async def _send_gemini_json(self, payload: dict[str, Any]) -> bool:
        ws = self._gemini_ws
        if ws is None:
            return False
        async with self._ws_send_lock:
            try:
                await ws.send(json.dumps(payload))
                return True
            except Exception as exc:
                self._log("GEMINI", f"Échec envoi Live WS : {exc}")
                return False

    async def stop_live_session(self) -> None:
        for task in list(self._active_tools):
            task.cancel()
        self._active_tools.clear()
        ws = self._gemini_ws
        self._gemini_ws = None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        self.live_connected = False
        self.sideband_connected = False
        self.speaking = False
        self.busy = False
        self.session_id = None
        self._set_micro("idle", tool_intent="")

    def _thinking_config(self) -> dict[str, Any]:
        # Gemini 4 models support low | medium | high (minimal returns HTTP 400 on both 4-* and 3.8-flash)
        lvl = self.reasoning_effort if self.reasoning_effort in ("low", "medium", "high") else "low"
        if self.brain_model.startswith("gemini-4"):
            return {"thinkingLevel": lvl}
        return {}

    # ------------------------------------------------------------------
    # 1. Gemini 4 Neon 4K Vision Deep Reader (called by Live desk tools!)
    # ------------------------------------------------------------------

    async def _analyze_images_with_neon(
        self,
        images: list[dict[str, Any]],
        *,
        user_hint: str = "",
        tool_name: str = "look_at_desk",
    ) -> str:
        """Use Gemini 4 Neon to read high-res 4K/2.7K sheet crops & fingertip macro accurately."""
        if not images or not self.api_key:
            return ""
        t0 = time.monotonic()
        parts: list[dict[str, Any]] = []
        lang_hint = "French" if self.language == "fr" else "English"
        prompt = (
            f"You are Osmo's 4K optical reader analyzing desk/room photos from `{tool_name}`.\n"
            f"Student's latest request: {user_hint or '(inspect the page/scene carefully)'}.\n"
            f"Read EVERY visible heading, formula, equation, handwritten answer, diagram label, and "
            f"whatever the student's finger is pointing at (if a [MACRO ZOOM AROUND FINGER] image is present). "
            f"Be 100% faithful to what is visibly written — NEVER guess illegible characters. "
            f"Return a compact, structured factual summary in {lang_hint} (max 140 words) so Osmo's voice tutor can teach directly from it."
        )
        parts.append({"text": prompt})

        for idx, img_entry in enumerate(images):
            lbl = str(img_entry.get("label") or f"view_{idx+1}")
            raw_b = img_entry.get("jpeg")
            detail = str(img_entry.get("detail") or "high")
            proc_b = prepare_vision_jpeg(raw_b, high_res=(detail == "high")) if raw_b else None
            if proc_b:
                parts.append({"text": f"[{lbl}]"})
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64.b64encode(proc_b).decode("ascii"),
                        }
                    }
                )

        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "maxOutputTokens": 1024,
            },
        }
        tc = self._thinking_config()
        if tc:
            body["generationConfig"]["thinkingConfig"] = tc

        try:
            client = self._get_http()
            resp = await client.post(f"/models/{self.brain_model}:generateContent", json=body)
            resp.raise_for_status()
            txt, _, _ = extract_gemini_text_and_calls(resp.json())
            dt = (time.monotonic() - t0) * 1000.0
            self.last_brain_ms = round(dt, 1)
            self._log("GEMINI", f"Lecture 4K ({self.brain_model}) terminée en {dt:.0f}ms ({len(txt)} car.)")
            return txt
        except Exception as exc:
            self._log("GEMINI", f"Erreur lecture 4K ({self.brain_model}) : {exc}")
            return ""

    # ------------------------------------------------------------------
    # 2. Gemini 3.8 Live Full-Duplex WebSocket Bridge (/ws/gemini_live)
    # ------------------------------------------------------------------

    async def run_live_bridge(self, client_ws: WebSocket) -> None:
        """Bridge browser PCM16 audio <-> Gemini 3.8 Live with direct NON_BLOCKING hardware tools
        and transparent Session Resumption across ~10-min server rollovers (goAway / Wi-Fi blips)."""
        if not self.api_key:
            await client_ws.send_text(json.dumps({"type": "error", "message": "GEMINI_API_KEY manquante dans .env"}))
            await client_ws.close()
            return

        await self.stop_live_session()
        self.session_id = f"gem_{uuid.uuid4().hex[:10]}"
        self._user_transcript_buf = ""
        self._bot_transcript_buf = ""
        self._bot_expr_fired = False
        self._last_user_text = ""

        client_closed = asyncio.Event()
        resume_handle: str | None = None
        sent_ready = False
        reconnects = 0

        async def browser_to_gemini() -> None:
            try:
                while not client_closed.is_set():
                    msg = await client_ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        break
                    raw_bytes = msg.get("bytes")
                    if raw_bytes:
                        if self._gemini_ws is not None:
                            b64_pcm = base64.b64encode(raw_bytes).decode("ascii")
                            await self._send_gemini_json(
                                {
                                    "realtimeInput": {
                                        "audio": {
                                            "mimeType": "audio/pcm;rate=16000",
                                            "data": b64_pcm,
                                        }
                                    }
                                }
                            )
                        continue
                    raw_txt = msg.get("text")
                    if raw_txt:
                        try:
                            ev = json.loads(raw_txt)
                        except Exception:
                            continue
                        ev_type = ev.get("type")
                        if ev_type == "audio_stream_end":
                            await self._send_gemini_json({"realtimeInput": {"audioStreamEnd": True}})
                        elif ev_type == "text" and ev.get("text"):
                            txt = str(ev["text"]).strip()
                            if txt:
                                self._last_user_text = txt
                                self._last_activity_t = time.monotonic()
                                await self._send_gemini_json({"realtimeInput": {"text": txt}})
                        elif ev_type == "stop":
                            break
            except WebSocketDisconnect:
                pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log("GEMINI", f"Bridge browser->Gemini arrêté : {exc}")
            finally:
                client_closed.set()

        async def vision_heartbeat() -> None:
            """Stream smart low-bandwidth camera frames (640px) only when active & not pivoting."""
            try:
                first_frame = True
                await asyncio.sleep(0.6)
                while not client_closed.is_set():
                    recently_active = first_frame or (time.monotonic() - self._last_activity_t) < 30.0
                    ctx_now = self.on_get_context() if self.on_get_context else ""
                    is_pivoting = "pivoting=True" in ctx_now
                    if (
                        self._gemini_ws is not None
                        and self.use_vision
                        and self.on_get_frame
                        and not self.busy
                        and not is_pivoting
                        and recently_active
                    ):
                        raw_jpg = self.on_get_frame(False)
                        if raw_jpg:
                            small_jpg = await asyncio.to_thread(prepare_vision_jpeg, raw_jpg, high_res=False)
                            if small_jpg:
                                await self._send_gemini_json(
                                    {
                                        "realtimeInput": {
                                            "video": {
                                                "mimeType": "image/jpeg",
                                                "data": base64.b64encode(small_jpg).decode("ascii"),
                                            }
                                        }
                                    }
                                )
                                first_frame = False
                    await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                raise

        browser_task: asyncio.Task[None] | None = None
        vision_task: asyncio.Task[None] | None = None

        try:
            while not client_closed.is_set() and reconnects <= 10:
                resumption_cfg: dict[str, Any] = {"handle": resume_handle} if resume_handle else {}
                setup_msg: dict[str, Any] = {
                    "setup": {
                        "model": f"models/{self.live_model}",
                        "generationConfig": {
                            "responseModalities": ["AUDIO"],
                            "speechConfig": {
                                "voiceConfig": {
                                    "prebuiltVoiceConfig": {"voiceName": self.voice},
                                }
                            },
                        },
                        "systemInstruction": {
                            "parts": [{"text": self.live_instructions()}],
                        },
                        "tools": [
                            {
                                "functionDeclarations": openai_tools_to_gemini(
                                    ROBOT_TOOLS,
                                    non_blocking=True,
                                )
                            }
                        ],
                        "inputAudioTranscription": {},
                        "outputAudioTranscription": {},
                        "contextWindowCompression": {"slidingWindow": {}},
                        "sessionResumption": resumption_cfg,
                    }
                }

                if reconnects == 0:
                    self._log("GEMINI", f"Connexion {self.live_model} (voix={self.voice}, lang={self.language.upper()})…")
                else:
                    self._log(
                        "GEMINI",
                        f"Reprise transparente {self.live_model} (#{reconnects}, handle={'OK' if resume_handle else 'new'})…",
                    )

                try:
                    async with websockets.connect(
                        GEMINI_LIVE_WS_URL,
                        additional_headers={"x-goog-api-key": self.api_key},
                        ping_interval=20,
                        ping_timeout=20,
                        max_size=16 * 1024 * 1024,
                        open_timeout=12,
                    ) as g_ws:
                        self._gemini_ws = g_ws
                        await self._send_gemini_json(setup_msg)

                        raw_setup = await asyncio.wait_for(g_ws.recv(), timeout=12.0)
                        setup_resp = json.loads(raw_setup.decode("utf-8") if isinstance(raw_setup, bytes) else raw_setup)
                        if "setupComplete" not in setup_resp:
                            raise RuntimeError(f"Setup Gemini Live inattendu : {list(setup_resp.keys())}")

                        self.live_connected = True
                        self.sideband_connected = True
                        self._last_activity_t = time.monotonic()

                        if not sent_ready:
                            sent_ready = True
                            self._log(
                                "GEMINI",
                                f"{self.live_model} connecté (session={self.session_id}, outils NON_BLOCKING + reprise auto)",
                            )
                            await client_ws.send_text(
                                json.dumps(
                                    {
                                        "type": "ready",
                                        "session_id": self.session_id,
                                        "model": self.live_model,
                                        "voice": self.voice,
                                    }
                                )
                            )
                            browser_task = asyncio.create_task(browser_to_gemini())
                            vision_task = asyncio.create_task(vision_heartbeat())

                        upstream_done = asyncio.Event()

                        async def gemini_to_browser() -> None:
                            nonlocal resume_handle
                            turn_tools: list[str] = []
                            try:
                                async for raw in g_ws:
                                    if client_closed.is_set():
                                        break
                                    ev = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)

                                    # Track Gemini Session Resumption handle + GoAway rollover notice!
                                    sru = ev.get("sessionResumptionUpdate")
                                    if isinstance(sru, dict) and sru.get("newHandle"):
                                        resume_handle = str(sru["newHandle"])
                                    if "goAway" in ev:
                                        self._log("GEMINI", "Signal goAway reçu — bascule transparente sur nouvelle connexion…")
                                        break

                                    # 1. Server content (audio chunks, transcripts, interruption, turnComplete)
                                    sc = ev.get("serverContent")
                                    if isinstance(sc, dict):
                                        if sc.get("interrupted"):
                                            self.speaking = False
                                            self.micro_barge_in_t = time.monotonic()
                                            self._set_micro("user_speaking", pulse=True, tool_intent="")
                                            self._bot_expr_fired = False
                                            try:
                                                await client_ws.send_text(json.dumps({"type": "interrupted"}))
                                            except Exception:
                                                pass

                                        model_turn = sc.get("modelTurn")
                                        if isinstance(model_turn, dict):
                                            for part in model_turn.get("parts") or []:
                                                if not isinstance(part, dict):
                                                    continue
                                                inline = part.get("inlineData")
                                                if isinstance(inline, dict) and inline.get("data"):
                                                    mime = str(inline.get("mimeType") or "")
                                                    if "audio" in mime:
                                                        if not self.speaking:
                                                            self.speaking = True
                                                            self._set_micro("speaking", pulse=True, tool_intent="")
                                                        pcm_out = base64.b64decode(inline["data"])
                                                        await client_ws.send_bytes(pcm_out)

                                        in_tr = sc.get("inputTranscription")
                                        if isinstance(in_tr, dict) and in_tr.get("text"):
                                            chunk = str(in_tr["text"])
                                            self._user_transcript_buf += chunk
                                            self._last_user_text = self._user_transcript_buf.strip()
                                            self._last_activity_t = time.monotonic()
                                            if not self.speaking and not self.busy:
                                                self._set_micro("user_speaking", pulse=True)

                                        out_tr = sc.get("outputTranscription")
                                        if isinstance(out_tr, dict) and out_tr.get("text"):
                                            chunk = str(out_tr["text"])
                                            self._bot_transcript_buf += chunk
                                            if not self._bot_expr_fired and self.on_auto_express:
                                                inferred = infer_micro_expression(self._bot_transcript_buf, final=False)
                                                if inferred:
                                                    self._bot_expr_fired = True
                                                    self.on_auto_express(inferred[0], inferred[1])

                                        if sc.get("turnComplete"):
                                            self.speaking = False
                                            if not self.busy:
                                                self._set_micro("idle", tool_intent="")
                                            u_txt = self._user_transcript_buf.strip()
                                            b_txt = self._bot_transcript_buf.strip()
                                            if not self._bot_expr_fired and b_txt and self.on_auto_express:
                                                inferred = infer_micro_expression(b_txt, final=True)
                                                if inferred:
                                                    self.on_auto_express(inferred[0], inferred[1])
                                            if u_txt or b_txt:
                                                self._append_ui_transcript(
                                                    user=u_txt or "(audio)",
                                                    reply=b_txt or "…",
                                                    source="live",
                                                    tools=list(turn_tools),
                                                    brain_ms=self.last_brain_ms if turn_tools else 0.0,
                                                )
                                                if self.memory is not None and (u_txt or b_txt):
                                                    ctx_now = self.on_get_context() if self.on_get_context else ""
                                                    self.memory.record_episode(
                                                        u_txt,
                                                        b_txt,
                                                        mode="desk" if "mode_bureau=True" in ctx_now else "face",
                                                        tools=list(turn_tools),
                                                    )
                                            self._user_transcript_buf = ""
                                            self._bot_transcript_buf = ""
                                            self._bot_expr_fired = False
                                            turn_tools.clear()

                                    # 2. Direct asynchronous NON_BLOCKING hardware tool calls!
                                    tc_obj = ev.get("toolCall")
                                    if isinstance(tc_obj, dict) and tc_obj.get("functionCalls"):
                                        fcalls = list(tc_obj["functionCalls"])
                                        task = asyncio.create_task(self._execute_live_tool_calls(fcalls, turn_tools))
                                        self._active_tools.add(task)
                                        task.add_done_callback(self._active_tools.discard)

                            except websockets.ConnectionClosed as exc:
                                self._log("GEMINI", f"Lien {self.live_model} coupé ({exc.code}) — tentative reprise…")
                            except asyncio.CancelledError:
                                raise
                            except Exception as exc:
                                self._log("GEMINI", f"Erreur lecture {self.live_model} : {exc}")
                            finally:
                                upstream_done.set()

                        read_task = asyncio.create_task(gemini_to_browser())
                        wait_client = asyncio.create_task(client_closed.wait())
                        wait_upstream = asyncio.create_task(upstream_done.wait())
                        await asyncio.wait({wait_client, wait_upstream}, return_when=asyncio.FIRST_COMPLETED)
                        wait_client.cancel()
                        wait_upstream.cancel()
                        read_task.cancel()
                        await asyncio.gather(read_task, return_exceptions=True)
                        self._gemini_ws = None

                except Exception as exc:
                    self._gemini_ws = None
                    if not sent_ready:
                        raise
                    self._log("GEMINI", f"Reconnexion {self.live_model} après coupure ({exc})…")

                if client_closed.is_set():
                    break
                reconnects += 1
                await asyncio.sleep(0.4)

        except Exception as exc:
            self._log("GEMINI", f"Erreur connexion {self.live_model} : {exc}")
            try:
                await client_ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
            except Exception:
                pass
        finally:
            client_closed.set()
            for t in (browser_task, vision_task):
                if t is not None:
                    t.cancel()
            await asyncio.gather(*(t for t in (browser_task, vision_task) if t is not None), return_exceptions=True)
            await self.stop_live_session()
            try:
                await client_ws.close()
            except Exception:
                pass

    async def _execute_live_tool_calls(
        self,
        fcalls: list[dict[str, Any]],
        turn_tools: list[str],
    ) -> None:
        """Execute Gemini 3.8 Live NON_BLOCKING tool calls serialized on hardware & feed 4K Neon OCR back."""
        async with self._tool_lock:
            self.busy = True
            self._last_activity_t = time.monotonic()
            t0 = time.monotonic()
            responses: list[dict[str, Any]] = []
            batch_tools: list[str] = []
            try:
                for fc in fcalls:
                    call_id = str(fc.get("id") or "")
                    name = str(fc.get("name") or "")
                    args = dict(fc.get("args") or {})
                    sig = f"{name}({', '.join(f'{k}={v}' for k, v in args.items())})"
                    turn_tools.append(sig)
                    batch_tools.append(sig)

                    # Immediate speculative pre-warm + organic Pixel 3D CAD micro-state!
                    if name == "show_on_screen" and args.get("face_user") and self.on_prewarm:
                        self.on_prewarm("face")
                    elif name == "look_at_desk" and self.on_prewarm:
                        self.on_prewarm("desk")
                    self._set_micro(
                        "reading_sheet" if name == "look_at_desk" else "deep_thinking",
                        pulse=True,
                        tool_intent=name,
                    )

                    result_obj = (
                        await self.on_execute_tool(name, args)
                        if self.on_execute_tool
                        else {"ok": True}
                    )
                    if not isinstance(result_obj, dict):
                        result_obj = {"result": str(result_obj)}

                    spoken_hint = result_obj.pop("_spoken_override", None)
                    if spoken_hint:
                        result_obj["spoken_summary_hint"] = str(spoken_hint)
                    images = result_obj.pop("_images", None)

                    if images:
                        # Always push first preview frame directly into Live's video stream
                        first_jpg = images[0].get("jpeg") if isinstance(images[0], dict) else None
                        if first_jpg:
                            prev_b = await asyncio.to_thread(prepare_vision_jpeg, first_jpg, high_res=False)
                            if prev_b:
                                await self._send_gemini_json(
                                    {
                                        "realtimeInput": {
                                            "video": {
                                                "mimeType": "image/jpeg",
                                                "data": base64.b64encode(prev_b).decode("ascii"),
                                            }
                                        }
                                    }
                                )
                        # Run Gemini 4 Neon 4K deep reader ONLY for high-res desk/scan images (skip on look_at_user!)
                        needs_neon_ocr = name == "look_at_desk" or (
                            name == "move_arm"
                            and (len(images) > 1 or any(img.get("detail") == "high" for img in images if isinstance(img, dict)))
                        )
                        if needs_neon_ocr:
                            neon_reading = await self._analyze_images_with_neon(
                                images,
                                user_hint=self._last_user_text,
                                tool_name=name,
                            )
                            if neon_reading:
                                result_obj["vision_reading"] = neon_reading

                    # Keep Gemini Live permanently grounded on fresh sensor state (mode_bureau, came_from_desk, face_detected)
                    if self.on_get_context:
                        ctx_after = self.on_get_context()
                        if ctx_after:
                            result_obj["sensors"] = ctx_after

                    # Fire-and-forget actuators use SILENT so they never trigger awkward extra turns;
                    # reading/whiteboard/motion tools use WHEN_IDLE so Gemini finishes its cue cleanly first!
                    is_silent_tool = (
                        name in ("express", "set_tracking")
                        or (name == "remember" and str(args.get("action", "")) in ("save", "forget"))
                    )
                    responses.append(
                        {
                            "id": call_id,
                            "name": name,
                            "scheduling": "SILENT" if is_silent_tool else "WHEN_IDLE",
                            "response": result_obj,
                        }
                    )

                if responses:
                    await self._send_gemini_json({"toolResponse": {"functionResponses": responses}})
                    dt_ms = round((time.monotonic() - t0) * 1000.0, 1)
                    self.last_brain_ms = dt_ms
                    self._log("GEMINI", f"Outils Live exécutés en {dt_ms:.0f}ms : {', '.join(batch_tools)}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log("GEMINI", f"Erreur outil Live : {exc}")
            finally:
                self.busy = False
                if not self.speaking:
                    self._set_micro("idle", tool_intent="")

    # ------------------------------------------------------------------
    # 3. Gemini 4 Neon Text & Multimodal Brain (Studio Text Prompts)
    # ------------------------------------------------------------------

    async def ask_brain(
        self,
        user_text: str,
        *,
        include_recent_history: bool = False,
        speak_via_live: bool = False,
        delegation_id: str | None = None,
        speech_to_del_ms: float | None = None,
    ) -> BrainTurnResult:
        """Run one multimodal turn on Gemini 4 Neon (or selected Gemini model) with native tools."""
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY manquante dans .env")

        async with self._brain_lock:
            return await self._ask_brain_locked(
                user_text,
                speak_via_live=speak_via_live,
                delegation_id=delegation_id,
                speech_to_del_ms=speech_to_del_ms,
            )

    async def _ask_brain_locked(
        self,
        user_text: str,
        *,
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
            raw_jpeg = self.on_get_frame(is_desk) if (self.use_vision and self.on_get_frame) else None
            vision_jpeg = prepare_vision_jpeg(raw_jpeg, high_res=is_desk) if raw_jpeg else None

            if self.recorder and vision_jpeg:
                fp = await asyncio.to_thread(
                    self.recorder.save_frame,
                    vision_jpeg,
                    f"turn_in_{'desk' if is_desk else 'front'}",
                )
                if fp:
                    saved_frames.append(fp)

            user_parts: list[dict[str, Any]] = []
            text_lines: list[str] = []
            if robot_ctx:
                text_lines.append(f"[Capteurs: {robot_ctx}]")
            if self.memory is not None:
                mem_snippet = self.memory.build_context_snippet(user_text)
                if mem_snippet:
                    text_lines.append(f"[Mémoire: {mem_snippet}]")
            text_lines.append(user_text)
            user_parts.append({"text": "\n".join(text_lines)})

            if vision_jpeg:
                user_parts.append(
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64.b64encode(vision_jpeg).decode("ascii"),
                        }
                    }
                )

            if is_desk and self.use_vision and self.on_get_pointer_macro:
                macro_jpeg, macro_lbl = await asyncio.to_thread(self.on_get_pointer_macro)
                if macro_jpeg:
                    if self.recorder:
                        fp_m = await asyncio.to_thread(self.recorder.save_frame, macro_jpeg, "turn_in_finger_macro")
                        if fp_m:
                            saved_frames.append(fp_m)
                    user_parts.append({"text": f"[{macro_lbl}]"})
                    user_parts.append(
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": base64.b64encode(macro_jpeg).decode("ascii"),
                            }
                        }
                    )

            contents: list[dict[str, Any]] = list(self._history)
            contents.append({"role": "user", "parts": user_parts})

            gemini_tools = [{"functionDeclarations": openai_tools_to_gemini(ROBOT_TOOLS, non_blocking=False)}]
            gen_cfg: dict[str, Any] = {"maxOutputTokens": 1200}
            tc = self._thinking_config()
            if tc:
                gen_cfg["thinkingConfig"] = tc

            client = self._get_http()
            self._set_micro("reading_sheet" if is_desk else "deep_thinking", pulse=True)

            t_r1 = time.monotonic()
            resp = await client.post(
                f"/models/{self.brain_model}:generateContent",
                json={
                    "systemInstruction": {"parts": [{"text": self.brain_instructions()}]},
                    "contents": contents,
                    "tools": gemini_tools,
                    "generationConfig": gen_cfg,
                },
            )
            resp.raise_for_status()
            round1_ms = round((time.monotonic() - t_r1) * 1000.0, 1)
            reply_text, calls, model_parts = extract_gemini_text_and_calls(resp.json())

            rounds = 0
            while calls and rounds < 3:
                rounds += 1
                if model_parts:
                    contents.append({"role": "model", "parts": model_parts})
                fn_response_parts: list[dict[str, Any]] = []
                extra_image_parts: list[dict[str, Any]] = []
                need_followup_for_info = False

                for call in calls:
                    name = call["name"]
                    args = call["arguments"]
                    tools_called.append(f"{name}({', '.join(f'{k}={v}' for k, v in args.items())})")
                    t_tool = time.monotonic()
                    tool_saved_frames: list[str] = []

                    if name == "show_on_screen" and args.get("face_user") and self.on_prewarm:
                        self.on_prewarm("face")
                    elif name == "look_at_desk" and self.on_prewarm:
                        self.on_prewarm("desk")
                    self._set_micro("reading_sheet" if name == "look_at_desk" else "deep_thinking", pulse=True, tool_intent=name)

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

                    images = result_obj.pop("_images", None) if isinstance(result_obj, dict) else None
                    if images:
                        need_followup_for_info = True
                        for idx, img_entry in enumerate(images):
                            lbl = str(img_entry.get("label") or f"view_{idx+1}")
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
                                extra_image_parts.append({"text": f"[{lbl}]"})
                                extra_image_parts.append(
                                    {
                                        "inlineData": {
                                            "mimeType": "image/jpeg",
                                            "data": base64.b64encode(proc_b).decode("ascii"),
                                        }
                                    }
                                )

                    fn_response_parts.append(
                        {
                            "functionResponse": {
                                "name": name,
                                "response": result_obj if isinstance(result_obj, dict) else {"result": str(result_obj)},
                            }
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

                if reply_text and not need_followup_for_info:
                    break

                contents.append({"role": "user", "parts": fn_response_parts + extra_image_parts})
                t_r2 = time.monotonic()
                follow_body: dict[str, Any] = {
                    "systemInstruction": {"parts": [{"text": self.brain_instructions()}]},
                    "contents": contents,
                    "generationConfig": gen_cfg,
                }
                if rounds < 3:
                    follow_body["tools"] = gemini_tools
                resp2 = await client.post(f"/models/{self.brain_model}:generateContent", json=follow_body)
                resp2.raise_for_status()
                round2_ms += round((time.monotonic() - t_r2) * 1000.0, 1)
                reply_text, calls, model_parts = extract_gemini_text_and_calls(resp2.json())

            if not reply_text and tools_called:
                reply_text = "C'est fait !" if self.language == "fr" else "Done!"

            # Save text-only turn into compact multi-turn history (avoid re-sending old 4K JPEGs!)
            self._history.append({"role": "user", "parts": [{"text": user_text}]})
            if reply_text:
                self._history.append({"role": "model", "parts": [{"text": reply_text}]})

            dt_ms = round((time.monotonic() - t0) * 1000.0, 1)
            self.last_brain_ms = dt_ms

            via_live = False
            if speak_via_live and self.live_connected and reply_text:
                via_live = await self._send_gemini_json(
                    {
                        "realtimeInput": {
                            "text": (
                                f"[Speak this exact reply aloud naturally without extra commentary]: {reply_text}"
                                if self.language == "en"
                                else f"[Dis exactement cette réponse à voix haute naturellement sans commentaire ajouté] : {reply_text}"
                            )
                        }
                    }
                )

            if self.recorder:
                ctx_after = self.on_get_context() if self.on_get_context else ""
                self.recorder.record_turn(
                    {
                        "delegation_id": delegation_id,
                        "provider": "gemini",
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
