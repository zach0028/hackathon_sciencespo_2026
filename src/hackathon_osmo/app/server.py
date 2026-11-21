"""FastAPI HTTP & WebSocket server exposing the hackathon_osmo Studio UI and voice endpoints."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hackathon_osmo.app.runtime import StudioRuntime
from hackathon_osmo.hardware.pixel_camera import find_adb, list_adb_devices
from hackathon_osmo.paths import STATIC_DIR

runtime = StudioRuntime()


def _compute_ui_version() -> str:
    h = hashlib.sha1()
    for name in ("face.html", "face.css", "face.js"):
        p = STATIC_DIR / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:10]


UI_VERSION = _compute_ui_version()
runtime.ui_version = UI_VERSION


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await runtime.start()
    try:
        yield
    finally:
        await runtime.stop()


app = FastAPI(title="hackathon_osmo Studio", lifespan=lifespan)


@app.middleware("http")
async def _no_cache_ui_middleware(request: Request, call_next: Any) -> Response:
    response: Response = await call_next(request)
    if request.url.path in ("/", "/face") or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/face", response_class=HTMLResponse)
async def pixel_face() -> HTMLResponse:
    """Pure #000000 OLED Adaptive Desk Companion interface for the Pixel 11 Pro screen."""
    return HTMLResponse((STATIC_DIR / "face.html").read_text(encoding="utf-8"))


@app.post("/api/pixel/launch_face")
async def api_launch_pixel_face(port: int = 8765) -> JSONResponse:
    """Expose Studio over adb reverse and launch the /face OLED Companion UI on the Pixel."""
    adb = find_adb()
    if not adb:
        return JSONResponse({"ok": False, "error": "adb introuvable"}, status_code=500)
    serial = runtime.pixel_serial
    if not serial:
        devs = [d for d in list_adb_devices() if d.state == "device"]
        serial = devs[0].serial if devs else None
    if not serial:
        return JSONResponse({"ok": False, "error": "Aucun Pixel connecté en ADB"}, status_code=400)

    url = f"http://127.0.0.1:{int(port)}/face?v={UI_VERSION}"
    all_serials = [d.serial for d in list_adb_devices() if d.state == "device"]
    if serial not in all_serials:
        all_serials.insert(0, serial)

    def _launch_sync() -> str:
        # Expose Studio over adb reverse on EVERY connected Pixel transport (USB + Wi-Fi!)
        for s in all_serials:
            subprocess.run([adb, "-s", s, "reverse", f"tcp:{int(port)}", f"tcp:{int(port)}"], check=False, capture_output=True)
        # Demo-ready display prep: wake screen + keep awake 30 min + lock Landscape orientation
        subprocess.run([adb, "-s", serial, "shell", "input", "keyevent", "KEYCODE_WAKEUP"], check=False, capture_output=True)
        subprocess.run(
            [adb, "-s", serial, "shell", "settings", "put", "system", "screen_off_timeout", "1800000"],
            check=False,
            capture_output=True,
        )
        subprocess.run(
            [adb, "-s", serial, "shell", "settings", "put", "system", "accelerometer_rotation", "0"],
            check=False,
            capture_output=True,
        )
        subprocess.run(
            [adb, "-s", serial, "shell", "settings", "put", "system", "user_rotation", "1"],
            check=False,
            capture_output=True,
        )
        # Prefer native fullscreen locked-landscape OsmoFace kiosk app if installed, else Chrome fallback
        pkg_check = subprocess.run(
            [adb, "-s", serial, "shell", "pm", "path", "com.osmo.face"],
            check=False,
            capture_output=True,
            text=True,
        )
        if pkg_check.returncode == 0 and "package:" in (pkg_check.stdout or ""):
            subprocess.run(
                [adb, "-s", serial, "shell", "am", "start", "-W", "-n", "com.osmo.face/.MainActivity", "--es", "url", url],
                check=True,
                capture_output=True,
            )
            return "native_app"
        subprocess.run(
            [adb, "-s", serial, "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", url],
            check=True,
            capture_output=True,
        )
        return "browser"

    try:
        launcher = await asyncio.to_thread(_launch_sync)
        runtime.arbiter.log("PIXEL", f"Interface OLED /face lancée sur Pixel ({serial}, mode={launcher})")
        return JSONResponse({"ok": True, "serial": serial, "url": url, "launcher": launcher})
    except Exception as exc:
        runtime.arbiter.log("PIXEL", f"Erreur lancement /face sur Pixel : {exc}")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    return runtime.snapshot()


@app.get("/api/telemetry/summary")
async def api_telemetry_summary() -> dict[str, Any]:
    return runtime.recorder.summary()


class MemoryCreatePayload(BaseModel):
    content: str
    category: str = "fact"
    pinned: bool = False


@app.get("/api/memory")
async def api_memory_list(q: str = "") -> dict[str, Any]:
    if q.strip():
        return {
            "stats": runtime.memory.stats(),
            "items": runtime.memory.search_memories(q.strip(), limit=20),
            "episodes": runtime.memory.search_episodes(q.strip(), limit=8),
        }
    return {
        "stats": runtime.memory.stats(),
        "items": runtime.memory.list_memories(limit=30),
    }


@app.post("/api/memory")
async def api_memory_create(payload: MemoryCreatePayload) -> JSONResponse:
    res = runtime.memory.save_memory(
        payload.content,
        category=payload.category,
        source="studio",
        pinned=payload.pinned,
    )
    if res.get("ok"):
        runtime.arbiter.log("MEM", f"Ajout manuel [{payload.category}] : « {payload.content[:60]} »")
    return JSONResponse(res, status_code=200 if res.get("ok") else 400)


@app.delete("/api/memory/{mem_id}")
async def api_memory_delete(mem_id: int) -> JSONResponse:
    deleted = runtime.memory.delete_memory(mem_id)
    if deleted:
        runtime.arbiter.log("MEM", f"Souvenir #{mem_id} supprimé")
    return JSONResponse({"ok": deleted, "id": mem_id}, status_code=200 if deleted else 404)


class LiveSessionOffer(BaseModel):
    sdp: str


class VoiceTextPayload(BaseModel):
    text: str


@app.post("/api/live/session")
async def api_live_session(payload: LiveSessionOffer) -> JSONResponse:
    try:
        res = await runtime.create_live_session(payload.sdp)
        return JSONResponse(res)
    except Exception as exc:
        runtime.arbiter.log("LIVE", f"Erreur création session GPT-Live : {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/live/stop")
async def api_live_stop() -> JSONResponse:
    await runtime.stop_live_session()
    return JSONResponse({"ok": True})


@app.post("/api/voice/text")
async def api_voice_text(payload: VoiceTextPayload) -> JSONResponse:
    try:
        res = await runtime.handle_text_turn(payload.text.strip())
        return JSONResponse(res.__dict__)
    except Exception as exc:
        runtime.arbiter.log("BRAIN", f"Erreur cerveau ({runtime.ai_provider.upper()}) : {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.websocket("/ws/gemini_live")
async def ws_gemini_live(ws: WebSocket) -> None:
    await ws.accept()
    await runtime.gemini_brain.run_live_bridge(ws)


@app.get("/video_feed")
async def video_feed() -> StreamingResponse:
    async def generate() -> AsyncIterator[bytes]:
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Connexion au Pixel 11 Pro...",
            (150, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (180, 200, 220),
            2,
        )
        _, ph_jpg = cv2.imencode(".jpg", placeholder)
        ph_bytes = ph_jpg.tobytes()

        while True:
            jpg = runtime._latest_jpeg or ph_bytes
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                + jpg
                + b"\r\n"
            )
            await asyncio.sleep(0.033)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws_video")
async def ws_video(ws: WebSocket) -> None:
    await ws.accept()
    last_jpg: bytes | None = None
    try:
        while True:
            jpg = runtime._latest_jpeg
            if jpg is not None and jpg is not last_jpg:
                last_jpg = jpg
                await ws.send_bytes(jpg)
            await asyncio.sleep(0.025)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()

    async def sender() -> None:
        while True:
            await ws.send_text(json.dumps(runtime.snapshot()))
            await asyncio.sleep(0.08)

    send_task = asyncio.create_task(sender())
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            action = msg.get("action")

            if action == "set_live_motors":
                enabled = bool(msg.get("enabled", False))
                runtime.arbiter.live_motors = enabled
                runtime.arbiter.log(
                    "MOTOR",
                    "Moteurs LIVE activés" if enabled else "Passage en mode DRY-RUN (moteurs coupés)",
                )
                if not enabled and runtime._osmo and runtime.osmo_connected:
                    await runtime._osmo.send_speed(0.0, 0.0)

            elif action == "set_control_mode":
                mode = msg.get("mode")
                if mode in ("speed", "relative", "joystick"):
                    runtime.arbiter.control_mode = mode
                    runtime.arbiter.log("MOTOR", f"Protocole moteur : {mode}")

            elif action == "voice_config":
                if "provider" in msg and str(msg["provider"]) in ("gemini", "openai"):
                    await runtime.set_ai_provider(str(msg["provider"]))
                if "brain_model" in msg:
                    runtime.brain.brain_model = str(msg["brain_model"])
                    if hasattr(runtime.brain, "_previous_response_id"):
                        runtime.brain._previous_response_id = None
                    runtime.arbiter.log("BRAIN", f"Cerveau ({runtime.ai_provider.upper()}) : {runtime.brain.brain_model}")
                if "reasoning_effort" in msg and msg["reasoning_effort"] in ("low", "medium", "high"):
                    runtime.brain.reasoning_effort = str(msg["reasoning_effort"])
                    runtime.arbiter.log("BRAIN", f"Reasoning effort : {runtime.brain.reasoning_effort}")
                if "voice" in msg:
                    runtime.brain.voice = str(msg["voice"])
                    runtime.arbiter.log("LIVE", f"Voix Live ({runtime.ai_provider.upper()}) : {runtime.brain.voice} (actif au prochain appel)")
                if "language" in msg and str(msg["language"]) in ("en", "fr"):
                    lang = str(msg["language"])
                    await runtime.openai_brain.set_language(lang)
                    await runtime.gemini_brain.set_language(lang)
                    runtime.arbiter.log(
                        "LIVE",
                        "Mode langue Osmo : FULL FRANÇAIS" if lang == "fr" else "Mode langue Osmo : FULL ENGLISH",
                    )
                if "use_vision" in msg:
                    val = bool(msg["use_vision"])
                    runtime.openai_brain.use_vision = val
                    runtime.gemini_brain.use_vision = val
                if msg.get("clear_history"):
                    runtime.brain.reset_memory()
                    runtime.arbiter.log("BRAIN", "Mémoire de conversation réinitialisée")

            elif action == "hilight":
                reaction = str(msg.get("reaction", "off"))
                if runtime.hilight.connected:
                    runtime.hilight.start_reaction(reaction)
                    runtime.arbiter.log("HILIGHT", f"Animation LED HiLight : {reaction}")

            elif action == "recenter":
                asyncio.create_task(runtime.recenter_landscape())

            elif action == "toggle_orientation":
                asyncio.create_task(runtime.toggle_orientation())

            elif action == "search_now":
                runtime.arbiter.trigger_search_now()

            elif action == "set_lost_policy":
                pol = str(msg.get("policy", "search"))
                if pol in ("search", "home", "last_pos"):
                    runtime.arbiter.config.lost_policy = pol
                    runtime.arbiter.log("FSM", f"Stratégie si visage perdu : {pol}")

            elif action == "gesture":
                gid = str(msg.get("id", ""))
                if not runtime.arbiter.live_motors:
                    runtime.arbiter.log(
                        "WARN",
                        "Geste simulé uniquement (Moteurs en DRY-RUN — clique sur Moteurs : DRY-RUN en haut à droite !)",
                    )
                runtime.arbiter.trigger_gesture(gid)
                if gid == "look_desk" and runtime.facing != "back":
                    await runtime.switch_camera("back")
                elif gid == "look_up" and runtime.facing != "front":
                    await runtime.switch_camera("front")

            elif action == "manual":
                yaw = float(msg.get("yaw", 0.0))
                pitch = float(msg.get("pitch", 0.0))
                if yaw == 0.0 and pitch == 0.0:
                    runtime.arbiter.stop_manual()
                else:
                    if not runtime.arbiter.live_motors:
                        runtime.arbiter.log(
                            "WARN",
                            "D-Pad ignoré côté moteurs (mode DRY-RUN actif — passe en Moteurs : LIVE !)",
                        )
                    runtime.arbiter.set_manual(yaw, pitch, hold_s=0.6)

            elif action == "camera":
                if "facing" in msg or "zoom" in msg:
                    await runtime.configure_camera(
                        facing=str(msg["facing"]) if "facing" in msg else None,
                        zoom=float(msg["zoom"]) if "zoom" in msg else None,
                        wait_frame=False,
                    )
                if "rotate_deg" in msg:
                    runtime.rotate_deg = int(msg["rotate_deg"]) % 360
                    runtime.arbiter.log("CAMERA", f"Rotation image : {runtime.rotate_deg}°")
                if "mirror" in msg:
                    runtime.mirror = bool(msg["mirror"])

            elif action == "enter_desk_mode":
                z = float(msg["zoom"]) if "zoom" in msg and msg["zoom"] is not None else None
                asyncio.create_task(runtime.enter_desk_reading_mode(zoom=z, auto_target=bool(msg.get("auto_target", True))))

            elif action == "return_face_mode":
                asyncio.create_task(runtime.return_to_face_mode())

            elif action == "save_face_pose":
                runtime.save_current_face_pose()

            elif action == "save_desk_pose":
                runtime.save_current_desk_pose()

            elif action == "toggle_screen_item":
                idx = int(msg.get("index", -1))
                items = runtime.screen_card.get("items") or []
                if 0 <= idx < len(items):
                    items[idx]["done"] = not bool(items[idx].get("done"))
                    if items and all(bool(it.get("done")) for it in items) and runtime.hilight.connected:
                        runtime.hilight.start_reaction("yes")

            elif action == "clear_screen_card":
                runtime._came_from_desk_for_board = False
                runtime.screen_card = {"mode": "none", "title": "", "items": [], "note": "", "came_from_desk": False, "updated_at": 0.0}

            elif action == "pid":
                cfg = runtime.arbiter.config
                if "kp" in msg:
                    new_kp = float(msg["kp"])
                    ratio = cfg.kp_pitch / max(1.0, cfg.kp_yaw)
                    cfg.kp = new_kp
                    cfg.kp_yaw = round(new_kp * 1.25, 1)
                    cfg.kp_pitch = round(cfg.kp_yaw * ratio, 1)
                for field_name in ("kd", "deadzone", "max_dps"):
                    if field_name in msg:
                        setattr(cfg, field_name, float(msg[field_name]))
                for bfield in ("swap_axes", "invert_yaw", "invert_pitch"):
                    if bfield in msg:
                        setattr(cfg, bfield, bool(msg[bfield]))
    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(description="hackathon_osmo Studio Web App")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"🚀 hackathon_osmo Studio → http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
