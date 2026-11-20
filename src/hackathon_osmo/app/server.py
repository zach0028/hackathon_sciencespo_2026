"""FastAPI HTTP & WebSocket server exposing the hackathon_osmo Studio UI and voice endpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hackathon_osmo.app.runtime import StudioRuntime
from hackathon_osmo.paths import STATIC_DIR

runtime = StudioRuntime()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await runtime.start()
    try:
        yield
    finally:
        await runtime.stop()


app = FastAPI(title="hackathon_osmo Studio", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    return runtime.snapshot()


class VoiceTextPayload(BaseModel):
    text: str


@app.post("/api/voice/audio")
async def api_voice_audio(audio: UploadFile = File(...)) -> JSONResponse:
    raw = await audio.read()
    try:
        res = await runtime.handle_voice_turn(audio_bytes=raw)
        return JSONResponse(res.__dict__)
    except Exception as exc:
        runtime.arbiter.log("VOICE", f"Erreur pipeline vocal : {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/voice/text")
async def api_voice_text(payload: VoiceTextPayload) -> JSONResponse:
    try:
        res = await runtime.handle_voice_turn(user_text=payload.text.strip())
        return JSONResponse(res.__dict__)
    except Exception as exc:
        runtime.arbiter.log("VOICE", f"Erreur pipeline vocal : {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)


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
                if "model" in msg:
                    runtime.brain.model = str(msg["model"])
                    runtime.arbiter.log("BRAIN", f"Modèle cerveau : {runtime.brain.model}")
                if "voice_id" in msg:
                    runtime.brain.voice_id = str(msg["voice_id"])
                    runtime.arbiter.log("VOICE", f"Voix Voxtral 3 : {runtime.brain.voice_id}")
                if "use_vision" in msg:
                    runtime.brain.use_vision = bool(msg["use_vision"])
                if "audio_output" in msg and msg["audio_output"] in ("mac", "browser"):
                    runtime.brain.audio_output = str(msg["audio_output"])
                    runtime.arbiter.log("VOICE", f"Sortie audio : {runtime.brain.audio_output}")
                if msg.get("clear_history"):
                    runtime.brain.history.clear()
                    runtime.brain.transcript.clear()
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
                if "facing" in msg:
                    await runtime.switch_camera(str(msg["facing"]))
                if "rotate_deg" in msg:
                    runtime.rotate_deg = int(msg["rotate_deg"]) % 360
                    runtime.arbiter.log("CAMERA", f"Rotation image : {runtime.rotate_deg}°")
                if "mirror" in msg:
                    runtime.mirror = bool(msg["mirror"])

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
