"""StudioRuntime: orchestrates Pixel 11 Pro camera/HiLight, Apple Vision, MotionArbiter, Osmo BLE, and VoiceBrain."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import cv2
import numpy as np

from hackathon_osmo.ai.voice_brain import FRENCH_VOICES, LLM_MODELS, VoiceBrain, VoiceTurnResult
from hackathon_osmo.control.motion import GESTURES, MotionArbiter, screen_to_motor
from hackathon_osmo.hardware.duml import CRC_CLASSIC, make_gimbal_joystick_packet
from hackathon_osmo.hardware.osmo_ble import OsmoBleClient, discover_osmo
from hackathon_osmo.hardware.pixel_camera import PixelCameraStream, list_adb_devices
from hackathon_osmo.hardware.pixel_hilight import PixelHiLight
from hackathon_osmo.paths import CALIBRATION_FILE
from hackathon_osmo.vision.face import AppleVisionFaceDetector, FaceBox


class StudioRuntime:
    """Owns the live hardware loops and state shared with the FastAPI Studio server."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.facing: str = "front"
        self.rotate_deg: int = 0  # Calibrated default Landscape orientation on OM3
        self.mirror: bool = True

        self.detector = AppleVisionFaceDetector()
        self.hilight = PixelHiLight()
        self.brain = VoiceBrain()
        self.arbiter = MotionArbiter()
        self.arbiter.on_request_recenter = self.recenter_landscape
        self.arbiter.on_light_reaction = self._trigger_light_reaction
        self._load_calibration()

        self.pixel_serial: str | None = None
        self.camera_status: str = "disconnected"
        self.osmo_name: str | None = None
        self.osmo_connected: bool = False
        self.osmo_packets: int = 0
        self.osmo_tx_packets: int = 0
        self.osmo_last_tx: str = ""
        self._hw_busy_until: float = 0.0

        self._stream: PixelCameraStream | None = None
        self._osmo: OsmoBleClient | None = None
        self._latest_jpeg: bytes | None = None
        self._latest_clean_jpeg: bytes | None = None
        self._latest_faces: list[FaceBox] = []
        self._fps_actual: float = 0.0
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []
        self._ble_seq = 1

    def _load_calibration(self) -> None:
        if not CALIBRATION_FILE.exists():
            return
        try:
            cal = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
            self.rotate_deg = int(cal.get("rotate_deg", 0))
            self.mirror = bool(cal.get("mirror", True))
            self.arbiter.control_mode = str(cal.get("control_mode", "speed"))
            cfg = self.arbiter.config
            cfg.cal_rotate_deg = self.rotate_deg
            cfg.cal_mirror = self.mirror
            cfg.swap_axes = bool(cal.get("swap_axes", True))
            cfg.invert_yaw = bool(cal.get("invert_yaw", True))
            cfg.invert_pitch = bool(cal.get("invert_pitch", True))
            cfg.kp_yaw = round(float(cal.get("kp_yaw", 58.6)), 1)
            cfg.kp_pitch = round(float(cal.get("kp_pitch", 31.4)), 1)
            cfg.kp = round((cfg.kp_yaw + cfg.kp_pitch) / 2.0, 1)
            cfg.kd = round(float(cal.get("kd", 3.6)), 1)
            cfg.deadzone = float(cal.get("deadzone", 0.09))
            cfg.max_dps = float(cal.get("max_dps", 55.0))
            self.arbiter.live_motors = True
            self.arbiter.log(
                "CALIB",
                f"Calibration 2x2 Paysage chargée : rot={self.rotate_deg}°, swap={cfg.swap_axes}, "
                f"inv=({cfg.invert_yaw},{cfg.invert_pitch}), Kp=({cfg.kp_yaw}/{cfg.kp_pitch})",
            )
        except Exception as exc:
            self.arbiter.log("CALIB", f"Erreur lecture calibration.json : {exc}")

    async def recenter_landscape(self) -> None:
        """Recenter gimbal to neutral level position, ensuring Landscape orientation by default."""
        if not self._osmo or not self.osmo_connected:
            return
        self._hw_busy_until = time.monotonic() + 2.6
        is_land = self._osmo.is_landscape()
        if is_land is False:
            self.arbiter.log("OSMO", "Nacelle en portrait détectée → bascule automatique en Paysage")
            await self._osmo.send_toggle_orientation()
            await asyncio.sleep(1.0)
        self.arbiter.log("OSMO", "Recentrage matériel en format Paysage (CmdId 0x4C)")
        await self._osmo.send_recenter()
        self.rotate_deg = self.arbiter.config.cal_rotate_deg
        await asyncio.sleep(1.2)
        self._hw_busy_until = 0.0

    async def toggle_orientation(self) -> None:
        """Toggle physical Landscape <-> Portrait on OM3."""
        if not self._osmo or not self.osmo_connected:
            return
        self._hw_busy_until = time.monotonic() + 1.5
        self.arbiter.log("OSMO", "Bascule physique Paysage ↔ Portrait (CmdId 0x3A)")
        await self._osmo.send_toggle_orientation()
        await asyncio.sleep(1.2)
        self._hw_busy_until = 0.0

    def _trigger_light_reaction(self, reaction: str) -> None:
        if self.hilight.connected:
            self.hilight.start_reaction(reaction)

    async def execute_ai_reaction(self, gesture: str | None, led: str | None) -> None:
        """Fire embodied robot gesture on DJI Osmo + HiLight LED from AI brain output."""
        if gesture == "recenter":
            asyncio.create_task(self.recenter_landscape())
        elif gesture == "search":
            self.arbiter.trigger_search_now()
        elif gesture in GESTURES:
            self.arbiter.trigger_gesture(gesture)
            if gesture == "look_desk" and self.facing != "back":
                asyncio.create_task(self.switch_camera("back"))
            elif gesture == "look_up" and self.facing != "front":
                asyncio.create_task(self.switch_camera("front"))

        if led and self.hilight.connected:
            self.hilight.start_reaction(led)

    async def handle_voice_turn(
        self,
        *,
        user_text: str | None = None,
        audio_bytes: bytes | None = None,
    ) -> VoiceTurnResult:
        self.brain.busy = True
        stt_ms = 0.0
        try:
            if user_text is None:
                if not audio_bytes:
                    raise ValueError("Audio vide")
                self.arbiter.log("VOICE", f"Transcription Voxtral 3 ({len(audio_bytes)//1024} Ko)...")
                user_text, stt_ms = await self.brain.transcribe(audio_bytes)
                if not user_text:
                    self.arbiter.log("VOICE", "Aucune parole détectée dans l'enregistrement")
                    return VoiceTurnResult(
                        user_text="",
                        reply_text="",
                        gesture=None,
                        led=None,
                        audio_b64=None,
                        stt_ms=stt_ms,
                        model=self.brain.model,
                    )

            self.arbiter.log("VOICE", f"Zach : « {user_text} » ({stt_ms:.0f}ms)")
            interp = self.arbiter.latest
            ctx = (
                f"Caméra={self.facing}, Visage détecté={interp.face_detected} "
                f"(pos x={interp.cx:+.2f}, y={interp.cy:+.2f}), état={interp.state}."
            )
            reply, gesture, led, llm_ms = await self.brain.think(
                user_text,
                frame_jpeg=self._latest_clean_jpeg,
                robot_context=ctx,
            )
            self.arbiter.log(
                "BRAIN",
                f"Osmo ({self.brain.model}, {llm_ms:.0f}ms) [geste={gesture or '-'} led={led or '-'}] : « {reply} »",
            )

            wav_bytes, audio_b64, tts_ms = await self.brain.synthesize(reply)

            # Fire physical gesture + HiLight LED in sync with voice playback
            await self.execute_ai_reaction(gesture, led)
            played_on_mac = False
            if wav_bytes and self.brain.audio_output == "mac":
                played_on_mac = True
                asyncio.create_task(self.brain.play_on_mac(wav_bytes))

            res = VoiceTurnResult(
                user_text=user_text,
                reply_text=reply,
                gesture=gesture,
                led=led,
                audio_b64=None if played_on_mac else audio_b64,
                played_on_mac=played_on_mac,
                stt_ms=stt_ms,
                llm_ms=llm_ms,
                tts_ms=tts_ms,
                model=self.brain.model,
            )
            self.brain.record_turn(res)
            return res
        finally:
            self.brain.busy = False

    async def start(self) -> None:
        self._running = True
        self.arbiter.log("SYSTEM", "Démarrage du studio hackathon_osmo")
        self._tasks = [
            asyncio.create_task(self._camera_loop(), name="camera-loop"),
            asyncio.create_task(self._osmo_loop(), name="osmo-loop"),
            asyncio.create_task(self._hilight_loop(), name="hilight-loop"),
        ]

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.hilight.disconnect()
        self._stop_camera_sync()
        if self._osmo:
            try:
                await self._osmo.send_speed(0.0, 0.0)
                await self._osmo.disconnect()
            except Exception:
                pass
            self._osmo = None

    def _stop_camera_sync(self) -> None:
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
            self._stream = None
        self.camera_status = "disconnected"

    async def switch_camera(self, facing: str) -> None:
        if facing not in ("front", "back"):
            return
        self.facing = facing
        self.mirror = facing == "front"
        self.arbiter.desk_mode = facing == "back"
        self.arbiter.log(
            "CAMERA",
            f"Bascule vers caméra {facing.upper()}"
            + (" (mode feuille : tracking visage en pause)" if facing == "back" else " (mode visage actif)"),
        )
        await asyncio.to_thread(self._stop_camera_sync)

    def _transform_frame(self, img: np.ndarray) -> np.ndarray:
        if self.rotate_deg == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate_deg == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif self.rotate_deg == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.mirror:
            img = cv2.flip(img, 1)
        return img

    def _render_overlay(self, img: np.ndarray, faces: list[FaceBox]) -> bytes:
        vis = img.copy()
        h, w = vis.shape[:2]
        cfg = self.arbiter.config
        interp = self.arbiter.latest

        tx = w // 2
        ty = int(h * (0.5 + cfg.target_y / 2.0))
        dz_x = int(w * cfg.deadzone / 2.0)
        dz_y = int(h * cfg.deadzone / 2.0)

        cv2.rectangle(vis, (tx - dz_x, ty - dz_y), (tx + dz_x, ty + dz_y), (255, 200, 60), 1)
        cv2.line(vis, (tx - 14, ty), (tx + 14, ty), (255, 200, 60), 1)
        cv2.line(vis, (tx, ty - 14), (tx, ty + 14), (255, 200, 60), 1)

        best = max(faces, key=lambda f: f.area) if faces else None
        for f in faces:
            x, y = int(f.x * w), int(f.y * h)
            fw, fh = int(f.w * w), int(f.h * h)
            is_best = f is best
            color = (80, 240, 140) if is_best else (140, 140, 140)
            cv2.rectangle(vis, (x, y), (x + fw, y + fh), color, 2 if is_best else 1)
            if is_best:
                fcx, fcy = int(f.cx * w), int(f.cy * h)
                cv2.circle(vis, (fcx, fcy), 6, (0, 230, 255), -1)
                cv2.arrowedLine(vis, (tx, ty), (fcx, fcy), (0, 210, 255), 2, tipLength=0.2)
                lbl = f"FACE {int(f.confidence * 100)}% ({interp.cx:+.2f}, {interp.cy:+.2f})"
                cv2.putText(
                    vis, lbl, (x, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA
                )

        mode_tag = "MOTEURS LIVE" if self.arbiter.live_motors else "DRY-RUN (SIMULATION)"
        hud_color = (80, 240, 140) if self.arbiter.live_motors else (80, 200, 255)
        cv2.rectangle(vis, (0, h - 38), (w, h), (15, 18, 24), -1)
        hud = (
            f"{mode_tag}  |  {interp.state}  |  "
            f"yaw={interp.cmd_yaw_dps:+5.1f} deg/s  pitch={interp.cmd_pitch_dps:+5.1f} deg/s  |  "
            f"{self._fps_actual:4.1f} fps  vision={interp.vision_ms:4.1f}ms"
        )
        cv2.putText(vis, hud, (14, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 1, cv2.LINE_AA)

        ok, jpg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return jpg.tobytes() if ok else b""

    async def _camera_loop(self) -> None:
        frame_times: list[float] = []
        last_ts = 0.0
        last_frame_mono = 0.0
        while self._running:
            if self._stream is None:
                devs = [d for d in list_adb_devices() if d.state == "device"]
                if not devs:
                    self.camera_status = "waiting_adb"
                    await asyncio.sleep(1.5)
                    continue
                self.pixel_serial = devs[0].serial
                self.camera_status = f"starting_{self.facing}"
                try:
                    stream = PixelCameraStream(
                        serial=self.pixel_serial,
                        width=self.width,
                        height=self.height,
                        fps=self.fps,
                        facing=self.facing,
                    )
                    await asyncio.to_thread(stream.start)
                    self._stream = stream
                    last_frame_mono = time.monotonic()
                    self.camera_status = f"live_{self.facing}"
                    self.arbiter.log(
                        "CAMERA",
                        f"Pixel 11 Pro ({self.pixel_serial}) flux {self.facing.upper()} {self.width}x{self.height}@{self.fps}fps actif",
                    )
                except Exception as exc:
                    self.camera_status = f"error: {exc}"
                    self.arbiter.log("CAMERA", f"Erreur caméra : {exc}")
                    await asyncio.sleep(2.0)
                    continue

            raw, ts = self._stream.read()
            if raw is None or ts == last_ts:
                if time.monotonic() - last_frame_mono > 3.0:
                    self.arbiter.log("CAMERA", "Flux caméra interrompu — reconnexion automatique...")
                    await asyncio.to_thread(self._stop_camera_sync)
                    self._fps_actual = 0.0
                    continue
                await asyncio.sleep(0.008)
                continue
            last_ts = ts
            last_frame_mono = time.monotonic()

            img = self._transform_frame(raw)
            t0 = time.monotonic()
            faces = await asyncio.to_thread(self.detector.detect, img)
            det_ms = (time.monotonic() - t0) * 1000.0

            self._latest_faces = faces
            self.arbiter.update_vision(faces, det_ms)

            if len(frame_times) % 3 == 0:
                small_clean = cv2.resize(img, (480, 270))
                ok_c, jpg_c = cv2.imencode(".jpg", small_clean, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok_c:
                    self._latest_clean_jpeg = jpg_c.tobytes()

            now = time.monotonic()
            frame_times.append(now)
            frame_times = [t for t in frame_times if now - t <= 1.0]
            self._fps_actual = float(len(frame_times))

            self._latest_jpeg = await asyncio.to_thread(self._render_overlay, img, faces)

    async def _hilight_loop(self) -> None:
        while self._running:
            if not self.hilight.connected:
                serial = self.pixel_serial
                if not serial:
                    devs = [d for d in list_adb_devices() if d.state == "device"]
                    serial = devs[0].serial if devs else None
                if serial:
                    try:
                        ok = await self.hilight.connect(serial)
                        if ok:
                            self.arbiter.log(
                                "HILIGHT",
                                f"Anneau LED HiLight Pixel 11 Pro connecté ({self.hilight.num_lights} LEDs RGB)",
                            )
                    except Exception as exc:
                        self.arbiter.log("HILIGHT", f"Erreur connexion HiLight : {exc}")
            await asyncio.sleep(2.0)

    async def _osmo_loop(self) -> None:
        while self._running:
            if self._osmo is None or not self.osmo_connected:
                try:
                    dev = await discover_osmo(timeout=3.0)
                    if dev is None:
                        await asyncio.sleep(2.0)
                        continue

                    def on_pkt(_pkt: object) -> None:
                        self.osmo_packets += 1

                    client = OsmoBleClient(dev, on_packet=on_pkt)
                    await client.connect()
                    self._osmo = client
                    self.osmo_name = dev.name or dev.address
                    self.osmo_connected = True
                    self.arbiter.log("OSMO", f"Connecté en BLE à {self.osmo_name}")
                except Exception as exc:
                    self.osmo_connected = False
                    self._osmo = None
                    self.arbiter.log("OSMO", f"Attente reconnexion BLE ({exc})")
                    await asyncio.sleep(2.0)
                    continue

            try:
                if self.arbiter.live_motors and time.monotonic() >= self._hw_busy_until:
                    interp = self.arbiter.latest
                    motor_yaw, motor_pitch = screen_to_motor(
                        interp.cmd_yaw_dps,
                        interp.cmd_pitch_dps,
                        self.arbiter.config,
                        self.rotate_deg,
                        self.mirror,
                    )

                    if self.arbiter.control_mode == "speed":
                        await self._osmo.send_speed(
                            yaw_dps=motor_yaw,
                            pitch_dps=motor_pitch,
                            roll_dps=interp.cmd_roll_dps,
                            duration_ds=2,
                        )
                        self.osmo_tx_packets += 1
                        self.osmo_last_tx = f"speed mY={motor_yaw:+.1f} mP={motor_pitch:+.1f}"
                    elif self.arbiter.control_mode == "relative":
                        if abs(motor_yaw) > 0.5 or abs(motor_pitch) > 0.5:
                            await self._osmo.send_relative(
                                yaw_deg=motor_yaw * 0.12,
                                pitch_deg=motor_pitch * 0.12,
                                duration_ds=2,
                            )
                            self.osmo_tx_packets += 1
                            self.osmo_last_tx = f"rel mY={motor_yaw*0.12:+.1f}° mP={motor_pitch*0.12:+.1f}°"
                    else:
                        j_yaw = int(max(364, min(1684, 1024 + motor_yaw * 6)))
                        j_pitch = int(max(364, min(1684, 1024 + motor_pitch * 6)))
                        self._ble_seq = (self._ble_seq + 1) & 0xFFFF or 1
                        pkt = make_gimbal_joystick_packet(
                            yaw=j_yaw,
                            pitch=j_pitch,
                            roll=1024,
                            seq=self._ble_seq,
                            profile=CRC_CLASSIC,
                        )
                        await self._osmo.send_raw(pkt)
                        self.osmo_tx_packets += 1
                        self.osmo_last_tx = f"joy yaw={j_yaw} pitch={j_pitch}"
                await asyncio.sleep(0.05)  # 20 Hz
            except Exception as exc:
                self.arbiter.log("OSMO", f"Lien BLE interrompu : {exc}")
                self.osmo_connected = False
                if self._osmo:
                    try:
                        await self._osmo.disconnect()
                    except Exception:
                        pass
                    self._osmo = None
                await asyncio.sleep(1.0)

    def snapshot(self) -> dict[str, Any]:
        att = self._osmo.last_attitude if self._osmo else None
        interp = self.arbiter.latest
        cfg = self.arbiter.config
        return {
            "camera": {
                "status": self.camera_status,
                "serial": self.pixel_serial,
                "facing": self.facing,
                "rotate_deg": self.rotate_deg,
                "mirror": self.mirror,
                "fps": round(self._fps_actual, 1),
            },
            "voice": {
                "ready": self.brain.ready,
                "busy": self.brain.busy,
                "speaking": self.brain.speaking,
                "model": self.brain.model,
                "voice_id": self.brain.voice_id,
                "use_vision": self.brain.use_vision,
                "audio_output": self.brain.audio_output,
                "models": LLM_MODELS,
                "voices": FRENCH_VOICES,
                "transcript": list(self.brain.transcript)[:12],
            },
            "hilight": {
                "connected": self.hilight.connected,
                "num_lights": self.hilight.num_lights,
                "effect": self.hilight.current_effect,
                "color": self.hilight.current_color,
            },
            "osmo": {
                "connected": self.osmo_connected,
                "name": self.osmo_name,
                "packets": self.osmo_packets,
                "tx_packets": self.osmo_tx_packets,
                "last_tx": self.osmo_last_tx,
                "live_motors": self.arbiter.live_motors,
                "control_mode": self.arbiter.control_mode,
                "is_landscape": self._osmo.is_landscape() if self._osmo else None,
                "attitude": (
                    {
                        "pitch": round(att.pitch_deg, 1),
                        "roll": round(att.roll_deg, 1),
                        "yaw": round(att.yaw_deg, 1),
                    }
                    if att
                    else None
                ),
            },
            "vision": {
                "face_detected": interp.face_detected,
                "cx": round(interp.cx, 3),
                "cy": round(interp.cy, 3),
                "area": round(interp.box_area, 4),
                "confidence": round(interp.confidence, 2),
                "err_x": round(interp.err_x, 3),
                "err_y": round(interp.err_y, 3),
                "vel_x": round(interp.vel_x, 2),
                "vel_y": round(interp.vel_y, 2),
                "vision_ms": interp.vision_ms,
                "state": interp.state,
                "summary": interp.summary,
                "active_gesture": interp.active_gesture,
                "cmd_yaw_dps": round(interp.cmd_yaw_dps, 1),
                "cmd_pitch_dps": round(interp.cmd_pitch_dps, 1),
            },
            "pid": {
                "kp": cfg.kp,
                "kd": cfg.kd,
                "deadzone": cfg.deadzone,
                "max_dps": cfg.max_dps,
                "swap_axes": cfg.swap_axes,
                "invert_yaw": cfg.invert_yaw,
                "invert_pitch": cfg.invert_pitch,
                "lost_policy": cfg.lost_policy,
            },
            "gestures": [{"id": k, "label": v[0]} for k, v in GESTURES.items()],
            "events": list(self.arbiter.events)[:25],
        }
