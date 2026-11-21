"""StudioRuntime: orchestrates Pixel 11 Pro camera/HiLight, Apple Vision, MotionArbiter, Osmo BLE, and VoiceBrain."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import cv2
import numpy as np

from hackathon_osmo.ai.live_brain import (
    BRAIN_MODELS,
    LIVE_VOICES,
    BrainTurnResult,
    OpenAILiveBrain,
)
from hackathon_osmo.control.motion import (
    GESTURES,
    MotionArbiter,
    angle_diff_deg,
    screen_to_motor,
    wrap_angle_deg,
)
from hackathon_osmo.hardware.duml import CRC_CLASSIC, make_gimbal_joystick_packet
from hackathon_osmo.hardware.osmo_ble import OsmoBleClient, discover_osmo
from hackathon_osmo.hardware.pixel_camera import PixelCameraStream, list_adb_devices
from hackathon_osmo.hardware.pixel_hilight import PixelHiLight
from hackathon_osmo.paths import CALIBRATION_FILE
from hackathon_osmo.vision.face import AppleVisionFaceDetector, FaceBox
from hackathon_osmo.vision.sheet import AppleVisionSheetDetector, SheetDetection


class StudioRuntime:
    """Owns the live hardware loops and state shared with the FastAPI Studio server."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.facing: str = "front"
        self.camera_zoom: float = 1.0
        self.camera_bitrate: int = 8_000_000
        self.rotate_deg: int = 0  # Calibrated default Landscape orientation on OM3
        self.mirror: bool = True
        self.desk_pose: dict[str, Any] = {
            "pitch_deg": 46.0,
            "roll_deg": -66.3,
            "yaw_offset_deg": 180.0,
            "width": 2688,
            "height": 1512,
            "bitrate": 16_000_000,
            "default_zoom": 1.0,
        }
        self._dialogue_yaw_deg: float | None = None
        self._camera_gen: int = 0
        self._frame_ready_event = asyncio.Event()

        self.detector = AppleVisionFaceDetector()
        self.sheet_detector = AppleVisionSheetDetector()
        self._latest_sheet: SheetDetection = SheetDetection()
        self._last_sheet_det_t: float = 0.0
        self.hilight = PixelHiLight()
        self.arbiter = MotionArbiter()
        self.arbiter.on_request_recenter = self.recenter_landscape
        self.arbiter.on_light_reaction = self._trigger_light_reaction
        self.brain = OpenAILiveBrain(
            on_execute_tool=self.execute_robot_tool,
            on_get_frame=self._get_latest_frame_jpeg,
            on_get_context=self._get_robot_context,
            on_log=self.arbiter.log,
            on_light=self._trigger_light_reaction,
        )
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
        self._latest_raw_bgr: np.ndarray | None = None
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
            if isinstance(cal.get("desk_pose"), dict):
                self.desk_pose.update(cal["desk_pose"])
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
        self.arbiter.ai_hold_mode = False
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

    def _get_latest_frame_jpeg(self, high_res: bool = False) -> bytes | None:
        """Encode the latest native camera frame on demand (up to 2048px q=90 in high_res OCR mode)."""
        raw = self._latest_raw_bgr
        if raw is None:
            return None
        h, w = raw.shape[:2]
        max_dim = 2048 if high_res else 640
        quality = 90 if high_res else 75
        scale = min(1.0, max_dim / max(1, max(h, w)))
        img = (
            cv2.resize(raw, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            if scale < 0.99
            else raw
        )
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return enc.tobytes() if ok else None

    def _get_robot_context(self) -> str:
        interp = self.arbiter.latest
        att = self._osmo.last_attitude if self._osmo else None
        att_txt = (
            f"IMU(pitch={att.pitch_deg:+.1f}°,roll={att.roll_deg:+.1f}°,yaw={att.yaw_deg:+.1f}°)"
            if att
            else "IMU=indispo"
        )
        return (
            f"camera={self.facing} ({self.width}x{self.height} zoom={self.camera_zoom:.2f}x), "
            f"mode_bureau={self.arbiter.desk_mode}, ai_hold={self.arbiter.ai_hold_mode}, "
            f"visage={interp.face_detected} (x={interp.cx:+.2f},y={interp.cy:+.2f}), "
            f"etat={interp.state}, moteurs_live={self.arbiter.live_motors}, "
            f"hilight={self.hilight.current_effect}, {att_txt}"
        )

    def save_current_desk_pose(self) -> dict[str, Any]:
        """Persist current Osmo IMU attitude as the calibrated desk-reading pose in calibration.json."""
        att = self._osmo.last_attitude if self._osmo else None
        if att is None:
            return {"ok": False, "error": "IMU Osmo indisponible"}
        self.desk_pose["pitch_deg"] = round(att.pitch_deg, 1)
        self.desk_pose["roll_deg"] = round(att.roll_deg, 1)
        if self._dialogue_yaw_deg is not None:
            self.desk_pose["yaw_offset_deg"] = round(angle_diff_deg(att.yaw_deg, self._dialogue_yaw_deg), 1)
        try:
            cal = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8")) if CALIBRATION_FILE.exists() else {}
            cal["desk_pose"] = self.desk_pose
            CALIBRATION_FILE.write_text(json.dumps(cal, indent=2), encoding="utf-8")
            self.arbiter.log(
                "CALIB",
                f"Pose feuille enregistrée : pitch={self.desk_pose['pitch_deg']:+.1f}°, "
                f"roll={self.desk_pose['roll_deg']:+.1f}°, yaw_off={self.desk_pose['yaw_offset_deg']:+.1f}°",
            )
            return {"ok": True, "desk_pose": self.desk_pose}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def _send_body_speed(self, pan_dps: float, tilt_dps: float) -> None:
        """Send speed in front-camera canonical body frame directly to Osmo motors."""
        if not self._osmo or not self.osmo_connected:
            return
        cfg = self.arbiter.config
        m_yaw, m_pitch = screen_to_motor(
            pan_dps, tilt_dps, cfg, cfg.cal_rotate_deg, cfg.cal_mirror, facing="front"
        )
        await self._osmo.send_speed(yaw_dps=m_yaw, pitch_dps=m_pitch, roll_dps=0.0, duration_ds=2)
        self.osmo_tx_packets += 1
        self.osmo_last_tx = f"imu-servo mY={m_yaw:+.1f} mP={m_pitch:+.1f}"

    async def _servo_pitch_to_imu(self, target_pitch_deg: float, timeout_s: float = 3.2) -> float | None:
        """Closed-loop IMU pitch servo: auto-verifies motor sign so it can NEVER tilt toward the ceiling."""
        if not self._osmo or not self.osmo_connected or self._osmo.last_attitude is None:
            return None
        cur = self._osmo.last_attitude.pitch_deg
        if abs(target_pitch_deg - cur) <= 2.5:
            return cur

        # Probe direction briefly to confirm real IMU sign
        sign = 1.0 if target_pitch_deg > cur else -1.0
        probe_start = cur
        for _ in range(4):
            await self._send_body_speed(0.0, sign * 38.0)
            await asyncio.sleep(0.05)
        cur_after = self._osmo.last_attitude.pitch_deg
        if abs(target_pitch_deg - cur_after) > abs(target_pitch_deg - probe_start) + 0.6:
            sign = -sign

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and self.osmo_connected:
            cur = self._osmo.last_attitude.pitch_deg
            err = target_pitch_deg - cur
            if abs(err) <= 2.2:
                break
            desired_dir = sign if (err > 0) == (target_pitch_deg > probe_start) else -sign
            spd = max(18.0, min(68.0, abs(err) * 2.6))
            await self._send_body_speed(0.0, desired_dir * spd)
            await asyncio.sleep(0.05)

        await self._send_body_speed(0.0, 0.0)
        return self._osmo.last_attitude.pitch_deg

    async def _servo_yaw_to_imu(self, target_yaw_deg: float, timeout_s: float = 4.2) -> float | None:
        """Closed-loop IMU yaw servo for 180° arm pivot with automatic mechanical endstop bypass."""
        if not self._osmo or not self.osmo_connected or self._osmo.last_attitude is None:
            return None
        cur = self._osmo.last_attitude.yaw_deg
        if abs(angle_diff_deg(target_yaw_deg, cur)) <= 5.0:
            return cur

        # Start turning clockwise; flip direction if mechanical hardstop is hit before reaching 180°
        direction = 1.0
        stall_since: float | None = None
        last_yaw = cur
        flipped = False
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline and self.osmo_connected:
            cur = self._osmo.last_attitude.yaw_deg
            err = abs(angle_diff_deg(target_yaw_deg, cur))
            if err <= 4.5:
                break

            spd = max(25.0, min(88.0, err * 2.8))
            await self._send_body_speed(direction * spd, 0.0)
            await asyncio.sleep(0.05)

            delta = abs(angle_diff_deg(cur, last_yaw))
            if delta < 0.8 and spd >= 45.0:
                if stall_since is None:
                    stall_since = time.monotonic()
                elif time.monotonic() - stall_since > 0.35 and not flipped:
                    # Mechanical limit reached on this side -> reverse and swing around the open side!
                    direction = -direction
                    flipped = True
                    stall_since = None
                    self.arbiter.log("ARM", "Butée mécanique Yaw détectée → contournement par l'autre sens")
            else:
                stall_since = None
            last_yaw = cur

        await self._send_body_speed(0.0, 0.0)
        return self._osmo.last_attitude.yaw_deg

    async def configure_camera(
        self,
        *,
        facing: str | None = None,
        zoom: float | None = None,
        width: int | None = None,
        height: int | None = None,
        bitrate: int | None = None,
        wait_frame: bool = True,
    ) -> bool:
        """Reconfigure Pixel camera facing/resolution/optical zoom and optionally await first fresh frame."""
        new_facing = facing if facing in ("front", "back") else self.facing
        new_zoom = float(zoom) if zoom is not None else self.camera_zoom
        z_min, z_max = (0.51, 10.0) if new_facing == "back" else (0.9, 4.0)
        new_zoom = round(max(z_min, min(z_max, new_zoom)), 2)

        if width is not None and height is not None:
            new_w, new_h = int(width), int(height)
        elif new_facing == "back":
            new_w = int(self.desk_pose.get("width", 2688))
            new_h = int(self.desk_pose.get("height", 1512))
        else:
            new_w, new_h = 1280, 720

        new_br = int(bitrate) if bitrate is not None else (
            int(self.desk_pose.get("bitrate", 16_000_000)) if new_facing == "back" else 8_000_000
        )

        changed = (
            new_facing != self.facing
            or abs(new_zoom - self.camera_zoom) > 0.01
            or new_w != self.width
            or new_h != self.height
            or new_br != self.camera_bitrate
        )

        self.facing = new_facing
        self.camera_zoom = new_zoom
        self.width = new_w
        self.height = new_h
        self.camera_bitrate = new_br
        self.mirror = new_facing == "front"
        self.arbiter.desk_mode = new_facing == "back"

        if not changed and self._stream is not None:
            return True

        self.arbiter.log(
            "CAMERA",
            f"Optique {new_facing.upper()} {new_w}x{new_h} · zoom={new_zoom:.2f}x",
        )
        self._frame_ready_event.clear()
        self._camera_gen += 1
        await asyncio.to_thread(self._stop_camera_sync)

        if wait_frame:
            try:
                await asyncio.wait_for(self._frame_ready_event.wait(), timeout=4.5)
                await asyncio.sleep(0.18)  # let auto-exposure/autofocus settle
                return True
            except asyncio.TimeoutError:
                return False
        return True

    def _detect_sheet_sync(self, bgr: np.ndarray) -> SheetDetection:
        """Run Apple Vision rectangle + text detector on a fast 960px copy of the frame."""
        h, w = bgr.shape[:2]
        scale = min(1.0, 960.0 / max(1, max(h, w)))
        small = (
            cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            if scale < 0.99
            else bgr
        )
        return self.sheet_detector.detect(small)

    async def enter_desk_reading_mode(
        self,
        *,
        zoom: float | None = None,
        auto_target: bool = True,
        ultra_hd: bool = False,
    ) -> dict[str, Any]:
        """Full Desk Reading transition:
        1. 180° Yaw pivot + IMU pitch to calibrated desk pose (+46.0°) in parallel with rear HD/4K startup
        2. Apple Vision sheet/text detection on the desk surface (falls back to 0.51x ultra-wide if not visible)
        3. Auto-centers the mechanical arm on the detected sheet and selects optimal optical zoom (1x/2x/3x/5x)
        4. Returns razor-sharp HD image (up to 2048px) + pre-extracted OCR lines to gpt-6.2-astra!
        """
        target_pitch = float(self.desk_pose.get("pitch_deg", 46.0))
        yaw_offset = float(self.desk_pose.get("yaw_offset_deg", 180.0))
        req_w, req_h = (3840, 2160) if ultra_hd else (
            int(self.desk_pose.get("width", 2688)),
            int(self.desk_pose.get("height", 1512)),
        )
        initial_zoom = float(zoom) if zoom is not None else float(self.desk_pose.get("default_zoom", 1.0))

        if not self.arbiter.live_motors:
            self.arbiter.live_motors = True

        self.arbiter.desk_mode = True
        self.arbiter.ai_hold_mode = True
        if self.hilight.connected:
            self.hilight.start_reaction("sheet_search")

        att = self._osmo.last_attitude if (self._osmo and self.osmo_connected) else None
        already_on_desk = self.facing == "back" and att is not None and abs(att.pitch_deg - target_pitch) < 16.0

        async def _pivot_arm() -> tuple[float | None, float | None]:
            if already_on_desk or not self._osmo or not self.osmo_connected or self._osmo.last_attitude is None:
                return (att.yaw_deg if att else None, att.pitch_deg if att else None)

            if self._dialogue_yaw_deg is None:
                self._dialogue_yaw_deg = self._osmo.last_attitude.yaw_deg
            target_yaw = wrap_angle_deg(self._dialogue_yaw_deg + yaw_offset)

            self._hw_busy_until = time.monotonic() + 8.5
            try:
                self.arbiter.log(
                    "ARM",
                    f"Pivot 180° mode feuille (Yaw → {target_yaw:+.0f}°, Pitch → {target_pitch:+.1f}°)",
                )
                if self._osmo.is_landscape() is False:
                    await self._osmo.send_toggle_orientation()
                    await asyncio.sleep(1.0)
                final_yaw = await self._servo_yaw_to_imu(target_yaw)
                final_pitch = await self._servo_pitch_to_imu(target_pitch)
                return final_yaw, final_pitch
            finally:
                await self._send_body_speed(0.0, 0.0)
                self._hw_busy_until = 0.0

        cam_task = asyncio.create_task(
            self.configure_camera(
                facing="back",
                zoom=initial_zoom,
                width=req_w,
                height=req_h,
                bitrate=20_000_000 if ultra_hd else int(self.desk_pose.get("bitrate", 16_000_000)),
                wait_frame=True,
            )
        )
        arm_task = asyncio.create_task(_pivot_arm())
        cam_ok, (final_yaw, final_pitch) = await asyncio.gather(cam_task, arm_task)

        await asyncio.sleep(0.28)
        sheet = SheetDetection()
        auto_zoom_applied = self.camera_zoom

        if auto_target and self._latest_raw_bgr is not None:
            sheet = await asyncio.to_thread(self._detect_sheet_sync, self._latest_raw_bgr)
            self._latest_sheet = sheet

            # If no sheet seen at 1.0x and user didn't force a zoom, widen to 0.51x ultra-wide to survey the desk!
            if not sheet.found and zoom is None and self.camera_zoom > 0.6:
                self.arbiter.log("SHEET", "Aucune feuille vue en 1x → passage ultra grand-angle 0.51x pour scanner le bureau")
                await self.configure_camera(facing="back", zoom=0.51, width=req_w, height=req_h, wait_frame=True)
                await asyncio.sleep(0.25)
                if self._latest_raw_bgr is not None:
                    sheet = await asyncio.to_thread(self._detect_sheet_sync, self._latest_raw_bgr)
                    self._latest_sheet = sheet

            if sheet.found:
                # 1. Mechanical auto-centering onto the sheet center (using rear-camera FOV ~ 65°/zoom)
                err_x = sheet.cx - 0.5
                err_y = sheet.cy - 0.5
                fov_h = 65.0 / max(0.5, self.camera_zoom)
                fov_v = 40.0 / max(0.5, self.camera_zoom)
                if abs(err_x) > 0.11 or abs(err_y) > 0.11:
                    pan_corr = max(-35.0, min(35.0, err_x * fov_h * 0.75))
                    tilt_corr = max(-22.0, min(22.0, err_y * fov_v * 0.75))
                    self.arbiter.log(
                        "SHEET",
                        f"Feuille détectée à ({sheet.cx:.2f}, {sheet.cy:.2f}) → ciblage bras ({pan_corr:+.0f}°, {tilt_corr:+.0f}°)",
                    )
                    await self.arbiter.move_relative_deg(pan_corr, tilt_corr, speed_dps=42.0, hold_after=True)
                    await asyncio.sleep(0.25)

                # 2. Automatic optical lens selection if zoom was not explicitly pinned
                if zoom is None:
                    # Target sheet filling ~60-70% of the frame width/height for razor-sharp OCR
                    span = max(sheet.w, sheet.h, 0.12)
                    desired_z = self.camera_zoom * (0.62 / span)
                    # Snap to clean Pixel 11 Pro optical/sensor-crop steps: 1.0x, 2.0x, 3.0x, 5.0x
                    if desired_z >= 3.8:
                        chosen_z = 5.0
                    elif desired_z >= 2.4:
                        chosen_z = 3.0
                    elif desired_z >= 1.5:
                        chosen_z = 2.0
                    else:
                        chosen_z = 1.0
                    if abs(chosen_z - self.camera_zoom) >= 0.35:
                        self.arbiter.log(
                            "SHEET",
                            f"Auto-zoom optique sur la feuille : {self.camera_zoom:.2f}x → {chosen_z:.1f}x",
                        )
                        await self.configure_camera(facing="back", zoom=chosen_z, width=req_w, height=req_h, wait_frame=True)
                        await asyncio.sleep(0.28)
                        auto_zoom_applied = self.camera_zoom
                        if self._latest_raw_bgr is not None:
                            sheet = await asyncio.to_thread(self._detect_sheet_sync, self._latest_raw_bgr)
                            self._latest_sheet = sheet

        if self.hilight.connected:
            self.hilight.start_reaction("sheet_locked" if sheet.found else "desk")

        jpeg_hd = self._get_latest_frame_jpeg(high_res=True)
        att_now = self._osmo.last_attitude if self._osmo else None
        res: dict[str, Any] = {
            "ok": True,
            "mode": "desk_reading",
            "camera_ready": cam_ok,
            "resolution": f"{self.width}x{self.height}",
            "zoom": auto_zoom_applied,
            "sheet_detected": sheet.found,
            "sheet_has_text": sheet.has_text,
            "sheet_ocr_preview": sheet.lines[:8],
            "imu_yaw": round(att_now.yaw_deg, 1) if att_now else (round(final_yaw, 1) if final_yaw is not None else None),
            "imu_pitch": round(att_now.pitch_deg, 1) if att_now else (round(final_pitch, 1) if final_pitch is not None else None),
        }
        if jpeg_hd:
            res["_images"] = [
                {
                    "label": f"Feuille caméra arrière ({self.width}x{self.height}, zoom optique {self.camera_zoom:.2f}x)",
                    "jpeg": jpeg_hd,
                    "detail": "high",
                }
            ]
        return res

    async def set_camera_zoom(self, zoom: float, *, capture_after: bool = True) -> dict[str, Any]:
        """Switch Pixel optical/hybrid zoom (0.51x..10x on rear camera) and return a fresh HD frame."""
        ok = await self.configure_camera(zoom=zoom, wait_frame=True)
        res: dict[str, Any] = {
            "ok": ok,
            "facing": self.facing,
            "resolution": f"{self.width}x{self.height}",
            "zoom": self.camera_zoom,
        }
        if capture_after:
            await asyncio.sleep(0.25)
            jpeg_hd = self._get_latest_frame_jpeg(high_res=True)
            if jpeg_hd:
                res["_images"] = [
                    {
                        "label": f"Capture optique {self.facing.upper()} {self.width}x{self.height} zoom {self.camera_zoom:.2f}x",
                        "jpeg": jpeg_hd,
                        "detail": "high",
                    }
                ]
        return res

    async def return_to_face_mode(self) -> dict[str, Any]:
        """Return from Desk Reading mode to Dialogue mode (front 720p camera, recenter to face, tracking ON)."""
        self.arbiter.log("ARM", "Retour en mode Dialogue (caméra avant + recentrage visage)")
        if self.hilight.connected:
            self.hilight.start_reaction("off")

        cam_task = asyncio.create_task(
            self.configure_camera(facing="front", zoom=1.0, width=1280, height=720, bitrate=8_000_000, wait_frame=True)
        )
        rec_task = asyncio.create_task(self.recenter_landscape())
        await asyncio.gather(cam_task, rec_task)

        self.arbiter.desk_mode = False
        self.arbiter.ai_hold_mode = False
        self._dialogue_yaw_deg = None
        return {"ok": True, "mode": "dialogue_face", "facing": "front", "zoom": 1.0}

    async def execute_robot_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Native OpenAI Responses tool executor for DJI Osmo Mobile 3 + Pixel 11 Pro HiLight/Camera."""
        if name == "enter_desk_reading_mode":
            z_raw = args.get("zoom")
            zoom = float(z_raw) if (z_raw is not None and float(z_raw) > 0) else None
            auto_target = bool(args.get("auto_target", True))
            ultra_hd = bool(args.get("ultra_hd", False))
            return await self.enter_desk_reading_mode(
                zoom=zoom,
                auto_target=auto_target,
                ultra_hd=ultra_hd,
            )

        if name == "set_camera_zoom":
            zoom = float(args.get("zoom", 1.0))
            capture_after = bool(args.get("capture_after", True))
            return await self.set_camera_zoom(zoom=zoom, capture_after=capture_after)

        if name == "return_to_face_mode":
            return await self.return_to_face_mode()

        if name == "move_gimbal":
            if not self.osmo_connected:
                return {"ok": False, "error": "Nacelle DJI Osmo non connectée"}
            if not self.arbiter.live_motors:
                self.arbiter.live_motors = True
                self.arbiter.log("ARM", "Moteurs activés automatiquement pour mouvement IA")
            pan_deg = float(args.get("pan_deg", 0.0))
            tilt_deg = float(args.get("tilt_deg", 0.0))
            speed_dps = float(args.get("speed_dps", 50.0))
            hold_pos = bool(args.get("hold_position", True))
            capture_after = bool(args.get("capture_after", True))

            pan_done, tilt_done, dur_s = await self.arbiter.move_relative_deg(
                pan_deg=pan_deg,
                tilt_deg=tilt_deg,
                speed_dps=speed_dps,
                hold_after=hold_pos,
            )
            res: dict[str, Any] = {
                "ok": True,
                "moved_pan_deg": pan_done,
                "moved_tilt_deg": tilt_done,
                "duration_s": dur_s,
                "holding_position": hold_pos,
            }
            if capture_after:
                await asyncio.sleep(0.25)
                is_desk = self.facing == "back"
                jpeg = self._get_latest_frame_jpeg(high_res=is_desk)
                if jpeg:
                    res["_images"] = [
                        {
                            "label": f"Vue après rotation bras (pan={pan_done:+.0f}°, tilt={tilt_done:+.0f}°)",
                            "jpeg": jpeg,
                            "detail": "high" if is_desk else "low",
                        }
                    ]
            return res

        if name == "scan_room":
            if not self.osmo_connected:
                return {"ok": False, "error": "Nacelle DJI Osmo non connectée"}
            if not self.arbiter.live_motors:
                self.arbiter.live_motors = True
            span_deg = max(80.0, min(220.0, float(args.get("span_deg", 150.0))))
            speed_dps = max(35.0, min(85.0, float(args.get("speed_dps", 55.0))))
            return_center = bool(args.get("return_to_center", True))
            half = span_deg / 2.0

            self.arbiter.ai_hold_mode = True
            if self.hilight.connected:
                self.hilight.start_reaction("search")
            self.arbiter.log("ARM", f"Balayage panoramique IA ({span_deg:.0f}° @ {speed_dps:.0f}°/s)")

            captures: list[dict[str, Any]] = []
            # 1. Sweep LEFT
            await self.arbiter.move_relative_deg(-half, 0.0, speed_dps, hold_after=True)
            await asyncio.sleep(0.28)
            j_left = self._get_latest_frame_jpeg(high_res=False)
            if j_left:
                captures.append({"label": f"GAUCHE (-{half:.0f}°)", "jpeg": j_left, "detail": "low"})

            # 2. Sweep CENTER
            await self.arbiter.move_relative_deg(+half, 0.0, speed_dps, hold_after=True)
            await asyncio.sleep(0.22)
            j_mid = self._get_latest_frame_jpeg(high_res=False)
            if j_mid:
                captures.append({"label": "CENTRE (0°)", "jpeg": j_mid, "detail": "low"})

            # 3. Sweep RIGHT
            await self.arbiter.move_relative_deg(+half, 0.0, speed_dps, hold_after=True)
            await asyncio.sleep(0.28)
            j_right = self._get_latest_frame_jpeg(high_res=False)
            if j_right:
                captures.append({"label": f"DROITE (+{half:.0f}°)", "jpeg": j_right, "detail": "low"})

            # 4. Return to center if requested
            if return_center:
                await self.arbiter.move_relative_deg(-half, 0.0, speed_dps, hold_after=False)
                self.arbiter.ai_hold_mode = False

            if self.hilight.connected:
                self.hilight.start_reaction("found")

            return {
                "ok": True,
                "scanned_span_deg": span_deg,
                "views_captured": len(captures),
                "returned_to_center": return_center,
                "_images": captures,
            }

        if name == "set_tracking_mode":
            mode = str(args.get("mode", "follow_face"))
            if mode == "follow_face":
                self.arbiter.ai_hold_mode = False
                self.arbiter.desk_mode = False
                self.arbiter.log("ARM", "Suivi visage automatique réactivé par l'IA")
                return {"ok": True, "mode": "follow_face"}
            if mode == "free_control":
                self.arbiter.ai_hold_mode = True
                self.arbiter.log("ARM", "Mode contrôle libre IA activé (suivi visage en pause)")
                return {"ok": True, "mode": "free_control"}
            return {"ok": False, "error": f"Mode inconnu: {mode}"}

        if name == "trigger_gesture":
            gesture = str(args.get("gesture", ""))
            if gesture not in GESTURES:
                return {"ok": False, "error": f"Geste inconnu: {gesture}"}
            if gesture == "look_up":
                self.arbiter.ai_hold_mode = False
            self.arbiter.trigger_gesture(gesture)
            if gesture == "look_desk" and self.facing != "back":
                asyncio.create_task(self.switch_camera("back"))
            elif gesture == "look_up" and self.facing != "front":
                asyncio.create_task(self.switch_camera("front"))
            return {"ok": True, "gesture": gesture}

        if name == "set_hilight":
            effect = str(args.get("effect", "off"))
            if self.hilight.connected:
                self.hilight.start_reaction(effect)
                self.arbiter.log("HILIGHT", f"Outil IA → LED HiLight : {effect}")
                return {"ok": True, "effect": effect}
            return {"ok": False, "error": "HiLight non connecté"}

        if name == "control_gimbal":
            action = str(args.get("action", ""))
            if action == "recenter":
                self.arbiter.ai_hold_mode = False
                asyncio.create_task(self.recenter_landscape())
                return {"ok": True, "action": "recenter"}
            if action == "search":
                self.arbiter.trigger_search_now()
                return {"ok": True, "action": "search"}
            if action == "toggle_orientation":
                asyncio.create_task(self.toggle_orientation())
                return {"ok": True, "action": "toggle_orientation"}
            return {"ok": False, "error": f"Action nacelle inconnue: {action}"}

        if name == "switch_camera":
            facing = str(args.get("facing", "front"))
            if facing not in ("front", "back"):
                return {"ok": False, "error": f"Caméra invalide: {facing}"}
            asyncio.create_task(self.switch_camera(facing))
            return {"ok": True, "facing": facing}

        return {"ok": False, "error": f"Outil inconnu: {name}"}

    async def create_live_session(self, sdp_offer: str) -> dict[str, str]:
        return await self.brain.create_live_session(sdp_offer)

    async def stop_live_session(self) -> None:
        await self.brain.stop_live_session()
        self.arbiter.log("LIVE", "Session GPT-Live raccrochée")

    async def handle_text_turn(self, user_text: str) -> BrainTurnResult:
        self.arbiter.log("BRAIN", f"Zach (texte) : « {user_text} »")
        return await self.brain.ask_brain(
            user_text,
            include_recent_history=True,
            speak_via_live=self.brain.sideband_connected,
        )

    async def start(self) -> None:
        self._running = True
        self.arbiter.log("SYSTEM", "Démarrage du studio hackathon_osmo (OpenAI GPT-Live + Astra)")
        self._tasks = [
            asyncio.create_task(self._camera_loop(), name="camera-loop"),
            asyncio.create_task(self._osmo_loop(), name="osmo-loop"),
            asyncio.create_task(self._hilight_loop(), name="hilight-loop"),
        ]

    async def stop(self) -> None:
        self._running = False
        await self.brain.close()
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
        await self.configure_camera(facing=facing, zoom=1.0, wait_frame=False)

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
        # Downscale preview copy to max 1280px wide so browser WebSocket stays buttery 28-30fps even in 4K/2.7K
        ih, iw = img.shape[:2]
        if iw > 1280:
            scale = 1280.0 / iw
            vis = cv2.resize(img, (1280, max(1, int(ih * scale))), interpolation=cv2.INTER_AREA)
        else:
            vis = img.copy()
        h, w = vis.shape[:2]
        cfg = self.arbiter.config
        interp = self.arbiter.latest

        tx = w // 2
        ty = int(h * (0.5 + cfg.target_y / 2.0))
        dz_x = int(w * cfg.deadzone / 2.0)
        dz_y = int(h * cfg.deadzone / 2.0)

        if self.arbiter.desk_mode:
            # Draw desk-reading reticle + live Apple Vision sheet & text overlay
            cv2.line(vis, (w // 2 - 16, h // 2), (w // 2 + 16, h // 2), (255, 210, 90), 1)
            cv2.line(vis, (w // 2, h // 2 - 16), (w // 2, h // 2 + 16), (255, 210, 90), 1)
            sheet = self._latest_sheet
            if sheet.found:
                if len(sheet.quad) == 4:
                    pts = np.array([[int(px * w), int(py * h)] for px, py in sheet.quad], dtype=np.int32)
                    cv2.polylines(vis, [pts], isClosed=True, color=(80, 240, 140), thickness=2)
                else:
                    sx, sy = int(sheet.x * w), int(sheet.y * h)
                    sw, sh = int(sheet.w * w), int(sheet.h * h)
                    cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), (80, 240, 140), 2)
                for bx, by, bw, bh in sheet.text_boxes[:16]:
                    cv2.rectangle(
                        vis,
                        (int(bx * w), int(by * h)),
                        (int((bx + bw) * w), int((by + bh) * h)),
                        (0, 220, 255),
                        1,
                    )
                scx, scy = int(sheet.cx * w), int(sheet.cy * h)
                cv2.circle(vis, (scx, scy), 6, (80, 240, 140), -1)
                lbl = f"FEUILLE ({len(sheet.lines)} lignes) · {self.width}x{self.height} · {self.camera_zoom:.2f}x"
                cv2.putText(vis, lbl, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 240, 140), 2, cv2.LINE_AA)
            else:
                lbl = f"MODE BUREAU · {self.width}x{self.height} · zoom {self.camera_zoom:.2f}x (recherche feuille...)"
                cv2.putText(vis, lbl, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 200, 80), 2, cv2.LINE_AA)
        else:
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
            f"{self.facing.upper()} {self.width}x{self.height} {self.camera_zoom:.1f}x  |  "
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
                        zoom=self.camera_zoom,
                        bitrate=self.camera_bitrate,
                    )
                    await asyncio.to_thread(stream.start)
                    self._stream = stream
                    last_frame_mono = time.monotonic()
                    self.camera_status = f"live_{self.facing}"
                    self.arbiter.log(
                        "CAMERA",
                        f"Pixel 11 Pro ({self.pixel_serial}) flux {self.facing.upper()} "
                        f"{self.width}x{self.height}@{self.fps}fps · zoom={self.camera_zoom:.2f}x actif",
                    )
                except Exception as exc:
                    self.camera_status = f"error: {exc}"
                    self.arbiter.log("CAMERA", f"Erreur caméra : {exc}")
                    await asyncio.sleep(2.0)
                    continue

            raw, ts = self._stream.read()
            if raw is None or ts == last_ts:
                if time.monotonic() - last_frame_mono > 3.5:
                    self.arbiter.log("CAMERA", "Flux caméra interrompu — reconnexion automatique...")
                    await asyncio.to_thread(self._stop_camera_sync)
                    self._fps_actual = 0.0
                    continue
                await asyncio.sleep(0.008)
                continue
            last_ts = ts
            last_frame_mono = time.monotonic()

            img = self._transform_frame(raw)
            self._latest_raw_bgr = img
            self._frame_ready_event.set()

            now = time.monotonic()
            if self.arbiter.desk_mode:
                faces = []
                det_ms = self._latest_sheet.vision_ms
                # Periodic live Apple Vision sheet/text detection (~2 Hz) without blocking video FPS
                if now - self._last_sheet_det_t > 0.55:
                    self._last_sheet_det_t = now
                    sheet = await asyncio.to_thread(self._detect_sheet_sync, img)
                    self._latest_sheet = sheet
                    det_ms = sheet.vision_ms
            else:
                t0 = time.monotonic()
                faces = await asyncio.to_thread(self.detector.detect, img)
                det_ms = (time.monotonic() - t0) * 1000.0

            self._latest_faces = faces
            self.arbiter.update_vision(faces, det_ms)

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
                        facing=self.facing,
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
                "width": self.width,
                "height": self.height,
                "zoom": self.camera_zoom,
                "rotate_deg": self.rotate_deg,
                "mirror": self.mirror,
                "fps": round(self._fps_actual, 1),
                "desk_pose": self.desk_pose,
                "sheet": {
                    "found": self._latest_sheet.found,
                    "has_text": self._latest_sheet.has_text,
                    "lines_count": len(self._latest_sheet.lines),
                    "cx": round(self._latest_sheet.cx, 2),
                    "cy": round(self._latest_sheet.cy, 2),
                },
            },
            "voice": {
                "ready": self.brain.ready,
                "live_connected": self.brain.live_connected,
                "sideband_connected": self.brain.sideband_connected,
                "session_id": self.brain.session_id,
                "busy": self.brain.busy,
                "speaking": self.brain.speaking,
                "live_model": self.brain.live_model,
                "brain_model": self.brain.brain_model,
                "reasoning_effort": self.brain.reasoning_effort,
                "voice": self.brain.voice,
                "use_vision": self.brain.use_vision,
                "last_brain_ms": self.brain.last_brain_ms,
                "models": BRAIN_MODELS,
                "voices": LIVE_VOICES,
                "transcript": list(self.brain.transcript)[:16],
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
