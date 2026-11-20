"""3-way sync test harness: Google Pixel 11 Pro (USB-C/ADB) + MacBook Pro + DJI Osmo Mobile 3 (BLE)."""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass

import cv2

from hackathon_osmo.hardware.duml import CRC_CLASSIC, make_gimbal_joystick_packet
from hackathon_osmo.hardware.osmo_ble import GimbalAttitude, OsmoBleClient, discover_osmo
from hackathon_osmo.hardware.pixel_camera import (
    PixelCameraStream,
    find_adb,
    find_ffmpeg,
    find_scrcpy,
    list_adb_devices,
    list_pixel_cameras,
)
from hackathon_osmo.vision.face import AppleVisionFaceDetector


@dataclass
class PidState:
    kp: float = 55.0
    kd: float = 12.0
    deadzone: float = 0.06
    max_dps: float = 45.0
    prev_err_x: float = 0.0
    prev_err_y: float = 0.0
    prev_t: float = 0.0

    def step(self, err_x: float, err_y: float, now: float) -> tuple[float, float]:
        dt = max(1e-3, now - self.prev_t) if self.prev_t > 0 else 1 / 30.0
        self.prev_t = now

        def axis(err: float, prev_err: float) -> float:
            if abs(err) < self.deadzone:
                return 0.0
            d_err = (err - prev_err) / dt
            val = self.kp * err + self.kd * d_err
            return max(-self.max_dps, min(self.max_dps, val))

        vx = axis(err_x, self.prev_err_x)
        vy = axis(err_y, self.prev_err_y)
        self.prev_err_x = err_x
        self.prev_err_y = err_y
        return vx, vy


async def cmd_status() -> None:
    print("┌────────────────────────────────────────────────────────────────┐")
    print("│          hackathon_osmo — Diagnostic de Synchro 3-en-1         │")
    print("└────────────────────────────────────────────────────────────────┘")

    # 1. Tools
    print("\n[1/3] Outils MacBook")
    print(f"  adb    : {find_adb() or 'MISSING'}")
    print(f"  scrcpy : {find_scrcpy() or 'MISSING'}")
    print(f"  ffmpeg : {find_ffmpeg() or 'MISSING'}")

    # 2. Pixel via ADB
    print("\n[2/3] Google Pixel 11 Pro (USB-C / ADB)")
    devs = list_adb_devices()
    if not devs:
        print("  ❌ Aucun appareil ADB détecté.")
    else:
        for d in devs:
            icon = "✅" if d.state == "device" else "⚠️"
            extra = f" ({d.model})" if d.model else ""
            print(f"  {icon} {d.serial}{extra} — état: {d.state}")
            if d.state == "unauthorized":
                print("     → Regarde l'écran du Pixel et clique sur « Autoriser le débogage USB ».")
            elif d.state == "device":
                cams = list_pixel_cameras(d.serial)
                lines = [ln for ln in cams.splitlines() if "--camera-id" in ln or "facing" in ln]
                print(f"     → Caméras détectées ({len(lines)}):")
                for ln in lines[:8]:
                    print(f"       {ln.strip()}")

    # 3. DJI Osmo Mobile 3 via BLE
    print("\n[3/3] DJI Osmo Mobile 3 (Bluetooth LE)")
    dev = await discover_osmo(timeout=4.0)
    if dev is None:
        print("  ❌ Osmo Mobile non détecté en BLE.")
        return

    print(f"  ✅ Détecté : {dev.name} ({dev.address})")
    pkt_count = 0

    def on_pkt(_pkt: object) -> None:
        nonlocal pkt_count
        pkt_count += 1

    async with OsmoBleClient(dev, on_packet=on_pkt) as client:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            await asyncio.sleep(0.1)
        att: GimbalAttitude | None = client.last_attitude
        print(f"  ✅ Paquets DUML reçus en 2s : {pkt_count}")
        if att:
            print(
                f"  ✅ Attitude temps réel : pitch={att.pitch_deg:+.1f}° | roll={att.roll_deg:+.1f}° | yaw={att.yaw_deg:+.1f}°"
            )
            if abs(att.roll_deg) > 25.0:
                print(
                    "  ⚠️  Roll important détecté : vérifie que l'Osmo est déplié et stabilisé avant de lancer motor-test."
                )


async def cmd_motor_test(mode: str, speed_dps: float, duration: float) -> None:
    dev = await discover_osmo(timeout=4.0)
    if dev is None:
        print("❌ Osmo Mobile non détecté en BLE.")
        return

    print(f"Connexion à {dev.name} ({dev.address}) — mode={mode}, vitesse={speed_dps}°/s, durée={duration}s")
    async with OsmoBleClient(dev) as client:
        await asyncio.sleep(0.6)
        start_att = client.last_attitude
        print(f"  Attitude initiale : {start_att}")

        async def drive(label: str, yaw_dps: float, pitch_dps: float, secs: float) -> None:
            print(f"  → {label} (yaw={yaw_dps:+.1f}°/s, pitch={pitch_dps:+.1f}°/s pendant {secs:.1f}s)...")
            t0 = time.monotonic()
            seq = 1
            while time.monotonic() - t0 < secs:
                if mode == "speed":
                    await client.send_speed(yaw_dps=yaw_dps, pitch_dps=pitch_dps)
                else:
                    # Joystick PWM mode (CmdId 0x01): center=1024, offset proportional to speed
                    j_yaw = int(max(364, min(1684, 1024 + yaw_dps * 6)))
                    j_pitch = int(max(364, min(1684, 1024 + pitch_dps * 6)))
                    seq += 1
                    pkt = make_gimbal_joystick_packet(
                        yaw=j_yaw, pitch=j_pitch, roll=1024, seq=seq, profile=CRC_CLASSIC
                    )
                    await client.send_raw(pkt)
                await asyncio.sleep(0.05)  # 20 Hz
            # Stop explicitly
            for _ in range(4):
                if mode == "speed":
                    await client.send_speed(0.0, 0.0)
                else:
                    seq += 1
                    await client.send_raw(
                        make_gimbal_joystick_packet(1024, 1024, 1024, seq=seq, profile=CRC_CLASSIC)
                    )
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.4)
            print(f"    Attitude après : {client.last_attitude}")

        await drive("YAW +", +speed_dps, 0.0, duration)
        await drive("YAW -", -speed_dps, 0.0, duration)
        await drive("PITCH +", 0.0, +speed_dps * 0.6, duration * 0.8)
        await drive("PITCH -", 0.0, -speed_dps * 0.6, duration * 0.8)
        print("✅ Test moteur terminé.")


async def cmd_track_test(
    source: str,
    facing: str,
    width: int,
    height: int,
    fps: int,
    dry_run: bool,
    invert_yaw: bool,
    invert_pitch: bool,
    headless_seconds: float | None,
) -> None:
    detector = AppleVisionFaceDetector()

    osmo_client: OsmoBleClient | None = None
    if not dry_run:
        dev = await discover_osmo(timeout=4.0)
        if dev is None:
            print("⚠️  Osmo non détecté : bascule automatique en --dry-run.")
            dry_run = True
        else:
            osmo_client = OsmoBleClient(dev)
            await osmo_client.connect()
            print(f"✅ Osmo connecté en BLE : {dev.name}")

    pid = PidState()
    t_start = time.monotonic()
    frames_seen = 0
    faces_seen = 0

    async def process_frame(frame: object) -> bool:
        nonlocal frames_seen, faces_seen
        import numpy as np

        img: np.ndarray = frame  # type: ignore[assignment]
        frames_seen += 1
        now = time.monotonic()
        h, w = img.shape[:2]

        t_det = time.monotonic()
        faces = detector.detect(img)
        det_ms = (time.monotonic() - t_det) * 1000.0

        vx, vy = 0.0, 0.0
        if faces:
            faces_seen += 1
            best = max(faces, key=lambda f: f.area)
            cx = best.cx * 2.0 - 1.0  # -1 (left) .. +1 (right)
            cy = best.cy * 2.0 - 1.0  # -1 (top) .. +1 (bottom)
            target_y = -0.10  # slight headroom
            err_x = cx
            err_y = cy - target_y
            vx, vy = pid.step(err_x, err_y, now)
            if invert_yaw:
                vx = -vx
            if invert_pitch:
                vy = -vy

            x, y = int(best.x * w), int(best.y * h)
            fw, fh = int(best.w * w), int(best.h * h)
            cv2.rectangle(img, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.circle(img, (x + fw // 2, y + fh // 2), 5, (0, 255, 255), -1)

        if osmo_client is not None and not dry_run:
            await osmo_client.send_speed(yaw_dps=vx, pitch_dps=vy)

        if headless_seconds is not None:
            if frames_seen == 1 or frames_seen % 15 == 0:
                print(
                    f"  frame={frames_seen:4d} ({w}x{h}) faces={faces_seen:4d} "
                    f"vision={det_ms:4.1f}ms cmd=(yaw={vx:+5.1f}°/s, pitch={vy:+5.1f}°/s)"
                )
            return (now - t_start) < headless_seconds

        # Draw deadzone + HUD
        dz_x, dz_y = int(w * pid.deadzone / 2), int(h * pid.deadzone / 2)
        cv2.rectangle(
            img,
            (w // 2 - dz_x, int(h * 0.45) - dz_y),
            (w // 2 + dz_x, int(h * 0.45) + dz_y),
            (255, 200, 0),
            1,
        )
        mode_lbl = "DRY-RUN" if dry_run else "LIVE BLE"
        cv2.putText(
            img,
            f"[{mode_lbl}] yaw={vx:+5.1f} deg/s  pitch={vy:+5.1f} deg/s  (q = quitter)",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow("hackathon_osmo — Face Tracking", img)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    try:
        if source == "pixel":
            devs = [d for d in list_adb_devices() if d.state == "device"]
            if not devs:
                print("❌ Pixel non autorisé ou absent en ADB. Vérifie `osmo-sync-test status`.")
                return
            serial = devs[0].serial
            print(f"Démarrage flux caméra Pixel ({serial}) {width}x{height}@{fps}fps ({facing})...")
            with PixelCameraStream(
                serial=serial, width=width, height=height, fps=fps, facing=facing
            ) as stream:
                for frame, _ts in stream.frames(timeout=12.0):
                    keep = await process_frame(frame)
                    if not keep:
                        break
        else:
            cam_idx = int(source) if source.isdigit() else 0
            cap = cv2.VideoCapture(cam_idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            if not cap.isOpened():
                print(f"❌ Impossible d'ouvrir la caméra {source}.")
                return
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        await asyncio.sleep(0.01)
                        continue
                    keep = await process_frame(frame)
                    if not keep:
                        break
            finally:
                cap.release()
    finally:
        cv2.destroyAllWindows()
        if osmo_client is not None:
            try:
                await osmo_client.send_speed(0.0, 0.0)
            except Exception:
                pass
            await osmo_client.disconnect()
        print(f"Terminé : {frames_seen} frames traitées, visage détecté sur {faces_seen} frames.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Banc de test synchro Pixel 11 Pro + MacBook + Osmo Mobile 3")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Vérifie l'état des 3 appareils (Pixel ADB, Mac, Osmo BLE)")

    p_mot = sub.add_parser("motor-test", help="Teste un mouvement moteur doux sur l'Osmo Mobile 3")
    p_mot.add_argument("--mode", choices=["speed", "joystick"], default="speed")
    p_mot.add_argument("--speed", type=float, default=20.0, help="Vitesse en deg/s")
    p_mot.add_argument("--duration", type=float, default=0.8, help="Durée par axe en secondes")

    p_trk = sub.add_parser("track-test", help="Lance la boucle Face Tracking + PID + Osmo BLE")
    p_trk.add_argument("--source", default="pixel", help="'pixel' (via ADB/scrcpy) ou index caméra macOS (ex. 0)")
    p_trk.add_argument("--facing", choices=["front", "back"], default="front")
    p_trk.add_argument("--width", type=int, default=1280)
    p_trk.add_argument("--height", type=int, default=720)
    p_trk.add_argument("--fps", type=int, default=30)
    p_trk.add_argument("--dry-run", action="store_true", help="N'envoie aucune commande moteur à l'Osmo")
    p_trk.add_argument("--invert-yaw", action="store_true")
    p_trk.add_argument("--invert-pitch", action="store_true")
    p_trk.add_argument("--headless-seconds", type=float, default=None, help="Mode sans fenêtre pendant N secondes")

    args = parser.parse_args()
    if args.cmd == "status":
        asyncio.run(cmd_status())
    elif args.cmd == "motor-test":
        asyncio.run(cmd_motor_test(args.mode, args.speed, args.duration))
    elif args.cmd == "track-test":
        asyncio.run(
            cmd_track_test(
                source=args.source,
                facing=args.facing,
                width=args.width,
                height=args.height,
                fps=args.fps,
                dry_run=args.dry_run,
                invert_yaw=args.invert_yaw,
                invert_pitch=args.invert_pitch,
                headless_seconds=args.headless_seconds,
            )
        )


if __name__ == "__main__":
    main()
