"""Automated 2x2 Jacobian optical-flow calibration for Pixel 11 Pro + DJI Osmo Mobile 3.

Automatically determines:
- Whether motor axes are swapped relative to the current image orientation (portrait vs landscape)
- Exact negative-feedback polarity (invert_yaw, invert_pitch)
- Critically damped per-axis PID gains (kp_yaw, kp_pitch, kd)
and saves everything into `calibration.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hackathon_osmo.hardware.osmo_ble import OsmoBleClient, discover_osmo
from hackathon_osmo.hardware.pixel_camera import PixelCameraStream, list_adb_devices
from hackathon_osmo.paths import CALIBRATION_FILE


def transform_frame(img: np.ndarray, rotate_deg: int, mirror: bool) -> np.ndarray:
    if rotate_deg == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotate_deg == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rotate_deg == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mirror:
        img = cv2.flip(img, 1)
    return img


def to_gray_f32(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Downsample preserving aspect ratio (~320px max dim) and apply Hann window."""
    h, w = img.shape[:2]
    scale = 320.0 / max(h, w)
    sw, sh = int(round(w * scale)), int(round(h * scale))
    small = cv2.resize(img, (sw, sh))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow((sw, sh), cv2.CV_32F)
    return (gray - float(gray.mean())) * win, sw, sh


async def measure_speed_pulse(
    stream: PixelCameraStream,
    client: OsmoBleClient,
    rotate_deg: int,
    mirror: bool,
    yaw_dps: float,
    pitch_dps: float,
    duration: float = 0.45,
) -> dict[str, Any]:
    prev_gray: np.ndarray | None = None
    sw = sh = 1
    last_ts = 0.0
    t_wait = time.monotonic()
    while time.monotonic() - t_wait < 2.5:
        f, ts = stream.read()
        if f is not None and ts != last_ts:
            prev_gray, sw, sh = to_gray_f32(transform_frame(f, rotate_deg, mirror))
            last_ts = ts
            break
        await asyncio.sleep(0.01)

    if prev_gray is None:
        raise RuntimeError("No camera frame received during calibration")

    total_dx = 0.0
    total_dy = 0.0
    samples = 0

    t0 = time.monotonic()
    while time.monotonic() - t0 < duration:
        await client.send_speed(yaw_dps=yaw_dps, pitch_dps=pitch_dps, duration_ds=2)
        await asyncio.sleep(0.04)
        f, ts = stream.read()
        if f is not None and ts != last_ts:
            last_ts = ts
            cur_gray, _, _ = to_gray_f32(transform_frame(f, rotate_deg, mirror))
            (dx, dy), resp = cv2.phaseCorrelate(prev_gray, cur_gray)
            if resp > 0.06:
                total_dx += dx
                total_dy += dy
                samples += 1
            prev_gray = cur_gray

    for _ in range(3):
        await client.send_speed(0.0, 0.0, duration_ds=1)
        await asyncio.sleep(0.04)
    await asyncio.sleep(0.30)

    for _ in range(4):
        f, ts = stream.read()
        if f is not None and ts != last_ts:
            last_ts = ts
            cur_gray, _, _ = to_gray_f32(transform_frame(f, rotate_deg, mirror))
            (dx, dy), resp = cv2.phaseCorrelate(prev_gray, cur_gray)
            if resp > 0.06:
                total_dx += dx
                total_dy += dy
                samples += 1
            prev_gray = cur_gray
        await asyncio.sleep(0.04)

    return {
        "yaw_cmd": yaw_dps,
        "pitch_cmd": pitch_dps,
        "duration": duration,
        "dx_px": round(total_dx, 2),
        "dy_px": round(total_dy, 2),
        "dx_norm": round(total_dx / (sw / 2.0), 3),
        "dy_norm": round(total_dy / (sh / 2.0), 3),
        "frame_size": f"{sw}x{sh}",
        "samples": samples,
    }


async def run_calibration(
    rotate_deg: int = 0, mirror: bool = False, facing: str = "back"
) -> dict[str, Any]:
    devs = [d for d in list_adb_devices() if d.state == "device"]
    if not devs:
        raise RuntimeError("Pixel 11 Pro non détecté en ADB.")
    serial = devs[0].serial

    osmo_dev = await discover_osmo(timeout=5.0)
    if osmo_dev is None:
        raise RuntimeError("DJI Osmo Mobile 3 non détecté en BLE.")

    print(f"Calibration 2x2 Jacobienne : Pixel ({serial}, {facing}) + Osmo ({osmo_dev.name})")
    print(f"Orientation cible : rotate={rotate_deg}°, mirror={mirror}\n")

    with PixelCameraStream(serial=serial, width=1280, height=720, fps=30, facing=facing) as stream:
        t0 = time.monotonic()
        while stream.read()[0] is None and time.monotonic() - t0 < 8.0:
            await asyncio.sleep(0.05)

        async with OsmoBleClient(osmo_dev) as client:
            await asyncio.sleep(0.4)

            # Axis A = Osmo send_speed(yaw_dps=...)
            print("[1/2] Mesure Axe A (yaw_dps ±35°/s)...")
            a_pos = await measure_speed_pulse(stream, client, rotate_deg, mirror, +35.0, 0.0, 0.45)
            a_neg = await measure_speed_pulse(stream, client, rotate_deg, mirror, -35.0, 0.0, 0.45)
            print("  A+ :", a_pos)
            print("  A- :", a_neg)

            # Axis B = Osmo send_speed(pitch_dps=...)
            print("[2/2] Mesure Axe B (pitch_dps ±35°/s)...")
            b_pos = await measure_speed_pulse(stream, client, rotate_deg, mirror, 0.0, +35.0, 0.45)
            b_neg = await measure_speed_pulse(stream, client, rotate_deg, mirror, 0.0, -35.0, 0.45)
            print("  B+ :", b_pos)
            print("  B- :", b_neg)

    # Average symmetric normalized shift per +35 deg/s over 0.45s (= 15.75 deg)
    a_dx = (a_pos["dx_norm"] - a_neg["dx_norm"]) / 2.0
    a_dy = (a_pos["dy_norm"] - a_neg["dy_norm"]) / 2.0
    b_dx = (b_pos["dx_norm"] - b_neg["dx_norm"]) / 2.0
    b_dy = (b_pos["dy_norm"] - b_neg["dy_norm"]) / 2.0

    # Determine whether Axis A controls horizontal (X) or vertical (Y) in the rotated image!
    # Normal (swap_axes=False): Axis A (yaw_dps) -> X, Axis B (pitch_dps) -> Y
    # Swapped (swap_axes=True): Axis B (pitch_dps) -> X, Axis A (yaw_dps) -> Y
    score_normal = abs(a_dx) + abs(b_dy)
    score_swapped = abs(b_dx) + abs(a_dy)
    swap_axes = score_swapped > score_normal

    if not swap_axes:
        horiz_resp = a_dx  # normalized dx produced by +35 deg/s on horizontal motor
        vert_resp = b_dy   # normalized dy produced by +35 deg/s on vertical motor
    else:
        horiz_resp = b_dx
        vert_resp = a_dy

    # Negative feedback rule:
    # - Face on RIGHT (err_x > 0) needs image content to shift LEFT (dx < 0)
    #   -> if +cmd gives dx > 0, invert_yaw = True
    # - Face at BOTTOM (err_y > 0) needs image content to shift UP (dy < 0)
    #   -> if +cmd gives dy > 0, invert_pitch = True
    invert_yaw = horiz_resp > 0
    invert_pitch = vert_resp > 0

    # Plant gain: norm_units / (deg/s * s)
    k_horiz = max(0.004, abs(horiz_resp) / (35.0 * 0.45))
    k_vert = max(0.004, abs(vert_resp) / (35.0 * 0.45))

    # Critically damped cinema tracking time constant tau ~ 0.80s (zero overshoot with ~80ms loop delay)
    kp_yaw = round(min(90.0, max(15.0, 1.0 / (0.80 * k_horiz))), 1)
    kp_pitch = round(min(90.0, max(15.0, 1.0 / (0.85 * k_vert))), 1)
    kp_avg = round((kp_yaw + kp_pitch) / 2.0, 1)

    cal = {
        "facing": facing,
        "rotate_deg": rotate_deg,
        "mirror": mirror,
        "control_mode": "speed",
        "swap_axes": bool(swap_axes),
        "invert_yaw": bool(invert_yaw),
        "invert_pitch": bool(invert_pitch),
        "kp": kp_avg,
        "kp_yaw": kp_yaw,
        "kp_pitch": kp_pitch,
        "kd": round(kp_avg * 0.08, 1),
        "deadzone": 0.09,
        "max_dps": 55.0,
        "measured": {
            "a_dx_norm": round(a_dx, 3),
            "a_dy_norm": round(a_dy, 3),
            "b_dx_norm": round(b_dx, 3),
            "b_dy_norm": round(b_dy, 3),
            "k_horiz": round(k_horiz, 4),
            "k_vert": round(k_vert, 4),
        },
    }
    CALIBRATION_FILE.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(f"\n✅ Calibration 2x2 sauvegardée dans {CALIBRATION_FILE}:")
    print(json.dumps(cal, indent=2))
    return cal


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-calibrate DJI Osmo Mobile 3 + Pixel 11 Pro")
    parser.add_argument("--facing", choices=["front", "back"], default="back")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--mirror", action="store_true", help="Mirror horizontal axis (default off for back camera)")
    args = parser.parse_args()
    asyncio.run(run_calibration(rotate_deg=args.rotate, mirror=args.mirror, facing=args.facing))


if __name__ == "__main__":
    main()
