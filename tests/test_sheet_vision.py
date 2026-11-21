"""Tests for desk sheet preprocessing, sharpness AF gate, and camera push caching."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from hackathon_osmo.ai.live_brain import prepare_vision_jpeg
from hackathon_osmo.hardware.pixel_camera import PixelCameraStream
from hackathon_osmo.vision.sheet import (
    PointingTarget,
    SheetDetection,
    _roi_paper_metrics,
    compute_sharpness,
    crop_pointing_macro,
    prepare_desk_reading_image,
    rotate_image,
    slice_desk_spread_tiles,
)


def _make_synthetic_page(w: int = 1600, h: int = 900) -> np.ndarray:
    img = np.full((h, w, 3), 235, dtype=np.uint8)
    for y in range(120, h - 120, 45):
        cv2.line(img, (200, y), (w - 200, y), (180, 180, 200), 1)
    cv2.putText(img, "Solution 1 -> After Desk", (320, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (30, 30, 160), 3)
    return img


def test_compute_sharpness_distinguishes_blur_from_focus() -> None:
    sharp = _make_synthetic_page()
    blurry = cv2.GaussianBlur(sharp, (31, 31), sigmaX=10.0)
    s_sharp = compute_sharpness(sharp)
    s_blur = compute_sharpness(blurry)
    assert s_sharp > 40.0
    assert s_blur < 10.0
    assert s_sharp > s_blur * 5.0


def test_rotate_image_270_swaps_axes_ccw() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    # Mark right edge white -> after 90° CCW, right edge becomes TOP edge
    img[:, -10:] = 255
    rot = rotate_image(img, 270)
    assert rot.shape == (200, 100, 3)
    assert rot[:10, :].mean() == 255.0
    assert rot[-10:, :].mean() == 0.0


def test_prepare_desk_reading_image_crops_wide_1x_sheet_and_orients_upright() -> None:
    img = _make_synthetic_page(2688, 1512)
    # Sheet occupies 18% of desk (like 017_turn_in_desk.jpg)
    sheet = SheetDetection(found=True, x=0.26, y=0.16, w=0.41, h=0.45, confidence=0.88, source="doc_seg")
    out, cropped = prepare_desk_reading_image(img, rotate_deg=270, sheet=sheet, auto_crop=True, enhance=True)
    assert cropped is True
    # Cropped upright sheet should be significantly smaller than full 1512x2688 rotated desk
    assert out.shape[0] < 2400
    assert out.shape[1] < 1500
    assert out.shape[0] > 600 and out.shape[1] > 600


def test_prepare_desk_reading_image_skips_crop_when_already_zoomed() -> None:
    img = _make_synthetic_page(2048, 1152)
    # Sheet occupies 45% (zoomed 2x shot -> must NOT clip edge words!)
    sheet = SheetDetection(found=True, x=0.10, y=0.10, w=0.75, h=0.60, confidence=0.95, source="doc_seg")
    out, cropped = prepare_desk_reading_image(img, rotate_deg=270, sheet=sheet, auto_crop=True, enhance=True)
    assert cropped is False
    assert out.shape == (2048, 1152, 3)  # rotated 270° cleanly with zero clipping


def test_prepare_vision_jpeg_passthrough_avoids_double_compression() -> None:
    img = _make_synthetic_page(1200, 800)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    assert ok
    raw = enc.tobytes()
    # Already <= 1600px in high_res mode -> exact same bytes returned!
    assert prepare_vision_jpeg(raw, high_res=True) is raw
    # In low-res mode (640px), must downscale
    low = prepare_vision_jpeg(raw, high_res=False)
    assert low is not None and low is not raw
    dec = cv2.imdecode(np.frombuffer(low, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert max(dec.shape[:2]) == 640


def test_pixel_camera_caches_scrcpy_server_push_per_serial() -> None:
    PixelCameraStream._pushed_devices.clear()
    cam = PixelCameraStream(serial="10.214.131.8:5555")
    cam._adb = "/usr/local/bin/adb"
    with patch("hackathon_osmo.hardware.pixel_camera.subprocess.run", return_value=MagicMock(returncode=0)) as m_run:
        cam._ensure_server_pushed("/tmp/scrcpy-server", "5.0.1")
        assert m_run.call_count == 2  # push + chmod 444
        # Second camera switch on same serial -> 0 subprocess calls!
        cam._ensure_server_pushed("/tmp/scrcpy-server", "5.0.1")
        assert m_run.call_count == 2
        # Forced retry re-pushes
        cam._ensure_server_pushed("/tmp/scrcpy-server", "5.0.1", force=True)
        assert m_run.call_count == 4
    PixelCameraStream._pushed_devices.clear()


def test_pixel_oled_face_assets_exist_and_serve_cleanly() -> None:
    from hackathon_osmo.paths import STATIC_DIR

    for fname in ("face.html", "face.css", "face.js", "face-manifest.json"):
        p = STATIC_DIR / fname
        assert p.exists(), f"Missing {fname}"
        assert p.stat().st_size > 100
    html = (STATIC_DIR / "face.html").read_text(encoding="utf-8")
    assert 'id="face-canvas"' in html
    assert 'id="desk-layer"' in html
    assert 'id="focus-layer"' in html


def test_roi_paper_metrics_separates_cream_paper_from_dark_laptop() -> None:
    frame = np.full((400, 600, 3), 65, dtype=np.uint8)  # dark laptop / desk
    frame[180:360, 120:480] = (205, 225, 235)  # warm cream paper in BGR
    luma_dark, _ = _roi_paper_metrics(frame, 0.0, 0.0, 0.5, 0.35)
    luma_paper, sat_paper = _roi_paper_metrics(frame, 0.25, 0.50, 0.5, 0.35)
    assert luma_dark < 120.0
    assert luma_paper > 180.0
    assert sat_paper < 80.0


def test_slice_desk_spread_tiles_produces_overlapping_left_and_right_pages() -> None:
    spread = _make_synthetic_page(2688, 1512)
    tiles = slice_desk_spread_tiles(spread, rotate_deg=270, enhance=True)
    assert len(tiles) == 2
    assert "GAUCHE" in tiles[0][0]
    assert "DROITE" in tiles[1][0]
    # Both tiles should be crisp high-res upright regions
    for _lbl, img in tiles:
        assert img.shape[0] >= 600 and img.shape[1] >= 400


def test_crop_pointing_macro_extracts_crisp_region_around_fingertip() -> None:
    page = _make_synthetic_page(1512, 2688)
    ptr = PointingTarget(found=True, tip_x=0.21, tip_y=0.35, target_x=0.24, target_y=0.29, confidence=0.85)
    patch = crop_pointing_macro(page, ptr, box_frac=0.42, enhance=True)
    assert patch is not None
    assert patch.shape[1] >= 500 and patch.shape[0] >= 900
    assert crop_pointing_macro(page, PointingTarget(found=False)) is None



