"""Unit tests for MotionArbiter, OneEuroFilter, and rotation-invariant screen_to_motor."""

from __future__ import annotations

import pytest

from hackathon_osmo.control.motion import (
    MotionArbiter,
    OneEuroFilter,
    PidConfig,
    angle_diff_deg,
    screen_to_motor,
    wrap_angle_deg,
)
from hackathon_osmo.vision.face import FaceBox


def _cal_cfg() -> PidConfig:
    """Calibrated landscape config matching real Pixel 11 Pro + OM3 setup."""
    return PidConfig(
        cal_rotate_deg=0,
        cal_mirror=True,
        swap_axes=True,
        invert_yaw=True,
        invert_pitch=True,
    )


def _transform_point(x: float, y: float, rotate_deg: int, mirror: bool) -> tuple[float, float]:
    """Simulate how a physical camera-frame vector (x, y) transforms under CW rotation + horizontal mirror."""
    # Step 1: CW rotation by rotate_deg
    r = rotate_deg % 360
    if r == 90:
        rx, ry = -y, x
    elif r == 180:
        rx, ry = -x, -y
    elif r == 270:
        rx, ry = y, -x
    else:
        rx, ry = x, y
    # Step 2: Horizontal mirror
    if mirror:
        rx = -rx
    return rx, ry


@pytest.mark.parametrize("rotate_deg", [0, 90, 180, 270])
@pytest.mark.parametrize("mirror", [False, True])
def test_screen_to_motor_invariant_to_rotation_and_mirror(rotate_deg: int, mirror: bool) -> None:
    """Regardless of preview rotation or mirror toggle, a physical face offset produces identical motor commands."""
    cfg = _cal_cfg()
    # Physical face offset in raw sensor frame (x_raw, y_raw)
    for raw_vec in [(30.0, 0.0), (-30.0, 0.0), (0.0, 25.0), (0.0, -25.0), (20.0, -15.0)]:
        # Reference screen vector at calibration settings (rotate=0, mirror=True)
        ref_sx, ref_sy = _transform_point(*raw_vec, rotate_deg=cfg.cal_rotate_deg, mirror=cfg.cal_mirror)
        expected_motor = screen_to_motor(ref_sx, ref_sy, cfg, cfg.cal_rotate_deg, cfg.cal_mirror)

        # Screen vector at current display settings
        cur_sx, cur_sy = _transform_point(*raw_vec, rotate_deg=rotate_deg, mirror=mirror)
        actual_motor = screen_to_motor(cur_sx, cur_sy, cfg, rotate_deg, mirror)

        assert actual_motor == pytest.approx(expected_motor, abs=1e-6)


def test_one_euro_filter_tracks_step_and_smooths_jitter() -> None:
    f = OneEuroFilter(min_cutoff=1.4, beta=2.2)
    x0, v0 = f.filter(0.0, 1 / 30)
    assert (x0, v0) == (0.0, 0.0)

    # Small jitter should be attenuated
    jitter_out = [f.filter(0.02 if i % 2 else -0.02, 1 / 30)[0] for i in range(10)]
    assert max(abs(v) for v in jitter_out) < 0.015

    # Large step should converge rapidly
    val = 0.0
    for _ in range(30):
        val, _ = f.filter(0.8, 1 / 30)
    assert val > 0.75


def test_sticky_face_selection_ignores_distant_background_face() -> None:
    arb = MotionArbiter()
    # Lock onto a face slightly left of center
    primary = FaceBox(x=0.30, y=0.35, w=0.20, h=0.20, confidence=0.95)
    arb.update_vision([primary], vision_ms=10.0, now=1.0)

    # A slightly larger face appears far away on the right — sticky selector should keep primary
    primary_moved = FaceBox(x=0.31, y=0.35, w=0.20, h=0.20, confidence=0.95)
    distant_bigger = FaceBox(x=0.75, y=0.30, w=0.23, h=0.23, confidence=0.98)
    interp = arb.update_vision([distant_bigger, primary_moved], vision_ms=11.0, now=1.033)
    assert interp.face_detected
    assert interp.cx < 0.0  # stayed on left face


def test_deadzone_centered_vs_tracking() -> None:
    arb = MotionArbiter()
    # Face right at target (cx=0.5 -> x=0, cy=0.46 -> y=-0.08 = target_y)
    centered = FaceBox(x=0.40, y=0.36, w=0.20, h=0.20, confidence=0.95)
    interp = arb.update_vision([centered], vision_ms=10.0, now=1.0)
    assert interp.state == "CENTERED"
    assert interp.cmd_yaw_dps == 0.0
    assert interp.cmd_pitch_dps == 0.0

    # Face far to the right -> should pan RIGHT (+yaw)
    right_face = FaceBox(x=0.75, y=0.36, w=0.20, h=0.20, confidence=0.95)
    interp2 = arb.update_vision([right_face], vision_ms=10.0, now=1.05)
    assert interp2.state == "TRACKING"
    assert interp2.cmd_yaw_dps > 0.0


def test_face_loss_fsm_transitions_and_reacquisition() -> None:
    reactions: list[str] = []
    arb = MotionArbiter(on_light_reaction=reactions.append)

    # 1. Track face exiting to the right
    face_right = FaceBox(x=0.75, y=0.35, w=0.20, h=0.20, confidence=0.95)
    arb.update_vision([face_right], vision_ms=10.0, now=10.0)

    # 2. Lost for 0.2s -> COASTING toward right
    c = arb.update_vision([], vision_ms=9.0, now=10.2)
    assert c.state == "COASTING"
    assert c.cmd_yaw_dps > 0.0

    # 3. Lost for 1.5s -> SEARCHING (first sweeps right where face exited)
    s1 = arb.update_vision([], vision_ms=9.0, now=11.5)
    assert s1.state == "SEARCHING"
    assert "search" in reactions
    assert s1.cmd_yaw_dps > 0.0

    # 4. Full sweep finishes (> 9.2s after search started at 11.5s -> t=21.0s) -> RETURNING_HOME
    h = arb.update_vision([], vision_ms=9.0, now=21.0)
    assert h.state == "RETURNING_HOME"
    assert "off" in reactions

    # 5. Next frame with no face -> IDLE_HOME
    idle = arb.update_vision([], vision_ms=9.0, now=21.1)
    assert idle.state == "IDLE_HOME"


def test_desk_mode_pauses_face_tracking() -> None:
    arb = MotionArbiter(desk_mode=True)
    face = FaceBox(x=0.75, y=0.35, w=0.20, h=0.20, confidence=0.95)
    interp = arb.update_vision([face], vision_ms=10.0, now=1.0)
    assert interp.state == "DESK_MODE"
    assert not interp.face_detected
    assert interp.cmd_yaw_dps == 0.0
    assert interp.cmd_pitch_dps == 0.0


def test_ai_hold_mode_prevents_face_tracking_override() -> None:
    arb = MotionArbiter(ai_hold_mode=True)
    face = FaceBox(x=0.80, y=0.35, w=0.20, h=0.20, confidence=0.95)
    interp = arb.update_vision([face], vision_ms=10.0, now=1.0)
    assert interp.state == "AI_HOLD"
    assert interp.cmd_yaw_dps == 0.0
    assert interp.cmd_pitch_dps == 0.0


def test_rear_camera_inverts_pitch_axis_vs_front_camera() -> None:
    """Tilting DOWN with the rear camera must drive the opposite physical pitch motor vs front camera."""
    cfg = _cal_cfg()
    front_m = screen_to_motor(0.0, 30.0, cfg, cur_rotate_deg=0, cur_mirror=True, facing="front")
    back_m = screen_to_motor(0.0, 30.0, cfg, cur_rotate_deg=0, cur_mirror=False, facing="back")
    assert back_m[0] == pytest.approx(-front_m[0], abs=1e-6)
    assert back_m[1] == pytest.approx(front_m[1], abs=1e-6)


def test_angle_wrap_and_shortest_diff() -> None:
    assert wrap_angle_deg(190.0) == pytest.approx(-170.0)
    assert wrap_angle_deg(-190.0) == pytest.approx(170.0)
    assert angle_diff_deg(170.0, -170.0) == pytest.approx(-20.0)
    assert angle_diff_deg(-170.0, 170.0) == pytest.approx(20.0)


def test_reset_face_loss_timers_prevents_spurious_search_after_desk_mode() -> None:
    arb = MotionArbiter()
    # Had face a long time ago, then went to desk mode
    arb.update_vision([FaceBox(x=0.4, y=0.4, w=0.2, h=0.2, confidence=0.95)], vision_ms=10.0, now=1.0)
    arb.desk_mode = True
    arb.update_vision([], vision_ms=5.0, now=50.0)

    # Exiting desk mode after calling reset_face_loss_timers must stay IDLE_HOME, not jump to SEARCHING
    arb.reset_face_loss_timers()
    arb.desk_mode = False
    interp = arb.update_vision([], vision_ms=10.0, now=50.1)
    assert interp.state == "IDLE_HOME"



