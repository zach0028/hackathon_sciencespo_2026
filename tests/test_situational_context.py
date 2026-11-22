"""Tests for Any-Situation Osmo: SituationalContext, SocialGroupTracker, active speaker, shelf horizon, and Scan-to-Lock."""

from __future__ import annotations

import asyncio
import time

import numpy as np

from hackathon_osmo.app.runtime import StudioRuntime
from hackathon_osmo.control.context import SituationalContext
from hackathon_osmo.control.motion import MotionArbiter
from hackathon_osmo.vision.face import FaceBox
from hackathon_osmo.vision.sheet import SheetDetection


def _face(cx: float, cy: float = 0.48, size: float = 0.22, mouth: float = 0.01) -> FaceBox:
    return FaceBox(
        x=cx - size / 2.0,
        y=cy - size / 2.0,
        w=size,
        h=size,
        confidence=0.96,
        mouth_open=mouth,
    )


def test_auto_context_switches_companion_to_social_and_back_with_hysteresis() -> None:
    ctx = SituationalContext()
    assert ctx.policy == "auto"
    assert ctx.active_mode == "companion"

    # Two people appear -> stays companion until 1.1s hysteresis completes (prevents brief passerby flaps)
    two = [_face(0.25), _face(0.75)]
    r0 = ctx.update(faces=two, desk_mode=False, sheet_found=False, now=10.0)
    assert r0.active_mode == "companion"
    assert not r0.mode_changed

    r1 = ctx.update(faces=two, desk_mode=False, sheet_found=False, now=11.3)
    assert r1.active_mode == "social"
    assert r1.mode_changed
    assert r1.people_count == 2

    # Dropping back to 1 person holds social briefly, then returns cleanly to companion after 5.5s
    one = [_face(0.5)]
    r2 = ctx.update(faces=one, desk_mode=False, sheet_found=False, now=13.0)
    assert r2.active_mode == "social"
    r3 = ctx.update(faces=one, desk_mode=False, sheet_found=False, now=19.0)
    assert r3.active_mode == "companion"
    assert r3.mode_changed


def test_social_group_tracker_follows_active_human_speaker_via_mouth_activity() -> None:
    ctx = SituationalContext()
    ctx.set_mode("social", now=0.0)

    # Person left (P1) stays quiet, Person right (P2) actively opens/closes mouth speaking
    mouth_seq_p2 = [0.02, 0.09, 0.03, 0.11, 0.04, 0.10]
    last_res = None
    for idx, m2 in enumerate(mouth_seq_p2):
        t = 1.0 + idx * 0.18
        faces = [_face(0.25, mouth=0.015), _face(0.76, mouth=m2)]
        last_res = ctx.update(faces=faces, desk_mode=False, sheet_found=False, now=t)

    assert last_res is not None
    assert last_res.active_speaker_id == "P2"
    assert last_res.active_person_id == "P2"
    assert last_res.preferred_face is not None
    assert last_res.preferred_face.cx > 0.70

    # MotionArbiter must honor preferred_face and steer RIGHT (+yaw) toward speaking P2!
    arb = MotionArbiter()
    interp = arb.update_vision(
        [_face(0.25), _face(0.76)],
        vision_ms=8.0,
        now=2.5,
        preferred_face=last_res.preferred_face,
        active_person_id=last_res.active_person_id,
    )
    assert interp.active_person_id == "P2"
    assert interp.cx > 0.4
    assert interp.cmd_yaw_dps > 0.0


def test_social_group_shared_gaze_cycles_when_osmo_speaks() -> None:
    ctx = SituationalContext()
    ctx.set_mode("social", now=0.0)
    faces = [_face(0.24, mouth=0.01), _face(0.78, mouth=0.01)]

    r0 = ctx.update(faces=faces, desk_mode=False, sheet_found=False, osmo_speaking=True, now=1.0)
    first_pid = r0.active_person_id
    assert first_pid in ("P1", "P2")

    # After 3.5s of Osmo speaking to the group, shared gaze smoothly shifts to the other person!
    r1 = ctx.update(faces=faces, desk_mode=False, sheet_found=False, osmo_speaking=True, now=4.6)
    assert r1.active_person_id != first_pid


def test_shelf_horizon_learning_and_session_desk_discovery() -> None:
    ctx = SituationalContext()
    # Placed high on a shelf: faces are consistently centered while IMU pitch is +18° downward
    for i in range(25):
        ctx.update(
            faces=[_face(0.5)],
            desk_mode=False,
            sheet_found=False,
            imu_pitch_deg=18.0,
            face_centered=True,
            now=float(i) * 0.2,
        )
    assert ctx.placement == "shelf"
    assert ctx.preferred_face_pitch(fallback_pitch=5.3) > 15.0

    # Moving to a study table & finding a sheet locks session pose and flips to study
    ctx.record_desk_found(pitch_deg=44.5, pan_roll_deg=-38.0, pivot_deg=128.0, now=10.0)
    assert ctx.placement == "desk"
    assert ctx.active_mode == "study"
    assert ctx.session_desk_pose is not None
    assert ctx.session_desk_pose["pitch_deg"] == 44.5
    assert "session_table_locked=True" in ctx.prompt_summary(now=10.2)

    # If Scan-to-Lock finds zero sheets on a shelf, record_no_desk_sheet sets honest sensor flag
    ctx2 = SituationalContext()
    ctx2.set_mode("auto", now=0.0)
    ctx2.active_mode = "study"
    ctx2.record_no_desk_sheet(now=5.0)
    assert ctx2.no_sheet_anywhere is True
    assert ctx2.placement == "shelf"
    assert ctx2.active_mode == "companion"
    assert "no_table_sheet_found=True" in ctx2.prompt_summary(now=5.1)


def test_runtime_set_situation_tool_focus_and_snapshot() -> None:
    rt = StudioRuntime()
    now = time.monotonic()
    faces = [_face(0.22), _face(0.80)]
    rt.context.update(faces=faces, desk_mode=False, sheet_found=False, now=now)

    res = asyncio.run(
        rt.execute_robot_tool(
            "set_situation",
            {"mode": "social", "focus_person": "right", "reason": "Saluer Jules à droite"},
        )
    )
    assert res["ok"] is True
    assert res["active_mode"] == "social"
    assert res["focused_person"] == "P2"

    snap = rt.snapshot()
    assert "situation" in snap
    assert snap["situation"]["active_mode"] == "social"
    assert snap["situation"]["active_target_id"] == "P2"
    ctx_str = rt._get_robot_context()
    assert "situation=social" in ctx_str


def test_runtime_adaptive_scan_to_lock_finds_sheet_or_reports_shelf() -> None:
    # Case 1: Sheet is off to the right on an uncalibrated desk -> Scan-to-Lock finds it on probe 2!
    rt = StudioRuntime()
    rt.osmo_connected = True
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    moved_pans: list[float] = []
    detect_calls = {"n": 0}

    async def fake_configure_camera(**kwargs):
        if "facing" in kwargs and kwargs["facing"]:
            rt.facing = kwargs["facing"]
        if "zoom" in kwargs and kwargs["zoom"] is not None:
            rt.camera_zoom = float(kwargs["zoom"])
        return True

    async def fake_pivot(pivot_deg: float, target_pitch: float):
        return (0.0, target_pitch)

    async def fake_wait_sharp(**_kwargs):
        return dummy_frame, 180.0

    async def fake_move_rel(pan_deg: float = 0.0, tilt_deg: float = 0.0, speed_dps: float = 50.0, *, hold_after: bool = True):
        moved_pans.append(pan_deg)
        return pan_deg, tilt_deg, 0.2

    def fake_detect(_img):
        detect_calls["n"] += 1
        # Miss on initial 1.0x, miss on 0.51x wide, miss on left probe (-36°), LOCK on right probe (+72°)!
        if detect_calls["n"] >= 4:
            return SheetDetection(found=True, x=0.25, y=0.275, w=0.50, h=0.45, confidence=0.92, source="bright_paper")
        return SheetDetection(found=False)

    rt.configure_camera = fake_configure_camera  # type: ignore[method-assign]
    rt._pivot_arm_to_desk = fake_pivot  # type: ignore[method-assign]
    rt._wait_sharp_frame = fake_wait_sharp  # type: ignore[method-assign]
    rt.arbiter.move_relative_deg = fake_move_rel  # type: ignore[method-assign]
    rt._detect_sheet_sync = fake_detect  # type: ignore[method-assign]

    res = asyncio.run(rt.enter_desk_reading_mode(auto_target=True))
    assert res["sheet_detected"] is True
    assert res["no_sheet_anywhere"] is False
    assert moved_pans[:2] == [-36.0, 72.0]
    assert rt.context.placement == "desk"
    assert rt.context.session_desk_pose is not None

    # Case 2: On a shelf (zero sheets anywhere) -> Scan-to-Lock finishes honestly with no_sheet_anywhere=True
    rt2 = StudioRuntime()
    rt2.osmo_connected = True
    rt2.configure_camera = fake_configure_camera  # type: ignore[method-assign]
    rt2._pivot_arm_to_desk = fake_pivot  # type: ignore[method-assign]
    rt2._wait_sharp_frame = fake_wait_sharp  # type: ignore[method-assign]
    rt2.arbiter.move_relative_deg = fake_move_rel  # type: ignore[method-assign]
    rt2._detect_sheet_sync = lambda _img: SheetDetection(found=False)  # type: ignore[method-assign]

    res2 = asyncio.run(rt2.enter_desk_reading_mode(auto_target=True))
    assert res2["sheet_detected"] is False
    assert res2["no_sheet_anywhere"] is True
    assert rt2.context.placement == "shelf"

