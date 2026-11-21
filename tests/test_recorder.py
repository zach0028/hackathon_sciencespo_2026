"""Unit tests for SessionRecorder (Flight Recorder telemetry, IMU traces, frame snapshots & latencies)."""

from __future__ import annotations

import json
from pathlib import Path

from hackathon_osmo.telemetry.recorder import SessionRecorder


def test_recorder_writes_events_imu_turns_frames_and_summary(tmp_path: Path) -> None:
    rec = SessionRecorder(base_dir=tmp_path, session_name="session_test_01")

    rec.record_event("ARM", "Pivot géodésique démarré", {"target_deg": 180})
    rec.record_imu_sample(
        pitch=49.3,
        roll=-46.7,
        yaw=176.3,
        quat=(0.7, 0.1, 0.2, 0.6),
        state="IMU_SERVO",
        facing="back",
        zoom=2.0,
        cmd_pan_dps=0.0,
        cmd_tilt_dps=-40.0,
        motor_yaw_dps=40.0,
        motor_pitch_dps=0.0,
        face_detected=False,
        sheet_found=True,
        sheet_lines=6,
    )
    frame_rel = rec.save_frame(b"\xff\xd8\xff\xe0FAKEJPEG", "tool look_at_desk hd")
    assert frame_rel == "frames/001_tool_look_at_desk_hd.jpg"
    assert (rec.session_dir / frame_rel).exists()

    rec.record_turn(
        {
            "model": "gpt-6.2-sol",
            "user_text": "Lis ma feuille",
            "reply_text": "Je lis les notes.",
            "tool_calls": [{"name": "look_at_desk", "duration_ms": 2100.0}],
            "frames": [frame_rel],
            "timings_ms": {"round1_ms": 1100.0, "tools_total_ms": 2100.0, "round2_ms": 950.0, "total_ms": 4150.0},
        }
    )
    rec.record_live_audio_latency(245.5)

    s = rec.summary()
    assert s["counts"]["turns"] == 1
    assert s["counts"]["events"] == 2
    assert s["counts"]["imu_samples"] == 1
    assert s["counts"]["frames_saved"] == 1
    assert s["latencies_ms"]["turn_total_p50"] == 4150.0
    assert s["latencies_ms"]["live_speech_start_p50"] == 245.5
    assert s["tool_usage"] == {"look_at_desk": 1}

    # Verify JSONL files are valid line-delimited JSON and latest symlink points to this session
    assert (tmp_path / "latest").resolve() == rec.session_dir.resolve()
    assert len(rec.events_path.read_text(encoding="utf-8").splitlines()) == 2
    imu_row = json.loads(rec.imu_path.read_text(encoding="utf-8").splitlines()[0])
    assert imu_row["pitch"] == 49.3 and imu_row["sheet_lines"] == 6
    turn_row = json.loads(rec.turns_path.read_text(encoding="utf-8").splitlines()[0])
    assert turn_row["turn"] == 1 and turn_row["timings_ms"]["round1_ms"] == 1100.0
