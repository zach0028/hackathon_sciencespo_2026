"""Lightweight non-blocking Flight Recorder for hackathon_osmo sessions.

Captures rich, structured diagnostics on disk under logs/session_<timestamp>/ (gitignored):
- events.jsonl    : every system/arm/camera/live/brain event with wall-clock + monotonic timing
- imu_trace.jsonl : 5 Hz mechanical trace (IMU pitch/roll/yaw/quat, motor commands, tracking state)
- turns.jsonl     : full interaction turns with stage-by-stage latency breakdown & sensor ground truth
- frames/*.jpg    : exact visual snapshots seen by the AI at turn input & after each arm/optics tool
- summary.json    : live aggregate stats (turn count, P50/P95 latencies, tool stats, error count)
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hackathon_osmo.paths import LOGS_DIR


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")[:40] or "frame"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return round(s[idx], 1)


class SessionRecorder:
    """Thread-safe structured telemetry recorder for post-session root-cause analysis."""

    def __init__(self, base_dir: Path | None = None, session_name: str | None = None) -> None:
        self.base_dir = base_dir or LOGS_DIR
        stamp = session_name or time.strftime("session_%Y-%m-%d_%H-%M-%S")
        self.session_id = stamp
        self.session_dir = self.base_dir / stamp
        self.frames_dir = self.session_dir / "frames"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.session_dir / "events.jsonl"
        self.imu_path = self.session_dir / "imu_trace.jsonl"
        self.turns_path = self.session_dir / "turns.jsonl"
        self.summary_path = self.session_dir / "summary.json"

        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self._frame_seq = 0
        self._turn_seq = 0
        self._event_count = 0
        self._imu_count = 0
        self._error_count = 0

        self._turn_total_ms: list[float] = []
        self._turn_round1_ms: list[float] = []
        self._turn_tools_ms: list[float] = []
        self._live_tts_start_ms: list[float] = []
        self._tool_counts: dict[str, int] = {}

        # Update logs/latest pointer for instant inspection
        latest_link = self.base_dir / "latest"
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(self.session_dir.name)
        except OSError:
            pass

        self._write_summary()

    def _ts_fields(self) -> dict[str, Any]:
        return {
            "ts_iso": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
            "t_rel_s": round(time.monotonic() - self._t0, 3),
        }

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)

    def record_event(self, tag: str, message: str, data: dict[str, Any] | None = None) -> None:
        rec: dict[str, Any] = {
            **self._ts_fields(),
            "tag": tag,
            "msg": message,
        }
        if data:
            rec["data"] = data
        self._append_jsonl(self.events_path, rec)
        with self._lock:
            self._event_count += 1
            if tag.upper() in ("ERROR", "WARN") or "erreur" in message.lower():
                self._error_count += 1

    def record_imu_sample(
        self,
        *,
        pitch: float | None,
        roll: float | None,
        yaw: float | None,
        quat: tuple[float, float, float, float] | None,
        state: str,
        facing: str,
        zoom: float,
        cmd_pan_dps: float,
        cmd_tilt_dps: float,
        motor_yaw_dps: float,
        motor_pitch_dps: float,
        face_detected: bool,
        face_xy: tuple[float, float] | None = None,
        sheet_found: bool = False,
        sheet_lines: int = 0,
    ) -> None:
        rec = {
            **self._ts_fields(),
            "pitch": round(pitch, 2) if pitch is not None else None,
            "roll": round(roll, 2) if roll is not None else None,
            "yaw": round(yaw, 2) if yaw is not None else None,
            "quat": [round(q, 4) for q in quat] if quat else None,
            "state": state,
            "facing": facing,
            "zoom": round(zoom, 2),
            "cmd": [round(cmd_pan_dps, 1), round(cmd_tilt_dps, 1)],
            "motor": [round(motor_yaw_dps, 1), round(motor_pitch_dps, 1)],
            "face": face_detected,
            "face_xy": [round(face_xy[0], 3), round(face_xy[1], 3)] if (face_detected and face_xy) else None,
            "sheet": sheet_found,
            "sheet_lines": sheet_lines,
        }
        self._append_jsonl(self.imu_path, rec)
        with self._lock:
            self._imu_count += 1

    def save_frame(self, jpeg_bytes: bytes | None, label: str) -> str | None:
        """Save a JPEG snapshot into frames/ and return its relative path (e.g. 'frames/001_turn_input.jpg')."""
        if not jpeg_bytes:
            return None
        with self._lock:
            self._frame_seq += 1
            seq = self._frame_seq
        fname = f"{seq:03d}_{_slug(label)}.jpg"
        fpath = self.frames_dir / fname
        try:
            fpath.write_bytes(jpeg_bytes)
            return f"frames/{fname}"
        except OSError:
            return None

    def record_turn(self, turn_data: dict[str, Any]) -> None:
        with self._lock:
            self._turn_seq += 1
            seq = self._turn_seq
        rec = {
            "turn": seq,
            **self._ts_fields(),
            **turn_data,
        }
        self._append_jsonl(self.turns_path, rec)

        timings = turn_data.get("timings_ms") or {}
        with self._lock:
            if isinstance(timings.get("total_ms"), (int, float)):
                self._turn_total_ms.append(float(timings["total_ms"]))
            if isinstance(timings.get("round1_ms"), (int, float)):
                self._turn_round1_ms.append(float(timings["round1_ms"]))
            if isinstance(timings.get("tools_total_ms"), (int, float)):
                self._turn_tools_ms.append(float(timings["tools_total_ms"]))
            for tc in turn_data.get("tool_calls") or []:
                tname = str(tc.get("name") or "unknown")
                self._tool_counts[tname] = self._tool_counts.get(tname, 0) + 1
        self._write_summary()

    def record_live_audio_latency(self, commentary_to_audio_ms: float) -> None:
        self.record_event(
            "LATENCY",
            f"GPT-Live parole démarrée {commentary_to_audio_ms:.0f} ms après commentaire cerveau",
            {"commentary_to_audio_ms": round(commentary_to_audio_ms, 1)},
        )
        with self._lock:
            self._live_tts_start_ms.append(float(commentary_to_audio_ms))
        self._write_summary()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "session_dir": str(self.session_dir),
                "uptime_s": round(time.monotonic() - self._t0, 1),
                "counts": {
                    "turns": self._turn_seq,
                    "events": self._event_count,
                    "imu_samples": self._imu_count,
                    "frames_saved": self._frame_seq,
                    "errors_or_warns": self._error_count,
                },
                "latencies_ms": {
                    "turn_total_p50": _percentile(self._turn_total_ms, 50),
                    "turn_total_p95": _percentile(self._turn_total_ms, 95),
                    "llm_round1_p50": _percentile(self._turn_round1_ms, 50),
                    "tools_total_p50": _percentile(self._turn_tools_ms, 50),
                    "live_speech_start_p50": _percentile(self._live_tts_start_ms, 50),
                },
                "tool_usage": dict(self._tool_counts),
            }

    def _write_summary(self) -> None:
        try:
            self.summary_path.write_text(
                json.dumps(self.summary(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass
