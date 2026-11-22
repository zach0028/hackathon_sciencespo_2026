"""Situational Context & Multi-Person Social Group Tracker for Any-Situation Osmo.

Enables Osmo to work seamlessly wherever it is placed — study desk, coffee table,
kitchen counter, or living-room shelf — without manual recalibration:
- Auto-detects situational mode (study / companion / social / ambient) with hysteresis
- Tracks multiple simultaneous people (P1, P2, ...) with ANE lip-activity speaker detection
- Implements ELEGNT shared group gaze (inclusive eye contact when addressing multiple people)
- Learns the current room's neutral face pitch (shelf vs desk) and dynamically discovered
  session desk pose (Scan-to-Lock) automatically.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from hackathon_osmo.vision.face import FaceBox

POLICY_MODES = ("auto", "study", "companion", "social", "ambient")
ACTIVE_MODES = ("study", "companion", "social", "ambient")
FOCUS_TARGETS = ("auto", "left", "right", "center", "next")


@dataclass
class TrackedPerson:
    """Spatial & speech-activity state of one person seen by Apple Vision."""

    pid: str
    cx: float
    cy: float
    area: float
    last_seen_t: float
    last_box: FaceBox | None = None
    mouth_history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=16))
    speech_activity: float = 0.0
    is_speaking: bool = False
    last_spoke_t: float = 0.0

    @property
    def side_label(self) -> str:
        if self.cx < 0.36:
            return "gauche"
        if self.cx > 0.64:
            return "droite"
        return "centre"

    def update_mouth(self, mouth_open: float, now: float) -> None:
        if mouth_open > 0.0:
            self.mouth_history.append((now, mouth_open))
        # Drop stale samples older than 1.0s
        while self.mouth_history and (now - self.mouth_history[0][0]) > 1.0:
            self.mouth_history.popleft()

        if len(self.mouth_history) >= 2:
            vals = [v for _, v in self.mouth_history]
            mean_v = sum(vals) / len(vals)
            var = sum((v - mean_v) ** 2 for v in vals) / len(vals)
            std = math.sqrt(var)
            peak = max(vals)
            # Lip motion variance + sustained opening above closed-mouth baseline (~0.025)
            activity = std * 2.4 + max(0.0, peak - 0.032) * 0.85
            self.speech_activity = round(activity, 4)
        elif len(self.mouth_history) == 1:
            _, v = self.mouth_history[0]
            self.speech_activity = round(max(0.0, v - 0.045) * 0.8, 4)
        else:
            self.speech_activity *= 0.82

        if self.speech_activity >= 0.022:
            self.is_speaking = True
            self.last_spoke_t = now
        elif (now - self.last_spoke_t) > 0.9:
            self.is_speaking = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.pid,
            "cx": round(self.cx, 3),
            "cy": round(self.cy, 3),
            "side": self.side_label,
            "speaking": self.is_speaking,
            "speech_activity": round(self.speech_activity, 3),
        }


class SocialGroupTracker:
    """Tracks multiple faces with stable spatial IDs, active-speaker detection, and ELEGNT shared gaze."""

    def __init__(self) -> None:
        self._people: dict[str, TrackedPerson] = {}
        self._next_id: int = 1
        self.active_target_id: str | None = None
        self.active_speaker_id: str | None = None
        self._last_switch_t: float = 0.0
        self._last_shared_cycle_t: float = 0.0
        self._manual_focus_id: str | None = None
        self._manual_focus_until: float = 0.0

    def reset(self) -> None:
        self._people.clear()
        self._next_id = 1
        self.active_target_id = None
        self.active_speaker_id = None
        self._manual_focus_id = None
        self._manual_focus_until = 0.0

    def visible_people(self, now: float, max_age_s: float = 0.75) -> list[TrackedPerson]:
        vis = [p for p in self._people.values() if (now - p.last_seen_t) <= max_age_s and p.last_box is not None]
        vis.sort(key=lambda p: p.cx)
        return vis

    def focus_by_hint(self, hint: str, now: float | None = None, hold_s: float = 12.0) -> str | None:
        """Direct gaze to 'left', 'right', 'center', 'next', a specific 'P1'/'P2', or 'auto'."""
        t = now if now is not None else time.monotonic()
        h = (hint or "auto").strip().lower()
        if h == "auto":
            self._manual_focus_id = None
            self._manual_focus_until = 0.0
            return self.active_target_id

        vis = self.visible_people(t, max_age_s=2.0)
        if not vis:
            return None

        chosen: TrackedPerson | None = None
        if h == "left":
            chosen = vis[0]
        elif h == "right":
            chosen = vis[-1]
        elif h == "center":
            chosen = min(vis, key=lambda p: abs(p.cx - 0.5))
        elif h == "next":
            ids = [p.pid for p in vis]
            if self.active_target_id in ids and len(ids) > 1:
                idx = (ids.index(self.active_target_id) + 1) % len(ids)
                chosen = vis[idx]
            else:
                chosen = vis[0]
        else:
            for p in vis:
                if p.pid.lower() == h:
                    chosen = p
                    break

        if chosen is not None:
            self._manual_focus_id = chosen.pid
            self._manual_focus_until = t + hold_s
            self.active_target_id = chosen.pid
            self._last_switch_t = t
            return chosen.pid
        return None

    def update(
        self,
        faces: list[FaceBox],
        *,
        now: float,
        osmo_speaking: bool = False,
        social_active: bool = False,
    ) -> tuple[FaceBox | None, str | None]:
        # 1. Expire people unseen for > 4.5s
        expired = [pid for pid, p in self._people.items() if (now - p.last_seen_t) > 4.5]
        for pid in expired:
            del self._people[pid]
            if self.active_target_id == pid:
                self.active_target_id = None
            if self._manual_focus_id == pid:
                self._manual_focus_id = None

        # Clear stale last_box references before matching this frame
        for p in self._people.values():
            p.last_box = None

        if not faces:
            self.active_speaker_id = None
            return None, None

        # 2. Greedy nearest-neighbor match between existing people and incoming faces
        unmatched_faces = list(range(len(faces)))
        matched_pids: set[str] = set()
        pairs: list[tuple[float, str, int]] = []
        for pid, p in self._people.items():
            for idx, f in enumerate(faces):
                dist = math.hypot(f.cx - p.cx, f.cy - p.cy)
                if dist <= 0.28:
                    pairs.append((dist, pid, idx))
        pairs.sort(key=lambda item: item[0])

        for _dist, pid, idx in pairs:
            if pid in matched_pids or idx not in unmatched_faces:
                continue
            matched_pids.add(pid)
            unmatched_faces.remove(idx)
            f = faces[idx]
            p = self._people[pid]
            p.cx = p.cx * 0.55 + f.cx * 0.45
            p.cy = p.cy * 0.55 + f.cy * 0.45
            p.area = f.area
            p.last_seen_t = now
            p.last_box = f
            p.update_mouth(f.mouth_open, now)

        # Create new TrackedPerson entries for newly appearing faces (ordered left-to-right)
        for idx in sorted(unmatched_faces, key=lambda i: faces[i].cx):
            f = faces[idx]
            pid = f"P{self._next_id}"
            self._next_id += 1
            p = TrackedPerson(
                pid=pid,
                cx=f.cx,
                cy=f.cy,
                area=f.area,
                last_seen_t=now,
                last_box=f,
            )
            p.update_mouth(f.mouth_open, now)
            self._people[pid] = p

        # Compact IDs cleanly back to P1 when only 0..1 person remains after long gaps
        vis = self.visible_people(now, max_age_s=0.1)
        if not vis:
            return None, None

        # 3. Identify active human speaker from ANE lip activity
        speakers = [p for p in vis if p.is_speaking]
        top_speaker = max(speakers, key=lambda p: p.speech_activity) if speakers else None
        self.active_speaker_id = top_speaker.pid if top_speaker else None

        # Single visible person -> straightforward track
        if len(vis) == 1:
            only = vis[0]
            self.active_target_id = only.pid
            return only.last_box, only.pid

        # 4. Multi-person target arbitration (2+ people visible!)
        vis_by_id = {p.pid: p for p in vis}

        # Priority A: Explicit manual/AI focus (unless expired)
        if self._manual_focus_id and now < self._manual_focus_until and self._manual_focus_id in vis_by_id:
            chosen = vis_by_id[self._manual_focus_id]
            self.active_target_id = chosen.pid
            return chosen.last_box, chosen.pid
        if now >= self._manual_focus_until:
            self._manual_focus_id = None

        # Priority B: Active human speaker (with 1.6s anti-ping-pong hysteresis)
        if top_speaker is not None:
            cur = vis_by_id.get(self.active_target_id or "")
            if cur is None:
                self.active_target_id = top_speaker.pid
                self._last_switch_t = now
            elif top_speaker.pid != cur.pid:
                clearly_louder = top_speaker.speech_activity > (cur.speech_activity + 0.012)
                if (now - self._last_switch_t) >= 1.6 and clearly_louder:
                    self.active_target_id = top_speaker.pid
                    self._last_switch_t = now
            chosen = vis_by_id[self.active_target_id]
            return chosen.last_box, chosen.pid

        # Priority C: ELEGNT Shared Group Gaze (inclusive eye-contact cycling when addressing a group!)
        if self.active_target_id not in vis_by_id:
            biggest = max(vis, key=lambda p: p.area)
            self.active_target_id = biggest.pid
            self._last_shared_cycle_t = now
            return biggest.last_box, biggest.pid

        if social_active:
            cycle_interval = 3.2 if osmo_speaking else 6.2
            if (now - self._last_shared_cycle_t) >= cycle_interval and (now - self._last_switch_t) >= 2.2:
                ids = [p.pid for p in vis]
                idx = (ids.index(self.active_target_id) + 1) % len(ids)
                self.active_target_id = ids[idx]
                self._last_shared_cycle_t = now
                self._last_switch_t = now

        chosen = vis_by_id[self.active_target_id]
        return chosen.last_box, chosen.pid


@dataclass
class ContextTickResult:
    """Per-frame output produced by SituationalContext.update()."""

    active_mode: str
    policy: str
    placement: str
    preferred_face: FaceBox | None
    active_person_id: str | None
    active_speaker_id: str | None
    people_count: int
    mode_changed: bool = False
    change_reason: str = ""


class SituationalContext:
    """Auto-adapting situational intelligence for Osmo (Study Desk / Companion / Social Group / Ambient)."""

    def __init__(self) -> None:
        self.policy: str = "auto"
        self.active_mode: str = "companion"
        self.placement: str = "unknown"  # "desk" | "shelf" | "unknown"
        self.social = SocialGroupTracker()

        # Adaptive room horizon & dynamically discovered table pose for this session
        self.inferred_face_pitch_deg: float | None = None
        self.session_desk_pose: dict[str, float] | None = None
        self.no_sheet_anywhere: bool = False
        self.last_reason: str = "Démarrage en mode auto-adaptatif"

        # Hysteresis timers so auto-mode never flaps rapidly
        self._multi_face_since: float | None = None
        self._single_face_since: float | None = None
        self._shelf_pitch_samples: int = 0
        self._last_mode_change_t: float = 0.0

    def set_mode(
        self,
        mode: str,
        *,
        focus_person: str = "auto",
        reason: str = "",
        now: float | None = None,
    ) -> dict[str, Any]:
        """Explicitly set situational policy ('auto'|'study'|'companion'|'social'|'ambient') and optional gaze target."""
        t = now if now is not None else time.monotonic()
        clean = (mode or "auto").strip().lower()
        if clean not in POLICY_MODES:
            return {"ok": False, "error": f"Mode situationnel inconnu : {mode}"}

        prev_active = self.active_mode
        self.policy = clean
        if clean != "auto":
            self.active_mode = clean
            self._last_mode_change_t = t
        elif self.active_mode == "ambient":
            self.active_mode = "companion"

        if clean == "study":
            self.no_sheet_anywhere = False

        focused_pid: str | None = None
        if focus_person and focus_person.strip().lower() != "auto":
            focused_pid = self.social.focus_by_hint(focus_person, now=t)
        elif clean != "social" and focus_person == "auto":
            self.social.focus_by_hint("auto", now=t)

        self.last_reason = reason.strip() or f"Mode {clean.upper()} activé"
        return {
            "ok": True,
            "policy": self.policy,
            "active_mode": self.active_mode,
            "previous_mode": prev_active,
            "placement": self.placement,
            "focused_person": focused_pid,
            "reason": self.last_reason,
        }

    def record_desk_found(
        self,
        *,
        pitch_deg: float | None,
        pan_roll_deg: float | None,
        pivot_deg: float | None = None,
        now: float | None = None,
    ) -> None:
        """Remember the dynamically discovered table/sheet pose for the current session."""
        t = now if now is not None else time.monotonic()
        self.placement = "desk"
        self.no_sheet_anywhere = False
        if pitch_deg is not None:
            pose = dict(self.session_desk_pose or {})
            pose["pitch_deg"] = round(max(28.0, min(65.0, float(pitch_deg))), 1)
            if pan_roll_deg is not None:
                pose["pan_roll_deg"] = round(float(pan_roll_deg), 1)
            if pivot_deg is not None:
                pose["pivot_deg"] = round(float(pivot_deg), 1)
            self.session_desk_pose = pose
        if self.policy == "auto" and self.active_mode != "study":
            self.active_mode = "study"
            self._last_mode_change_t = t
            self.last_reason = "Feuille verrouillée sur la table → mode Devoir"

    def record_no_desk_sheet(self, now: float | None = None) -> None:
        """Note that an adaptive desk scan found zero sheets around Osmo (likely on a shelf/counter)."""
        t = now if now is not None else time.monotonic()
        self.no_sheet_anywhere = True
        if self.placement == "unknown":
            self.placement = "shelf"
        if self.policy == "auto" and self.active_mode == "study":
            vis = self.social.visible_people(t, max_age_s=2.0)
            self.active_mode = "social" if len(vis) >= 2 else "companion"
            self._last_mode_change_t = t
            self.last_reason = "Aucune feuille sous Osmo → retour mode Compagnon / Étagère"

    def preferred_face_pitch(self, fallback_pitch: float = 5.3) -> float:
        """Return the learned room horizon pitch (adapts automatically on shelves vs desks!)."""
        if self.inferred_face_pitch_deg is not None:
            return round(self.inferred_face_pitch_deg, 1)
        return round(float(fallback_pitch), 1)

    def update(
        self,
        *,
        faces: list[FaceBox],
        desk_mode: bool,
        sheet_found: bool,
        osmo_speaking: bool = False,
        imu_pitch_deg: float | None = None,
        face_centered: bool = False,
        now: float | None = None,
    ) -> ContextTickResult:
        t = now if now is not None else time.monotonic()
        prev_mode = self.active_mode
        change_reason = ""

        # 1. Learn real room horizon pitch when stably looking at a person's face!
        if not desk_mode and face_centered and imu_pitch_deg is not None and -25.0 <= imu_pitch_deg <= 32.0:
            if self.inferred_face_pitch_deg is None:
                self.inferred_face_pitch_deg = float(imu_pitch_deg)
            else:
                self.inferred_face_pitch_deg = self.inferred_face_pitch_deg * 0.94 + float(imu_pitch_deg) * 0.06
            # Sustained downward gaze while tracking faces (pitch >= +14°) strongly hints Osmo sits high on a shelf!
            if imu_pitch_deg >= 14.0 and not sheet_found:
                self._shelf_pitch_samples += 1
                if self._shelf_pitch_samples >= 20 and self.placement == "unknown":
                    self.placement = "shelf"
            else:
                self._shelf_pitch_samples = max(0, self._shelf_pitch_samples - 1)

        # 2. Automatic situational transitions (only when policy == "auto")
        n_faces = len(faces) if not desk_mode else 0
        if self.policy == "auto":
            if desk_mode or sheet_found:
                self._multi_face_since = None
                self._single_face_since = None
                if self.active_mode != "study":
                    self.active_mode = "study"
                    change_reason = "Mode bureau / feuille détecté → Devoir"
                if sheet_found:
                    self.placement = "desk"
            elif n_faces >= 2:
                self._single_face_since = None
                if self._multi_face_since is None:
                    self._multi_face_since = t
                if (t - self._multi_face_since) >= 1.1 and self.active_mode != "social":
                    self.active_mode = "social"
                    change_reason = f"{n_faces} personnes détectées → mode Salon / Multi-visages"
            else:
                self._multi_face_since = None
                if self.active_mode == "social":
                    if self._single_face_since is None:
                        self._single_face_since = t
                    if (t - self._single_face_since) >= 5.5:
                        self.active_mode = "companion"
                        change_reason = "Retour en conversation individuelle → mode Compagnon"
                else:
                    self._single_face_since = None

        mode_changed = self.active_mode != prev_mode
        if mode_changed:
            self._last_mode_change_t = t
            self.last_reason = change_reason

        # 3. Update multi-person social tracker & decide preferred gaze target
        preferred_face: FaceBox | None = None
        active_pid: str | None = None
        if not desk_mode:
            target_box, pid = self.social.update(
                faces,
                now=t,
                osmo_speaking=osmo_speaking,
                social_active=(self.active_mode == "social"),
            )
            active_pid = pid
            # Override MotionArbiter's sticky picker whenever 2+ faces are visible or explicit focus active
            if len(faces) >= 2 or self.social._manual_focus_id is not None:
                preferred_face = target_box

        return ContextTickResult(
            active_mode=self.active_mode,
            policy=self.policy,
            placement=self.placement,
            preferred_face=preferred_face,
            active_person_id=active_pid,
            active_speaker_id=self.social.active_speaker_id,
            people_count=n_faces,
            mode_changed=mode_changed,
            change_reason=change_reason,
        )

    def prompt_summary(self, now: float | None = None) -> str:
        t = now if now is not None else time.monotonic()
        vis = self.social.visible_people(t, max_age_s=1.2)
        parts = [
            f"situation={self.active_mode} (policy={self.policy}, placement={self.placement})",
            f"people_visible={len(vis)}",
        ]
        if len(vis) >= 2:
            desc = " / ".join(
                f"{p.pid}:{p.side_label}{'*SPEAKING*' if p.is_speaking else ''}" for p in vis
            )
            parts.append(f"group=[{desc}]")
            if self.social.active_target_id:
                parts.append(f"gaze_target={self.social.active_target_id}")
        if self.session_desk_pose:
            parts.append(f"session_table_locked=True(pitch={self.session_desk_pose.get('pitch_deg'):+.1f}°)")
        if self.no_sheet_anywhere:
            parts.append("no_table_sheet_found=True")
        return ", ".join(parts)

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        t = now if now is not None else time.monotonic()
        vis = self.social.visible_people(t, max_age_s=1.2)
        return {
            "policy": self.policy,
            "active_mode": self.active_mode,
            "placement": self.placement,
            "people_count": len(vis),
            "people": [p.to_dict() for p in vis],
            "active_target_id": self.social.active_target_id,
            "active_speaker_id": self.social.active_speaker_id,
            "inferred_face_pitch": (
                round(self.inferred_face_pitch_deg, 1) if self.inferred_face_pitch_deg is not None else None
            ),
            "session_desk_locked": self.session_desk_pose is not None,
            "no_sheet_anywhere": self.no_sheet_anywhere,
            "reason": self.last_reason,
        }
