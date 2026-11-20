"""Motion Arbiter v2: One-Euro filtered cinema PID, rotation-invariant screen->motor mapping,
sticky multi-face tracking, expressive gestures, and full face-loss recovery FSM.

Canonical Screen-Space convention used everywhere inside this module:
- pan_right_dps > 0  ==> pan camera RIGHT  (brings right-side face toward center)
- pan_right_dps < 0  ==> pan camera LEFT   (brings left-side face toward center)
- tilt_down_dps > 0  ==> tilt camera DOWN  (brings bottom face toward center / looks at desk)
- tilt_down_dps < 0  ==> tilt camera UP    (brings top face toward center / looks up)
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from hackathon_osmo.vision.face import FaceBox


class OneEuroFilter:
    """Adaptive low-pass filter: rock-steady when still, low-latency when moving fast."""

    def __init__(self, min_cutoff: float = 1.4, beta: float = 2.2, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(1e-3, cutoff))
        return 1.0 / (1.0 + tau / max(1e-3, dt))

    def filter(self, x: float, dt: float) -> tuple[float, float]:
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            return x, 0.0
        dx = (x - self._x_prev) / max(1e-3, dt)
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat, dx_hat


@dataclass
class PidConfig:
    kp: float = 45.0
    kp_yaw: float = 55.0
    kp_pitch: float = 30.0
    kd: float = 3.5
    deadzone: float = 0.09
    max_dps: float = 55.0
    max_accel_dps2: float = 150.0  # slew-rate limit for buttery cinema ramps
    target_y: float = -0.08  # slight headroom (rule of thirds)
    # Calibrated reference mapping (measured at cal_rotate_deg / cal_mirror)
    cal_rotate_deg: int = 0
    cal_mirror: bool = True
    swap_axes: bool = True
    invert_yaw: bool = True
    invert_pitch: bool = True
    # Lost-face recovery policy: "search" | "home" | "last_pos"
    lost_policy: str = "search"


@dataclass
class TrackingInterpretation:
    face_detected: bool = False
    faces_count: int = 0
    cx: float = 0.0  # -1 (left) .. +1 (right)
    cy: float = 0.0  # -1 (top)  .. +1 (bottom)
    box_area: float = 0.0
    confidence: float = 0.0
    err_x: float = 0.0
    err_y: float = 0.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    # States: CENTERED | TRACKING | COASTING | SEARCHING | RETURNING_HOME | IDLE_HOME | GESTURE | MANUAL | DESK_MODE
    state: str = "IDLE_HOME"
    summary: str = "En attente d'un visage..."
    # Canonical screen-space velocities: +yaw = pan RIGHT, +pitch = tilt DOWN
    cmd_yaw_dps: float = 0.0
    cmd_pitch_dps: float = 0.0
    cmd_roll_dps: float = 0.0
    active_gesture: str | None = None
    vision_ms: float = 0.0
    last_exit_dir: str = "center"


@dataclass
class GestureStep:
    """Step in canonical screen-space: +pan_right_dps = RIGHT, +tilt_down_dps = DOWN."""

    pan_right_dps: float
    tilt_down_dps: float
    roll_dps: float
    duration_s: float


GESTURES: dict[str, tuple[str, list[GestureStep]]] = {
    "nod_yes": (
        "Hochement OUI (haut / bas)",
        [
            # Two bold, unmistakable vertical nods (down -> up -> down -> up -> level)
            GestureStep(0.0, +64.0, 0.0, 0.27),
            GestureStep(0.0, -72.0, 0.0, 0.30),
            GestureStep(0.0, +68.0, 0.0, 0.28),
            GestureStep(0.0, -68.0, 0.0, 0.28),
            GestureStep(0.0, +24.0, 0.0, 0.16),
            GestureStep(0.0, 0.0, 0.0, 0.12),
        ],
    ),
    "shake_no": (
        "Secousse NON (gauche / droite)",
        [
            # Three crisp, wide horizontal shakes (left -> right -> left -> right -> center)
            GestureStep(-85.0, 0.0, 0.0, 0.23),
            GestureStep(+96.0, 0.0, 0.0, 0.27),
            GestureStep(-96.0, 0.0, 0.0, 0.27),
            GestureStep(+96.0, 0.0, 0.0, 0.27),
            GestureStep(-76.0, 0.0, 0.0, 0.22),
            GestureStep(0.0, 0.0, 0.0, 0.12),
        ],
    ),
    "curious_tilt": (
        "Inclinaison curieuse",
        [
            GestureStep(+38.0, -26.0, 0.0, 0.38),
            GestureStep(0.0, 0.0, 0.0, 0.50),
            GestureStep(-38.0, +26.0, 0.0, 0.38),
            GestureStep(0.0, 0.0, 0.0, 0.12),
        ],
    ),
    "look_desk": (
        "Inclinaison vers la feuille (bureau)",
        [
            GestureStep(0.0, +48.0, 0.0, 0.90),
            GestureStep(0.0, 0.0, 0.0, 0.15),
        ],
    ),
    "look_up": (
        "Redressement vers le visage",
        [
            GestureStep(0.0, -48.0, 0.0, 0.90),
            GestureStep(0.0, 0.0, 0.0, 0.15),
        ],
    ),
}


def screen_to_motor(
    pan_right_dps: float,
    tilt_down_dps: float,
    cfg: PidConfig,
    cur_rotate_deg: int,
    cur_mirror: bool,
) -> tuple[float, float]:
    """Convert canonical screen-space (pan_right, tilt_down) into physical Osmo (motor_yaw, motor_pitch).

    Automatically compensates for any runtime change in rotate_deg or mirror relative to calibration,
    guaranteeing movements NEVER invert when the user rotates or flips the video preview.
    """
    sx, sy = pan_right_dps, tilt_down_dps

    # Exact inverse of forward image pipeline: v_cur = M_cur * R_cur * v_raw
    # => v_cal = M_cal * R_cal * R_cur^-1 * M_cur^-1 * v_cur
    # (Mirror and 90°/270° rotation do not commute, so M_cur^-1 and M_cal must bracket the rotation.)

    # 1. Undo current horizontal mirror (M_cur^-1)
    if cur_mirror:
        sx = -sx

    # 2. Undo rotation difference delta = (cur_rotate - cal_rotate) CW in screen coords (R_cal * R_cur^-1)
    delta = (cur_rotate_deg - cfg.cal_rotate_deg) % 360
    if delta == 90:
        cal_x, cal_y = sy, -sx
    elif delta == 180:
        cal_x, cal_y = -sx, -sy
    elif delta == 270:
        cal_x, cal_y = -sy, sx
    else:
        cal_x, cal_y = sx, sy

    # 3. Re-apply calibration horizontal mirror (M_cal)
    if cfg.cal_mirror:
        cal_x = -cal_x

    # 4. Apply calibrated polarity (single source of truth)
    h_cmd = -cal_x if cfg.invert_yaw else cal_x
    v_cmd = -cal_y if cfg.invert_pitch else cal_y

    # 5. Route to physical DUML wire axes (swap_axes handles portrait/landscape motor mapping)
    if cfg.swap_axes:
        return v_cmd, h_cmd  # (motor_yaw_dps, motor_pitch_dps)
    return h_cmd, v_cmd


@dataclass
class MotionArbiter:
    """Arbitrates Priority 1 (Manual/Gesture) > Priority 2 (Face PID) > Priority 3 (Search/Home FSM)."""

    config: PidConfig = field(default_factory=PidConfig)
    live_motors: bool = False
    control_mode: str = "speed"
    desk_mode: bool = False  # True when using back camera on sheet/table
    events: deque[dict[str, str]] = field(default_factory=lambda: deque(maxlen=80))

    # Filters & tracking memory
    _fx: OneEuroFilter = field(default_factory=lambda: OneEuroFilter(min_cutoff=1.4, beta=2.4))
    _fy: OneEuroFilter = field(default_factory=lambda: OneEuroFilter(min_cutoff=1.4, beta=2.4))
    _prev_err_x: float = 0.0
    _prev_err_y: float = 0.0
    _out_yaw: float = 0.0
    _out_pitch: float = 0.0
    _prev_t: float = 0.0
    _last_face_t: float = 0.0
    _last_known_x: float = 0.0
    _last_known_y: float = 0.0
    _last_known_area: float = 0.0
    _last_exit_x: float = 0.0  # -1 (exited left) .. +1 (exited right)
    _last_exit_y: float = 0.0
    _had_face: bool = False

    # Lost-face recovery FSM state
    _search_start_t: float = 0.0
    _search_active: bool = False
    _home_requested: bool = False
    _returned_home: bool = False
    on_request_recenter: Callable[[], Awaitable[None]] | None = None
    on_light_reaction: Callable[[str], None] | None = None

    # Priority-1 override state
    _override_name: str | None = None
    _override_cmd: tuple[float, float, float] = (0.0, 0.0, 0.0)
    _override_task: asyncio.Task[None] | None = None
    _manual_cmd: tuple[float, float, float] = (0.0, 0.0, 0.0)
    _manual_until: float = 0.0

    latest: TrackingInterpretation = field(default_factory=TrackingInterpretation)

    def log(self, tag: str, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.events.appendleft({"ts": ts, "tag": tag, "msg": message})

    def _slew_limit(self, target: float, current: float, dt: float) -> float:
        max_step = self.config.max_accel_dps2 * dt
        diff = target - current
        if abs(diff) <= max_step:
            return target
        return current + math.copysign(max_step, diff)

    def _select_sticky_face(self, faces: list[FaceBox]) -> FaceBox:
        """Pick the primary face with spatial hysteresis so background faces don't hijack tracking."""
        if len(faces) == 1 or not self._had_face:
            return max(faces, key=lambda f: f.area)

        def score(f: FaceBox) -> float:
            fx = f.cx * 2.0 - 1.0
            fy = f.cy * 2.0 - 1.0
            dist = math.hypot(fx - self._last_known_x, fy - self._last_known_y)
            return f.area * math.exp(-2.2 * dist)

        return max(faces, key=score)

    def _compute_pid(self, err: float, prev_err: float, dt: float, kp: float) -> float:
        dz = self.config.deadzone
        if abs(err) <= dz:
            return 0.0
        # Soft-knee deadzone + slightly progressive nonlinear curve (gentle near center, snappy at edge)
        mag = abs(err) - dz
        err_eff = math.copysign(mag ** 1.12, err)
        d_err = max(-3.0, min(3.0, (err - prev_err) / max(1e-3, dt)))
        out = kp * err_eff + self.config.kd * d_err
        return max(-self.config.max_dps, min(self.config.max_dps, out))

    def _compute_search_sweep(self, elapsed: float) -> tuple[float, float, str] | None:
        """Panoramic search trajectory:
        Phase 1 (0..2.2s): look toward the side where the face last exited
        Phase 2 (2.2..6.6s): smooth wide sweep across to the opposite side
        Phase 3 (6.6..10.2s): return sweep across center
        Returns (pan_right_dps, tilt_down_dps, description) or None when scan completes.
        """
        first_dir = +1.0 if self._last_exit_x >= 0 else -1.0
        dir_label = "droite" if first_dir > 0 else "gauche"
        opp_label = "gauche" if first_dir > 0 else "droite"
        sweep_speed = 32.0

        # Gently pull vertical tilt slightly back toward horizon if face exited high/low
        pitch_nudge = max(-10.0, min(10.0, self._last_exit_y * 12.0)) if elapsed < 1.2 else 0.0

        if elapsed < 2.2:
            return first_dir * sweep_speed, pitch_nudge, f"Recherche côté {dir_label} (dernière sortie)"
        if elapsed < 6.6:
            return -first_dir * sweep_speed, 0.0, f"Balayage panoramique vers la {opp_label}"
        if elapsed < 9.2:
            return first_dir * sweep_speed, 0.0, "Balayage retour vers le centre"
        return None

    def update_vision(
        self, faces: list[FaceBox], vision_ms: float, now: float | None = None
    ) -> TrackingInterpretation:
        t = now if now is not None else time.monotonic()
        dt = max(1e-3, min(0.25, t - self._prev_t)) if self._prev_t > 0 else 1 / 30.0
        self._prev_t = t

        interp = TrackingInterpretation(
            vision_ms=round(vision_ms, 1),
            faces_count=len(faces),
            last_exit_dir="droite" if self._last_exit_x > 0.1 else ("gauche" if self._last_exit_x < -0.1 else "centre"),
        )

        # 1. Process face detection (sticky selection + One-Euro filter)
        if faces and not self.desk_mode:
            best = self._select_sticky_face(faces)
            raw_x = best.cx * 2.0 - 1.0
            raw_y = best.cy * 2.0 - 1.0

            if not self._had_face or self._search_active or self._returned_home:
                self._fx.reset()
                self._fy.reset()
                if self._search_active:
                    self.log("SEARCH", "Visage retrouvé pendant le balayage — verrouillage !")
                    if self.on_light_reaction:
                        self.on_light_reaction("found")
                elif not self._had_face:
                    self.log("VISION", "Visage verrouillé par Apple Vision")
                self._had_face = True
                self._search_active = False
                self._home_requested = False
                self._returned_home = False

            sx, vx = self._fx.filter(raw_x, dt)
            sy, vy = self._fy.filter(raw_y, dt)
            self._last_known_x = sx
            self._last_known_y = sy
            self._last_known_area = best.area
            self._last_exit_x = sx
            self._last_exit_y = sy - self.config.target_y
            self._last_face_t = t

            interp.face_detected = True
            interp.cx = sx
            interp.cy = sy
            interp.box_area = best.area
            interp.confidence = best.confidence
            interp.err_x = sx
            interp.err_y = sy - self.config.target_y
            interp.vel_x = vx
            interp.vel_y = vy

        # 2. Arbitrate command in canonical Screen-Space (+yaw = RIGHT, +pitch = DOWN)
        target_yaw = 0.0
        target_pitch = 0.0
        target_roll = 0.0
        bypass_slew = False

        if self._override_name is not None:
            target_yaw, target_pitch, target_roll = self._override_cmd
            bypass_slew = True
            interp.state = "GESTURE"
            interp.active_gesture = self._override_name
            interp.summary = f"Geste expressif : {self._override_name}"

        elif t < self._manual_until:
            target_yaw, target_pitch, target_roll = self._manual_cmd
            bypass_slew = True
            interp.state = "MANUAL"
            interp.summary = f"Pilotage manuel (pan={target_yaw:+.0f}°/s, tilt={target_pitch:+.0f}°/s)"

        elif self.desk_mode:
            interp.state = "DESK_MODE"
            interp.summary = "Mode Feuille / Caméra arrière — tracking visage en pause"
            self._prev_err_x = self._prev_err_y = 0.0

        elif interp.face_detected:
            target_yaw = self._compute_pid(interp.err_x, self._prev_err_x, dt, self.config.kp_yaw)
            target_pitch = self._compute_pid(interp.err_y, self._prev_err_y, dt, self.config.kp_pitch)
            self._prev_err_x = interp.err_x
            self._prev_err_y = interp.err_y

            if abs(target_yaw) < 0.5 and abs(target_pitch) < 0.5:
                interp.state = "CENTERED"
                interp.summary = "Visage centré dans la zone morte — nacelle stable"
            else:
                interp.state = "TRACKING"
                hints: list[str] = []
                if interp.err_x > self.config.deadzone:
                    hints.append("à droite")
                elif interp.err_x < -self.config.deadzone:
                    hints.append("à gauche")
                if interp.err_y > self.config.deadzone:
                    hints.append("en bas")
                elif interp.err_y < -self.config.deadzone:
                    hints.append("en haut")
                dir_txt = " et ".join(hints) if hints else "en mouvement"
                interp.summary = f"Visage {dir_txt} → suivi fluide"

        else:
            # Face lost FSM: COASTING -> SEARCHING / LAST_POS / RETURNING_HOME -> IDLE_HOME
            self._prev_err_x = self._prev_err_y = 0.0
            lost_for = (t - self._last_face_t) if self._last_face_t > 0 else 999.0

            if self._had_face and lost_for < 1.2:
                interp.state = "COASTING"
                if lost_for < 0.35 and (abs(self._last_exit_x) > 0.35 or abs(self._last_exit_y) > 0.35):
                    # Gentle predictive coast toward exit direction
                    target_yaw = max(-22.0, min(22.0, self._last_exit_x * 24.0))
                    target_pitch = max(-14.0, min(14.0, self._last_exit_y * 16.0))
                    interp.summary = f"Visage sorti ({interp.last_exit_dir}) — poursuite douce..."
                else:
                    interp.summary = f"Visage brièvement masqué ({lost_for:.1f}s) — maintien"

            elif self._had_face and self.config.lost_policy == "last_pos":
                interp.state = "IDLE_HOME"
                interp.summary = "Visage perdu — maintien sur la dernière position connue"

            elif self._had_face and self.config.lost_policy == "home":
                if not self._home_requested:
                    self._home_requested = True
                    self._returned_home = True
                    self.log("HOME", "Visage perdu — retour position nacelle paysage")
                    if self.on_request_recenter:
                        asyncio.create_task(self.on_request_recenter())
                interp.state = "IDLE_HOME"
                interp.summary = "Retour position nacelle centrée (paysage) — en attente"

            elif self._had_face and self.config.lost_policy == "search" and not self._returned_home:
                if not self._search_active:
                    self._search_active = True
                    self._search_start_t = t
                    self.log("SEARCH", f"Visage perdu (> 1.2s) — démarrage balayage côté {interp.last_exit_dir}")
                    if self.on_light_reaction:
                        self.on_light_reaction("search")
                sweep = self._compute_search_sweep(t - self._search_start_t)
                if sweep is not None:
                    target_yaw, target_pitch, desc = sweep
                    interp.state = "SEARCHING"
                    interp.summary = desc
                else:
                    # Completed full panoramic search with no face -> recenter home in landscape!
                    self._search_active = False
                    self._returned_home = True
                    self._had_face = False
                    if self.on_light_reaction:
                        self.on_light_reaction("off")
                    if not self._home_requested:
                        self._home_requested = True
                        self.log("HOME", "Balayage terminé sans visage — recentrage nacelle paysage")
                        if self.on_request_recenter:
                            asyncio.create_task(self.on_request_recenter())
                    interp.state = "RETURNING_HOME"
                    interp.summary = "Aucun visage trouvé — recentrage nacelle paysage"
            else:
                interp.state = "IDLE_HOME"
                interp.summary = "En veille centrée (paysage) — en attente d'un visage"

        # 3. Apply smooth acceleration limiter (slew-rate) for cinema-grade motion
        if bypass_slew:
            self._out_yaw = target_yaw
            self._out_pitch = target_pitch
        else:
            self._out_yaw = self._slew_limit(target_yaw, self._out_yaw, dt)
            self._out_pitch = self._slew_limit(target_pitch, self._out_pitch, dt)
            if abs(target_yaw) < 0.1 and abs(self._out_yaw) < 1.2:
                self._out_yaw = 0.0
            if abs(target_pitch) < 0.1 and abs(self._out_pitch) < 1.2:
                self._out_pitch = 0.0

        interp.cmd_yaw_dps = self._out_yaw
        interp.cmd_pitch_dps = self._out_pitch
        interp.cmd_roll_dps = target_roll

        self.latest = interp
        return interp

    def set_manual(self, pan_right_dps: float, tilt_down_dps: float, hold_s: float = 0.45) -> None:
        self._manual_cmd = (pan_right_dps, tilt_down_dps, 0.0)
        self._manual_until = time.monotonic() + hold_s

    def stop_manual(self) -> None:
        self._manual_cmd = (0.0, 0.0, 0.0)
        self._manual_until = 0.0
        self._out_yaw = 0.0
        self._out_pitch = 0.0

    def trigger_search_now(self) -> None:
        """Manually trigger panoramic search sweep."""
        self._had_face = True
        self._returned_home = False
        self._home_requested = False
        self._search_active = True
        self._search_start_t = time.monotonic()
        self._last_face_t = time.monotonic() - 2.0
        self.log("SEARCH", "Balayage panoramique déclenché manuellement")
        if self.on_light_reaction:
            self.on_light_reaction("search")

    def trigger_gesture(
        self,
        gesture_key: str,
        on_done: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        if gesture_key not in GESTURES:
            return False
        if self._override_task and not self._override_task.done():
            self._override_task.cancel()

        label, steps = GESTURES[gesture_key]
        self.log("GESTURE", f"Déclenchement : {label}")

        # Synchronize rear Pixel 11 Pro HiLight LED color with the expressive gesture
        if self.on_light_reaction:
            light_map = {
                "nod_yes": "yes",
                "shake_no": "no",
                "curious_tilt": "curious",
                "look_desk": "desk",
                "look_up": "off",
            }
            if gesture_key in light_map:
                self.on_light_reaction(light_map[gesture_key])

        async def _run() -> None:
            self._override_name = label
            try:
                for step in steps:
                    self._override_cmd = (step.pan_right_dps, step.tilt_down_dps, step.roll_dps)
                    await asyncio.sleep(step.duration_s)
            finally:
                self._override_cmd = (0.0, 0.0, 0.0)
                self._override_name = None
                self.log("GESTURE", f"Fin du geste : {label}")
                if on_done:
                    await on_done()

        self._override_task = asyncio.create_task(_run())
        return True
