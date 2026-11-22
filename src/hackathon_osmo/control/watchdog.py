"""Motor Watchdog & Anti-Interruption Controller for DJI Osmo Mobile 3.

Prevents the 3 real-world motor failure modes observed across 28+ live sessions:
1. Mechanical end-stop stall -> eases torque ONLY in the blocked direction after
   1.4s of zero IMU motion so DJI's internal overcurrent breaker never trips,
   WITHOUT artificial fixed pitch/yaw clamps!
2. Inactivity standby -> schedules gentle 6s BLE speed keepalive + 45s DUML
   Resume (CmdId=0x0D 0x2AB5) heartbeats.
3. Unexpected limp/standby drop -> distinguishes true motor collapse from tall-user
   upward tilt, runs bounded automatic BLE wake recovery (max 2 attempts before
   entering a cool-down thermal_protection state that keeps voice/OLED alive!).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class WatchdogDecision:
    motor_yaw: float
    motor_pitch: float
    motor_state: str  # "active" | "stall_relief" | "standby_recovering" | "thermal_protection"
    action: str | None = None  # None | "wake_motors" | "keepalive_speed" | "keepalive_resume"
    alert: str | None = None


@dataclass
class MotorWatchdog:
    """Pure, deterministic motor resilience state machine (unit-testable without BLE)."""

    # Stall relief parameters: catch slow PID end-stop pushing (>= 10 deg/s, e.g. 18-20 deg/s at upper pitch stop)
    # after 2.2s (>2 OM3 1 Hz IMU packets) BEFORE DJI's internal overcurrent breaker ever trips!
    stall_cmd_threshold_dps: float = 10.0
    stall_min_delta_deg: float = 1.0
    stall_trigger_s: float = 2.2
    stall_backoff_s: float = 2.8

    # Anti-sleep BLE keepalive (ONLY safe 0-speed CmdId=0x0C! NEVER send CmdId=0x0D 0x2AB5 periodically:
    # empirical IMU logs proved 0x0D 0x2AB5 is a Standby TOGGLE on OM3 that puts awake motors to sleep!)
    keepalive_speed_interval_s: float = 6.0
    keepalive_resume_interval_s: float = 0.0  # disabled permanently (toggle command!)

    # Limp / standby detection (passive alert only by default — never surprise-recenter mid-tracking!)
    auto_wake_enabled: bool = False
    limp_drop_delta_deg: float = 22.0
    limp_pitch_floor_deg: float = -30.5
    max_wake_attempts: int = 2
    wake_retry_interval_s: float = 6.5
    thermal_cooldown_s: float = 25.0

    # Internal state
    motor_state: str = "active"
    _prev_pitch: float | None = None
    _prev_roll: float | None = None
    _prev_att_t: float = 0.0
    _last_speed_tx_t: float = 0.0
    _last_resume_tx_t: float = 0.0

    # Stall tracking per motor axis (yaw = vertical pitch on OM3 when swap_axes=True, pitch = horizontal pan)
    _stall_start_yaw_t: float = 0.0
    _stall_ref_pitch: float | None = None
    _stall_blocked_yaw_sign: int = 0
    _stall_blocked_yaw_until: float = 0.0

    _stall_start_pitch_t: float = 0.0
    _stall_ref_roll: float | None = None
    _stall_blocked_pitch_sign: int = 0
    _stall_blocked_pitch_until: float = 0.0

    # Limp & wake tracking
    _limp_since_t: float = 0.0
    _wake_attempts: int = 0
    _last_wake_t: float = 0.0
    _thermal_until_t: float = 0.0
    _active_history: list[tuple[float, float]] = field(default_factory=list)

    def reset(self) -> None:
        """Clear stall blocks and thermal latches (e.g. after manual WAKE MOTORS or BLE reconnect)."""
        self.motor_state = "active"
        self._stall_start_yaw_t = 0.0
        self._stall_start_pitch_t = 0.0
        self._stall_blocked_yaw_sign = 0
        self._stall_blocked_yaw_until = 0.0
        self._stall_blocked_pitch_sign = 0
        self._stall_blocked_pitch_until = 0.0
        self._limp_since_t = 0.0
        self._wake_attempts = 0
        self._thermal_until_t = 0.0

    def note_ble_tx(self, now: float | None = None, *, is_resume: bool = False) -> None:
        t = now if now is not None else time.monotonic()
        self._last_speed_tx_t = t
        if is_resume:
            self._last_resume_tx_t = t

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        d = (a - b + 180.0) % 360.0 - 180.0
        return abs(d)

    def _update_stall_axis(
        self,
        cmd_val: float,
        imu_angle: float | None,
        now: float,
        *,
        axis: str,
    ) -> float:
        """Ease torque ONLY in the stalled sign direction if motor pushes an end-stop > 1.4s."""
        if axis == "yaw":
            blocked_sign = self._stall_blocked_yaw_sign
            blocked_until = self._stall_blocked_yaw_until
            start_t = self._stall_start_yaw_t
            ref_ang = self._stall_ref_pitch
        else:
            blocked_sign = self._stall_blocked_pitch_sign
            blocked_until = self._stall_blocked_pitch_until
            start_t = self._stall_start_pitch_t
            ref_ang = self._stall_ref_roll

        cur_sign = 1 if cmd_val > 0.5 else (-1 if cmd_val < -0.5 else 0)

        # If currently backed off in this exact sign direction, hold 0 torque (allow opposite sign immediately!)
        if blocked_sign != 0 and now < blocked_until:
            if cur_sign == blocked_sign:
                return 0.0
            if cur_sign == -blocked_sign:
                # Reversing away from the end-stop clears the block immediately
                blocked_sign = 0
                blocked_until = 0.0
        elif blocked_sign != 0 and now >= blocked_until:
            blocked_sign = 0
            blocked_until = 0.0

        # Track sustained high torque with zero IMU progress
        if imu_angle is None or abs(cmd_val) < self.stall_cmd_threshold_dps:
            start_t = 0.0
            ref_ang = None
        else:
            if start_t <= 0.0 or ref_ang is None:
                start_t = now
                ref_ang = imu_angle
            elif self._angle_diff(imu_angle, ref_ang) >= self.stall_min_delta_deg:
                # Gimbal IS moving cleanly — reset timer
                start_t = now
                ref_ang = imu_angle
            elif (now - start_t) >= self.stall_trigger_s:
                # Stalled against physical end-stop! Back off ONLY this direction
                blocked_sign = cur_sign
                blocked_until = now + self.stall_backoff_s
                start_t = 0.0
                ref_ang = None
                cmd_val = 0.0

        if axis == "yaw":
            self._stall_blocked_yaw_sign = blocked_sign
            self._stall_blocked_yaw_until = blocked_until
            self._stall_start_yaw_t = start_t
            self._stall_ref_pitch = ref_ang
        else:
            self._stall_blocked_pitch_sign = blocked_sign
            self._stall_blocked_pitch_until = blocked_until
            self._stall_start_pitch_t = start_t
            self._stall_ref_roll = ref_ang

        return cmd_val

    def evaluate(
        self,
        *,
        motor_yaw: float,
        motor_pitch: float,
        pitch_deg: float | None,
        roll_deg: float | None,
        is_stuck_flag: bool = False,
        desk_mode: bool = False,
        now: float | None = None,
    ) -> WatchdogDecision:
        """Inspect live IMU & desired motor speeds, returning safe speeds + wake/keepalive actions."""
        t = now if now is not None else time.monotonic()
        if self._last_speed_tx_t <= 0.0:
            self._last_speed_tx_t = t
        if self._last_resume_tx_t <= 0.0:
            self._last_resume_tx_t = t

        alert: str | None = None
        action: str | None = None

        # 1. Maintain 2.5s pitch history to spot sudden freefall limp collapse (+6° -> -45°)
        sudden_drop = False
        if pitch_deg is not None:
            self._active_history.append((t, pitch_deg))
            cutoff = t - 2.6
            self._active_history = [(ts, p) for ts, p in self._active_history if ts >= cutoff]
            if not desk_mode and pitch_deg < self.limp_pitch_floor_deg and self._active_history:
                max_recent = max(p for _, p in self._active_history)
                if (max_recent - pitch_deg) >= self.limp_drop_delta_deg:
                    sudden_drop = True

        # 2. Detect confirmed limp/standby state vs normal active operation
        collapsed_pose = (
            not desk_mode
            and pitch_deg is not None
            and pitch_deg < -32.0
            and roll_deg is not None
            and abs(roll_deg) > 35.0
        )
        limp_detected = bool(is_stuck_flag or sudden_drop)
        if collapsed_pose:
            if self._limp_since_t <= 0.0:
                self._limp_since_t = t
            elif (t - self._limp_since_t) >= 2.0:
                limp_detected = True
        elif pitch_deg is not None and pitch_deg > -24.0:
            # Motors clearly holding upright again — full automatic recovery!
            self._limp_since_t = 0.0
            self._wake_attempts = 0
            self._thermal_until_t = 0.0
            if self.motor_state in ("standby_recovering", "thermal_protection"):
                self.motor_state = "active"
                alert = "Moteurs DJI OM3 réveillés et actifs !"

        if limp_detected and not self.auto_wake_enabled:
            prev_state = self.motor_state
            self.motor_state = "thermal_protection"
            limp_alert = (
                f"Moteurs DJI OM3 en veille/protection (pitch={pitch_deg:+.1f}°) — "
                "commandes suspendues. Clique WAKE MOTORS (ou 1× bouton M)."
                if prev_state != "thermal_protection" and pitch_deg is not None
                else None
            )
            return WatchdogDecision(
                motor_yaw=0.0,
                motor_pitch=0.0,
                motor_state=self.motor_state,
                action=None,
                alert=limp_alert,
            )

        if limp_detected and self.auto_wake_enabled:
            if t < self._thermal_until_t:
                # Cool-down active: don't hammer overheated motors, keep Studio/voice alive
                self.motor_state = "thermal_protection"
                return WatchdogDecision(
                    motor_yaw=0.0,
                    motor_pitch=0.0,
                    motor_state=self.motor_state,
                    action=None,
                    alert=None,
                )
            if self._wake_attempts < self.max_wake_attempts:
                if (t - self._last_wake_t) >= self.wake_retry_interval_s:
                    self._wake_attempts += 1
                    self._last_wake_t = t
                    self.motor_state = "standby_recovering"
                    action = "wake_motors"
                    alert = (
                        f"Décrochage/veille OM3 détecté (pitch={pitch_deg:+.1f}°) — "
                        f"réveil BLE automatique #{self._wake_attempts}/{self.max_wake_attempts} (CmdId 0x0D 0x2AB5)…"
                        if pitch_deg is not None
                        else f"Réveil BLE automatique OM3 #{self._wake_attempts}/{self.max_wake_attempts}…"
                    )
                    return WatchdogDecision(
                        motor_yaw=0.0,
                        motor_pitch=0.0,
                        motor_state=self.motor_state,
                        action=action,
                        alert=alert,
                    )
            else:
                # Exhausted automatic BLE wakes -> hardware thermal/battery protection!
                self.motor_state = "thermal_protection"
                self._thermal_until_t = t + self.thermal_cooldown_s
                self._wake_attempts = 0
                alert = (
                    "Protection thermique/batterie DJI OM3 active — vérifie l'équilibrage du Pixel "
                    "et la batterie, puis appuie 1× sur le bouton M (ou clique WAKE MOTORS)."
                )
                return WatchdogDecision(
                    motor_yaw=0.0,
                    motor_pitch=0.0,
                    motor_state=self.motor_state,
                    action=None,
                    alert=alert,
                )

        # 3. Apply mechanical end-stop stall relief (no fixed angle clamps!)
        # Note: on OM3 with swap_axes=True, motor_yaw drives vertical pitch_deg and motor_pitch drives horizontal pan/roll_deg.
        safe_yaw = self._update_stall_axis(motor_yaw, pitch_deg, t, axis="yaw")
        safe_pitch = self._update_stall_axis(motor_pitch, roll_deg, t, axis="pitch")

        in_stall_relief = (
            (self._stall_blocked_yaw_sign != 0 and t < self._stall_blocked_yaw_until)
            or (self._stall_blocked_pitch_sign != 0 and t < self._stall_blocked_pitch_until)
        )
        if in_stall_relief:
            if self.motor_state != "stall_relief":
                alert = "Butée mécanique détectée par l'IMU — couple relâché dans cette direction (anti-surchauffe OM3)"
            self.motor_state = "stall_relief"
        elif self.motor_state not in ("standby_recovering", "thermal_protection"):
            self.motor_state = "active"

        # 4. Safe anti-sleep BLE heartbeat when idle (ONLY 0-speed CmdId=0x0C, NEVER 0x0D toggle!)
        is_idle = abs(safe_yaw) < 0.2 and abs(safe_pitch) < 0.2
        if is_idle:
            if (t - self._last_speed_tx_t) >= self.keepalive_speed_interval_s:
                action = "keepalive_speed"
                self._last_speed_tx_t = t
        else:
            self._last_speed_tx_t = t

        self._prev_pitch = pitch_deg
        self._prev_roll = roll_deg
        self._prev_att_t = t

        return WatchdogDecision(
            motor_yaw=safe_yaw,
            motor_pitch=safe_pitch,
            motor_state=self.motor_state,
            action=action,
            alert=alert,
        )
