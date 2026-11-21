"""Tests for DUML motor wake/resume packets and MotorWatchdog resilience scenarios."""

from __future__ import annotations

import struct

from hackathon_osmo.control.watchdog import MotorWatchdog
from hackathon_osmo.hardware.duml import (
    CRC_CLASSIC,
    MOTOR_RESUME_MAGIC,
    MOTOR_SUSPEND_MAGIC,
    DumlStreamParser,
    make_gimbal_resume_packet,
    make_gimbal_set_mode_packet,
    make_gimbal_suspend_packet,
)


def test_duml_resume_suspend_and_mode_packets_roundtrip() -> None:
    parser = DumlStreamParser()
    raw_resume = make_gimbal_resume_packet(seq=7, profile=CRC_CLASSIC)
    raw_suspend = make_gimbal_suspend_packet(seq=8, profile=CRC_CLASSIC)
    raw_mode = make_gimbal_set_mode_packet(mode=0, reset=False, seq=9, profile=CRC_CLASSIC)

    pkts = list(parser.feed(raw_resume + raw_suspend + raw_mode))
    assert len(pkts) == 3

    p_res, p_sus, p_mod = pkts
    assert p_res.crc8_ok and p_res.crc16_ok
    assert (p_res.sender, p_res.receiver, p_res.cmd_set, p_res.cmd_id) == (0x02, 0x04, 0x04, 0x0D)
    assert struct.unpack("<H", p_res.payload)[0] == MOTOR_RESUME_MAGIC

    assert p_sus.crc8_ok and p_sus.crc16_ok
    assert (p_sus.cmd_set, p_sus.cmd_id) == (0x04, 0x0D)
    assert struct.unpack("<H", p_sus.payload)[0] == MOTOR_SUSPEND_MAGIC

    assert p_mod.crc8_ok and p_mod.crc16_ok
    assert (p_mod.cmd_set, p_mod.cmd_id) == (0x04, 0x4C)
    assert p_mod.payload == b"\x00\x00"


def test_watchdog_stall_relief_backs_off_only_blocked_direction() -> None:
    wd = MotorWatchdog(stall_trigger_s=1.4, stall_backoff_s=2.5)

    # Normal motion: IMU moves smoothly -> zero clamping
    d0 = wd.evaluate(motor_yaw=45.0, motor_pitch=0.0, pitch_deg=-10.0, roll_deg=0.0, now=10.0)
    assert d0.motor_yaw == 45.0
    assert d0.motor_state == "active"

    # Hits physical end-stop at -29°: IMU stuck at -29.0° for 1.5s despite +45°/s command
    wd.evaluate(motor_yaw=45.0, motor_pitch=0.0, pitch_deg=-29.0, roll_deg=0.0, now=11.0)
    d_stall = wd.evaluate(motor_yaw=45.0, motor_pitch=0.0, pitch_deg=-29.1, roll_deg=0.0, now=12.5)
    assert d_stall.motor_yaw == 0.0
    assert d_stall.motor_state == "stall_relief"

    # Opposite direction (-30°/s away from end-stop) works IMMEDIATELY and clears block!
    d_rev = wd.evaluate(motor_yaw=-30.0, motor_pitch=0.0, pitch_deg=-29.0, roll_deg=0.0, now=12.7)
    assert d_rev.motor_yaw == -30.0
    assert d_rev.motor_state == "active"


def test_watchdog_detects_sudden_limp_drop_and_bounds_wake_attempts() -> None:
    wd = MotorWatchdog(auto_wake_enabled=True, max_wake_attempts=2, wake_retry_interval_s=6.0, thermal_cooldown_s=25.0)

    # Normal upright pose
    d1 = wd.evaluate(motor_yaw=0.0, motor_pitch=0.0, pitch_deg=5.0, roll_deg=0.0, now=100.0)
    assert d1.motor_state == "active" and d1.action is None

    # Tall user gradual tilt to -28° -> NO false positive!
    d_tall = wd.evaluate(motor_yaw=0.0, motor_pitch=0.0, pitch_deg=-28.0, roll_deg=0.0, now=103.5)
    assert d_tall.motor_state == "active" and d_tall.action is None

    # Sudden freefall collapse (+5° -> -44° in 0.8s, matching logged OM3 drops)
    wd.evaluate(motor_yaw=0.0, motor_pitch=0.0, pitch_deg=5.0, roll_deg=0.0, now=110.0)
    d_drop1 = wd.evaluate(motor_yaw=0.0, motor_pitch=0.0, pitch_deg=-44.0, roll_deg=-62.0, now=110.8)
    assert d_drop1.action == "wake_motors"
    assert d_drop1.motor_state == "standby_recovering"

    # Still collapsed 7s later -> 2nd wake attempt
    d_drop2 = wd.evaluate(motor_yaw=0.0, motor_pitch=0.0, pitch_deg=-44.0, roll_deg=-62.0, now=117.5)
    assert d_drop2.action == "wake_motors"

    # Still collapsed after 2 attempts -> thermal_protection cool-down (stop hammering!)
    d_prot = wd.evaluate(motor_yaw=0.0, motor_pitch=0.0, pitch_deg=-44.0, roll_deg=-62.0, now=124.5)
    assert d_prot.motor_state == "thermal_protection"
    assert d_prot.action is None

    # User re-energizes handle -> pitch returns to +4° -> instant recovery to active!
    d_rec = wd.evaluate(motor_yaw=0.0, motor_pitch=0.0, pitch_deg=4.0, roll_deg=2.0, now=126.0)
    assert d_rec.motor_state == "active"


def test_watchdog_schedules_anti_sleep_keepalives() -> None:
    wd = MotorWatchdog(keepalive_speed_interval_s=6.0, keepalive_resume_interval_s=45.0)
    wd.note_ble_tx(now=10.0, is_resume=True)

    d_quiet = wd.evaluate(motor_yaw=0.0, motor_pitch=0.0, pitch_deg=4.0, roll_deg=0.0, now=13.0)
    assert d_quiet.action is None

    d_spd = wd.evaluate(motor_yaw=0.0, motor_pitch=0.0, pitch_deg=4.0, roll_deg=0.0, now=16.5)
    assert d_spd.action == "keepalive_speed"

    d_res = wd.evaluate(motor_yaw=0.0, motor_pitch=0.0, pitch_deg=4.0, roll_deg=0.0, now=56.0)
    assert d_res.action == "keepalive_resume"
