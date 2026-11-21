"""DJI DUML v1 binary protocol encoder/decoder with CRC-8 and CRC-16 support.

Supports both known DJI CRC seed pairs:
- DUML classic (OM3 / Mavic / dji-firmware-tools): CRC8 init=0x77, CRC16 init=0x3692
- DUML modern  (Pocket 3 / Action / newer BLE):    CRC8 init=0xEE, CRC16 init=0x496C

Both use the exact same reflected polynomials:
- CRC-8:  poly 0x31 (reflected 0x8C)
- CRC-16: poly 0x1021 (reflected 0x8408)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterator


def _make_crc8_table() -> tuple[int, ...]:
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            if c & 1:
                c = (c >> 1) ^ 0x8C  # reflected 0x31
            else:
                c >>= 1
        table.append(c & 0xFF)
    return tuple(table)


def _make_crc16_table() -> tuple[int, ...]:
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            if c & 1:
                c = (c >> 1) ^ 0x8408  # reflected 0x1021
            else:
                c >>= 1
        table.append(c & 0xFFFF)
    return tuple(table)


CRC8_TABLE = _make_crc8_table()
CRC16_TABLE = _make_crc16_table()


@dataclass(frozen=True)
class CrcProfile:
    name: str
    crc8_init: int
    crc16_init: int


CRC_CLASSIC = CrcProfile("classic", crc8_init=0x77, crc16_init=0x3692)
CRC_MODERN = CrcProfile("modern", crc8_init=0xEE, crc16_init=0x496C)
ALL_CRC_PROFILES = (CRC_CLASSIC, CRC_MODERN)


def crc8(data: bytes, init: int = CRC_CLASSIC.crc8_init) -> int:
    crc = init & 0xFF
    for b in data:
        crc = CRC8_TABLE[(crc ^ b) & 0xFF]
    return crc


def crc16(data: bytes, init: int = CRC_CLASSIC.crc16_init) -> int:
    crc = init & 0xFFFF
    for b in data:
        crc = (crc >> 8) ^ CRC16_TABLE[(crc ^ b) & 0xFF]
    return crc


@dataclass
class DumlPacket:
    sender: int
    receiver: int
    seq: int
    cmd_type: int
    cmd_set: int
    cmd_id: int
    payload: bytes
    version: int = 1
    crc_profile: str = "unknown"
    crc8_ok: bool = False
    crc16_ok: bool = False

    def encode(self, profile: CrcProfile = CRC_CLASSIC) -> bytes:
        total_len = 13 + len(self.payload)
        if total_len > 0x3FF:
            raise ValueError(f"DUML packet too long: {total_len}")
        ver_len = ((self.version & 0x3F) << 10) | (total_len & 0x3FF)
        header_pre = struct.pack("<BH", 0x55, ver_len)
        h_crc = crc8(header_pre, init=profile.crc8_init)
        body = (
            header_pre
            + bytes([h_crc])
            + struct.pack(
                "<BBHBBB",
                self.sender & 0xFF,
                self.receiver & 0xFF,
                self.seq & 0xFFFF,
                self.cmd_type & 0xFF,
                self.cmd_set & 0xFF,
                self.cmd_id & 0xFF,
            )
            + self.payload
        )
        p_crc = crc16(body, init=profile.crc16_init)
        return body + struct.pack("<H", p_crc)


class DumlStreamParser:
    """Incremental parser extracting DUML frames from arbitrary BLE notification chunks."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> Iterator[DumlPacket]:
        self._buf.extend(data)
        while len(self._buf) >= 4:
            if self._buf[0] != 0x55:
                idx = self._buf.find(b"\x55")
                if idx == -1:
                    self._buf.clear()
                    return
                del self._buf[:idx]
                if len(self._buf) < 4:
                    return

            ver_len = self._buf[1] | (self._buf[2] << 8)
            length = ver_len & 0x3FF
            version = (ver_len >> 10) & 0x3F

            # Validate header CRC against known profiles
            header_crc = self._buf[3]
            matched_profile: CrcProfile | None = None
            for prof in ALL_CRC_PROFILES:
                if crc8(bytes(self._buf[:3]), init=prof.crc8_init) == header_crc:
                    matched_profile = prof
                    break

            if matched_profile is None or length < 13:
                # False 0x55 start byte; skip one byte and resync
                del self._buf[0]
                continue

            if len(self._buf) < length:
                return  # wait for more bytes

            raw = bytes(self._buf[:length])
            del self._buf[:length]

            expected_crc16 = struct.unpack_from("<H", raw, length - 2)[0]
            computed_crc16 = crc16(raw[:-2], init=matched_profile.crc16_init)
            crc16_ok = computed_crc16 == expected_crc16

            sender, receiver, seq, cmd_type, cmd_set, cmd_id = struct.unpack_from(
                "<BBHBBB", raw, 4
            )
            payload = raw[11 : length - 2]

            yield DumlPacket(
                sender=sender,
                receiver=receiver,
                seq=seq,
                cmd_type=cmd_type,
                cmd_set=cmd_set,
                cmd_id=cmd_id,
                payload=payload,
                version=version,
                crc_profile=matched_profile.name,
                crc8_ok=True,
                crc16_ok=crc16_ok,
            )


# --- Gimbal command helpers (CmdSet 0x04, verified against DJI Osmo Mobile 2/3/4) ---

ROTATION_MODE_RELATIVE = 0x04
ROTATION_MODE_ABSOLUTE = 0x05
ROTATION_MODE_SPEED = 0x80

# Reverse-engineered DJI Osmo DUML Gimbal Suspend/Resume magic constants (CmdSet=0x04, CmdId=0x0D)
MOTOR_RESUME_MAGIC = 0x2AB5
MOTOR_SUSPEND_MAGIC = 0x7EF2


def make_gimbal_resume_packet(
    seq: int = 1,
    sender: int = 0x02,
    receiver: int = 0x04,
    profile: CrcProfile = CRC_CLASSIC,
) -> bytes:
    """Build DUML Gimbal Resume / Exit-Standby packet (CmdSet=0x04, CmdId=0x0D, payload=0x2AB5 LE)."""
    pkt = DumlPacket(
        sender=sender,
        receiver=receiver,
        seq=seq,
        cmd_type=0x40,
        cmd_set=0x04,
        cmd_id=0x0D,
        payload=struct.pack("<H", MOTOR_RESUME_MAGIC),
    )
    return pkt.encode(profile=profile)


def make_gimbal_suspend_packet(
    seq: int = 1,
    sender: int = 0x02,
    receiver: int = 0x04,
    profile: CrcProfile = CRC_CLASSIC,
) -> bytes:
    """Build DUML Gimbal Suspend / Standby packet (CmdSet=0x04, CmdId=0x0D, payload=0x7EF2 LE)."""
    pkt = DumlPacket(
        sender=sender,
        receiver=receiver,
        seq=seq,
        cmd_type=0x40,
        cmd_set=0x04,
        cmd_id=0x0D,
        payload=struct.pack("<H", MOTOR_SUSPEND_MAGIC),
    )
    return pkt.encode(profile=profile)


def make_gimbal_set_mode_packet(
    mode: int = 0x00,
    reset: bool = False,
    seq: int = 1,
    sender: int = 0x02,
    receiver: int = 0x04,
    profile: CrcProfile = CRC_CLASSIC,
) -> bytes:
    """Build DUML Gimbal Mode/Reset packet (CmdSet=0x04, CmdId=0x4C: mode=0 Follow, 1 Lock, 2 FPV)."""
    pkt = DumlPacket(
        sender=sender,
        receiver=receiver,
        seq=seq,
        cmd_type=0x40,
        cmd_set=0x04,
        cmd_id=0x4C,
        payload=bytes([mode & 0xFF, 0x01 if reset else 0x00]),
    )
    return pkt.encode(profile=profile)


def make_gimbal_rotate_packet(
    yaw: int = 0,
    pitch: int = 0,
    roll: int = 0,
    duration_ds: int = 2,
    mode: int = ROTATION_MODE_SPEED,
    seq: int = 1,
    sender: int = 0x02,
    receiver: int = 0x04,
    profile: CrcProfile = CRC_CLASSIC,
) -> bytes:
    """Build verified DJI Osmo Mobile 3 rotation packet (21 bytes total, len=0x15).

    Calibrated wire payload (8 bytes): <hhhBB = (pitch, roll, yaw, mode, duration_ds)
    - slot0 = pitch (vertical tilt), slot1 = roll, slot2 = yaw (horizontal pan)
    - mode=0x80 (SPEED):    CmdId=0x0C, speeds in 0.1 deg/s (e.g. -1500..1500)
    - mode=0x04 (RELATIVE): CmdId=0x14, delta angles in 0.1 deg
    - mode=0x05 (ABSOLUTE): CmdId=0x14, target angles in 0.1 deg
    - duration_ds: duration in tenths of a second (e.g. 2 = 0.2s, 10 = 1.0s)
    """
    cmd_id = 0x0C if mode == ROTATION_MODE_SPEED else 0x14
    payload = struct.pack(
        "<hhhBB",
        max(-1800, min(1800, int(pitch))),
        max(-1800, min(1800, int(roll))),
        max(-1800, min(1800, int(yaw))),
        mode & 0xFF,
        max(0, min(255, int(duration_ds))),
    )
    pkt = DumlPacket(
        sender=sender,
        receiver=receiver,
        seq=seq,
        cmd_type=0x00,
        cmd_set=0x04,
        cmd_id=cmd_id,
        payload=payload,
    )
    return pkt.encode(profile=profile)


def make_gimbal_speed_packet(
    yaw_speed: int = 0,
    pitch_speed: int = 0,
    roll_speed: int = 0,
    duration_ds: int = 2,
    enable: bool = True,
    seq: int = 1,
    sender: int = 0x02,
    receiver: int = 0x04,
    profile: CrcProfile = CRC_CLASSIC,
) -> bytes:
    """Build DUML Gimbal Speed Control packet (CmdSet=0x04, CmdId=0x0C, 21 bytes)."""
    if not enable:
        yaw_speed = pitch_speed = roll_speed = 0
    return make_gimbal_rotate_packet(
        yaw=yaw_speed,
        pitch=pitch_speed,
        roll=roll_speed,
        duration_ds=duration_ds,
        mode=ROTATION_MODE_SPEED,
        seq=seq,
        sender=sender,
        receiver=receiver,
        profile=profile,
    )


def make_gimbal_joystick_packet(
    yaw: int = 1024,
    pitch: int = 1024,
    roll: int = 1024,
    seq: int = 1,
    sender: int = 0x02,
    receiver: int = 0x04,
    profile: CrcProfile = CRC_CLASSIC,
) -> bytes:
    """Build DUML Gimbal Joystick/PWM packet (CmdSet=0x04, CmdId=0x01).

    Center = 1024, typical range 364..1684.
    """
    payload = struct.pack("<HHH", yaw, roll, pitch)
    pkt = DumlPacket(
        sender=sender,
        receiver=receiver,
        seq=seq,
        cmd_type=0x40,
        cmd_set=0x04,
        cmd_id=0x01,
        payload=payload,
    )
    return pkt.encode(profile=profile)
