"""Async BLE client for DJI Osmo Mobile (tested on OM3-4008VF)."""

from __future__ import annotations

import asyncio
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Callable

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice

from hackathon_osmo.hardware.duml import (
    CRC_CLASSIC,
    ROTATION_MODE_ABSOLUTE,
    ROTATION_MODE_RELATIVE,
    CrcProfile,
    DumlPacket,
    DumlStreamParser,
    make_gimbal_resume_packet,
    make_gimbal_rotate_packet,
    make_gimbal_set_mode_packet,
    make_gimbal_speed_packet,
)

SERVICE_DJI_FFF0 = "0000fff0-0000-1000-8000-00805f9b34fb"
CHAR_NOTIFY_FFF4 = "0000fff4-0000-1000-8000-00805f9b34fb"
CHAR_WRITE_FFF5 = "0000fff5-0000-1000-8000-00805f9b34fb"


@dataclass
class GimbalAttitude:
    pitch_deg: float
    roll_deg: float
    yaw_deg: float
    quat: tuple[float, float, float, float] | None = None  # normalized (qw, qx, qy, qz)
    updated_at: float = field(default_factory=time.monotonic)


def quat_angle_deg(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> float:
    """True 3D geodesic rotation angle in degrees [0..180] between two unit quaternions (no Euler coupling)."""
    dot = abs(q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3])
    return math.degrees(2.0 * math.acos(max(0.0, min(1.0, dot))))


async def discover_osmo(
    name_prefix: str = "OM", timeout: float = 5.0
) -> BLEDevice | None:
    """Find the DJI Osmo Mobile: first check already-connected macOS peripherals, then scan."""
    # 1. Instant lookup for peripherals already connected at macOS Bluetooth level
    try:
        import CoreBluetooth
        from bleak.backends.corebluetooth.CentralManagerDelegate import (
            CentralManagerDelegate,
        )

        mgr = CentralManagerDelegate()
        await asyncio.sleep(0.25)
        uuids = [CoreBluetooth.CBUUID.UUIDWithString_("FFF0")]
        for peripheral in mgr.central_manager.retrieveConnectedPeripheralsWithServices_(
            uuids
        ):
            name = peripheral.name() or ""
            if not name_prefix or name.upper().startswith(name_prefix.upper()):
                return BLEDevice(
                    peripheral.identifier().UUIDString(),
                    name,
                    (peripheral, mgr),
                )
    except Exception:
        pass

    # 2. Fallback to active BLE advertising scan
    found = await BleakScanner.discover(timeout=timeout, return_adv=True)
    candidates: list[tuple[int, BLEDevice]] = []
    for _addr, (dev, adv) in found.items():
        name = dev.name or adv.local_name or ""
        uuids = [u.lower() for u in (adv.service_uuids or [])]
        if SERVICE_DJI_FFF0 in uuids or name.upper().startswith(name_prefix.upper()):
            candidates.append((adv.rssi, dev))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


class OsmoBleClient:
    """High-level BLE controller for DJI Osmo Mobile over GATT fff4/fff5."""

    def __init__(
        self,
        device_or_address: BLEDevice | str,
        profile: CrcProfile = CRC_CLASSIC,
        on_packet: Callable[[DumlPacket], None] | None = None,
    ) -> None:
        self.address = (
            device_or_address.address
            if isinstance(device_or_address, BLEDevice)
            else device_or_address
        )
        self.profile = profile
        self.on_packet = on_packet
        self._client = BleakClient(device_or_address, timeout=10.0)
        self._parser = DumlStreamParser()
        self._seq = 1
        self.last_attitude: GimbalAttitude | None = None
        self.last_rx_at: float = 0.0
        self.connected_at: float = 0.0

    async def __aenter__(self) -> OsmoBleClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.disconnect()

    async def connect(self) -> None:
        await self._client.connect()
        try:
            # macOS CoreBluetooth occasionally needs a brief beat after connect before GATT services settle
            try:
                await self._client.start_notify(CHAR_NOTIFY_FFF4, self._handle_notify)
            except Exception:
                await asyncio.sleep(0.35)
                await self._client.start_notify(CHAR_NOTIFY_FFF4, self._handle_notify)
            now = time.monotonic()
            self.connected_at = now
            self.last_rx_at = now
        except Exception:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            raise

    async def disconnect(self) -> None:
        if self._client.is_connected:
            try:
                await self._client.stop_notify(CHAR_NOTIFY_FFF4)
            except Exception:
                pass
            await self._client.disconnect()

    def _next_seq(self) -> int:
        s = self._seq
        self._seq = (self._seq + 1) & 0xFFFF
        if self._seq == 0:
            self._seq = 1
        return s

    def _handle_notify(self, _sender: object, data: bytearray) -> None:
        self.last_rx_at = time.monotonic()
        for pkt in self._parser.feed(bytes(data)):
            if pkt.cmd_set == 0x04 and pkt.cmd_id == 0x05 and len(pkt.payload) >= 40:
                # Live IMU quaternion (4 x float32 LE at offset 24)
                qx, qy, qz, qw = struct.unpack_from("<ffff", pkt.payload, 24)
                norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
                if 0.9 < norm < 1.1:
                    sinr = 2.0 * (qw * qx + qy * qz)
                    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
                    roll = math.degrees(math.atan2(sinr, cosr))
                    sinp = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
                    pitch = math.degrees(math.asin(sinp))
                    siny = 2.0 * (qw * qz + qx * qy)
                    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
                    yaw = math.degrees(math.atan2(siny, cosy))
                    self.last_attitude = GimbalAttitude(
                        pitch_deg=pitch,
                        roll_deg=roll,
                        yaw_deg=yaw,
                        quat=(qw / norm, qx / norm, qy / norm, qz / norm),
                    )
            elif pkt.cmd_set == 0x04 and pkt.cmd_id == 0x05 and len(pkt.payload) >= 6:
                pitch, roll, yaw = struct.unpack_from("<hhh", pkt.payload, 0)
                self.last_attitude = GimbalAttitude(
                    pitch_deg=pitch / 10.0,
                    roll_deg=roll / 10.0,
                    yaw_deg=yaw / 10.0,
                )
            if self.on_packet:
                self.on_packet(pkt)

    async def send_raw(self, data: bytes) -> None:
        # DJI Osmo Mobile 2/3/4 fff5 RX buffer expects <= 20-byte chunks even if OS negotiates larger ATT MTU
        chunk_size = 20
        for i in range(0, len(data), chunk_size):
            await self._client.write_gatt_char(
                CHAR_WRITE_FFF5, data[i : i + chunk_size], response=False
            )

    async def send_recenter(self) -> None:
        """Recenter gimbal to neutral position (verified OM3 CmdSet=0x04 CmdId=0x4C payload=0101)."""
        pkt = DumlPacket(
            sender=0x02,
            receiver=0x04,
            seq=self._next_seq(),
            cmd_type=0x40,
            cmd_set=0x04,
            cmd_id=0x4C,
            payload=b"\x01\x01",
        ).encode(profile=self.profile)
        await self.send_raw(pkt)

    async def send_resume_motors(self) -> None:
        """Send DUML Resume/Exit-Standby command (CmdSet=0x04, CmdId=0x0D, payload=0x2AB5 LE)."""
        pkt = make_gimbal_resume_packet(seq=self._next_seq(), profile=self.profile)
        await self.send_raw(pkt)

    async def send_follow_mode(self) -> None:
        """Ensure Gimbal Follow mode is active (CmdSet=0x04, CmdId=0x4C, mode=0x00)."""
        pkt = make_gimbal_set_mode_packet(mode=0x00, reset=False, seq=self._next_seq(), profile=self.profile)
        await self.send_raw(pkt)

    async def wake_motors_sequence(self) -> None:
        """Multi-stage BLE wake sequence to exit OM3 standby and re-energize motor control."""
        await self.send_resume_motors()
        await asyncio.sleep(0.06)
        await self.send_resume_motors()
        await asyncio.sleep(0.06)
        await self.send_follow_mode()
        await asyncio.sleep(0.06)
        await self.send_recenter()
        await asyncio.sleep(0.08)
        await self.send_speed(0.0, 0.0, 0.0, duration_ds=2)

    async def send_toggle_orientation(self) -> None:
        """Toggle Landscape <-> Portrait 90 deg roll (verified OM3 CmdSet=0x04 CmdId=0x3A payload=0100)."""
        pkt = DumlPacket(
            sender=0x02,
            receiver=0x04,
            seq=self._next_seq(),
            cmd_type=0x40,
            cmd_set=0x04,
            cmd_id=0x3A,
            payload=b"\x01\x00",
        ).encode(profile=self.profile)
        await self.send_raw(pkt)

    def is_landscape(self) -> bool | None:
        """Heuristic from live IMU quaternion on OM3:
        - Landscape has |yaw| > 135°
        - True upright Portrait has 70° <= |yaw| <= 115° AND |pitch| < 25°
        - Limp/standby tilt returns None so recenter NEVER fires a spurious 90° orientation flip!
        """
        if self.last_attitude is None:
            return None
        ay = abs(self.last_attitude.yaw_deg)
        if ay > 135.0:
            return True
        if 70.0 <= ay <= 115.0 and abs(self.last_attitude.pitch_deg) < 25.0:
            return False
        return None

    async def send_speed(
        self,
        yaw_dps: float = 0.0,
        pitch_dps: float = 0.0,
        roll_dps: float = 0.0,
        duration_ds: int = 2,
    ) -> None:
        """Send angular speed in degrees/sec (verified OM3 CmdSet=0x04 CmdId=0x0C)."""
        frame = make_gimbal_speed_packet(
            yaw_speed=int(max(-1800, min(1800, round(yaw_dps * 10)))),
            pitch_speed=int(max(-1800, min(1800, round(pitch_dps * 10)))),
            roll_speed=int(max(-1800, min(1800, round(roll_dps * 10)))),
            duration_ds=duration_ds,
            enable=True,
            seq=self._next_seq(),
            profile=self.profile,
        )
        await self.send_raw(frame)

    async def send_relative(
        self,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        duration_ds: int = 5,
    ) -> None:
        """Rotate relative to current angle (verified OM3 CmdSet=0x04 CmdId=0x14)."""
        frame = make_gimbal_rotate_packet(
            yaw=int(round(yaw_deg * 10)),
            pitch=int(round(pitch_deg * 10)),
            roll=int(round(roll_deg * 10)),
            duration_ds=duration_ds,
            mode=ROTATION_MODE_RELATIVE,
            seq=self._next_seq(),
            profile=self.profile,
        )
        await self.send_raw(frame)

    async def send_absolute(
        self,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        duration_ds: int = 8,
    ) -> None:
        """Rotate to absolute angle (verified OM3 CmdSet=0x04 CmdId=0x14)."""
        frame = make_gimbal_rotate_packet(
            yaw=int(round(yaw_deg * 10)),
            pitch=int(round(pitch_deg * 10)),
            roll=int(round(roll_deg * 10)),
            duration_ds=duration_ds,
            mode=ROTATION_MODE_ABSOLUTE,
            seq=self._next_seq(),
            profile=self.profile,
        )
        await self.send_raw(frame)
