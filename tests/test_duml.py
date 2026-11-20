"""Unit tests for the DJI DUML v1 binary protocol (CRC-8/CRC-16, OM3 wire order, stream parser)."""

from __future__ import annotations

import struct

from hackathon_osmo.hardware.duml import (
    CRC_CLASSIC,
    CRC_MODERN,
    ROTATION_MODE_ABSOLUTE,
    ROTATION_MODE_RELATIVE,
    ROTATION_MODE_SPEED,
    DumlPacket,
    DumlStreamParser,
    crc8,
    crc16,
    make_gimbal_rotate_packet,
    make_gimbal_speed_packet,
)


def test_om3_speed_packet_header_and_length() -> None:
    raw = make_gimbal_speed_packet(yaw_speed=200, pitch_speed=-150, roll_speed=0, duration_ds=2, seq=1)
    assert len(raw) == 21
    # Verified OM3 header: SOF=0x55, ver=1 + len=21 (0x0415 little-endian -> 15 04), CRC8=0xa9
    assert raw[:4] == bytes([0x55, 0x15, 0x04, 0xA9])
    assert crc8(raw[:3], init=CRC_CLASSIC.crc8_init) == 0xA9
    expected_crc16 = struct.unpack_from("<H", raw, 19)[0]
    assert crc16(raw[:-2], init=CRC_CLASSIC.crc16_init) == expected_crc16


def test_om3_wire_order_is_pitch_roll_yaw() -> None:
    raw = make_gimbal_rotate_packet(
        yaw=300,
        pitch=-120,
        roll=45,
        duration_ds=5,
        mode=ROTATION_MODE_SPEED,
        seq=7,
    )
    parser = DumlStreamParser()
    packets = list(parser.feed(raw))
    assert len(packets) == 1
    pkt = packets[0]
    assert pkt.crc8_ok and pkt.crc16_ok
    assert pkt.crc_profile == "classic"
    assert pkt.cmd_set == 0x04
    assert pkt.cmd_id == 0x0C
    assert pkt.cmd_type == 0x00
    assert pkt.seq == 7
    pitch, roll, yaw, mode, duration_ds = struct.unpack("<hhhBB", pkt.payload)
    assert (pitch, roll, yaw, mode, duration_ds) == (-120, 45, 300, ROTATION_MODE_SPEED, 5)


def test_relative_and_absolute_modes_use_cmd_id_0x14() -> None:
    rel = make_gimbal_rotate_packet(yaw=100, mode=ROTATION_MODE_RELATIVE, seq=2)
    abs_pkt = make_gimbal_rotate_packet(pitch=50, mode=ROTATION_MODE_ABSOLUTE, seq=3)
    parser = DumlStreamParser()
    pkts = list(parser.feed(rel + abs_pkt))
    assert len(pkts) == 2
    assert pkts[0].cmd_id == 0x14
    assert pkts[1].cmd_id == 0x14
    assert pkts[0].crc16_ok and pkts[1].crc16_ok


def test_speed_packet_disable_zeroes_axes() -> None:
    raw = make_gimbal_speed_packet(yaw_speed=500, pitch_speed=-400, roll_speed=100, enable=False)
    pkt = list(DumlStreamParser().feed(raw))[0]
    pitch, roll, yaw, mode, _ = struct.unpack("<hhhBB", pkt.payload)
    assert (pitch, roll, yaw, mode) == (0, 0, 0, ROTATION_MODE_SPEED)


def test_stream_parser_handles_garbage_prefix_and_ble_chunking() -> None:
    p1 = make_gimbal_speed_packet(yaw_speed=100, seq=10)
    p2 = DumlPacket(
        sender=0x04,
        receiver=0x02,
        seq=11,
        cmd_type=0x00,
        cmd_set=0x04,
        cmd_id=0x4C,
        payload=b"\x01\x01",
    ).encode(profile=CRC_MODERN)

    # Prefix with noise containing a false 0x55 start byte, then feed byte-by-byte / 7-byte chunks
    wire = b"\x00\x55\xff\x01\x99" + p1 + b"\xaa\xbb" + p2
    parser = DumlStreamParser()
    out: list[DumlPacket] = []
    for i in range(0, len(wire), 7):
        out.extend(parser.feed(wire[i : i + 7]))

    assert len(out) == 2
    assert out[0].seq == 10 and out[0].crc_profile == "classic" and out[0].crc16_ok
    assert out[1].seq == 11 and out[1].crc_profile == "modern" and out[1].crc16_ok
    assert out[1].payload == b"\x01\x01"
