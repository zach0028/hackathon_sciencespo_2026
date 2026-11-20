"""CLI probe to inspect live DUML telemetry from DJI Osmo Mobile over BLE."""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import Counter

from hackathon_osmo.hardware.duml import DumlPacket
from hackathon_osmo.hardware.osmo_ble import OsmoBleClient, discover_osmo


async def run_listen(duration: float, address: str | None) -> None:
    target: object
    if address is None:
        print("Looking for DJI Osmo Mobile over BLE...")
        dev = await discover_osmo(timeout=5.0)
        if dev is None:
            print("No DJI Osmo Mobile found. Make sure it is powered on and DJI Mimo is closed.")
            return
        target = dev
        label = f"{dev.name} ({dev.address})"
        print(f"Found: {label}")
    else:
        target = address
        label = address

    counts: Counter[tuple[int, int, int, int, str, bool]] = Counter()
    samples: dict[tuple[int, int, int, int], bytes] = {}
    raw_chunks = 0

    def on_packet(pkt: DumlPacket) -> None:
        key = (pkt.sender, pkt.receiver, pkt.cmd_set, pkt.cmd_id, pkt.crc_profile, pkt.crc16_ok)
        counts[key] += 1
        short_key = (pkt.sender, pkt.receiver, pkt.cmd_set, pkt.cmd_id)
        if short_key not in samples:
            samples[short_key] = pkt.payload

    client = OsmoBleClient(target, on_packet=on_packet)  # type: ignore[arg-type]

    orig_handle = client._handle_notify

    def wrapped_handle(sender: object, data: bytearray) -> None:
        nonlocal raw_chunks
        raw_chunks += 1
        if raw_chunks <= 5:
            print(f"  [raw #{raw_chunks}] ({len(data)}B): {bytes(data).hex()}")
        orig_handle(sender, data)

    client._handle_notify = wrapped_handle  # type: ignore[method-assign]

    print(f"Connecting to {label} and listening on fff4 for {duration:.1f}s...")
    async with client:
        print("Connected! Listening...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration:
            await asyncio.sleep(0.2)
            if client.last_attitude is not None:
                att = client.last_attitude
                print(
                    f"\r  Attitude: pitch={att.pitch_deg:+6.1f}°  roll={att.roll_deg:+6.1f}°  yaw={att.yaw_deg:+6.1f}°   ",
                    end="",
                    flush=True,
                )
        print()

    print(f"\nReceived {raw_chunks} raw BLE notifications, {sum(counts.values())} parsed DUML packets:")
    for (snd, rcv, cset, cid, prof, ok16), n in counts.most_common():
        sample = samples[(snd, rcv, cset, cid)].hex()
        print(
            f"  src=0x{snd:02x} -> dst=0x{rcv:02x} | cmdSet=0x{cset:02x} cmdId=0x{cid:02x} "
            f"| crc={prof} (crc16_ok={ok16}) | count={n:3d} | payload={sample[:48]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe DJI Osmo Mobile BLE telemetry")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds to listen")
    parser.add_argument("--address", type=str, default=None, help="Explicit BLE UUID/address")
    args = parser.parse_args()
    asyncio.run(run_listen(duration=args.duration, address=args.address))


if __name__ == "__main__":
    main()
