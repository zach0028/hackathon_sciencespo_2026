"""Async RGB controller for Google Pixel 11 Pro rear HiLight 8-LED ring over ADB.

Drives all 8 individual ARGB segments at up to 30 fps via Android 17 LightsManager.
Supports both transient expressive reactions (yes/no/curious/found) and ambient
robot states (centered/tracking/search/sheet_search/sheet_locked/idle).
"""

from __future__ import annotations

import asyncio
import math
import subprocess

from hackathon_osmo.hardware.pixel_camera import find_adb
from hackathon_osmo.paths import HILIGHT_JAR as JAR_LOCAL_PATH

JAR_REMOTE_PATH = "/data/local/tmp/hilight.jar"

TRANSIENT_REACTIONS = {"yes", "no", "curious", "found"}


def _scale_rgb(r: int, g: int, b: int, intensity: float) -> str:
    k = max(0.0, min(1.0, intensity))
    return f"{int(r * k):02X}{int(g * k):02X}{int(b * k):02X}"


class PixelHiLight:
    """Persistent stdin-driven controller for the Pixel 11 Pro 8-segment ARGB HiLight ring."""

    def __init__(self) -> None:
        self.serial: str | None = None
        self.connected: bool = False
        self.num_lights: int = 8
        self.current_effect: str = "off"
        self.current_color: str = "#000000"
        self._ambient_state: str = "off"
        self._in_transient: bool = False
        self._proc: asyncio.subprocess.Process | None = None
        self._anim_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def connect(self, serial: str) -> bool:
        adb = find_adb()
        if not adb or not JAR_LOCAL_PATH.exists():
            return False

        await self.disconnect()
        self.serial = serial

        await asyncio.to_thread(
            subprocess.run,
            [adb, "-s", serial, "push", str(JAR_LOCAL_PATH), JAR_REMOTE_PATH],
            capture_output=True,
            check=True,
        )
        await asyncio.to_thread(
            subprocess.run,
            [adb, "-s", serial, "shell", "chmod", "444", JAR_REMOTE_PATH],
            capture_output=True,
            check=True,
        )

        self._proc = await asyncio.create_subprocess_exec(
            adb,
            "-s",
            serial,
            "shell",
            f"CLASSPATH={JAR_REMOTE_PATH} app_process / HiLightBridge",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert self._proc.stdout is not None
        try:
            ready_line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=5.0)
        except asyncio.TimeoutError:
            await self.disconnect()
            return False

        text = ready_line.decode(errors="ignore").strip()
        if not text.startswith("READY"):
            await self.disconnect()
            return False

        try:
            self.num_lights = int(text.split("lights=")[1].split()[0])
        except Exception:
            self.num_lights = 8

        for _ in range(self.num_lights):
            try:
                await asyncio.wait_for(self._proc.stdout.readline(), timeout=0.5)
            except asyncio.TimeoutError:
                break

        self.connected = True
        return True

    async def disconnect(self) -> None:
        self._cancel_anim()
        self.connected = False
        if self._proc is not None:
            try:
                if self._proc.stdin and not self._proc.stdin.is_closing():
                    self._proc.stdin.write(b"OFF\nQUIT\n")
                    await self._proc.stdin.drain()
            except Exception:
                pass
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=1.5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def _cancel_anim(self) -> None:
        if self._anim_task and not self._anim_task.done():
            self._anim_task.cancel()
        self._anim_task = None
        self._in_transient = False

    async def _send_cmd(self, cmd: str) -> bool:
        async with self._lock:
            if not self.connected or self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
                return False
            if self._proc.returncode is not None:
                self.connected = False
                return False
            try:
                self._proc.stdin.write((cmd + "\n").encode("ascii"))
                await self._proc.stdin.drain()
                resp = await asyncio.wait_for(self._proc.stdout.readline(), timeout=1.0)
                return bool(resp)
            except Exception:
                self.connected = False
                return False

    async def set_off(self) -> None:
        self._cancel_anim()
        self._ambient_state = "off"
        self.current_effect = "off"
        self.current_color = "#000000"
        await self._send_cmd("OFF")

    async def set_solid(self, hex_rgb: str, label: str = "solid") -> None:
        clean = hex_rgb.lstrip("#").upper()
        self.current_effect = label
        self.current_color = f"#{clean}"
        await self._send_cmd(f"SOLID {clean}")

    def set_ambient(self, state: str) -> None:
        """Set continuous robot state animation (no-op if unchanged or transient playing)."""
        if state == self._ambient_state and not self._in_transient:
            return
        self._ambient_state = state
        if self._in_transient:
            return
        self._launch(state, transient=False)

    def start_reaction(self, reaction: str) -> None:
        """Trigger an immediate reaction (transient reactions override ambient and resume it after)."""
        is_trans = reaction in TRANSIENT_REACTIONS
        if not is_trans:
            self._ambient_state = reaction
        self._launch(reaction, transient=is_trans)

    def _launch(self, name: str, transient: bool) -> None:
        self._cancel_anim()
        self._in_transient = transient

        coros = {
            "yes": self._anim_yes,
            "no": self._anim_no,
            "curious": self._anim_curious,
            "found": self._anim_found,
            "search": self._anim_search,
            "centered": self._anim_centered,
            "tracking": self._anim_tracking,
            "sheet_search": self._anim_sheet_search,
            "sheet_locked": self._anim_sheet_locked,
            "desk": self._anim_sheet_locked,
            "idle": self._anim_idle,
            "off": self._anim_off,
        }
        fn = coros.get(name, self._anim_off)

        async def _runner() -> None:
            try:
                await fn()
            except asyncio.CancelledError:
                return
            if transient:
                self._in_transient = False
                # Smoothly restore ambient robot state after transient gesture finishes
                if self._ambient_state and self._ambient_state not in TRANSIENT_REACTIONS:
                    self._launch(self._ambient_state, transient=False)

        self._anim_task = asyncio.create_task(_runner())

    # --- Transient expressive reactions ---

    async def _anim_yes(self) -> None:
        """Bright emerald GREEN pulses synchronized with the YES nod (~1.5s)."""
        self.current_effect = "yes_green"
        self.current_color = "#00FF44"
        for _ in range(3):
            for step in range(8):
                k = 0.25 + 0.75 * math.sin(step / 7.0 * math.pi)
                await self._send_cmd(f"SOLID {_scale_rgb(0, 255, 68, k)}")
                await asyncio.sleep(0.055)
        await self._send_cmd("SOLID 00FF44")
        await asyncio.sleep(0.22)

    async def _anim_no(self) -> None:
        """Sharp vivid RED alternating sweep synchronized with the NO head shake (~1.5s)."""
        self.current_effect = "no_red"
        self.current_color = "#FF1212"
        left_red = "SEG FF1212 FF1212 FF1212 FF1212 180000 180000 180000 180000"
        right_red = "SEG 180000 180000 180000 180000 FF1212 FF1212 FF1212 FF1212"
        for _ in range(3):
            await self._send_cmd(left_red)
            await asyncio.sleep(0.18)
            await self._send_cmd("SOLID FF1212")
            await asyncio.sleep(0.07)
            await self._send_cmd(right_red)
            await asyncio.sleep(0.18)
        await self._send_cmd("SOLID FF1212")
        await asyncio.sleep(0.22)

    async def _anim_curious(self) -> None:
        """Warm AMBER + VIOLET rotating thinking wave (~1.4s)."""
        self.current_effect = "curious_amber"
        self.current_color = "#FFB820"
        n = max(1, self.num_lights)
        for tick in range(24):
            segs = []
            for i in range(n):
                phase = (i - tick) % n
                if phase == 0:
                    segs.append("FFB820")
                elif phase in (1, n - 1):
                    segs.append("663A00")
                elif phase == n // 2:
                    segs.append("9D4BFF")
                else:
                    segs.append("120A24")
            await self._send_cmd("SEG " + " ".join(segs))
            await asyncio.sleep(0.06)

    async def _anim_found(self) -> None:
        """Quick bright emerald confirmation flash when face is re-acquired."""
        self.current_effect = "found_green"
        self.current_color = "#00FF66"
        await self._send_cmd("SOLID 00EE55")
        await asyncio.sleep(0.32)

    # --- Ambient continuous robot states ---

    async def _anim_centered(self) -> None:
        """Calm, alive soft TEAL/CYAN breathing when locked & centered on your face."""
        self.current_effect = "centered_calm"
        self.current_color = "#10C8B8"
        step = 0
        while True:
            k = 0.14 + 0.14 * (0.5 + 0.5 * math.sin(step * 0.18))
            await self._send_cmd(f"SOLID {_scale_rgb(16, 220, 195, k)}")
            step += 1
            await asyncio.sleep(0.09)

    async def _anim_tracking(self) -> None:
        """Crisp SKY BLUE / CYAN glow while actively following your movement."""
        self.current_effect = "tracking_cyan"
        self.current_color = "#20B8FF"
        await self._send_cmd("SOLID 1478CC")
        while True:
            await asyncio.sleep(1.0)

    async def _anim_search(self) -> None:
        """Rotating CYAN radar comet while sweeping the room for your face."""
        self.current_effect = "search_comet"
        self.current_color = "#00D0FF"
        n = max(1, self.num_lights)
        tick = 0
        while True:
            segs = []
            for i in range(n):
                d = (i - tick) % n
                if d == 0:
                    segs.append("00E0FF")
                elif d == 1:
                    segs.append("005588")
                elif d == 2:
                    segs.append("001830")
                else:
                    segs.append("000000")
            await self._send_cmd("SEG " + " ".join(segs))
            tick = (tick + 1) % n
            await asyncio.sleep(0.075)

    async def _anim_sheet_search(self) -> None:
        """Warm AMBER pulse while tilting down to scan the desk for a written sheet."""
        self.current_effect = "sheet_scan"
        self.current_color = "#FFAA22"
        step = 0
        while True:
            k = 0.20 + 0.55 * (0.5 + 0.5 * math.sin(step * 0.35))
            await self._send_cmd(f"SOLID {_scale_rgb(255, 175, 35, k)}")
            step += 1
            await asyncio.sleep(0.07)

    async def _anim_sheet_locked(self) -> None:
        """Bright warm WHITE ring-light illuminating the sheet on the desk."""
        self.current_effect = "sheet_ringlight"
        self.current_color = "#FFF5E0"
        await self._send_cmd("SOLID FFF5E0")
        while True:
            await asyncio.sleep(1.0)

    async def _anim_idle(self) -> None:
        """Subtle peaceful INDIGO standby halo when centered at home."""
        self.current_effect = "idle_indigo"
        self.current_color = "#3038AA"
        await self._send_cmd("SOLID 161C55")
        while True:
            await asyncio.sleep(1.0)

    async def _anim_off(self) -> None:
        self.current_effect = "off"
        self.current_color = "#000000"
        await self._send_cmd("OFF")
