"""ADB & scrcpy-server raw H.264 camera streaming for Google Pixel 11 Pro over USB-C.

Uses in-process PyAV (libavcodec) H.264 decoding for ultra-low latency (<20 ms, zero probe delay).
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np


@dataclass
class AdbDevice:
    serial: str
    state: str
    model: str | None = None
    product: str | None = None


def find_adb() -> str | None:
    return shutil.which("adb") or (
        "/Users/zacharieelbaz/Library/Android/sdk/platform-tools/adb"
        if Path("/Users/zacharieelbaz/Library/Android/sdk/platform-tools/adb").exists()
        else None
    )


def find_scrcpy() -> str | None:
    return shutil.which("scrcpy")


_SCRCPY_SERVER_CACHE: tuple[str, str] | None = None


def find_scrcpy_server() -> tuple[str, str]:
    """Return (path_to_scrcpy_server, version_string), cached across camera restarts."""
    global _SCRCPY_SERVER_CACHE
    if _SCRCPY_SERVER_CACHE is not None and Path(_SCRCPY_SERVER_CACHE[0]).exists():
        return _SCRCPY_SERVER_CACHE
    scrcpy = find_scrcpy()
    if not scrcpy:
        raise RuntimeError("scrcpy not found on PATH")
    res = subprocess.run([scrcpy, "--version"], capture_output=True, text=True, check=True)
    first_line = res.stdout.splitlines()[0]
    version = first_line.split()[1]
    resolved = Path(scrcpy).resolve()
    server_path = resolved.parent.parent / "share" / "scrcpy" / "scrcpy-server"
    if not server_path.exists():
        raise RuntimeError(f"scrcpy-server not found at {server_path}")
    _SCRCPY_SERVER_CACHE = (str(server_path), version)
    return _SCRCPY_SERVER_CACHE


def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def list_adb_devices() -> list[AdbDevice]:
    adb = find_adb()
    if not adb:
        return []
    res = subprocess.run([adb, "devices", "-l"], capture_output=True, text=True, check=False)
    devices: list[AdbDevice] = []
    for line in res.stdout.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        serial, state = parts[0], parts[1]
        meta = dict(p.split(":", 1) for p in parts[2:] if ":" in p)
        devices.append(
            AdbDevice(
                serial=serial,
                state=state,
                model=meta.get("model"),
                product=meta.get("product"),
            )
        )
    return devices


def list_pixel_cameras(serial: str | None = None) -> str:
    scrcpy = find_scrcpy()
    if not scrcpy:
        return "scrcpy not found"
    cmd = [scrcpy, "--list-cameras"]
    if serial:
        cmd.extend(["-s", serial])
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=15)
    return (res.stdout + "\n" + res.stderr).strip()


class PixelCameraStream:
    """Ultra-low-latency camera stream from Android via scrcpy-server raw H.264 -> PyAV -> numpy BGR."""

    # Avoid re-pushing scrcpy-server.jar + chmod on every camera/zoom switch (~350-500ms saved each time)
    _pushed_devices: set[tuple[str | None, str]] = set()
    _push_lock = threading.Lock()

    def __init__(
        self,
        serial: str | None = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        facing: str = "front",
        camera_id: str | None = None,
        zoom: float = 1.0,
        bitrate: int | None = None,
        port: int = 27183,
    ) -> None:
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.facing = facing
        self.camera_id = camera_id
        self.zoom = float(zoom)
        self.bitrate = bitrate
        self.port = port
        self._adb = find_adb()
        self._server_proc: subprocess.Popen[bytes] | None = None
        self._sock: socket.socket | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_ts: float = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._decode_thread: threading.Thread | None = None

    def _adb_cmd(self, *args: str) -> list[str]:
        assert self._adb is not None
        cmd = [self._adb]
        if self.serial:
            cmd.extend(["-s", self.serial])
        cmd.extend(args)
        return cmd

    def _ensure_server_pushed(self, server_path: str, version: str, *, force: bool = False) -> None:
        key = (self.serial, version)
        with self._push_lock:
            if not force and key in self._pushed_devices:
                return
            subprocess.run(
                self._adb_cmd("push", server_path, "/data/local/tmp/scrcpy-server.jar"),
                capture_output=True,
                check=True,
            )
            subprocess.run(
                self._adb_cmd("shell", "chmod", "444", "/data/local/tmp/scrcpy-server.jar"),
                capture_output=True,
                check=True,
            )
            self._pushed_devices.add(key)

    def start(self) -> None:
        if not self._adb:
            raise RuntimeError("adb must be installed.")
        server_path, version = find_scrcpy_server()
        was_cached = (self.serial, version) in self._pushed_devices

        for attempt in range(2):
            # 1. Push scrcpy-server once per device (must be read-only on Android 14+ W^X) & forward socket
            self._ensure_server_pushed(server_path, version, force=(attempt > 0))
            subprocess.run(
                self._adb_cmd("forward", f"tcp:{self.port}", "localabstract:scrcpy"),
                capture_output=True,
                check=True,
            )

            # 2. Launch scrcpy-server in raw H.264 camera streaming mode
            server_args = [
                "CLASSPATH=/data/local/tmp/scrcpy-server.jar",
                "app_process",
                "/",
                "com.genymobile.scrcpy.Server",
                version,
                "video_source=camera",
                f"camera_size={self.width}x{self.height}",
                f"camera_fps={self.fps}",
                "video_codec=h264",
                "audio=false",
                "control=false",
                "raw_stream=true",
                "tunnel_forward=true",
                "cleanup=false",
                "log_level=warn",
            ]
            if self.camera_id is not None:
                server_args.append(f"camera_id={self.camera_id}")
            else:
                server_args.append(f"camera_facing={self.facing}")
            if self.bitrate is not None and self.bitrate > 0:
                server_args.append(f"video_bit_rate={int(self.bitrate)}")
            # Apply native optical/hybrid zoom (Pixel 11 Pro back: 0.51..30, front: 0.9..10)
            z_min, z_max = (0.51, 10.0) if self.facing == "back" else (0.9, 4.0)
            clamped_zoom = max(z_min, min(z_max, self.zoom))
            if abs(clamped_zoom - 1.0) > 0.01 or self.facing == "back":
                server_args.append(f"camera_zoom={clamped_zoom:.2f}")

            self._server_proc = subprocess.Popen(
                self._adb_cmd("shell", " ".join(server_args)),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            # 3. Connect TCP socket (fast 30ms polling until scrcpy-server binds abstract socket)
            sock: socket.socket | None = None
            first = b""
            deadline = time.monotonic() + (4.5 if (was_cached and attempt == 0) else 8.0)
            while time.monotonic() < deadline:
                if self._server_proc.poll() is not None:
                    break
                try:
                    s = socket.create_connection(("127.0.0.1", self.port), timeout=0.6)
                    s.settimeout(1.5)
                    chunk = s.recv(4096)
                    if chunk:
                        sock = s
                        first = chunk
                        break
                    s.close()
                except OSError:
                    pass
                time.sleep(0.03)

            if sock is not None:
                self._sock = sock
                self._sock.settimeout(1.0)
                self._stop.clear()
                self._decode_thread = threading.Thread(
                    target=self._decode_loop, args=(first,), daemon=True
                )
                self._decode_thread.start()
                return

            err = ""
            if self._server_proc and self._server_proc.stderr:
                try:
                    err = self._server_proc.stderr.read().decode(errors="ignore")
                except Exception:
                    pass
            self.stop()
            if was_cached and attempt == 0:
                with self._push_lock:
                    self._pushed_devices.discard((self.serial, version))
                continue
            raise RuntimeError(f"Timed out waiting for scrcpy camera stream. {err}")

    def _decode_loop(self, initial_chunk: bytes) -> None:
        assert self._sock is not None
        codec = av.CodecContext.create("h264", "r")
        codec.thread_type = "AUTO"

        def feed(data: bytes) -> None:
            for pkt in codec.parse(data):
                try:
                    for frame in codec.decode(pkt):
                        bgr = frame.to_ndarray(format="bgr24")
                        with self._lock:
                            self._latest_frame = bgr
                            self._latest_ts = time.monotonic()
                except av.error.InvalidDataError:
                    continue

        feed(initial_chunk)
        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            feed(chunk)

    def read(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return self._latest_frame, self._latest_ts

    def frames(self, timeout: float = 10.0) -> Iterator[tuple[np.ndarray, float]]:
        deadline = time.monotonic() + timeout
        last_ts = 0.0
        while not self._stop.is_set():
            frame, ts = self.read()
            if frame is not None and ts != last_ts:
                last_ts = ts
                yield frame, ts
            else:
                if last_ts == 0.0 and time.monotonic() > deadline:
                    break
                time.sleep(0.004)

    def stop(self) -> None:
        self._stop.set()
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._server_proc and self._server_proc.poll() is None:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
        if self._adb:
            subprocess.run(
                self._adb_cmd("forward", "--remove", f"tcp:{self.port}"),
                capture_output=True,
                check=False,
            )

    def __enter__(self) -> PixelCameraStream:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
