# Osmo — Embodied AI Desk Tutor

**Osmo** is a real-time embodied AI private tutor built for the **Sciences Po 2026 Hackathon**. It turns a motorized **DJI Osmo Mobile 3** gimbal, a **Google Pixel 11 Pro** (multi-lens 4K camera, native OLED whiteboard app, and 8-LED RGB HiLight ring), and an **Apple Silicon Mac** into an attentive, low-latency desktop companion that reads handwritten notebooks, tracks where you point your finger, displays interactive exercises and definitions on its OLED screen, and teaches in **Full English or Full French** via a 1-click toggle.

---

## Highlights

- **Dual-Provider Real-Time Voice & Vision Engine (`AI : GEMINI` ↔ `AI : OPENAI`)** — Switch providers in 1 click from the Studio header:
  - **Google Gemini (Default)** — **Gemini 3.8 Live** (`gemini-3.8-live`) full-duplex WebSocket audio (16 kHz PCM in / 24 kHz PCM out via AEC-coupled `<audio>`, instant barge-in flush, context window compression, and transparent **Session Resumption** across `goAway` rollovers) with **direct asynchronous `NON_BLOCKING` hardware tools** (`SILENT` for gestures/memory, `WHEN_IDLE` for vision/whiteboards) + **Gemini 4 Hydrogen** (`gemini-4-hydrogen`, 2× faster 4K OCR) / **Gemini 4 Neon / Argon / Carbon** for razor-sharp 4K sheet reading and text prompts.
  - **OpenAI (1-Click Fallback)** — `gpt-live-1.1` over WebRTC (UDP/Opus with instant barge-in) paired via sideband WebSocket to a delegated multimodal brain (`gpt-6.2-sol` / `gpt-6.2-astra`) with **SSE Speculative Pre-Warm** (~50 ms).
- **1-Click Full Bilingual Mode (`LANG : EN` ↔ `LANG : FR`)** — Switches voice instructions, 4K reasoning brain, OLED study cards, and bilingual micro-expressions cleanly across both providers.
- **4K Handwritten Sheet Reading & 11 ms Finger-Pointing Macro** — Apple Neural Engine (`VNDetectDocumentSegmentationRequest` + bright-paper luminance filtering) locks onto notebooks while ignoring dark laptops, rotates upright, gates autofocus sharpness via Laplacian variance, splits open 2-page spreads into 4K HD tiles, and automatically attaches a macro close-up around the student's fingertip via `VNDetectHumanHandPoseRequest` (~11 ms ANE).
- **Native Android OLED Companion (`Osmo Face`, 16 KB APK)** — Locked-landscape, edge-to-edge immersive fullscreen WebView kiosk app (`com.osmo.face`) displaying a 120 Hz analytical 3D CAD superellipsoid face coupled to live gimbal IMU & motor velocity, plus **6 proactive study whiteboards**:
  - `DEFINITION` (hero definition + formula / example / common trap)
  - `CONCEPT` (smart label pills for timelines, derivations, comparisons, essay outlines, vocab)
  - `EXERCISE` (`Given` strip + `Q1/Q2` cards + collapsible tappable `HINT ▸`)
  - `MINI-EXAM` (point-weighted questions `[2 pts]` + hidden `RUBRIC / HINT ▸`)
  - `CHECKLIST` (finger-tappable to-do list with live progress bar & celebratory HiLight pulse)
  - `FEEDBACK` (structured `✓ OK` / `△ FIX` / `→ NEXT` grading after inspecting written work)
- **Automatic Desk ↔ Face "Show & Tell" Loop** — Osmo proactively puts dense definitions, formulas, and exercises on its OLED screen, swings toward you to let you copy/solve, and tracks `came_from_desk=True` so a simple *"Got it, I noted it"* swings it right back to your worksheet in one hop (`read_now=false`, 0 Round-2 delay).
- **Organic Micro-Expressions, Local Memory & Hardware Resilience (`MotorWatchdog`)**:
  - Real-time `< 0.1 ms` micro-expression engine triggers synchronized DJI OM3 nods/shakes/curious tilts (with 2.5 s anti-double-trigger cooldown), 8-LED RGB HiLight animations, and 3D CAD eye expressions — backed by SQLite + FTS5 persistent memory (`< 1 ms` recall) and a full Flight Recorder (`turns.jsonl`, `events.jsonl`, `imu_trace.jsonl`, captured frames).
  - **Anti-Sleep & Auto-Heal**: DUML `CmdSet=0x04 CmdId=0x0D` (`0x2AB5` Resume) + dual-tier BLE keepalive (`6 s` speed / `45 s` resume), 1-click `WAKE MOTORS`, IMU-based mechanical end-stop stall relief, silent CoreBluetooth freeze reconnect (`> 4.5 s`), Wireless ADB Hotspot auto-reconnect, `scrcpy` Camera2 HAL zombie cleanup, and 4K → 2.7K automatic encoder fallback.

---

## System Architecture

```text
┌───────────────────────────────────┐     Wireless ADB (no cable)     ┌───────────────────────────────────────────────┐
│        Google Pixel 11 Pro        │ ── H.264 (scrcpy -> PyAV) ────► │          MacBook Pro (Apple Silicon)          │
│           (Android 17)            │                                 │                 Python 3.12                   │
│                                   │ ◄── adb reverse :8765/face ──── │                                               │
│  • Native App: Osmo Face (16 KB)  │     (WS 120Hz CAD + 6 boards)   │  1. Vision   Apple Vision ANE (Face, Sheet,   │
│    Locked Landscape OLED Kiosk    │                                 │              Hand-Pose Finger Macro ~11 ms)   │
│  • Optics: Front 720p / Rear 4K   │ ◄── stdin -> hilight.jar ────── │  2. Control  One-Euro + PID + MotorWatchdog   │
│    (0.51x .. 5x optical zoom)     │     (LightsManager, 30 fps)     │              Quaternion Closed-Loop Pivots    │
│  • 8-LED RGB HiLight Ring         │                                 │  3. AI Dual  [GEMINI] 3.8 Live (NON_BLOCKING) │
└─────────────────▲─────────────────┘                                 │                      + 4 Hydrogen / Neon 4K   │
                  │ physically moves                                  │              [OPENAI] GPT-Live 1.1 + 6.2 Sol  │
┌─────────────────┴─────────────────┐       BLE GATT (fff4/fff5)      │  4. Memory   SQLite + FTS5 + Flight Recorder  │
│        DJI Osmo Mobile 3          │ ◄── DUML v1 (0x0C/0x0D/0x4C) ───┴───────────────────────┬───────────────────────┘
│         (Bluetooth LE)            │ ─── Live IMU Quaternions (0x05) ────────────────────────┘
└───────────────────────────────────┘
```

### Dual-Provider Voice & Tool Pipelines

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ GEMINI MODE (Default ⚡) — Direct NON_BLOCKING Live Tools + 4K Reader      │
│                                                                            │
│  Browser Mic (16 kHz PCM) ──► /ws/gemini_live ──► gemini-3.8-live          │
│                                                    ├── 7 tools NON_BLOCKING│
│                                                    │   (SILENT / WHEN_IDLE)│
│                                                    └── 4K desk OCR ──►     │
│                                                        gemini-4-hydrogen   │
│  Browser Audio (24 kHz PCM + AEC + Barge-In) ◄── sessionResumption auto    │
├────────────────────────────────────────────────────────────────────────────┤
│ OPENAI MODE (1-Click Fallback) — WebRTC + SSE Speculative Pre-Warm (~50ms) │
│                                                                            │
│  Browser WebRTC ──► gpt-live-1.1 ──► Sideband ──► gpt-6.2-sol (SSE tools)  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```text
hackathon_osmo/
├── android/
│   ├── hilight/                     # 3.5 KB DEX micro-daemon driving Android 17 LightsManager (8 RGB LEDs)
│   │   ├── HiLightBridge.java
│   │   ├── build.sh
│   │   └── hilight.jar
│   └── osmo_face/                   # 16 KB native locked-landscape fullscreen WebView kiosk app (com.osmo.face)
│       ├── AndroidManifest.xml
│       ├── build.sh                 # 2-second Gradle-free build (aapt2 + javac + d8 + zipalign + apksigner)
│       ├── res/values/styles.xml
│       └── src/com/osmo/face/MainActivity.java
├── src/hackathon_osmo/
│   ├── paths.py                     # Centralized paths (.env, calibration.json, data/, logs/, static/)
│   ├── hardware/
│   │   ├── duml.py                  # DJI DUML v1 encoder/decoder (CRC-8/16, 0x0C speed, 0x0D resume/suspend, 0x4C)
│   │   ├── osmo_ble.py              # Bleak CoreBluetooth client (IMU quaternions, wake sequence, freeze watchdog)
│   │   ├── pixel_camera.py          # Wireless Pixel camera via scrcpy + PyAV (auto ADB reconnect, zombie cleanup)
│   │   └── pixel_hilight.py         # 8-LED RGB animations & reactive ring-light controller
│   ├── vision/
│   │   ├── face.py                  # Apple Vision ANE face detection (VNDetectFaceRectanglesRequest)
│   │   └── sheet.py                 # Document segmentation, bright-paper filter, 4K spread slicer, finger macro
│   ├── control/
│   │   ├── motion.py                # One-Euro filter, PID, rotation-invariant screen->motor, gestures, search FSM
│   │   ├── watchdog.py              # MotorWatchdog (end-stop stall relief, anti-sleep keepalive, limp detection)
│   │   └── calibrate.py             # Automatic 2x2 Jacobian camera-to-motor calibration (calibration.json)
│   ├── ai/
│   │   ├── gemini_brain.py          # Gemini 3.8 Live WS bridge (NON_BLOCKING + sessionResumption) + Gemini 4 4K brain
│   │   ├── live_brain.py            # OpenAI GPT-Live 1.1 WebRTC + Sideband, SSE pre-warm, 7 strict tools
│   │   └── memory.py                # Local SQLite + FTS5 persistent tutor memory & episode logger
│   ├── telemetry/
│   │   └── recorder.py              # Session Flight Recorder (turns, events, 10 Hz IMU trace, vision frames)
│   ├── app/
│   │   ├── runtime.py               # StudioRuntime (dual AI provider, vision, gimbal, HiLight, watchdog & tools)
│   │   ├── server.py                # FastAPI server (REST, /ws, /ws_video, /ws/gemini_live, /face)
│   │   └── static/                  # Mac Studio dashboard (index.html/studio.js/voice.js) + Pixel OLED (face.*)
│   └── tools/
│       ├── probe.py                 # OM3 BLE scanner & DUML sniffer
│       └── sync_test.py             # Quick Pixel + Osmo synchronization test
├── tests/                           # 56 unit tests (DUML, Motion, Watchdog, GeminiBrain, LiveBrain, Memory, Sheet)
├── calibration.json                 # Active landscape Jacobian + calibrated face/desk IMU poses
└── pyproject.toml                   # Python 3.12 dependencies & CLI entry points (uv)
```

---

## Requirements & Setup

### Prerequisites
- **macOS (Apple Silicon)** — uses Apple Vision Framework (`pyobjc-framework-vision`) on the Neural Engine.
- **Python 3.12** managed with [`uv`](https://docs.astral.sh/uv/).
- **Android Platform Tools & scrcpy**:
  ```bash
  brew install android-platform-tools scrcpy
  ```
- **Hardware**:
  - **DJI Osmo Mobile 3** (Bluetooth LE)
  - **Google Pixel** (tested on Pixel 11 Pro, Android 17) paired over **Wireless ADB** (`adb tcpip 5555 && adb connect <pixel-ip>:5555`).
  > **Hardware balance tip:** The DJI Osmo Mobile 3 payload cap is `200 g ± 30 g`. Remove thick phone cases, unplug the USB-C cable once Wireless ADB is active, and slide the Pixel in the clamp (bringing the heavy rear camera bar closer to the clamp center) until it balances level with motors powered off. This prevents static motor overheating and hardware protection cutouts.

### Configuration

Create a `.env` file at the repository root (strictly gitignored). Set either or both keys — Osmo defaults to **Gemini** when `GEMINI_API_KEY` is present and lets you switch in 1 click in the Studio header:

```env
GEMINI_API_KEY=...
OPENAI_API_KEY=sk-...
```

### Install & Run

```bash
# 1. Install dependencies
uv sync

# 2. Launch Osmo Studio (Mac dashboard: http://127.0.0.1:8765)
uv run osmo-app

# 3. (Optional) Build & install the native fullscreen landscape Android app on the Pixel
./android/osmo_face/build.sh
adb install -r android/osmo_face/build/OsmoFace.apk

# 4. Launch the OLED Companion directly on the Pixel (uses Osmo Face if installed, Chrome fallback)
curl -X POST http://127.0.0.1:8765/api/pixel/launch_face
```

---

## CLI Tools & Testing

```bash
# Run full unit test suite (56 tests)
uv run pytest -v

# Recalibrate 2x2 Jacobian camera <-> motor mapping
uv run osmo-calibrate

# Diagnostic utilities (BLE probe & sync check)
uv run osmo-probe --duration 5
uv run osmo-sync-test

# Rebuild Android DEX HiLight bridge (only if HiLightBridge.java is modified)
./android/hilight/build.sh
```

### Previewing the Pixel OLED UI Without Hardware

Open `http://127.0.0.1:8765/face?demo=1` in any browser to inspect the 120 Hz 3D CAD superellipsoid head, organic micro-states, and all 6 interactive study boards (`DEF`, `CONCEPT`, `EXERCISE`, `EXAM`, `FEEDBACK`, `TO-DO`).

---

## Technical Notes & Security

- **DJI Osmo Mobile 3 BLE Protocol (`CmdSet=0x04`)**:
  - GATT writes on `fff5` are chunked to `<= 20` bytes; `CmdId=0x0C` speed packets stay within `<= 96 °/s` with `roll = 0` (`> 100 °/s` is silently dropped by the OM3 firmware).
  - `CmdId=0x0D` (`0x2AB5` LE) sends the DUML Resume/Exit-Standby heartbeat (`WAKE MOTORS` + 45 s idle keepalive) alongside a quiet 6 s `send_speed(0, 0)` keepalive.
- **Public Repository Security**:
  - Studio binds exclusively to `127.0.0.1` (loopback only).
  - Both `GEMINI_API_KEY` and `OPENAI_API_KEY` remain 100% server-side in Python (the browser connects only to local `/ws/gemini_live` or `/api/live/session`).
  - `.env`, `logs/`, `data/`, `*.apk`, and `*.keystore` are strictly excluded via `.gitignore`.
