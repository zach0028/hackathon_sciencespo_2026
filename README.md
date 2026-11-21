# Osmo — Embodied AI Desk Tutor

**Osmo** is a real-time embodied AI private tutor built for the **Sciences Po 2026 Hackathon**. It turns a motorized **DJI Osmo Mobile 3** gimbal, a **Google Pixel 11 Pro** (multi-lens rear camera, OLED whiteboard screen, and 8-LED RGB HiLight ring), and an **Apple Silicon Mac** into an attentive, low-latency desktop companion that reads handwritten notebooks, tracks where you point your finger, displays interactive exercises and definitions on its OLED screen, and speaks in calm, pedagogical English.

---

## Highlights

- **Full-Duplex Voice + Vision Brain** — OpenAI `gpt-live-1.1` over WebRTC (UDP/Opus with instant barge-in) paired via sideband WebSocket to a delegated multimodal brain (`gpt-6.2-sol` / `gpt-6.2-astra`) with 7 native hardware, vision, whiteboard, and memory tools.
- **Zero-Latency SSE Speculative Pre-Warm** — Parses OpenAI `/v1/responses` Server-Sent Events in real time: as soon as a tool intent (`look_at_desk`, `look_at_user`, or `show_on_screen` with `face_user=true` as its first JSON key) streams in (~50 ms), Osmo starts pivoting the gimbal and switching optics **in parallel** with LLM generation, eliminating 1.5–5 s of perceived latency per turn.
- **4K Handwritten Sheet Reading & 11 ms Finger-Pointing Macro** — Apple Neural Engine (`VNDetectDocumentSegmentationRequest` + bright-paper luminance filtering) locks onto notebooks while ignoring dark laptops, rotates upright, gates autofocus sharpness via Laplacian variance, splits open 2-page spreads into 4K HD tiles, and automatically attaches a macro close-up around Zach's fingertip via `VNDetectHumanHandPoseRequest` (~11 ms ANE).
- **Native Android OLED Companion (`Osmo Face`, 16 KB APK)** — Locked-landscape, edge-to-edge immersive fullscreen WebView kiosk app (`com.osmo.face`) displaying a 120 Hz analytical 3D CAD superellipsoid face coupled to live gimbal IMU & motor velocity, plus **6 proactive study whiteboards**:
  - `DEFINITION` (hero definition + formula / example / common trap)
  - `CONCEPT` (smart label pills for timelines, derivations, comparisons, essay outlines, vocab)
  - `EXERCISE` (`Given` strip + `Q1/Q2` cards + collapsible tappable `HINT ▸`)
  - `MINI-EXAM` (point-weighted questions `[2 pts]` + hidden `RUBRIC / HINT ▸`)
  - `CHECKLIST` (finger-tappable to-do list with live progress bar & celebratory HiLight pulse)
  - `FEEDBACK` (structured `✓ OK` / `△ FIX` / `→ NEXT` grading after inspecting written work)
- **Automatic Desk ↔ Face "Show & Tell" Loop** — Osmo proactively puts dense definitions, formulas, and exercises on its OLED screen, swings toward you to let you copy/solve, and tracks `came_from_desk=True` so a simple *"Got it, I noted it"* swings it right back to your worksheet in one hop (`read_now=false`, 0 Round-2 delay).
- **Organic Physical Micro-Expressions & Local Memory** — Real-time `< 0.1 ms` micro-expression engine triggers synchronized DJI OM3 nods/shakes/curious tilts, 8-LED RGB HiLight animations, and 3D CAD eye expressions as Osmo speaks — backed by a local SQLite + FTS5 persistent memory (`< 1 ms` recall) and a full Flight Recorder (`turns.jsonl`, `events.jsonl`, `imu_trace.jsonl`, captured frames).

---

## System Architecture

```text
┌───────────────────────────────────┐     Wireless ADB (no cable)     ┌───────────────────────────────────────────────┐
│        Google Pixel 11 Pro        │ ── H.264 (scrcpy -> PyAV) ────► │          MacBook Pro (Apple Silicon)          │
│           (Android 17)            │                                 │                 Python 3.12                   │
│                                   │ ◄── adb reverse :8765/face ──── │                                               │
│  • Native App: Osmo Face (16 KB)  │     (WS 120Hz CAD + 6 boards)   │  1. Vision   Apple Vision ANE (Face, Sheet,   │
│    Locked Landscape OLED Kiosk    │                                 │              Hand-Pose Finger Macro ~11 ms)   │
│  • Optics: Front 720p / Rear 4K   │ ◄── stdin -> hilight.jar ────── │  2. Control  One-Euro + PID + Gestures + FSM  │
│    (0.51x .. 5x optical zoom)     │     (LightsManager, 30 fps)     │              Quaternion Closed-Loop Pivots    │
│  • 8-LED RGB HiLight Ring         │                                 │  3. Brain    GPT-Live 1.1 (WebRTC+Sideband) + │
└─────────────────▲─────────────────┘                                 │              GPT-6.2 Sol/Astra (SSE Pre-Warm) │
                  │ physically moves                                  │  4. Memory   SQLite + FTS5 + Flight Recorder  │
┌─────────────────┴─────────────────┐       BLE GATT (fff4/fff5)      │  5. Studio   FastAPI Dashboard (:8765)        │
│        DJI Osmo Mobile 3          │ ◄── DUML v1 (21 B, CRC-8/16) ───┴───────────────────────┬───────────────────────┘
│           (OM3-4008VF)            │ ─── Live IMU Quaternions ───────────────────────────────┘
└───────────────────────────────────┘
```

### Fast Show & Tell Choreography (SSE Pre-Warm)

```text
  DESK READING (4K Rear Camera)
       │  "Give me the definition of Qmax to write down"
       ▼
  t + 50ms   SSE streams `{"face_user":true` ──▶ [=== Parallel 125°/s Pan + Pitch Lift + Front Cam (~1.3s) ===]
             (runs WHILE the brain finishes generating title, definition, formula & spoken_summary)
  t + 2.7s   OLED screen is ALREADY facing the student ──▶ Definition card snaps on screen (0 Round-2 delay!)
       │
       │  Student copies cleanly, then says: "Okay good, got it!"
       ▼
  t + 50ms   `came_from_desk=True` + `read_now=false` ──▶ [=== Parallel Pivot back to sheet (~1.8s, 0 Round-2) ===]
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
│   │   ├── duml.py                  # DJI DUML v1 binary encoder/decoder (CRC-8 / CRC-16)
│   │   ├── osmo_ble.py              # Bleak CoreBluetooth client (speed, recenter, IMU quaternion parser)
│   │   ├── pixel_camera.py          # Wireless Pixel camera via scrcpy-server + PyAV (push-cached, gen-synced)
│   │   └── pixel_hilight.py         # 8-LED RGB animations & reactive ring-light controller
│   ├── vision/
│   │   ├── face.py                  # Apple Vision ANE face detection (VNDetectFaceRectanglesRequest)
│   │   └── sheet.py                 # Document segmentation, bright-paper filter, 4K spread slicer, finger macro
│   ├── control/
│   │   ├── motion.py                # One-Euro filter, PID, rotation-invariant screen->motor, gestures, search FSM
│   │   └── calibrate.py             # Automatic 2x2 Jacobian camera-to-motor calibration (calibration.json)
│   ├── ai/
│   │   ├── live_brain.py            # GPT-Live 1.1 WebRTC + Sideband, SSE pre-warm, 7 strict tools, auto-expressions
│   │   └── memory.py                # Local SQLite + FTS5 persistent tutor memory & episode logger
│   ├── telemetry/
│   │   └── recorder.py              # Session Flight Recorder (turns, events, 10 Hz IMU trace, vision frames)
│   ├── app/
│   │   ├── runtime.py               # StudioRuntime (async vision, gimbal, HiLight, pre-warm & tool executor)
│   │   ├── server.py                # FastAPI server (REST, /ws, /ws_video, /face, /api/pixel/launch_face)
│   │   └── static/                  # Mac Studio dashboard (index.html/studio.js/voice.js) + Pixel OLED (face.*)
│   └── tools/
│       ├── probe.py                 # OM3 BLE scanner & DUML sniffer
│       └── sync_test.py             # Quick Pixel + Osmo synchronization test
├── tests/                           # 48 unit tests (DUML, MotionArbiter, LiveBrain, Memory, SheetVision, Recorder)
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
  > **Hardware tip:** Unplug the USB-C cable once Wireless ADB is active and balance the Pixel horizontally in the clamp with motors powered off — this prevents USB cable drag from tripping the OM3's motor overload protection.

### Configuration

Create a `.env` file at the repository root (gitignored):

```env
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
# Run full unit test suite (48 tests)
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

## Technical Notes & Empirical Findings

- **DJI Osmo Mobile 3 BLE (`CmdId 0x0C`)**: Packets must be chunked to `<= 20` bytes on characteristic `fff5`, speeds should stay within `<= 96 °/s` with `roll = 0` for reliable gesture execution, and idle `send_speed(0, 0)` traffic is silenced after 2 brake packets to keep the BLE link stable.
- **No Secrets Committed**: `.env`, `logs/`, `data/`, and Android `build/` / `*.keystore` are strictly gitignored.
