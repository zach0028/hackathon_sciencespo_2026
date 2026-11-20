# hackathon_osmo

Compagnon robotique de bureau temps réel synchronisant un **Google Pixel 11 Pro** (caméra + anneau LED RGB HiLight), un **MacBook Pro (Apple Silicon)** (vision ANE, contrôle moteur, voix & LLM) et une nacelle **DJI Osmo Mobile 3** (BLE DUML v1).

## Architecture globale

```text
┌──────────────────────────────┐        ADB Wi-Fi (sans câble)        ┌─────────────────────────────────────────────┐
│    Google Pixel 11 Pro       │ ─── H.264 (scrcpy-server -> PyAV) ─► │            MacBook Pro (Python 3.12)        │
│       (Android 17)           │                                      │                                             │
│  • Caméra avant / arrière    │ ◄── stdin -> HiLightBridge.jar ───── │  1. Vision   Apple Vision ANE (~10–14 ms)   │
│  • Anneau 8 LEDs RGB HiLight │        (LightsManager, 30 fps)       │  2. Contrôle One-Euro + PID + Gestes + FSM  │
└──────────────▲───────────────┘                                      │  3. Cerveau  Voxtral 3 STT/TTS + OpenRouter │
               │ bouge physiquement                                   │  4. Studio   FastAPI + WS vidéo/télémétrie  │
┌──────────────┴───────────────┐         BLE GATT (fff4/fff5)         └──────────────────────┬──────────────────────┘
│     DJI Osmo Mobile 3        │ ◄── Paquets DUML v1 (21 B, CRC) ────────────────────────────┘
│        (OM3-4008VF)          │ ─── Quaternions IMU temps réel ───►
│   [USB-C = charge 5V seul]   │
└──────────────────────────────┘
```

## Organisation du dépôt

```text
hackathon_osmo/
├── android/hilight/              # Micro-daemon Java DEX (3,5 Ko) pour les 8 LEDs RGB Android 17
│   ├── HiLightBridge.java
│   ├── build.sh
│   └── hilight.jar
├── src/hackathon_osmo/
│   ├── paths.py                  # Chemins centralisés (.env, calibration.json, hilight.jar, static)
│   ├── hardware/                 # Pilotes bas niveau & protocoles matériels
│   │   ├── duml.py               #   Encodeur/décodeur binaire DJI DUML v1 (CRC-8/CRC-16)
│   │   ├── osmo_ble.py           #   Client BLE CoreBluetooth/Bleak (vitesse, recentrage, IMU)
│   │   ├── pixel_camera.py       #   Capture caméra Pixel sans fil via scrcpy-server + PyAV
│   │   └── pixel_hilight.py      #   Animations LED RGB (réactions expressives & états ambiants)
│   ├── vision/                   # Perception temps réel sur Apple Neural Engine
│   │   ├── face.py               #   Détection de visages (VNDetectFaceRectanglesRequest)
│   │   └── sheet.py              #   Détection feuille écrite sur le bureau (Phase 2)
│   ├── control/                  # Boucle d'asservissement & calibration
│   │   ├── motion.py             #   One-Euro, PID, mapping invariant screen->motor, gestes, FSM
│   │   └── calibrate.py          #   Calibration automatique Jacobienne 2×2 (calibration.json)
│   ├── ai/                       # Pipeline conversationnelle multimodale
│   │   └── voice_brain.py        #   Mistral Voxtral 3 (STT/TTS) + OpenRouter Vision LLM + afplay
│   ├── app/                      # Orchestrateur temps réel & Studio Web
│   │   ├── runtime.py            #   StudioRuntime (boucles vision, moteur, voix, LEDs)
│   │   ├── server.py             #   Serveur FastAPI (REST + WebSockets /ws, /ws_video)
│   │   └── static/               #   UI Studio découpée (index.html, styles.css, studio.js, voice.js)
│   └── tools/                    # Utilitaires CLI de diagnostic
│       ├── probe.py              #   Scan BLE OM3 + sniffing DUML
│       └── sync_test.py          #   Test rapide de synchronisation Pixel + Osmo
├── tests/                        # Tests unitaires (DUML, MotionArbiter, VoiceBrain)
├── calibration.json              # Calibration Jacobienne active (paysage)
└── pyproject.toml                # Dépendances & points d'entrée CLI (uv)
```

## Prérequis & configuration

- **macOS (Apple Silicon)** avec `uv`, `adb` et `scrcpy` (`brew install android-platform-tools scrcpy`).
- **Variables d'environnement** dans `.env` (ignoré par Git) :
  - `MISTRAL_API_KEY` — STT (`voxtral-mini-transcribe-3-0`) et TTS (`voxtral-mini-tts-3-0`)
  - `OPENROUTER_API_KEY` — Cerveau Vision LLM (`openai/gpt-6.2-sol`, `google/gemini-2.5-flash`)
- **Pixel 11 Pro** en ADB sans fil (débrancher le câble USB-C une fois appairé pour éviter la mise en veille surcharge des moteurs OM3).

## Commandes

```bash
# Installer / synchroniser l'environnement (Python 3.12)
uv sync

# Lancer le Studio Web (http://127.0.0.1:8765)
uv run osmo-app

# Recalibrer automatiquement le mapping caméra <-> moteurs
uv run osmo-calibrate

# Outils de diagnostic BLE / synchronisation
uv run osmo-probe --duration 5
uv run osmo-sync-test

# Lancer la suite de tests unitaires
uv run pytest -v
```

### Recompiler le binaire HiLight Android (si `HiLightBridge.java` est modifié)

```bash
./android/hilight/build.sh
```
