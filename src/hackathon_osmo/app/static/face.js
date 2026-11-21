/* Osmo Adaptive Desk Companion — 120Hz OLED Canvas + HiLight Mirror + Touch Gestures */

(() => {
  const canvas = document.getElementById("face-canvas");
  const ctx = canvas.getContext("2d");

  const elConn = document.getElementById("conn-dot");
  const elStatePill = document.getElementById("state-pill");
  const elOptPill = document.getElementById("opt-pill");
  const elClockMini = document.getElementById("clock-mini");
  const elFocusLayer = document.getElementById("focus-layer");
  const elFocusClock = document.getElementById("focus-clock");
  const elCaptionWrap = document.getElementById("caption-wrap");
  const elCaptionText = document.getElementById("caption-text");
  const elDeskLayer = document.getElementById("desk-layer");
  const elDeskLed = document.getElementById("desk-led-dot");
  const elDeskStatus = document.getElementById("desk-status-label");
  const elDeskOptics = document.getElementById("desk-optics");
  const elDeskReply = document.getElementById("desk-reply");
  const elDeskScanBar = document.getElementById("desk-scan-bar");
  const elToast = document.getElementById("toast");
  const elDemoDock = document.getElementById("demo-dock");
  const btnDemoToggle = document.getElementById("btn-demo-toggle");
  const btnFs = document.getElementById("btn-fs");

  // --- Wake Lock (keeps Pixel OLED awake during desk sessions) ---
  let wakeLock = null;
  async function requestWakeLock() {
    try {
      if ("wakeLock" in navigator && !wakeLock) {
        wakeLock = await navigator.wakeLock.request("screen");
        wakeLock.addEventListener("release", () => { wakeLock = null; });
      }
    } catch (_) {}
  }
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") requestWakeLock();
  });

  function haptic(pattern) {
    try {
      if (navigator.vibrate) navigator.vibrate(pattern);
    } catch (_) {}
  }

  let toastTimer = null;
  function showToast(msg) {
    elToast.textContent = msg;
    elToast.classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => elToast.classList.add("hidden"), 1800);
  }

  // --- Color palette synchronized with rear 8-LED HiLight ring ---
  const MOOD_COLORS = {
    idle: [220, 236, 255],
    listen: [56, 189, 248],     // cyan
    think: [251, 191, 36],      // amber
    speak: [125, 211, 252],     // warm sky
    yes: [52, 211, 153],        // emerald
    no: [248, 113, 113],        // coral red
    curious: [251, 191, 36],    // amber
    search: [34, 211, 238],     // bright cyan
    desk: [253, 224, 150],      // warm reading white
    desk_locked: [52, 211, 153] // sheet locked emerald
  };

  // --- Live state mirrored from StudioRuntime /ws ---
  let ws = null;
  let wsConnected = false;
  let demoMode = new URLSearchParams(location.search).get("demo");
  let manualDnd = false;
  let lastActivityAt = performance.now();
  let lastReplyText = "";
  let lastReplyAt = 0;

  const live = {
    mode: "dialogue", // "focus" | "dialogue" | "desk"
    mood: "idle",     // "idle" | "listen" | "think" | "speak" | "yes" | "no" | "curious" | "search" | "desk"
    gazeX: 0,         // -1..1 target eye gaze
    gazeY: 0,
    speaking: false,
    busy: false,
    faceDetected: false,
    deskMode: false,
    sheetFound: false,
    zoom: 1.0,
    width: 1280,
    height: 720,
    hilightColor: null,
    reply: "",
    stateLabel: "PRÊT"
  };

  // --- Spring-physics eye state (smooth 120Hz motion) ---
  const anim = {
    gx: 0, gy: 0,
    openness: 1.0,
    smile: 0.0,
    frown: 0.0,
    asymmetry: 0.0,
    eyeScale: 1.0,
    r: 220, g: 236, b: 255,
    haloAlpha: 0.14,
    blinkTimer: 2.5,
    blinkPhase: 0, // 0 = open, >0 = blinking
    waveAmp: 0
  };

  // --- Clock updater ---
  function updateClock() {
    const now = new Date();
    const txt = now.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
    elClockMini.textContent = txt;
    elFocusClock.textContent = txt;
  }
  updateClock();
  setInterval(updateClock, 5000);

  // --- Apply backend StudioRuntime snapshot -> Adaptive Mode ---
  function applySnapshot(s) {
    if (demoMode) return;
    const cam = s.camera || {};
    const vis = s.vision || {};
    const voice = s.voice || {};
    const hi = s.hilight || {};

    const isDesk = Boolean(cam.facing === "back" || vis.state === "DESK_MODE");
    const sheetFound = Boolean(cam.sheet && cam.sheet.found);
    const activeGesture = vis.active_gesture || "";
    const speaking = Boolean(voice.speaking);
    const busy = Boolean(voice.busy);
    const liveOn = Boolean(voice.live_connected);
    const searching = vis.state === "SEARCHING";

    // Detect fresh assistant reply
    const topTurn = Array.isArray(voice.transcript) && voice.transcript.length > 0 ? voice.transcript[0] : null;
    const reply = topTurn && topTurn.reply ? String(topTurn.reply).trim() : "";
    if (reply && reply !== lastReplyText) {
      lastReplyText = reply;
      lastReplyAt = performance.now();
      lastActivityAt = performance.now();
      if (isDesk) haptic([14, 30, 14]);
    }

    if (speaking || busy || activeGesture || searching || isDesk || liveOn) {
      lastActivityAt = performance.now();
    }

    // Track HiLight rear LED color when available
    live.hilightColor = Array.isArray(hi.color) && hi.color.length === 3 ? hi.color : null;
    live.speaking = speaking;
    live.busy = busy;
    live.faceDetected = Boolean(vis.face_detected);
    live.deskMode = isDesk;
    live.sheetFound = sheetFound;
    live.zoom = Number(cam.zoom || 1.0);
    live.width = Number(cam.width || 1280);
    live.height = Number(cam.height || 720);
    live.reply = reply;

    // Eye gaze follows Zach's face (cx, cy) or active search sweep
    if (searching) {
      const sweepDir = Math.sign(vis.cmd_yaw_dps || 0) || Math.sin(performance.now() * 0.004);
      live.gazeX = sweepDir * 0.85;
      live.gazeY = -0.05;
    } else if (busy) {
      live.gazeX = 0.35;
      live.gazeY = -0.35;
    } else if (vis.face_detected) {
      live.gazeX = Math.max(-0.9, Math.min(0.9, Number(vis.cx || 0) * 1.15));
      live.gazeY = Math.max(-0.7, Math.min(0.7, Number(vis.cy || 0) * 1.1));
    } else {
      live.gazeX = 0;
      live.gazeY = 0;
    }

    // Choose adaptive display mode: Desk HUD > Focus (DND or idle > 25s) > Dialogue Face
    const idleForMs = performance.now() - lastActivityAt;
    if (isDesk) {
      live.mode = "desk";
      live.mood = sheetFound ? "desk_locked" : "desk";
      live.stateLabel = busy ? "ANALYSE FEUILLE" : sheetFound ? "FEUILLE VERROUILLÉE" : "MODE BUREAU";
    } else if (manualDnd || (!liveOn && !speaking && !busy && !activeGesture && !searching && idleForMs > 25000)) {
      live.mode = "focus";
      live.mood = "idle";
      live.stateLabel = "MODE DISCRET";
    } else {
      live.mode = "dialogue";
      if (activeGesture === "nod_yes") {
        live.mood = "yes";
        live.stateLabel = "OUI";
      } else if (activeGesture === "shake_no") {
        live.mood = "no";
        live.stateLabel = "NON";
      } else if (activeGesture === "curious_tilt") {
        live.mood = "curious";
        live.stateLabel = "CURIEUX";
      } else if (searching) {
        live.mood = "search";
        live.stateLabel = "RECHERCHE VISAGE";
      } else if (busy) {
        live.mood = "think";
        live.stateLabel = "RÉFLEXION…";
      } else if (speaking) {
        live.mood = "speak";
        live.stateLabel = "OSMO PARLE";
      } else if (liveOn) {
        live.mood = "listen";
        live.stateLabel = vis.face_detected ? "À L'ÉCOUTE" : "EN ATTENTE";
      } else {
        live.mood = "idle";
        live.stateLabel = vis.face_detected ? "PRÉSENT" : "PRÊT";
      }
    }

    syncDomLayers();
  }

  function syncDomLayers() {
    document.body.dataset.mode = live.mode;
    elStatePill.textContent = live.stateLabel;

    // Optics pill
    if (live.deskMode) {
      const resTag = live.width >= 3800 ? "4K" : live.width >= 2500 ? "2.7K" : `${live.width}p`;
      elOptPill.textContent = `${live.zoom.toFixed(live.zoom % 1 ? 2 : 1)}× · ${resTag}`;
      elOptPill.classList.remove("hidden");
    } else {
      elOptPill.classList.add("hidden");
    }

    // Focus layer
    elFocusLayer.classList.toggle("hidden", live.mode !== "focus");

    // Desk contextual HUD
    const showDesk = live.mode === "desk";
    elDeskLayer.classList.toggle("hidden", !showDesk);
    if (showDesk) {
      elDeskLed.classList.toggle("locked", live.sheetFound && !live.busy);
      elDeskStatus.textContent = live.busy
        ? "ANALYSE DE LA FEUILLE…"
        : live.sheetFound
          ? "FEUILLE VERROUILLÉE"
          : "CAMÉRA BUREAU ACTIVE";
      elDeskOptics.textContent = `${live.zoom.toFixed(live.zoom % 1 ? 2 : 1)}× · ${live.width}×${live.height}`;
      elDeskReply.textContent = live.reply || "Pose ta feuille devant la caméra et pose-moi ta question.";
      elDeskScanBar.classList.toggle("hidden", !live.busy);
    }

    // Live caption in Dialogue mode (show while speaking or for 6s after new reply)
    const showCaption =
      live.mode === "dialogue" &&
      Boolean(live.reply) &&
      (live.speaking || performance.now() - lastReplyAt < 6500);
    elCaptionWrap.classList.toggle("hidden", !showCaption);
    if (showCaption) {
      elCaptionText.textContent = live.reply;
    }
  }

  // --- WebSocket bridge to StudioRuntime ---
  function sendWs(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }

  function connectWs() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws`);
    ws.onopen = () => {
      wsConnected = true;
      elConn.classList.add("ok");
    };
    ws.onmessage = (ev) => {
      try {
        applySnapshot(JSON.parse(ev.data));
      } catch (_) {}
    };
    ws.onclose = () => {
      wsConnected = false;
      elConn.classList.remove("ok");
      setTimeout(connectWs, 1500);
    };
  }
  connectWs();

  // --- Touch gestures on Pixel OLED surface ---
  let touchStartY = 0;
  let touchStartT = 0;
  let lastTapT = 0;

  document.addEventListener("pointerdown", (e) => {
    if (e.target.closest("#demo-dock") || e.target.closest("#btn-demo-toggle")) return;
    requestWakeLock();
    touchStartY = e.clientY;
    touchStartT = performance.now();
  });

  document.addEventListener("pointerup", (e) => {
    if (e.target.closest("#demo-dock") || e.target.closest("#btn-demo-toggle")) return;
    const dy = e.clientY - touchStartY;
    const dt = performance.now() - touchStartT;

    // Swipe down -> Quiet Focus / DND mode
    if (dy > 75 && dt < 500) {
      manualDnd = true;
      live.mode = "focus";
      live.mood = "idle";
      live.stateLabel = "MODE DISCRET";
      sendWs({ action: "hilight", reaction: "off" });
      haptic(20);
      showToast("Mode discret activé");
      syncDomLayers();
      return;
    }

    // Swipe up -> Wake & fullscreen
    if (dy < -75 && dt < 500) {
      manualDnd = false;
      lastActivityAt = performance.now();
      live.mode = live.deskMode ? "desk" : "dialogue";
      if (!document.fullscreenElement && document.documentElement.requestFullscreen) {
        document.documentElement.requestFullscreen().catch(() => {});
      }
      haptic([12, 25, 12]);
      showToast("Osmo réveillé");
      syncDomLayers();
      return;
    }

    // Tap / Double-tap
    if (Math.abs(dy) < 24 && dt < 320) {
      const now = performance.now();
      if (now - lastTapT < 340) {
        // Double tap -> Toggle Face <-> Desk Reading!
        lastTapT = 0;
        manualDnd = false;
        lastActivityAt = now;
        if (live.deskMode) {
          sendWs({ action: "return_face_mode" });
          haptic([16, 40, 22]);
          showToast("Retour vers ton visage");
        } else {
          sendWs({ action: "enter_desk_mode" });
          haptic([16, 40, 22]);
          showToast("Passage en lecture feuille");
        }
      } else {
        lastTapT = now;
        manualDnd = false;
        lastActivityAt = now;
        // Playful micro-blink + subtle haptic acknowledgement
        anim.blinkPhase = 0.01;
        haptic(12);
        if (live.mode === "focus") {
          live.mode = "dialogue";
          syncDomLayers();
        }
      }
    }
  });

  // --- Demo / Preview mode (accessible via ?demo=1 or bottom-right micro dot) ---
  function applyDemo(preset) {
    if (preset === "live") {
      demoMode = null;
      manualDnd = false;
      showToast("Mode Live connecté");
      return;
    }
    demoMode = preset;
    manualDnd = false;
    lastActivityAt = performance.now();
    live.deskMode = preset === "desk";
    live.sheetFound = preset === "desk";
    live.speaking = preset === "speak";
    live.busy = preset === "think";
    live.zoom = preset === "desk" ? 1.0 : 1.0;
    live.width = preset === "desk" ? 2688 : 1280;
    live.height = preset === "desk" ? 1512 : 720;
    live.hilightColor = null;

    if (preset === "focus") {
      live.mode = "focus";
      live.mood = "idle";
      live.stateLabel = "MODE DISCRET";
      live.reply = "";
    } else if (preset === "desk") {
      live.mode = "desk";
      live.mood = "desk_locked";
      live.stateLabel = "FEUILLE VERROUILLÉE";
      live.reply = "« What catch your attention? » → « What catches your attention? » · « become biger » → « gets bigger ».";
    } else {
      live.mode = "dialogue";
      live.mood = preset;
      const labels = {
        listen: "À L'ÉCOUTE",
        think: "RÉFLEXION…",
        speak: "OSMO PARLE",
        yes: "OUI",
        no: "NON",
        curious: "CURIEUX",
        search: "RECHERCHE VISAGE"
      };
      live.stateLabel = labels[preset] || "PRÊT";
      live.gazeX = preset === "think" ? 0.35 : preset === "curious" ? -0.2 : 0;
      live.gazeY = preset === "think" ? -0.35 : 0;
      live.reply = preset === "speak" ? "Je vois bien ton schéma rouge avec Solution 1 et Solution 2." : "";
      lastReplyAt = performance.now();
    }
    syncDomLayers();
  }

  btnDemoToggle.addEventListener("click", () => {
    elDemoDock.classList.toggle("hidden");
  });

  elDemoDock.querySelectorAll("button[data-demo]").forEach((btn) => {
    btn.addEventListener("click", () => {
      elDemoDock.querySelectorAll("button[data-demo]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      applyDemo(btn.dataset.demo);
    });
  });

  btnFs.addEventListener("click", () => {
    requestWakeLock();
    if (!document.fullscreenElement && document.documentElement.requestFullscreen) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else if (document.exitFullscreen) {
      document.exitFullscreen().catch(() => {});
    }
  });

  if (demoMode) {
    elDemoDock.classList.remove("hidden");
    applyDemo(demoMode === "1" ? "listen" : demoMode);
  }

  // --- Canvas resize (Retina / Pixel 11 Pro high-DPI crispness) ---
  let W = 0, H = 0, DPR = 1;
  function resize() {
    DPR = Math.min(window.devicePixelRatio || 1, 3);
    W = window.innerWidth;
    H = window.innerHeight;
    canvas.width = Math.round(W * DPR);
    canvas.height = Math.round(H * DPR);
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }
  window.addEventListener("resize", resize);
  resize();

  // --- Helper: rounded pill eye ---
  function drawRoundedRect(x, y, w, h, r) {
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  // --- 120Hz Render Loop ---
  let lastT = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - lastT) / 1000);
    lastT = now;
    const t = now * 0.001;

    // 1. Target mood parameters
    const targetColor =
      (live.hilightColor && (live.hilightColor[0] + live.hilightColor[1] + live.hilightColor[2] > 25))
        ? live.hilightColor
        : (MOOD_COLORS[live.mood] || MOOD_COLORS.idle);

    const k = Math.min(1, dt * 8.5);
    anim.r += (targetColor[0] - anim.r) * k;
    anim.g += (targetColor[1] - anim.g) * k;
    anim.b += (targetColor[2] - anim.b) * k;

    const targetHalo = live.mode === "focus" ? 0.04 : live.speaking ? 0.24 : live.busy ? 0.22 : 0.14;
    anim.haloAlpha += (targetHalo - anim.haloAlpha) * k;

    // Update CSS edge halo mirroring HiLight
    const rootStyle = document.documentElement.style;
    rootStyle.setProperty("--halo-r", Math.round(anim.r));
    rootStyle.setProperty("--halo-g", Math.round(anim.g));
    rootStyle.setProperty("--halo-b", Math.round(anim.b));
    rootStyle.setProperty("--halo-alpha", anim.haloAlpha.toFixed(3));

    // Smooth gaze
    const targetGx = live.mood === "search" ? Math.sin(t * 3.1) * 0.85 : live.gazeX;
    const targetGy = live.gazeY + Math.sin(t * 1.1) * 0.025;
    anim.gx += (targetGx - anim.gx) * Math.min(1, dt * 10);
    anim.gy += (targetGy - anim.gy) * Math.min(1, dt * 10);

    // Expressive morph targets
    const targetSmile = live.mood === "yes" ? 1.0 : live.mood === "speak" ? 0.22 : 0.0;
    const targetFrown = live.mood === "no" ? 1.0 : 0.0;
    const targetAsym = live.mood === "curious" ? 1.0 : 0.0;
    const targetWave = live.speaking ? 1.0 : live.mood === "listen" ? 0.16 : 0.0;
    anim.smile += (targetSmile - anim.smile) * k;
    anim.frown += (targetFrown - anim.frown) * k;
    anim.asymmetry += (targetAsym - anim.asymmetry) * k;
    anim.waveAmp += (targetWave - anim.waveAmp) * Math.min(1, dt * 11);

    // Autonomous natural blinking
    anim.blinkTimer -= dt;
    if (anim.blinkTimer <= 0 && anim.blinkPhase === 0) {
      anim.blinkPhase = 0.001;
      anim.blinkTimer = 2.8 + Math.random() * 3.4;
    }
    let blinkFactor = 1.0;
    if (anim.blinkPhase > 0) {
      anim.blinkPhase += dt * 7.2;
      if (anim.blinkPhase >= 1.0) {
        anim.blinkPhase = 0;
      } else {
        blinkFactor = Math.max(0.08, Math.abs(Math.cos(anim.blinkPhase * Math.PI)));
      }
    }

    // Clear pure OLED black
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, W, H);

    // Mode A (Focus): just a tiny quiet breathing dot beneath clock
    if (live.mode === "focus") {
      const pulse = 0.35 + 0.25 * Math.sin(t * 0.9);
      ctx.beginPath();
      ctx.arc(W * 0.5, H * 0.72, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(244, 247, 251, ${(pulse * 0.45).toFixed(3)})`;
      ctx.fill();
      requestAnimationFrame(frame);
      return;
    }

    // Mode C (Desk HUD): subtle background HiLight glow only (card carries the legibility)
    if (live.mode === "desk") {
      requestAnimationFrame(frame);
      return;
    }

    // Mode B (Dialogue): Expressive Alive Eyes + Living Voice Wave
    const minDim = Math.min(W, H);
    const eyeW = minDim * 0.21;
    const eyeHBase = minDim * 0.28;
    const spacing = minDim * 0.22;
    const baseY = H * 0.44;
    const shiftX = anim.gx * (minDim * 0.13);
    const shiftY = anim.gy * (minDim * 0.09);

    const rgbStr = `${Math.round(anim.r)}, ${Math.round(anim.g)}, ${Math.round(anim.b)}`;
    const eyeColor = live.mood === "idle" ? "#eef6ff" : `rgb(${rgbStr})`;

    // Soft glow behind eyes
    ctx.save();
    ctx.shadowColor = `rgba(${rgbStr}, 0.55)`;
    ctx.shadowBlur = minDim * 0.08;

    [-1, 1].forEach((side) => {
      const isRight = side === 1;
      const scaleMod = isRight ? 1.0 + anim.asymmetry * 0.16 : 1.0 - anim.asymmetry * 0.08;
      const breath = 1.0 + Math.sin(t * 2.1) * 0.018;
      const ew = eyeW * scaleMod;
      const eh = Math.max(8, eyeHBase * scaleMod * breath * blinkFactor);
      const cx = W * 0.5 + side * spacing + shiftX;
      const cy = baseY + shiftY + (isRight ? -anim.asymmetry * 10 : anim.asymmetry * 6);

      ctx.save();
      ctx.translate(cx, cy);

      if (anim.smile > 0.55) {
        // Happy curved arc eye (^^) for nod_yes
        ctx.strokeStyle = eyeColor;
        ctx.lineWidth = Math.max(10, minDim * 0.036);
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.arc(0, eh * 0.18, ew * 0.52, Math.PI * 1.15, Math.PI * 1.85);
        ctx.stroke();
      } else {
        // Sleek OLED geometric pill eye
        ctx.fillStyle = eyeColor;
        drawRoundedRect(-ew / 2, -eh / 2, ew, eh, ew * 0.42);
        ctx.fill();

        // Subtle inner iris highlight for depth
        if (eh > 26) {
          ctx.fillStyle = "rgba(255, 255, 255, 0.32)";
          ctx.beginPath();
          ctx.arc(ew * 0.16 + anim.gx * 5, -eh * 0.22 + anim.gy * 4, ew * 0.13, 0, Math.PI * 2);
          ctx.fill();
        }

        // Skeptical / "no" top eyelid slant
        if (anim.frown > 0.08) {
          ctx.fillStyle = "#000000";
          ctx.beginPath();
          const cut = eh * 0.44 * anim.frown;
          if (side === -1) {
            ctx.moveTo(-ew * 0.6, -eh * 0.6);
            ctx.lineTo(ew * 0.6, -eh * 0.6);
            ctx.lineTo(ew * 0.6, -eh * 0.5 + cut);
            ctx.lineTo(-ew * 0.6, -eh * 0.5 - cut * 0.1);
          } else {
            ctx.moveTo(-ew * 0.6, -eh * 0.6);
            ctx.lineTo(ew * 0.6, -eh * 0.6);
            ctx.lineTo(ew * 0.6, -eh * 0.5 - cut * 0.1);
            ctx.lineTo(-ew * 0.6, -eh * 0.5 + cut);
          }
          ctx.closePath();
          ctx.fill();
        }

        // Subtle bottom-cheek smile lift when speaking
        if (anim.smile > 0.08 && anim.smile <= 0.55) {
          ctx.fillStyle = "#000000";
          ctx.beginPath();
          ctx.arc(0, eh * 0.72, ew * 0.62, Math.PI * 1.1, Math.PI * 1.9);
          ctx.fill();
        }
      }

      ctx.restore();
    });

    // Thinking orbital arc around center when Brain is working
    if (live.busy) {
      ctx.save();
      ctx.strokeStyle = `rgba(${rgbStr}, 0.75)`;
      ctx.lineWidth = 3.5;
      ctx.lineCap = "round";
      const rOrbit = minDim * 0.055;
      const a0 = t * 4.2;
      ctx.beginPath();
      ctx.arc(W * 0.5, H * 0.76, rOrbit, a0, a0 + Math.PI * 1.25);
      ctx.stroke();
      ctx.restore();
    }

    // Living organic voice wave beneath eyes when listening / speaking
    if (anim.waveAmp > 0.03 && !live.busy) {
      const waveW = minDim * 0.54;
      const waveY = H * 0.74;
      const amp = minDim * 0.055 * anim.waveAmp;
      ctx.save();
      ctx.strokeStyle = `rgba(${rgbStr}, ${(0.35 + 0.55 * anim.waveAmp).toFixed(2)})`;
      ctx.lineWidth = Math.max(3, minDim * 0.011);
      ctx.lineCap = "round";
      ctx.beginPath();
      const steps = 48;
      for (let i = 0; i <= steps; i++) {
        const u = i / steps;
        const env = Math.sin(u * Math.PI);
        const x = W * 0.5 - waveW / 2 + u * waveW;
        const y =
          waveY +
          env *
            amp *
            (Math.sin(u * 11.0 - t * 9.5) * 0.62 +
              Math.sin(u * 19.0 + t * 6.2) * 0.38);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }

    ctx.restore();
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
})();
