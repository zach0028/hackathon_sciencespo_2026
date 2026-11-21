/* Osmo 3D Wireframe Superellipsoid Head · Hatched Vector-Node Eyes · marked.dev / CAD Craft */

(() => {
  const canvas = document.getElementById("face-canvas");
  const ctx = canvas.getContext("2d");

  const elConn = document.getElementById("conn-dot");
  const elStatePill = document.getElementById("state-pill");
  const elPoseTelemetry = document.getElementById("pose-telemetry");
  const elOptPill = document.getElementById("opt-pill");
  const elClockMini = document.getElementById("clock-mini");
  const elFocusLayer = document.getElementById("focus-layer");
  const elFocusClock = document.getElementById("focus-clock");

  // Clean Desk Reading Strip
  const elDeskStrip = document.getElementById("desk-strip");
  const elDeskLed = document.getElementById("desk-led-dot");
  const elDeskStatus = document.getElementById("desk-status-label");
  const elDeskOptics = document.getElementById("desk-optics");
  const elDeskHiLight = document.getElementById("desk-hilight-label");
  const elDeskScanBar = document.getElementById("desk-scan-bar");
  const btnDeskToFace = document.getElementById("btn-desk-to-face");

  // Full-Bleed Interactive OLED Study Board (Exercise / Checklist / Concept)
  const elBoardLayer = document.getElementById("board-layer");
  const elBoardCard = document.getElementById("board-card");
  const elBoardModeBadge = document.getElementById("board-mode-badge");
  const elBoardTitle = document.getElementById("board-title");
  const elBoardProgressPill = document.getElementById("board-progress-pill");
  const elBoardProgressTrack = document.getElementById("board-progress-track");
  const elBoardProgressFill = document.getElementById("board-progress-fill");
  const elBoardGiven = document.getElementById("board-given");
  const elBoardGivenText = document.getElementById("board-given-text");
  const elBoardList = document.getElementById("board-list");
  const elBoardHintWrap = document.getElementById("board-hint-wrap");
  const btnToggleHint = document.getElementById("btn-toggle-hint");
  const elBoardHintBox = document.getElementById("board-hint-box");
  const elBoardNote = document.getElementById("board-note");
  const elBoardFootGuide = document.getElementById("board-foot-guide");
  const btnBoardPivot = document.getElementById("btn-board-pivot");
  const btnClearCard = document.getElementById("btn-clear-card");
  const btnCloseBoard = document.getElementById("btn-close-board");

  const elToast = document.getElementById("toast");
  const elDemoDock = document.getElementById("demo-dock");
  const btnDemoToggle = document.getElementById("btn-demo-toggle");
  const btnFs = document.getElementById("btn-fs");

  // --- Screen Wake Lock + Landscape Orientation Lock + Haptics ---
  let wakeLock = null;
  function lockLandscape() {
    try {
      if (screen.orientation && screen.orientation.lock) {
        screen.orientation.lock("landscape").catch(() => {});
      }
    } catch (_) {}
  }
  async function requestWakeLock() {
    lockLandscape();
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
  lockLandscape();

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
    toastTimer = setTimeout(() => elToast.classList.add("hidden"), 1700);
  }

  // --- Surgical monochrome-first accent colors (mirrors rear HiLight 8-LED ring on bottom edge rim) ---
  const RIM_COLORS = {
    idle: [96, 165, 250],        // crisp blueprint steel blue
    listen: [96, 165, 250],      // blueprint blue
    attentive: [56, 189, 248],   // bright attentive cyan (user speaking!)
    barge_in: [224, 242, 254],   // crisp flash on interruption
    think: [245, 158, 11],       // technical amber (thinking delta!)
    intent_desk: [251, 191, 36], // warm amber pre-action anticipation
    focusing: [167, 139, 250],   // crisp violet AF squint
    reading: [251, 191, 36],     // warm reading saccades
    formulating: [96, 165, 250], // eager pre-speech return
    speak: [125, 211, 252],      // ice cyan syllable oscilloscope
    greet: [52, 211, 153],       // warm emerald arrival greet
    drowsy: [148, 163, 184],     // soft slate drowsy transition
    yes: [52, 211, 153],         // emerald
    no: [248, 113, 113],         // coral
    curious: [251, 191, 36],     // warm amber
    search: [56, 189, 248],      // radar cyan
    desk: [251, 191, 36],        // warm amber scan
    desk_locked: [52, 211, 153]  // locked emerald
  };

  // --- Live state mirrored from StudioRuntime /ws ---
  let ws = null;
  let knownUiVersion = null;
  let lastBoardSig = "";
  let lastCardUpdatedAt = 0;
  let hintRevealed = false;
  let demoMode = new URLSearchParams(location.search).get("demo");
  let manualDnd = false;
  let lastActivityAt = performance.now();
  let lastReplyText = "";
  let lastReplyAt = 0;

  // Interactive pointer/hover gaze (marked.dev micro-interaction!)
  const pointer = {
    active: false,
    x: 0, // -1..1
    y: 0, // -1..1
    lastMoveAt: 0
  };

  const live = {
    mode: "dialogue", // "focus" | "dialogue" | "desk"
    mood: "listen",
    gazeX: 0,
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
    hilightEffect: "off",
    reply: "",
    stateLabel: "READY",
    // Real-time mechanical arm synchronization (DJI Osmo Mobile 3 IMU + motor velocity)
    armConnected: false,
    armPivoting: false,
    armPanDps: 0,
    armTiltDps: 0,
    armRelPanDeg: 0,
    armRelPitchDeg: 0,
    // Zero-latency organic micro-states (SSE thinking/tool/speech deltas + optical AF + face arrival)
    microPhase: "idle",
    toolIntent: "",
    opticalPhase: "",
    lastPulseSeq: 0,
    pulseEnergy: 0,      // 0..1 decays smoothly; kicked by every real SSE/transcript delta!
    thinkCursor: 0,      // advances along the 18 eye CAD nodes on each thinking delta!
    bargeInUntil: 0,
    greetUntil: 0,
    faceLostAt: 0,
    drowsyLevel: 0,      // 0..1 smooth lid droop between 14s and 25s idle
    // Interactive OLED checklist / concept card pushed via show_on_screen tool
    screenCard: { mode: "none", title: "", items: [], note: "" }
  };

  function wrapDeg(deg) {
    let d = ((deg + 180) % 360 + 360) % 360 - 180;
    return d;
  }

  // --- 2nd-order Spring-Damper Physics for ultra-organic 3D head motion ---
  function makeSpring(val = 0, stiffness = 95, damping = 13) {
    return { v: val, vel: 0, k: stiffness, d: damping };
  }
  function stepSpring(s, target, dt) {
    const force = (target - s.v) * s.k - s.vel * s.d;
    s.vel += force * dt;
    s.v += s.vel * dt;
    return s.v;
  }

  const sp = {
    yaw: makeSpring(-0.34, 72, 11.5),    // rad (natural 3/4 sculptural angle like reference!)
    pitch: makeSpring(-0.14, 72, 11.5),  // rad (slight upward dome tilt showing 3D volume)
    roll: makeSpring(-0.04, 65, 11.0),   // rad (expressive head cock)
    scale: makeSpring(1.0, 68, 12.0),    // head size (shrinks smoothly in Desk mode!)
    centerX: makeSpring(0.5, 62, 12.0),  // normalized screen X (moves left in Desk mode!)
    centerY: makeSpring(0.5, 62, 12.0),
    eyeOpen: makeSpring(1.0, 140, 16.0),
    smile: makeSpring(0.0, 95, 14.0),
    frown: makeSpring(0.0, 95, 14.0),
    asym: makeSpring(0.0, 90, 13.0),
    wave: makeSpring(0.0, 110, 15.0)
  };

  const anim = {
    r: 96, g: 165, b: 250,
    haloAlpha: 0.08,
    wireAlpha: 0.72,
    blinkTimer: 2.6,
    blinkPhase: 0,
    gestureStart: 0
  };

  // --- Clock ---
  function updateClock() {
    const txt = new Date().toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
    elClockMini.textContent = txt;
    elFocusClock.textContent = txt;
  }
  updateClock();
  setInterval(updateClock, 5000);

  // --- Apply StudioRuntime snapshot ---
  function applySnapshot(s) {
    // Auto-reload Pixel OLED immediately whenever backend UI assets are updated!
    if (s && s.ui_version) {
      if (knownUiVersion === null) {
        knownUiVersion = s.ui_version;
      } else if (s.ui_version !== knownUiVersion) {
        location.reload();
        return;
      }
    }
    if (demoMode) return;
    const cam = s.camera || {};
    const vis = s.vision || {};
    const voice = s.voice || {};
    const hi = s.hilight || {};
    const osmo = s.osmo || {};

    const isDesk = Boolean(cam.facing === "back" || vis.state === "DESK_MODE");
    const sheetFound = Boolean(cam.sheet && cam.sheet.found);
    const activeGesture = vis.active_gesture || "";
    const speaking = Boolean(voice.speaking);
    const busy = Boolean(voice.busy);
    const liveOn = Boolean(voice.live_connected);
    const searching = vis.state === "SEARCHING";

    // Zero-latency micro-state deltas from OpenAI streaming & local optical AF
    const micro = voice.micro || {};
    const nowMs = performance.now();
    live.microPhase = String(micro.phase || "idle");
    live.toolIntent = String(micro.tool_intent || "");
    live.opticalPhase = String(cam.optical_phase || "");

    const seq = Number(micro.pulse_seq || 0);
    if (seq !== live.lastPulseSeq) {
      live.lastPulseSeq = seq;
      live.pulseEnergy = 1.0;
      live.thinkCursor = (live.thinkCursor + 1) % 18;
      lastActivityAt = nowMs;
    }
    if (micro.barge_in_age_ms !== null && micro.barge_in_age_ms !== undefined && micro.barge_in_age_ms < 420) {
      if (nowMs > live.bargeInUntil) {
        live.bargeInUntil = nowMs + 420;
        anim.blinkPhase = 0.001;
        sp.pitch.vel -= 2.4; // crisp natural micro-recoil on interruption!
      }
    }

    // Local 0ms arrival greet: when user returns in front of camera after > 8s absence
    const wasFace = live.faceDetected;
    const nowFace = Boolean(vis.face_detected);
    if (!nowFace && wasFace) {
      live.faceLostAt = nowMs;
    } else if (nowFace && !wasFace && live.faceLostAt > 0 && (nowMs - live.faceLostAt) > 8000 && !isDesk) {
      live.greetUntil = nowMs + 950;
      anim.gestureStart = nowMs;
      lastActivityAt = nowMs;
    }

    // Real-time mechanical arm coupling (IMU attitude delta + live motor velocity)
    live.armConnected = Boolean(osmo.connected);
    live.armPivoting = Boolean(osmo.pivoting);
    const bodyCmd = osmo.body_cmd || {};
    live.armPanDps = Number(bodyCmd.pan_dps ?? vis.cmd_yaw_dps ?? 0);
    live.armTiltDps = Number(bodyCmd.tilt_dps ?? vis.cmd_pitch_dps ?? 0);
    if (osmo.attitude && osmo.face_pose) {
      const refPan = Number(osmo.face_pose.pan_roll_deg ?? 179.3);
      const refPitch = Number(osmo.face_pose.pitch_deg ?? 5.3);
      live.armRelPanDeg = wrapDeg(Number(osmo.attitude.roll ?? refPan) - refPan);
      live.armRelPitchDeg = Number(osmo.attitude.pitch ?? refPitch) - refPitch;
    } else {
      live.armRelPanDeg = 0;
      live.armRelPitchDeg = 0;
    }

    const topTurn = Array.isArray(voice.transcript) && voice.transcript.length > 0 ? voice.transcript[0] : null;
    const reply = topTurn && topTurn.reply ? String(topTurn.reply).trim() : "";
    if (reply && reply !== lastReplyText) {
      lastReplyText = reply;
      lastReplyAt = performance.now();
      lastActivityAt = performance.now();
      if (isDesk) haptic([12, 28, 12]);
    }

    if (speaking || busy || activeGesture || searching || isDesk || liveOn || live.armPivoting || Math.abs(live.armPanDps) > 4 || Math.abs(live.armTiltDps) > 4) {
      lastActivityAt = performance.now();
    }

    // Parse HiLight color whether hex string ("#34d399") or [r,g,b] tuple
    if (typeof hi.color === "string" && /^#[0-9a-fA-F]{6}$/.test(hi.color)) {
      live.hilightColor = [
        parseInt(hi.color.slice(1, 3), 16),
        parseInt(hi.color.slice(3, 5), 16),
        parseInt(hi.color.slice(5, 7), 16)
      ];
    } else if (Array.isArray(hi.color) && hi.color.length === 3) {
      live.hilightColor = hi.color;
    } else {
      live.hilightColor = null;
    }
    live.hilightEffect = String(hi.effect || "off");
    live.speaking = speaking;
    live.busy = busy;
    live.faceDetected = Boolean(vis.face_detected);
    live.deskMode = isDesk;
    live.sheetFound = sheetFound;
    live.zoom = Number(cam.zoom || 1.0);
    live.width = Number(cam.width || 1280);
    live.height = Number(cam.height || 720);
    live.reply = reply;
    if (s.screen && typeof s.screen === "object") {
      live.screenCard = s.screen;
    }

    if (searching) {
      const dir = Math.sign(vis.cmd_yaw_dps || 0) || Math.sin(performance.now() * 0.003);
      live.gazeX = dir * 0.9;
      live.gazeY = 0;
    } else if (vis.face_detected) {
      live.gazeX = Math.max(-1, Math.min(1, Number(vis.cx || 0) * 1.15));
      live.gazeY = Math.max(-1, Math.min(1, Number(vis.cy || 0) * 1.15));
    } else {
      live.gazeX = 0;
      live.gazeY = 0;
    }

    const idleForMs = performance.now() - lastActivityAt;
    const drowsy = !manualDnd && !liveOn && !speaking && !busy && !isDesk && !searching && idleForMs > 14000 && idleForMs <= 25000;
    live.drowsyLevel = drowsy ? Math.min(1, (idleForMs - 14000) / 11000) : 0;

    const hasScreenCard = live.screenCard && live.screenCard.mode && live.screenCard.mode !== "none";
    if (hasScreenCard) {
      live.mode = "board";
      const cmode = live.screenCard.mode;
      if (cmode === "checklist") {
        live.mood = "desk_locked";
        live.stateLabel = "CHECKLIST";
      } else if (cmode === "feedback") {
        live.mood = "yes";
        live.stateLabel = "FEEDBACK";
      } else if (cmode === "definition") {
        live.mood = "attentive";
        live.stateLabel = "DEFINITION";
      } else if (cmode === "exam") {
        live.mood = "focusing";
        live.stateLabel = "MINI-EXAM";
      } else if (cmode === "concept") {
        live.mood = "curious";
        live.stateLabel = "EXPLANATION";
      } else {
        live.mood = "listen";
        live.stateLabel = "EXERCISE";
      }
    } else if (isDesk) {
      live.mode = "desk";
      if (live.opticalPhase === "focusing") {
        live.mood = "focusing";
        live.stateLabel = "AF // FOCUSING";
      } else if (live.microPhase === "reading_sheet" || busy) {
        live.mood = "reading";
        live.stateLabel = "READING SHEET";
      } else {
        live.mood = sheetFound ? "desk_locked" : "desk";
        live.stateLabel = sheetFound ? "SHEET LOCKED" : "DESK MODE";
      }
    } else if (manualDnd || (!liveOn && !speaking && !busy && !activeGesture && !searching && idleForMs > 25000)) {
      live.mode = "focus";
      live.mood = "idle";
      live.stateLabel = "QUIET";
    } else {
      live.mode = "dialogue";
      if (activeGesture === "nod_yes") {
        if (live.mood !== "yes") anim.gestureStart = performance.now();
        live.mood = "yes";
        live.stateLabel = "YES";
      } else if (activeGesture === "shake_no") {
        if (live.mood !== "no") anim.gestureStart = performance.now();
        live.mood = "no";
        live.stateLabel = "NO";
      } else if (activeGesture === "curious_tilt") {
        live.mood = "curious";
        live.stateLabel = "CURIOUS";
      } else if (performance.now() < live.bargeInUntil) {
        live.mood = "barge_in";
        live.stateLabel = "BARGE-IN";
      } else if (live.opticalPhase === "focusing") {
        live.mood = "focusing";
        live.stateLabel = "AF // FOCUSING";
      } else if (searching) {
        live.mood = "search";
        live.stateLabel = "SCANNING";
      } else if (live.microPhase === "tool_intent" && live.toolIntent === "look_at_desk") {
        live.mood = "intent_desk";
        live.stateLabel = "INTENT // DESK";
      } else if (live.microPhase === "reading_sheet") {
        live.mood = "reading";
        live.stateLabel = "READING";
      } else if (live.microPhase === "formulating") {
        live.mood = "formulating";
        live.stateLabel = "FORMULATING";
      } else if (live.microPhase === "deep_thinking" || busy) {
        live.mood = "think";
        live.stateLabel = live.microPhase === "deep_thinking" ? "THINKING Δ" : "THINKING";
      } else if (speaking) {
        live.mood = "speak";
        live.stateLabel = "SPEAKING";
      } else if (live.microPhase === "user_speaking") {
        live.mood = "attentive";
        live.stateLabel = "ATTENTIVE";
      } else if (performance.now() < live.greetUntil) {
        live.mood = "greet";
        live.stateLabel = "WELCOME BACK";
      } else if (drowsy) {
        live.mood = "drowsy";
        live.stateLabel = "DROWSING";
      } else if (liveOn) {
        live.mood = "listen";
        live.stateLabel = vis.face_detected ? "LISTENING" : "STANDBY";
      } else {
        live.mood = "idle";
        live.stateLabel = vis.face_detected ? "TRACKING" : "READY";
      }
    }

    syncDomLayers();
  }

  function escHtml(str) {
    return String(str || "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function renderBoard(card) {
    const mode = card.mode || "exercise";
    const items = Array.isArray(card.items) ? card.items : [];
    const updatedAt = Number(card.updated_at || 0);
    if (updatedAt && updatedAt !== lastCardUpdatedAt) {
      lastCardUpdatedAt = updatedAt;
      hintRevealed = false;
      haptic([12, 24, 14]);
      sp.pitch.vel += 1.5; // crisp arrival micro-nod!
    }

    const sig = JSON.stringify({
      m: mode,
      t: card.title || "",
      n: card.note || "",
      h: hintRevealed,
      d: live.deskMode,
      c: Boolean(card.came_from_desk),
      i: items.map((x) => [x.text, Boolean(x.done)])
    });
    if (sig === lastBoardSig) return;
    lastBoardSig = sig;

    elBoardCard.dataset.cardMode = mode;
    elBoardModeBadge.textContent = mode === "exam" ? "MINI-EXAM" : mode.toUpperCase();
    elBoardTitle.textContent =
      card.title ||
      (mode === "definition"
        ? "KEY DEFINITION"
        : mode === "exam"
          ? "MINI-EXAM"
          : mode === "exercise"
            ? "PRACTICE EXERCISE"
            : mode === "feedback"
              ? "WORK CHECK · FEEDBACK"
              : mode === "checklist"
                ? "STUDY CHECKLIST"
                : "KEY EXPLANATION");

    // Checklist completion progress bar
    if (mode === "checklist" && items.length > 0) {
      const doneCount = items.filter((it) => Boolean(it.done)).length;
      elBoardProgressPill.textContent = `${doneCount} / ${items.length}`;
      elBoardProgressPill.classList.remove("hidden");
      elBoardProgressTrack.classList.remove("hidden");
      elBoardProgressFill.style.width = `${Math.round((doneCount / items.length) * 100)}%`;
    } else {
      elBoardProgressPill.classList.add("hidden");
      elBoardProgressTrack.classList.add("hidden");
    }

    // Separate "Given / Data" strip in Exercise & Exam modes for effortless scanning
    const givenParts = [];
    const displayRows = [];
    items.forEach((it, origIdx) => {
      const txt = String(it.text || "").trim();
      if (!txt) return;
      if (mode === "exercise" || mode === "exam") {
        const mGiven = txt.match(/^(?:given|data|context|values|time)\s*[:—-]\s*(.+)$/i);
        if (mGiven) {
          givenParts.push(mGiven[1].trim());
          return;
        }
      }
      displayRows.push({ text: txt, done: Boolean(it.done), origIdx });
    });

    if ((mode === "exercise" || mode === "exam") && givenParts.length > 0) {
      elBoardGivenText.textContent = givenParts.join("  ·  ");
      elBoardGiven.classList.remove("hidden");
    } else {
      elBoardGiven.classList.add("hidden");
    }

    // Render Definition / Exercise / Exam [pts] / Feedback ✓△→ / Checklist / Concept rows
    elBoardList.innerHTML = displayRows.map((row, rowIdx) => {
      let tag = `0${rowIdx + 1}`;
      let bodyText = row.text;
      let extraCls = "";
      let ptsBadge = "";

      if (mode === "checklist") {
        tag = row.done ? "[✓]" : "[ ]";
      } else if (mode === "definition") {
        const mDef = bodyText.match(/^(definition|def|formula|equation|example|ex|trap|watch out|key)\s*[:—-]\s*(.+)$/i);
        if (mDef) {
          const lbl = mDef[1].toUpperCase();
          tag = lbl === "DEF" ? "DEF" : lbl.startsWith("WATCH") ? "TRAP" : lbl;
          bodyText = mDef[2].trim();
          if (tag === "DEF" || tag === "DEFINITION") { tag = "DEF"; extraCls = "def-hero"; }
          else if (tag === "TRAP") extraCls = "def-trap";
        } else if (rowIdx === 0) {
          tag = "DEF";
          extraCls = "def-hero";
        } else {
          tag = "NOTE";
        }
      } else if (mode === "feedback") {
        if (/^(?:ok|good|correct|✓)\s*[:—-]?\s*/i.test(bodyText)) {
          tag = "✓";
          extraCls = "fb-ok";
          bodyText = bodyText.replace(/^(?:ok|good|correct|✓)\s*[:—-]\s*/i, "").trim();
        } else if (/^(?:fix|tip|watch|correction|✗|△)\s*[:—-]?\s*/i.test(bodyText)) {
          tag = "△";
          extraCls = "fb-fix";
          bodyText = bodyText.replace(/^(?:fix|tip|watch|correction|✗|△)\s*[:—-]\s*/i, "").trim();
        } else if (/^(?:next|try|→)\s*[:—-]?\s*/i.test(bodyText)) {
          tag = "→";
          extraCls = "fb-next";
          bodyText = bodyText.replace(/^(?:next|try|→)\s*[:—-]\s*/i, "").trim();
        } else {
          tag = "·";
        }
      } else if (mode === "exercise" || mode === "exam") {
        const mQ = bodyText.match(/^(Q\d+|Part\s*[A-Z]|\d+)(?:\s*[\[\(](\d+(?:\.\d+)?\s*pts?)[\]\)])?[\)\.\:\-]\s*(.+)$/i);
        if (mQ) {
          tag = mQ[1].toUpperCase().replace(/^(\d+)$/, "Q$1");
          if (mQ[2]) ptsBadge = `<span class="item-pts">${escHtml(mQ[2].toUpperCase())}</span>`;
          bodyText = mQ[3].trim();
        } else {
          tag = `Q${rowIdx + 1}`;
        }
        // Also catch trailing [2 pts] / (3 pts) at end of question line
        if (!ptsBadge && mode === "exam") {
          const mTrail = bodyText.match(/[\[\(](\d+(?:\.\d+)?\s*pts?)[\]\)]\s*$/i);
          if (mTrail) {
            ptsBadge = `<span class="item-pts">${escHtml(mTrail[1].toUpperCase())}</span>`;
            bodyText = bodyText.slice(0, mTrail.index).trim();
          }
        }
      }
      return `<li class="board-item ${extraCls} ${row.done ? "done" : ""}" data-idx="${row.origIdx}"><span class="item-tag">${escHtml(tag)}</span><span class="item-text">${escHtml(bodyText)}</span>${ptsBadge}</li>`;
    }).join("");

    // Exercise/Exam collapsible Hint/Rubric vs Definition/Concept/Checklist/Feedback bottom Note
    const noteTxt = String(card.note || "").trim();
    if ((mode === "exercise" || mode === "exam") && noteTxt) {
      elBoardHintWrap.classList.remove("hidden");
      elBoardNote.classList.add("hidden");
      const hLbl = mode === "exam" ? "RUBRIC / HINT" : "HINT";
      btnToggleHint.textContent = hintRevealed ? `${hLbl} ▾ HIDE` : `${hLbl} ▸ TAP TO REVEAL`;
      elBoardHintBox.textContent = noteTxt;
      elBoardHintBox.classList.toggle("hidden", !hintRevealed);
    } else if (noteTxt) {
      elBoardHintWrap.classList.add("hidden");
      elBoardNote.classList.remove("hidden");
      elBoardNote.textContent = noteTxt;
    } else {
      elBoardHintWrap.classList.add("hidden");
      elBoardNote.classList.add("hidden");
    }

    const fromDesk = Boolean(card.came_from_desk);
    elBoardFootGuide.textContent = fromDesk
      ? "NOTED? SAY 'GOT IT' TO RETURN TO YOUR SHEET"
      : mode === "exercise" || mode === "exam"
        ? (noteTxt ? "SOLVE ON PAPER · TAP HINT IF STUCK" : "SOLVE ON PAPER · SAY CHECK MY SHEET")
        : mode === "definition"
          ? "COPY THIS DEFINITION · SAY 'GOT IT' WHEN READY"
          : mode === "feedback"
            ? "REVIEW FEEDBACK · READY FOR NEXT STEP"
            : mode === "checklist"
              ? "TAP ITEMS TO CHECK OFF"
              : "OSMO // OLED TUTOR BOARD";

    btnBoardPivot.textContent = live.deskMode
      ? "FACE ME ↗"
      : fromDesk
        ? "BACK TO SHEET ↘"
        : "LOOK AT SHEET ↘";
  }

  function syncDomLayers() {
    document.body.dataset.mode = live.mode;
    elStatePill.textContent = live.stateLabel;

    if (live.deskMode) {
      const resTag = live.width >= 3800 ? "4K" : live.width >= 2500 ? "2.7K" : `${live.width}p`;
      elOptPill.textContent = `${live.zoom.toFixed(live.zoom % 1 ? 2 : 1)}× · ${resTag}`;
      elOptPill.classList.remove("hidden");
    } else {
      elOptPill.classList.add("hidden");
    }

    elFocusLayer.classList.toggle("hidden", live.mode !== "focus");

    // Clean Desk Reading Strip (ONLY when reading desk with no board card)
    const showDeskStrip = live.mode === "desk";
    elDeskStrip.classList.toggle("hidden", !showDeskStrip);
    if (showDeskStrip) {
      elDeskLed.classList.toggle("locked", live.sheetFound && !live.busy);
      elDeskStatus.textContent = live.busy
        ? "DESK // ANALYZING SHEET"
        : live.sheetFound
          ? "DESK // SHEET LOCKED"
          : "DESK // CAMERA ACTIVE";
      const resTag = live.width >= 3800 ? "4K" : live.width >= 2500 ? "2.7K" : `${live.width}p`;
      elDeskOptics.textContent = `${live.zoom.toFixed(live.zoom % 1 ? 2 : 1)}× · ${resTag}`;
      elDeskHiLight.textContent = `HILIGHT // ${live.hilightEffect.toUpperCase()}`;
      elDeskScanBar.classList.toggle("hidden", !live.busy);
    }

    // Full-Bleed OLED Study Board (Exercise / Checklist / Concept)
    const showBoard = live.mode === "board";
    elBoardLayer.classList.toggle("hidden", !showBoard);
    if (showBoard) {
      renderBoard(live.screenCard || { mode: "exercise", title: "", items: [], note: "" });
    } else {
      lastBoardSig = "";
    }
  }

  // --- WebSocket connection ---
  function sendWs(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload));
  }
  function connectWs() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws`);
    ws.onopen = () => elConn.classList.add("ok");
    ws.onmessage = (ev) => {
      try { applySnapshot(JSON.parse(ev.data)); } catch (_) {}
    };
    ws.onclose = () => {
      elConn.classList.remove("ok");
      setTimeout(connectWs, 1500);
    };
  }
  connectWs();

  // --- Interactive Pointer Hover + Touch Gestures ---
  let touchStartY = 0, touchStartT = 0, lastTapT = 0;

  document.addEventListener("pointermove", (e) => {
    if (e.target.closest("#demo-dock")) return;
    pointer.active = true;
    pointer.x = ((e.clientX / Math.max(1, window.innerWidth)) - 0.5) * 2;
    pointer.y = ((e.clientY / Math.max(1, window.innerHeight)) - 0.5) * 2;
    pointer.lastMoveAt = performance.now();
  });

  document.addEventListener("pointerleave", () => {
    pointer.active = false;
  });

  function isUiControl(target) {
    return Boolean(
      target.closest("#demo-dock") ||
      target.closest("#btn-demo-toggle") ||
      target.closest("#board-card") ||
      target.closest("#desk-strip")
    );
  }

  document.addEventListener("pointerdown", (e) => {
    if (isUiControl(e.target)) return;
    requestWakeLock();
    touchStartY = e.clientY;
    touchStartT = performance.now();
  });

  // Interactive tap-to-check on Pixel OLED checklist items!
  elBoardList.addEventListener("click", (e) => {
    const li = e.target.closest(".board-item");
    if (!li || !live.screenCard || live.screenCard.mode !== "checklist") return;
    const idx = Number(li.dataset.idx);
    if (Number.isNaN(idx)) return;
    haptic(14);
    sp.pitch.vel += 1.5; // happy micro-nod on check-off!
    if (demoMode && live.screenCard.items[idx]) {
      live.screenCard.items[idx].done = !live.screenCard.items[idx].done;
      syncDomLayers();
    } else {
      sendWs({ action: "toggle_screen_item", index: idx });
    }
  });

  // Exercise collapsible Hint button
  btnToggleHint.addEventListener("click", () => {
    hintRevealed = !hintRevealed;
    haptic(12);
    sp.pitch.vel += 1.2;
    syncDomLayers();
  });

  function dismissBoard() {
    haptic(12);
    live.screenCard = { mode: "none", title: "", items: [], note: "" };
    if (!demoMode) sendWs({ action: "clear_screen_card" });
    live.mode = live.deskMode ? "desk" : "dialogue";
    syncDomLayers();
  }

  btnClearCard.addEventListener("click", dismissBoard);
  btnCloseBoard.addEventListener("click", dismissBoard);

  // Instant Show & Tell hardware pivot buttons on Pixel OLED!
  btnBoardPivot.addEventListener("click", () => {
    haptic([14, 30, 18]);
    if (live.deskMode) {
      if (!demoMode) sendWs({ action: "return_face_mode" });
      showToast("TURNING TO FACE");
    } else {
      if (!demoMode) sendWs({ action: "enter_desk_mode" });
      showToast("LOOKING AT SHEET");
    }
  });

  btnDeskToFace.addEventListener("click", () => {
    haptic([14, 30, 18]);
    if (!demoMode) sendWs({ action: "return_face_mode" });
    showToast("TURNING TO FACE");
  });

  document.addEventListener("pointerup", (e) => {
    if (isUiControl(e.target)) return;
    const dy = e.clientY - touchStartY;
    const dt = performance.now() - touchStartT;

    if (dy > 75 && dt < 500) {
      manualDnd = true;
      live.mode = "focus";
      live.mood = "idle";
      live.stateLabel = "QUIET";
      sendWs({ action: "hilight", reaction: "off" });
      haptic(18);
      showToast("QUIET MODE");
      syncDomLayers();
      return;
    }

    if (dy < -75 && dt < 500) {
      manualDnd = false;
      lastActivityAt = performance.now();
      live.mode = live.deskMode ? "desk" : "dialogue";
      if (!document.fullscreenElement && document.documentElement.requestFullscreen) {
        document.documentElement.requestFullscreen().catch(() => {});
      }
      haptic([12, 24, 12]);
      showToast("AWAKE");
      syncDomLayers();
      return;
    }

    if (Math.abs(dy) < 22 && dt < 320) {
      const now = performance.now();
      if (now - lastTapT < 340) {
        lastTapT = 0;
        manualDnd = false;
        lastActivityAt = now;
        if (live.deskMode) {
          sendWs({ action: "return_face_mode" });
          haptic([16, 40, 22]);
          showToast("MODE // FACE");
        } else {
          sendWs({ action: "enter_desk_mode" });
          haptic([16, 40, 22]);
          showToast("MODE // DESK");
        }
      } else {
        lastTapT = now;
        manualDnd = false;
        lastActivityAt = now;
        anim.blinkPhase = 0.001;
        // Playful spring impulse on tap
        sp.pitch.vel += 1.4;
        haptic(10);
        if (live.mode === "focus") {
          live.mode = "dialogue";
          syncDomLayers();
        }
      }
    }
  });

  // --- Demo / Preview presets ---
  function applyDemo(preset) {
    if (preset === "live") {
      demoMode = null;
      manualDnd = false;
      showToast("WS // LIVE");
      return;
    }
    demoMode = preset;
    manualDnd = false;
    lastActivityAt = performance.now();
    anim.gestureStart = performance.now();
    live.deskMode = preset === "desk";
    live.sheetFound = preset === "desk";
    live.speaking = preset === "speak";
    live.busy = preset === "think" || preset === "reading";
    live.opticalPhase = preset === "focusing" ? "focusing" : "";
    live.microPhase = preset === "think" ? "deep_thinking" : preset === "attentive" ? "user_speaking" : preset;
    live.toolIntent = preset === "intent_desk" ? "look_at_desk" : "";
    live.drowsyLevel = preset === "drowsy" ? 0.65 : 0;
    live.pulseEnergy = 1.0;
    live.zoom = 1.0;
    live.width = preset === "desk" ? 2688 : 1280;
    live.height = preset === "desk" ? 1512 : 720;
    live.hilightColor = null;
    live.hilightEffect = preset === "desk" ? "sheet_ringlight" : preset;

    if (preset === "barge_in") {
      live.bargeInUntil = performance.now() + 900;
      anim.blinkPhase = 0.001;
      sp.pitch.vel -= 2.4;
    }
    if (preset === "greet") {
      live.greetUntil = performance.now() + 1400;
      sp.pitch.vel += 1.8;
    }

    if (preset === "focus") {
      live.mode = "focus";
      live.mood = "idle";
      live.stateLabel = "QUIET";
      live.reply = "";
      live.screenCard = { mode: "none", title: "", items: [], note: "" };
    } else if (preset === "definition") {
      live.mode = "board";
      live.mood = "attentive";
      live.stateLabel = "DEFINITION";
      live.screenCard = {
        mode: "definition",
        title: "QMAX · PEAK CONTACT HEAT FLUX",
        items: [
          { text: "Maximum rate of conductive heat transferred from warm skin into a cooler fabric at first contact.", done: false },
          { text: "Formula: Pmax = Qmax × A   (W/cm² × cm² = W)", done: false },
          { text: "Trap: High Qmax measures instant cool touch, NOT how long cooling lasts.", done: false }
        ],
        note: "Higher Qmax → faster initial heat draw from skin → cooler sensation",
        came_from_desk: true,
        updated_at: Date.now()
      };
    } else if (preset === "exam") {
      live.mode = "board";
      live.mood = "focusing";
      live.stateLabel = "MINI-EXAM";
      hintRevealed = false;
      live.screenCard = {
        mode: "exam",
        title: "MINI-EXAM · THERMAL TRANSFER (5 PTS)",
        items: [
          { text: "Given: ICENYL Qmax = 0.18 W/cm² · Cotton = 0.10 W/cm² · Area A = 12 cm²", done: false },
          { text: "Q1 [2 pts]: Explain why ICENYL feels cooler than cotton at first touch.", done: false },
          { text: "Q2 [3 pts]: Compute peak heat power difference ΔPmax between both fabrics.", done: false }
        ],
        note: "Rubric: 2 pts conduction direction + rate; 3 pts ΔP = (0.18 − 0.10) × 12 = 0.96 W",
        came_from_desk: true,
        updated_at: Date.now()
      };
    } else if (preset === "exercise") {
      live.mode = "board";
      live.mood = "listen";
      live.stateLabel = "EXERCISE";
      hintRevealed = false;
      live.screenCard = {
        mode: "exercise",
        title: "BAC PRACTICE · HEAT TRANSFER",
        items: [
          { text: "Given: ICENYL Qmax = 0.18 W/cm² · Contact area A = 10 cm²", done: false },
          { text: "Q1: Which direction does conductive heat flow when your hand touches the fabric?", done: false },
          { text: "Q2: Calculate the peak instantaneous heat power Pmax in watts.", done: false }
        ],
        note: "Use Pmax = Qmax × A (check that cm² cancels cleanly!)",
        updated_at: Date.now()
      };
    } else if (preset === "feedback") {
      live.mode = "board";
      live.mood = "yes";
      live.stateLabel = "FEEDBACK";
      live.screenCard = {
        mode: "feedback",
        title: "WORK CHECK · HEAT TRANSFER",
        items: [
          { text: "OK: Direction of heat flow (skin → fabric → air) is spot on", done: false },
          { text: "FIX: Pmax = 0.18 × 10 = 1.8 W (watch the decimal placement)", done: false },
          { text: "NEXT: Compute total heat energy transferred over 5 seconds", done: false }
        ],
        note: "Strong physical intuition — one tiny decimal slip to fix!",
        updated_at: Date.now()
      };
    } else if (preset === "checklist") {
      live.mode = "board";
      live.mood = "desk_locked";
      live.stateLabel = "CHECKLIST";
      live.screenCard = {
        mode: "checklist",
        title: "THERMO REVISION · QMAX",
        items: [
          { text: "Draw heat flux arrow: skin → fabric → air", done: true },
          { text: "Compare Qmax: Cotton vs ICENYL (0.18 W/cm²)", done: false },
          { text: "Explain instant touch vs sustained cooling", done: false }
        ],
        note: "Qmax = peak conductive heat flux at first contact",
        updated_at: Date.now()
      };
    } else if (preset === "concept") {
      live.mode = "board";
      live.mood = "curious";
      live.stateLabel = "CONCEPT";
      live.screenCard = {
        mode: "concept",
        title: "COOL-TOUCH FABRIC PHYSICS",
        items: [
          { text: "Higher Qmax means faster initial heat draw from skin → cooler feel", done: false },
          { text: "Conduction dominates the first second; evaporation & airflow sustain cooling", done: false },
          { text: "ICENYL reaches 0.18 W/cm² vs ~0.10 W/cm² for standard cotton", done: false }
        ],
        note: "P = Qmax × Area   (W = W/cm² × cm²)",
        updated_at: Date.now()
      };
    } else if (preset === "desk") {
      live.mode = "desk";
      live.mood = "desk_locked";
      live.stateLabel = "SHEET LOCKED";
      live.screenCard = { mode: "none", title: "", items: [], note: "" };
    } else {
      live.mode = "dialogue";
      live.screenCard = { mode: "none", title: "", items: [], note: "" };
      live.mood = preset;
      const labels = {
        listen: "LISTENING",
        attentive: "ATTENTIVE · INPUT Δ",
        barge_in: "BARGE-IN",
        think: "THINKING Δ",
        intent_desk: "INTENT // DESK",
        focusing: "AF // FOCUSING",
        reading: "READING SACCADES",
        formulating: "FORMULATING",
        speak: "SPEAKING",
        greet: "WELCOME BACK",
        drowsy: "DROWSING",
        yes: "YES",
        no: "NO",
        curious: "CURIOUS",
        search: "SCANNING"
      };
      live.stateLabel = labels[preset] || "READY";
      live.reply = preset === "speak" ? "Je vois ton schéma en rouge avec Solution 1 et Solution 2." : "";
      lastReplyAt = performance.now();
    }
    syncDomLayers();
  }

  btnDemoToggle.addEventListener("click", () => elDemoDock.classList.toggle("hidden"));
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
    const initPreset = demoMode === "1" ? "listen" : demoMode;
    applyDemo(initPreset);
    const activeBtn = elDemoDock.querySelector(`button[data-demo="${initPreset}"]`);
    if (activeBtn) activeBtn.classList.add("active");
  }

  // --- High-DPI Canvas Resize ---
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

  // =========================================================================
  // 3D SUPERELLIPSOID (ROUNDED CUBE) MATH — Analytical Support Silhouette
  // =========================================================================
  const LP = 3.75; // superellipsoid power (3.75 = crisp industrial rounded cube!)
  const CAM_DIST = 4.6;

  // Build 3x3 rotation matrix R = Rz(roll) * Rx(pitch) * Ry(yaw)
  function rotMatrix(yaw, pitch, roll) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp_ = Math.sin(pitch);
    const cr = Math.cos(roll), sr = Math.sin(roll);
    return [
      cr * cy - sr * sp_ * sy,  -sr * cp,  cr * sy + sr * sp_ * cy,
      sr * cy + cr * sp_ * sy,   cr * cp,  sr * sy - cr * sp_ * cy,
      -cp * sy,                  sp_,      cp * cy
    ];
  }

  function mulR(R, x, y, z) {
    return [
      R[0] * x + R[1] * y + R[2] * z,
      R[3] * x + R[4] * y + R[5] * z,
      R[6] * x + R[7] * y + R[8] * z
    ];
  }

  function mulRT(R, x, y, z) {
    return [
      R[0] * x + R[3] * y + R[6] * z,
      R[1] * x + R[4] * y + R[7] * z,
      R[2] * x + R[5] * y + R[8] * z
    ];
  }

  // Normalize any 3D vector onto the L_p superellipsoid surface (|x|^p + |y|^p + |z|^p = 1)
  function toSuperSurface(x, y, z) {
    const n = Math.pow(
      Math.pow(Math.abs(x), LP) + Math.pow(Math.abs(y), LP) + Math.pow(Math.abs(z), LP),
      1 / LP
    ) || 1;
    return [x / n, y / n, z / n];
  }

  // Outward normal of L_p surface in local coordinates
  function superNormal(x, y, z) {
    const p1 = LP - 1;
    const nx = Math.sign(x) * Math.pow(Math.abs(x), p1);
    const ny = Math.sign(y) * Math.pow(Math.abs(y), p1);
    const nz = Math.sign(z) * Math.pow(Math.abs(z), p1);
    const len = Math.hypot(nx, ny, nz) || 1;
    return [nx / len, ny / len, nz / len];
  }

  // Perspective projection (local -> world -> screen)
  function projectWorld(wx, wy, wz, cx, cy, scalePx) {
    const persp = CAM_DIST / (CAM_DIST - wz);
    return [cx + wx * scalePx * persp, cy + wy * scalePx * persp, wz];
  }

  // Exact analytical PERSPECTIVE silhouette of rotated L_p rounded cube!
  // Solves dual-norm support against perspective frustum plane normal (cos a, sin a, r / CAM_DIST)
  // in 3 fixed-point iterations -> 100% smooth, notch-free perspective contour at any 3D angle.
  function computeSilhouette(R, cx, cy, scalePx, samples = 144) {
    const pts = [];
    const invP1 = 1 / (LP - 1);
    for (let i = 0; i < samples; i++) {
      const a = (i / samples) * Math.PI * 2;
      const dx = Math.cos(a);
      const dy = Math.sin(a);
      let dz = 1.0 / CAM_DIST;
      let wx = 0, wy = 0, wz = 0;
      for (let it = 0; it < 3; it++) {
        const [ax, ay, az] = mulRT(R, dx, dy, dz);
        const lx = Math.sign(ax) * Math.pow(Math.abs(ax), invP1);
        const ly = Math.sign(ay) * Math.pow(Math.abs(ay), invP1);
        const lz = Math.sign(az) * Math.pow(Math.abs(az), invP1);
        const [sx, sy, sz] = toSuperSurface(lx, ly, lz);
        [wx, wy, wz] = mulR(R, sx, sy, sz);
        const rProj = (wx * dx + wy * dy) / (CAM_DIST - wz);
        dz = Math.max(0.05, Math.min(0.6, rProj));
      }
      const [px, py] = projectWorld(wx, wy, wz, cx, cy, scalePx);
      pts.push({ x: px, y: py, a, dy });
    }
    return pts;
  }

  // Draw dashed 3D meridian curve clipped strictly inside the silhouette on the front hemisphere
  function drawMeridian(R, axis, cx, cy, scalePx, strokeColor, silPath) {
    const steps = 120;
    ctx.save();
    ctx.clip(silPath);
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.05;
    ctx.setLineDash([3.0, 4.2]);
    ctx.beginPath();
    let drawing = false;

    for (let i = 0; i <= steps; i++) {
      const t = (i / steps) * Math.PI * 2;
      let rawX = 0, rawY = 0, rawZ = 0;
      if (axis === "vertical") {
        rawX = 0; rawY = Math.sin(t); rawZ = Math.cos(t);
      } else {
        rawX = Math.sin(t); rawY = 0; rawZ = Math.cos(t);
      }
      const [sx, sy, sz] = toSuperSurface(rawX, rawY, rawZ);
      const [nx, ny, nz] = superNormal(sx, sy, sz);
      const [wx, wy, wz] = mulR(R, sx, sy, sz);
      const [wnx, wny, wnz] = mulR(R, nx, ny, nz);
      const dot = wnx * (0 - wx) + wny * (0 - wy) + wnz * (CAM_DIST - wz);
      if (dot > 0.02) {
        const [px, py] = projectWorld(wx, wy, wz, cx, cy, scalePx);
        if (!drawing) {
          ctx.moveTo(px, py);
          drawing = true;
        } else {
          ctx.lineTo(px, py);
        }
      } else {
        drawing = false;
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  // Draw one architectural CAD eye on the curved 3D front face (45° hatching + vector perimeter nodes!)
  // Uses dedicated Path2D so clip & perimeter stroke NEVER collide.
  function drawDraftingEye(R, side, cx, cy, scalePx, eyeOpts) {
    const { openFactor, smile, frown, scaleMod, strokeAlpha, nodeColor, hatchColor, rimRgb } = eyeOpts;
    // Tight, iconic vertical capsule proportions matching the 3D CAD reference
    const ex = side * 0.235;
    const ey = 0.06 + (side === 1 ? -eyeOpts.asym * 0.04 : eyeOpts.asym * 0.025);
    const rx = 0.102 * scaleMod;
    const ry = Math.max(0.022, 0.215 * scaleMod * openFactor);

    const contour2D = [];
    const N = 18; // crisp evenly spaced CAD control-point nodes around each eye
    for (let i = 0; i < N; i++) {
      const a = (i / N) * Math.PI * 2;
      const ca = Math.cos(a), sa = Math.sin(a);
      let u = Math.sign(ca) * Math.pow(Math.abs(ca), 0.72) * rx;
      let v = Math.sign(sa) * Math.pow(Math.abs(sa), 0.72) * ry;

      if (smile > 0.05) {
        const normU = u / Math.max(0.01, rx);
        v -= (1 - normU * normU) * 0.095 * smile;
        if (sa > 0) v -= sa * ry * 0.45 * smile;
      }
      if (frown > 0.05 && sa < 0) {
        const slant = side === -1 ? (u / rx) : -(u / rx);
        v += (0.05 + slant * 0.05) * frown;
      }

      const [sx, sy, sz] = toSuperSurface(ex + u, ey + v, 1.0);
      const [wx, wy, wz] = mulR(R, sx, sy, sz);
      const [px, py] = projectWorld(wx, wy, wz, cx, cy, scalePx);
      contour2D.push({ x: px, y: py });
    }

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    const eyePath = new Path2D();
    contour2D.forEach((p, idx) => {
      if (idx === 0) eyePath.moveTo(p.x, p.y);
      else eyePath.lineTo(p.x, p.y);
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    });
    eyePath.closePath();

    // 1. Matte dark eye fill
    ctx.save();
    ctx.fillStyle = "rgba(10, 12, 16, 0.94)";
    ctx.fill(eyePath);

    // 2. Strictly clipped 45° diagonal architectural hatching INSIDE eyePath ONLY
    ctx.save();
    ctx.clip(eyePath);
    ctx.strokeStyle = hatchColor;
    ctx.lineWidth = 1.05;
    const step = Math.max(4.5, scalePx * 0.038);
    const span = (maxX - minX) + (maxY - minY);
    ctx.beginPath();
    for (let d = -span; d <= span; d += step) {
      ctx.moveTo(minX + d, maxY + 4);
      ctx.lineTo(minX + d + (maxY - minY) + 8, minY - 4);
    }
    ctx.stroke();
    ctx.restore();

    // 3. Crisp perimeter wireframe around eyePath
    ctx.strokeStyle = `rgba(210, 222, 238, ${(0.56 * strokeAlpha).toFixed(3)})`;
    ctx.lineWidth = 1.05;
    ctx.stroke(eyePath);

    // 4. Tiny hollow circular CAD vector nodes (○) along eye perimeter
    // During THINKING Δ / ATTENTIVE, a luminous token-driven wave travels around the control vertices!
    const nodeR = Math.max(1.55, Math.min(2.6, scalePx * 0.0145));
    const travelIdx = eyeOpts.travelIdx;
    for (let i = 0; i < contour2D.length; i++) {
      const p = contour2D[i];
      const dist = travelIdx >= 0
        ? Math.min(Math.abs(i - travelIdx), contour2D.length - Math.abs(i - travelIdx))
        : 99;
      const lit = dist <= 1;
      ctx.beginPath();
      ctx.arc(p.x, p.y, lit ? nodeR * 1.35 : nodeR, 0, Math.PI * 2);
      ctx.fillStyle = lit ? `rgba(${rimRgb}, 0.92)` : "#07080b";
      ctx.fill();
      ctx.strokeStyle = lit ? `rgba(${rimRgb}, 1.0)` : nodeColor;
      ctx.lineWidth = 1.05;
      ctx.stroke();
    }

    ctx.restore();
  }

  // =========================================================================
  // 120Hz RENDER LOOP
  // =========================================================================
  let lastT = performance.now();

  function frame(now) {
    const dt = Math.min(0.05, (now - lastT) / 1000);
    lastT = now;
    const t = now * 0.001;

    // 1. Target HiLight-synced accent rim color
    const targetRgb =
      live.hilightColor && (live.hilightColor[0] + live.hilightColor[1] + live.hilightColor[2] > 25)
        ? live.hilightColor
        : (RIM_COLORS[live.mood] || RIM_COLORS.idle);
    const ck = Math.min(1, dt * 7.5);
    anim.r += (targetRgb[0] - anim.r) * ck;
    anim.g += (targetRgb[1] - anim.g) * ck;
    anim.b += (targetRgb[2] - anim.b) * ck;

    const targetHalo = live.mode === "focus" ? 0.02 : live.speaking || live.busy ? 0.14 : 0.08;
    anim.haloAlpha += (targetHalo - anim.haloAlpha) * ck;
    const targetWire = live.mode === "focus" ? 0.24 : 0.78;
    anim.wireAlpha += (targetWire - anim.wireAlpha) * ck;

    const rimRgb = `${Math.round(anim.r)}, ${Math.round(anim.g)}, ${Math.round(anim.b)}`;
    const rs = document.documentElement.style;
    rs.setProperty("--rim-r", Math.round(anim.r));
    rs.setProperty("--rim-g", Math.round(anim.g));
    rs.setProperty("--rim-b", Math.round(anim.b));
    rs.setProperty("--halo-alpha", anim.haloAlpha.toFixed(3));

    // 2. Compute target 3D head orientation (yaw, pitch, roll) + layout position
    const pointerFresh = pointer.active && (now - pointer.lastMoveAt < 5000);
    const inputX = pointerFresh ? pointer.x : live.gazeX;
    const inputY = pointerFresh ? pointer.y : live.gazeY;

    let tYaw = -0.32 + inputX * 0.42 + Math.sin(t * 0.65) * 0.03;
    let tPitch = -0.12 + inputY * 0.28 + Math.cos(t * 0.52) * 0.02;
    let tRoll = -inputX * 0.05;
    let tScale = 1.0;
    let tCx = 0.5;
    let tCy = 0.49;

    // Synchronize 3D CAD cube directly with physical DJI Osmo Mobile 3 arm (IMU + motor lead-in!)
    if (!demoMode && live.armConnected) {
      const deg2rad = Math.PI / 180;
      // 1. Live motor velocity anticipation (pronounced 0ms kinetic follow-through!)
      const velYaw = Math.max(-0.56, Math.min(0.56, live.armPanDps * 0.0068));
      const velPitch = Math.max(-0.46, Math.min(0.46, live.armTiltDps * 0.0068));
      const velBank = Math.max(-0.26, Math.min(0.26, -live.armPanDps * 0.0030));

      // 2. Real physical IMU attitude coupling relative to calibrated neutral face_pose
      const imuYaw = Math.max(-0.45, Math.min(0.45, live.armRelPanDeg * deg2rad * 0.55));
      const imuPitch = Math.max(-0.35, Math.min(0.55, live.armRelPitchDeg * deg2rad * 0.72));

      tYaw = (live.faceDetected ? inputX * 0.22 : 0) + imuYaw + velYaw;
      tPitch = (live.faceDetected ? inputY * 0.16 : 0) + imuPitch + velPitch;
      tRoll = velBank - inputX * 0.04;
    }

    // Decay token-pulse energy smoothly (re-energized continuously in demo for think/speak/attentive)
    if (demoMode && (live.mood === "think" || live.mood === "speak" || live.mood === "attentive" || live.mood === "reading")) {
      live.pulseEnergy = 0.55 + 0.45 * Math.abs(Math.sin(t * 7.5));
      live.thinkCursor = Math.floor(t * 14) % 18;
    } else {
      live.pulseEnergy = Math.max(0, live.pulseEnergy - dt * 2.4);
    }

    const gElapsed = (now - anim.gestureStart) * 0.001;

    if (live.mode === "focus") {
      tYaw = -0.16 + Math.sin(t * 0.25) * 0.04;
      tPitch = 0.14;
      tRoll = 0.0;
      tScale = 0.78;
      tCy = 0.44;
    } else if (live.mode === "board") {
      // Dock cleanly into the left 21% rail facing right toward the OLED Study Board!
      const narrow = W < 640;
      tYaw = 0.34 + Math.sin(t * 0.75) * 0.04;
      tPitch = -0.04 + Math.cos(t * 0.6) * 0.025 + (live.speaking ? Math.sin(t * 4.8) * 0.03 : 0);
      tRoll = 0.02;
      tScale = narrow ? 0.001 : 0.43;
      tCx = narrow ? 0.10 : 0.108;
      tCy = 0.51;
    } else if (live.mode === "desk") {
      // Full-screen sculptural 3D CAD head pitched down reading the sheet (zero text wall blocking it!)
      const deskPitchRad = !demoMode && live.armConnected
        ? Math.max(0.22, Math.min(0.55, live.armRelPitchDeg * (Math.PI / 180) * 0.85))
        : 0.34;
      const readSweep = live.mood === "reading" ? (((t * 1.15) % 1) * 0.30 - 0.15) : Math.sin(t * 0.8) * 0.05;
      const readLine = live.mood === "reading" ? (Math.floor(t * 1.15) % 3) * 0.045 : 0;
      tYaw = readSweep;
      tPitch = deskPitchRad + readLine;
      tRoll = -0.04;
      tScale = 0.88;
      tCx = 0.50;
      tCy = 0.44;
    } else if (demoMode || !live.armConnected) {
      if (live.mood === "attentive") {
        // Lean slightly in + attentive head tilt while user is speaking!
        tScale = 1.045 + live.pulseEnergy * 0.015;
        tPitch += 0.05;
        tRoll = 0.07;
      } else if (live.mood === "barge_in") {
        // Crisp startled recoil on mid-sentence interruption
        tPitch -= 0.15;
        tScale = 1.04;
      } else if (live.mood === "think") {
        // Reflective glance up-right driven by thinking deltas
        tYaw = 0.28 + Math.sin(t * 1.6) * 0.07;
        tPitch = -0.22 + Math.cos(t * 1.2) * 0.05;
        tRoll = 0.09;
      } else if (live.mood === "intent_desk") {
        // Pre-action anticipation: glances downward toward the sheet ~400ms before motors fire!
        tYaw = -0.42;
        tPitch = 0.30;
        tRoll = -0.06;
      } else if (live.mood === "focusing") {
        // Autofocus squint lean-in
        tScale = 1.05;
        tPitch = 0.10;
      } else if (live.mood === "reading") {
        // Authentic human left-to-right reading saccades + carriage return line by line!
        const saw = (t * 1.1) % 1;
        const lineStep = Math.floor(t * 1.1) % 3;
        tYaw = -0.26 + saw * 0.44;
        tPitch = 0.16 + lineStep * 0.06;
        tRoll = -0.03;
      } else if (live.mood === "formulating") {
        // Snaps eye contact cleanly back to user ~500ms before voice starts
        tYaw = 0.0;
        tPitch = -0.05;
        tScale = 1.02;
      } else if (live.mood === "greet") {
        // Warm welcome nod when face returns after absence
        tPitch = -0.04 + Math.sin(gElapsed * 8.5) * 0.18 * Math.exp(-gElapsed * 1.6);
        tScale = 1.03;
      } else if (live.mood === "drowsy") {
        // Gentle droop before quiet mode
        tPitch = 0.12 + live.drowsyLevel * 0.08;
        tScale = 1.0 - live.drowsyLevel * 0.12;
      } else if (live.mood === "yes") {
        tPitch = 0.08 + Math.sin(gElapsed * 10.0) * 0.34 * Math.exp(-gElapsed * 0.32);
        tScale = 1.07;
      } else if (live.mood === "no") {
        tYaw = Math.sin(gElapsed * 11.5) * 0.46 * Math.exp(-gElapsed * 0.32);
        tRoll = -Math.sin(gElapsed * 11.5) * 0.10 * Math.exp(-gElapsed * 0.32);
      } else if (live.mood === "curious") {
        tYaw = -0.28;
        tPitch = -0.12;
        tRoll = 0.32;
        tScale = 1.05;
      } else if (live.mood === "search") {
        tYaw = Math.sin(t * 2.2) * 0.56;
        tPitch = 0.04 + Math.sin(t * 4.4) * 0.05;
      } else if (live.mood === "speak") {
        tPitch += Math.sin(t * 4.8) * 0.05;
        tRoll += Math.sin(t * 2.4) * 0.04;
      }
    } else {
      // When arm IS physically connected, layer pronounced 3D CAD reactions in sync with the physical gimbal!
      if (live.mood === "yes") {
        tPitch += Math.sin(gElapsed * 10.0) * 0.28 * Math.exp(-gElapsed * 0.35);
        tScale = 1.07;
      } else if (live.mood === "no") {
        tYaw += Math.sin(gElapsed * 11.5) * 0.36 * Math.exp(-gElapsed * 0.35);
        tRoll -= Math.sin(gElapsed * 11.5) * 0.09 * Math.exp(-gElapsed * 0.35);
      } else if (live.mood === "curious") {
        tYaw += 0.18;
        tPitch -= 0.12;
        tRoll += 0.32;
        tScale = 1.05;
      } else if (live.mood === "attentive") {
        tScale = 1.07;
        tRoll += 0.11;
      } else if (live.mood === "barge_in") {
        tPitch -= 0.20;
        tScale = 1.06;
      } else if (live.mood === "think") {
        tYaw += 0.24;
        tPitch -= 0.18;
        tRoll += 0.11;
      } else if (live.mood === "intent_desk") {
        tYaw -= 0.28;
        tPitch += 0.28;
      } else if (live.mood === "reading") {
        tYaw += (((t * 1.1) % 1) * 0.30 - 0.15);
      } else if (live.mood === "greet") {
        tPitch += Math.sin(gElapsed * 9.0) * 0.24 * Math.exp(-gElapsed * 1.4);
        tScale = 1.06;
      } else if (live.mood === "speak") {
        tPitch += Math.sin(t * 4.8) * 0.04;
        tRoll += Math.sin(t * 2.4) * 0.03;
      }
    }

    const yaw = stepSpring(sp.yaw, tYaw, dt);
    const pitch = stepSpring(sp.pitch, tPitch, dt);
    const roll = stepSpring(sp.roll, tRoll, dt);
    const scale = stepSpring(sp.scale, tScale, dt);
    const cxNorm = stepSpring(sp.centerX, tCx, dt);
    const cyNorm = stepSpring(sp.centerY, tCy, dt);

    // Update live monospace angle readout in top-right HUD (shows real ARM IMU when connected!)
    if (!demoMode && live.armConnected) {
      const ap = live.armRelPanDeg.toFixed(0);
      const at = live.armRelPitchDeg.toFixed(0);
      const spd = Math.round(Math.hypot(live.armPanDps, live.armTiltDps));
      elPoseTelemetry.textContent = `ARM ΔY ${live.armRelPanDeg >= 0 ? "+" : ""}${ap}° · ΔP ${live.armRelPitchDeg >= 0 ? "+" : ""}${at}° · ${spd}°/s`;
    } else {
      const yDeg = (yaw * 180 / Math.PI).toFixed(1).padStart(5, " ");
      const pDeg = (pitch * 180 / Math.PI).toFixed(1).padStart(5, " ");
      elPoseTelemetry.textContent = `Y ${yaw >= 0 ? "+" : ""}${yDeg.trim()}° · P ${pitch >= 0 ? "+" : ""}${pDeg.trim()}°`;
    }

    // Autonomous blinking
    anim.blinkTimer -= dt;
    if (anim.blinkTimer <= 0 && anim.blinkPhase === 0 && live.mode !== "focus") {
      anim.blinkPhase = 0.001;
      anim.blinkTimer = 3.0 + Math.random() * 3.8;
    }
    let blinkMult = 1.0;
    if (anim.blinkPhase > 0) {
      anim.blinkPhase += dt * 7.8;
      if (anim.blinkPhase >= 1.0) {
        anim.blinkPhase = 0;
      } else {
        blinkMult = Math.max(0.08, Math.abs(Math.cos(anim.blinkPhase * Math.PI)));
      }
    }

    let baseOpen = 1.0;
    if (live.mode === "focus") baseOpen = 0.14;
    else if (live.mood === "drowsy") baseOpen = Math.max(0.24, 1.0 - live.drowsyLevel * 0.72);
    else if (live.mood === "focusing") baseOpen = 0.56; // autofocus squint!
    else if (live.mood === "attentive" || live.mood === "barge_in") baseOpen = 1.12; // attentive widen!

    const targetOpen = baseOpen * blinkMult;
    const eyeOpen = stepSpring(sp.eyeOpen, targetOpen, dt);
    const smile = stepSpring(
      sp.smile,
      (live.mood === "yes" || live.mood === "greet") ? 1.0 : live.mood === "speak" ? 0.18 : 0.0,
      dt
    );
    const frown = stepSpring(sp.frown, live.mood === "no" ? 1.0 : 0.0, dt);
    const asym = stepSpring(sp.asym, live.mood === "curious" ? 1.0 : live.mood === "attentive" ? 0.25 : 0.0, dt);
    // Syllable-driven speaking wave: pulses on REAL output_transcript.delta energy!
    const speakTarget = live.speaking
      ? (0.35 + 0.65 * Math.max(0.2, live.pulseEnergy))
      : (live.mood === "attentive" ? 0.22 + 0.25 * live.pulseEnergy : live.mood === "listen" ? 0.10 : 0.0);
    const wave = stepSpring(sp.wave, speakTarget, dt);

    // 3. Clear pure obsidian background
    ctx.fillStyle = "#07080b";
    ctx.fillRect(0, 0, W, H);

    // Subtle drafting grid dots (ultra-faint marked.dev / CAD vibe)
    if (live.mode !== "focus") {
      ctx.fillStyle = "rgba(232, 237, 245, 0.045)";
      const gStep = 36;
      for (let gx = gStep; gx < W; gx += gStep) {
        for (let gy = gStep; gy < H; gy += gStep) {
          ctx.fillRect(gx, gy, 1, 1);
        }
      }
    }

    if (scale < 0.05) {
      requestAnimationFrame(frame);
      return;
    }

    // 4. Render 3D Wireframe Superellipsoid Head
    const minDim = Math.min(W, H);
    const scalePx = minDim * 0.255 * scale;
    const cx = W * cxNorm;
    const cy = H * cyNorm;
    const R = rotMatrix(yaw, pitch, roll);

    const sil = computeSilhouette(R, cx, cy, scalePx, 144);
    const silPath = new Path2D();
    sil.forEach((p, idx) => (idx === 0 ? silPath.moveTo(p.x, p.y) : silPath.lineTo(p.x, p.y)));
    silPath.closePath();

    // Subtle matte fill inside 3D head silhouette
    ctx.save();
    const gradFill = ctx.createLinearGradient(cx, cy - scalePx, cx, cy + scalePx);
    gradFill.addColorStop(0, "rgba(18, 21, 28, 0.92)");
    gradFill.addColorStop(1, "rgba(10, 12, 16, 0.96)");
    ctx.fillStyle = gradFill;
    ctx.fill(silPath);

    // Crisp outer 1px wireframe contour
    ctx.strokeStyle = `rgba(218, 228, 242, ${(0.58 * anim.wireAlpha).toFixed(3)})`;
    ctx.lineWidth = 1.15;
    ctx.stroke(silPath);

    // Surgical colored bottom-edge specular rim (mirrors HiLight 8-LED color)
    ctx.beginPath();
    let startedRim = false;
    for (let i = 0; i < sil.length; i++) {
      const p = sil[i];
      if (p.dy > 0.42) {
        if (!startedRim) { ctx.moveTo(p.x, p.y); startedRim = true; }
        else ctx.lineTo(p.x, p.y);
      } else {
        startedRim = false;
      }
    }
    ctx.strokeStyle = `rgba(${rimRgb}, ${(0.88 * anim.wireAlpha).toFixed(3)})`;
    ctx.lineWidth = 1.75;
    ctx.shadowColor = `rgba(${rimRgb}, 0.65)`;
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.restore();

    // 5. Dashed 3D construction meridians clipped cleanly inside silPath
    const merColor = `rgba(188, 200, 218, ${(0.30 * anim.wireAlpha).toFixed(3)})`;
    drawMeridian(R, "vertical", cx, cy, scalePx, merColor, silPath);
    drawMeridian(R, "horizontal", cx, cy, scalePx, merColor, silPath);

    // 6. Crisp hollow center origin node (◦) at front pole (0, 0, 1)
    const [owx, owy, owz] = mulR(R, 0, 0, 1.0);
    if (owz > 0.1) {
      const [ox, oy] = projectWorld(owx, owy, owz, cx, cy, scalePx);
      ctx.save();
      ctx.beginPath();
      ctx.arc(ox, oy, 2.8, 0, Math.PI * 2);
      ctx.fillStyle = "#07080b";
      ctx.fill();
      ctx.strokeStyle = `rgba(220, 230, 244, ${(0.65 * anim.wireAlpha).toFixed(3)})`;
      ctx.lineWidth = 1.1;
      ctx.stroke();
      ctx.restore();

      // Thinking / Focusing reticle around center origin node
      if (live.mood === "focusing") {
        // Autofocus precision bracket pulsing cleanly around origin
        const rAf = scalePx * (0.14 + 0.03 * Math.sin(t * 14.0));
        ctx.save();
        ctx.strokeStyle = `rgba(${rimRgb}, 0.92)`;
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.arc(ox, oy, rAf, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      } else if (live.busy || live.mood === "think" || live.mood === "intent_desk") {
        ctx.save();
        ctx.strokeStyle = `rgba(${rimRgb}, 0.85)`;
        ctx.lineWidth = 1.3;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.arc(ox, oy, scalePx * 0.18, t * 3.5, t * 3.5 + Math.PI * 1.35);
        ctx.stroke();
        ctx.restore();
      }
    }

    // 7. Two 3D-projected CAD eyes (45° hatching + token-animated perimeter vector nodes!)
    const hatchColor = `rgba(198, 212, 232, ${(0.42 * anim.wireAlpha).toFixed(3)})`;
    const nodeColor = `rgba(218, 228, 244, ${(0.76 * anim.wireAlpha).toFixed(3)})`;
    const showTravel = (
      live.mood === "think" ||
      live.mood === "attentive" ||
      live.mood === "reading" ||
      live.mood === "intent_desk"
    ) && (live.pulseEnergy > 0.08 || live.busy);
    const activeVertexIdx = showTravel ? live.thinkCursor : -1;

    [-1, 1].forEach((side) => {
      const scaleMod = side === 1 ? 1.0 + asym * 0.16 : 1.0 - asym * 0.08;
      drawDraftingEye(R, side, cx, cy, scalePx, {
        openFactor: Math.max(0.08, eyeOpen),
        smile,
        frown,
        asym,
        scaleMod,
        strokeAlpha: anim.wireAlpha,
        nodeColor,
        hatchColor,
        rimRgb,
        travelIdx: activeVertexIdx
      });
    });

    // 8. Speaking state: subtle technical oscilloscope ticks along lower cube face
    if (wave > 0.05 && (live.mode === "dialogue" || live.mode === "board")) {
      ctx.save();
      const bars = 13;
      const halfSpan = 0.34;
      for (let i = 0; i < bars; i++) {
        const u = (i / (bars - 1)) * 2 - 1;
        const env = 1 - u * u;
        const hBar = (0.015 + env * 0.075 * Math.abs(Math.sin(t * 9.5 + i * 0.9))) * wave;
        const [sx1, sy1, sz1] = toSuperSurface(u * halfSpan, 0.56 - hBar, 1.0);
        const [sx2, sy2, sz2] = toSuperSurface(u * halfSpan, 0.56 + hBar, 1.0);
        const [w1x, w1y, w1z] = mulR(R, sx1, sy1, sz1);
        const [w2x, w2y, w2z] = mulR(R, sx2, sy2, sz2);
        if (w1z > 0.1) {
          const [p1x, p1y] = projectWorld(w1x, w1y, w1z, cx, cy, scalePx);
          const [p2x, p2y] = projectWorld(w2x, w2y, w2z, cx, cy, scalePx);
          ctx.strokeStyle = `rgba(${rimRgb}, ${(0.45 + 0.5 * env).toFixed(2)})`;
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.moveTo(p1x, p1y);
          ctx.lineTo(p2x, p2y);
          ctx.stroke();
        }
      }
      ctx.restore();
    }

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
})();
