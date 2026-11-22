import * as THREE from 'three';

const LAMP_PRESETS = {
  off: { color: 0xffcf87, intensity: 0.0, label: 'LAMP OFF' },
  warm: { color: 0xffcf82, intensity: 1.05, label: 'WARM LAMP · 2900K' },
  study: { color: 0xfff0d6, intensity: 1.32, label: 'STUDY MAX · 3800K' },
  cozy: { color: 0xff9a44, intensity: 0.68, label: 'COZY AMBER · 2300K' },
  sunset: { color: 0xff7b29, intensity: 0.0, label: 'SUNSET FADE' },
};

const DEMO_BOARDS = {
  definition: {
    mode: 'definition',
    title: 'QMAX · EFFUSIVITÉ THERMIQUE',
    items: [
      'Mesure la capacité d\'un matériau à absorber rapidement la chaleur.',
      'Formula: e = √(k · ρ · c)  [W·s½/(m²·K)]',
      'Example: Alu (24 000) >> Bois (450) → sensation froide',
      'Trap: À 20 °C, le métal et le bois ont la MÊME température !',
    ],
    note: 'Dis « c\'est bon » pour revenir sur ton cahier',
  },
  exercise: {
    mode: 'exercise',
    title: 'EXERCICE · ANALYSE FINANCIÈRE',
    items: [
      'Given: EBIT = 120 M€, impôt t = 25 %, D&A = 18 M€, CapEx = 28 M€',
      'Q1: Calcule le Free Cash Flow si ΔBFR = +5 M€.',
      'Q2: Que se passe-t-il si les stock-options doublent ?',
    ],
    note: 'HINT ▸ FCF = EBIT(1 - t) + D&A - CapEx - ΔBFR',
  },
  exam: {
    mode: 'exam',
    title: 'MINI-EXAM · MICROÉCONOMIE (5 PTS)',
    items: [
      'Q1 [2 pts]: Définis l\'équilibre concurrentiel E* (P*, Q*).',
      'Q2 [3 pts]: Effet d\'une taxe unitaire sur le surplus.',
    ],
    note: 'RUBRIC ▸ 2 pts graphique propre · 3 pts perte sèche',
  },
  checklist: {
    mode: 'checklist',
    title: 'PLAN DE RÉVISION · CE SOIR',
    items: [
      { text: 'Relire Chapitre 11 (Bénéfice ajusté)', done: true },
      { text: 'Résoudre TD n°4 — Effusivité Qmax', done: true },
      { text: 'Faire le mini-examen chronométré (5 min)', done: false },
      { text: 'Pause café / musique 5 min', done: false },
    ],
    note: '2 / 4 TERMINÉS · CONTINUE COMME ÇA !',
  },
  feedback: {
    mode: 'feedback',
    title: 'CORRECTION COPIE · Q1 & Q2',
    items: [
      'OK: Calcul exact : e_alu = √(575 910 000) ≈ 23 998 !',
      'OK: Bonne comparaison d\'ordre de grandeur (×50 vs bois).',
      'FIX: Précise l\'unité [W·s½ / (m²·K)] sur la ligne finale.',
      'NEXT: Prêt pour la question 3 sur la diffusivité ?',
    ],
    note: 'SCORE ESTIMÉ : 4.5 / 5 · EXCELLENT TRAVAIL',
  },
};

/**
 * Complete Apple ELEGNT Movement & Behavior Prototyping Engine
 * Drives the articulated DJI OM3 + Pixel 11 Pro OLED + 8-LED Volumetric HiLight Lamp.
 */
export class ElegntEngine {
  constructor(rig, desk, oled, sound) {
    this.rig = rig;
    this.desk = desk;
    this.oled = oled;
    this.sound = sound;

    // Core ELEGNT toggles
    this.movementStyle = 'elegnt'; // 'elegnt' | 'robotic'
    this.expressiveness = 'companion'; // 'focus' | 'companion' | 'playful'
    this.lampPreset = 'off'; // 'off' | 'warm' | 'study' | 'cozy' | 'sunset'
    this.followHandsLamp = true;
    this.multiFaceMode = false;
    this.musicMode = false;
    this.liveBridgeEnabled = false;

    // Live joint physics state
    this.curPan = 0;
    this.curPitch = 6;
    this.curRoll = 0;
    this.velPan = 0;
    this.velPitch = 0;
    this.velRoll = 0;
    this.prevVelPitch = 0;

    this.targetPan = 0;
    this.targetPitch = 6;
    this.targetRoll = 0;

    // Eye-lead delay buffer (moves OLED eyes 150ms BEFORE the mechanical gimbal!)
    this.eyeLeadTimer = 0;
    this.pendingPan = null;
    this.pendingPitch = null;
    this.eyeLookX = 0;
    this.eyeLookY = 0;
    this.proxScale = 1.0;

    // HiLight filament smooth state
    this.lampIntensity = 0;
    this.targetLampIntensity = 0;
    this.lampColor = new THREE.Color(0xffcf82);
    this.targetLampColor = new THREE.Color(0xffcf82);
    this.ringEffect = 'idle'; // idle | lamp | yes | no | curious | search | celebrate | nudge | beat
    this.ringEffectUntil = 0;
    this.beatPulse = 0;

    // Active choreography timeline
    this.time = 0;
    this.scenario = 'idle_face'; // idle_face | desk_lamp | show_tell | triadic | nudge | social | music
    this.choreo = null;
    this.reaction = null;
    this.speaking = false;
    this.statusText = 'COMPANION · READY';
    this.accentHex = '#60a5fa';
    this.expression = 'neutral';
    this.torquePct = 16;

    // Multi-face gaze alternator
    this._socialTarget = 'student';
    this._nextSocialSwap = 3.2;

    // Live hardware/Gemini WebSocket bridge
    this._liveWs = null;
    this.liveBridgeStatus = 'OFF';
    this.onLog = null;

    this.sound.onBeat = (strength) => {
      this.beatPulse = Math.max(this.beatPulse, strength);
    };
  }

  _log(msg) {
    if (this.onLog) this.onLog(msg);
  }

  _amp() {
    if (this.movementStyle === 'robotic') return 0.0;
    if (this.expressiveness === 'focus') return 0.55;
    if (this.expressiveness === 'playful') return 1.35;
    return 1.0;
  }

  setMovementStyle(style) {
    this.movementStyle = style === 'robotic' ? 'robotic' : 'elegnt';
    this._log(
      this.movementStyle === 'elegnt'
        ? 'Mode ELEGNT Expressif : Eye-Lead 150ms + courbes Pixar + fausse proxémique actifs'
        : 'Mode Robotique Fonctionnel : trajectoires linéaires directes (comparaison Apple ELEGNT)'
    );
  }

  setExpressiveness(level) {
    this.expressiveness = level;
    this.sound.enabled = level !== 'focus';
    this._log(`Curseur d'expressivité : ${level.toUpperCase()}`);
  }

  setBoardPreset(mode) {
    if (!mode || mode === 'none') {
      this.oled.setBoard({ mode: 'none' });
      this._log('Écran OLED Pixel remis en mode Visage 3D');
      return;
    }
    const board = DEMO_BOARDS[mode];
    if (board) {
      this.oled.setBoard(structuredClone(board));
      this.sound.playCue('card');
      this._log(`Carte OLED affichée : [${mode.toUpperCase()}] ${board.title}`);
    }
  }

  /**
   * Command a target pose with ELEGNT 150ms Eye-Lead anticipation!
   */
  _commandPose(panDeg, pitchDeg, { immediate = false } = {}) {
    const dPan = panDeg - this.curPan;
    const dPitch = pitchDeg - this.curPitch;

    if (!immediate && this.movementStyle === 'elegnt' && (Math.abs(dPan) > 14 || Math.abs(dPitch) > 12)) {
      // Eyes look FIRST toward where the gimbal is about to move!
      this.eyeLookX = Math.max(-1, Math.min(1, dPan / 55));
      this.eyeLookY = Math.max(-1, Math.min(1, -dPitch / 40));
      this.pendingPan = panDeg;
      this.pendingPitch = pitchDeg;
      this.eyeLeadTimer = 0.16;
    } else {
      this.pendingPan = null;
      this.pendingPitch = null;
      this.eyeLeadTimer = 0;
      this.targetPan = panDeg;
      this.targetPitch = pitchDeg;
    }
  }

  // ===========================================================================
  // SCENARIO 1: ELEGNT WARM DESK LAMP MODE (HiLight 8-LEDs + Follow-My-Hands!)
  // ===========================================================================
  setLampMode(preset = 'warm') {
    this.choreo = null;
    this.setMusicMode(false);
    this.oled.setNudge('');

    if (preset === 'off') {
      this.lampPreset = 'off';
      this.targetLampIntensity = 0.0;
      this.scenario = 'idle_face';
      this.expression = 'neutral';
      this.accentHex = '#60a5fa';
      this.statusText = 'FACE TRACKING · READY';
      this.ringEffect = 'idle';
      this.sound.playCue('lamp_off');
      const aim = this.rig.computeFrontAimAngles(this.desk.studentMarker.position);
      this._commandPose(aim.panDeg, aim.pitchDeg);
      this._log('Lampe HiLight éteinte → retour doux face à Zach');
      return;
    }

    if (preset === 'sunset') {
      this.lampPreset = 'sunset';
      this.targetLampColor.setHex(LAMP_PRESETS.sunset.color);
      this.targetLampIntensity = 0.0;
      this.statusText = 'SUNSET · EXTINCTION DOUCE';
      this.accentHex = '#ff944d';
      this.expression = 'sleepy';
      this.sound.playCue('lamp_off');
      this._log('Mode Sunset : fondu ambre chaleureux jusqu\'à extinction complète');
      return;
    }

    const cfg = LAMP_PRESETS[preset] || LAMP_PRESETS.warm;
    this.lampPreset = preset;
    this.scenario = 'desk_lamp';
    this.targetLampColor.setHex(cfg.color);
    this.targetLampIntensity = cfg.intensity;
    this.ringEffect = 'lamp';
    this.expression = 'lamp';
    this.accentHex = '#ffd088';
    this.statusText = cfg.label;
    this.oled.setBoard({ mode: 'none' });
    this.sound.playCue('lamp_on');

    const deskPt = this.desk.getActiveDeskTarget();
    const aim = this.rig.computeRearLampAimAngles(deskPt);
    this._commandPose(aim.panDeg, aim.pitchDeg);
    this._log(`Mode Lampe ELEGNT [${cfg.label}] — faisceau chaud braqué sur le cahier (déplace le cahier ou clique dessus !)`);
  }

  // ===========================================================================
  // SCENARIO 2: FULL SHOW & TELL LOOP (Desk 4K Read -> OLED Card -> Return!)
  // ===========================================================================
  runShowAndTell(boardMode = 'definition') {
    this.setMusicMode(false);
    this.oled.setNudge('');
    this.scenario = 'show_tell';
    this.choreo = {
      name: 'show_tell',
      t: 0,
      stage: 0,
      boardMode,
    };
    this._log('Chorégraphie Show & Tell lancée : Lecture 4K + Lampe → Eye-Lead → Fiche OLED → Retour cahier');
  }

  // ===========================================================================
  // SCENARIO 3: TRIADIC THINKING GAZE (Zach <-> Notebook <-> Zach)
  // ===========================================================================
  runTriadicThinking() {
    this.setMusicMode(false);
    this.oled.setNudge('');
    this.scenario = 'triadic';
    this.choreo = { name: 'triadic', t: 0, stage: 0 };
    this._log('Regard Triadique ELEGNT : coup d\'œil vers ta feuille pendant la réflexion puis retour eye-contact !');
  }

  // ===========================================================================
  // SCENARIO 4: SILENT PROACTIVE NUDGE (Apple ELEGNT non-intrusive prompt!)
  // ===========================================================================
  runSilentNudge() {
    this.setMusicMode(false);
    this.scenario = 'nudge';
    this.choreo = { name: 'nudge', t: 0, stage: 0 };
    this._log('Nudge silencieux ELEGNT : léger regard curieux + halo ambre + badge OLED discret sans couper ta concentration');
  }

  // ===========================================================================
  // SCENARIO 5: SOCIAL MULTI-FACE GAZE (Student + Guest/Jury alternation!)
  // ===========================================================================
  toggleMultiFace(forceState = null) {
    this.multiFaceMode = forceState !== null ? Boolean(forceState) : !this.multiFaceMode;
    this.desk.guestMarker.visible = this.multiFaceMode;
    if (this.multiFaceMode) {
      this.choreo = null;
      this.lampPreset = 'off';
      this.targetLampIntensity = 0;
      this.scenario = 'social';
      this._socialTarget = 'guest';
      this._nextSocialSwap = this.time + 2.8;
      this.statusText = 'SOCIAL · MULTI-VISAGES';
      this.accentHex = '#34d399';
      this.expression = 'happy';
      this.sound.playCue('yes');
      this._log('Mode Social Multi-Visages activé : Osmo alterne son regard entre Zach et l\'invité (déplace-les à la souris !)');
    } else {
      this.scenario = 'idle_face';
      this.statusText = 'COMPANION · READY';
      this.accentHex = '#60a5fa';
      this.expression = 'neutral';
      this._log('Mode Social Multi-Visages désactivé');
    }
    return this.multiFaceMode;
  }

  // ===========================================================================
  // SCENARIO 6: MUSIC BEAT MODE (Pan-only rhythmic sway — ZERO Pitch stress!)
  // ===========================================================================
  setMusicMode(enable) {
    this.musicMode = Boolean(enable);
    this.sound.toggleMusic(this.musicMode);
    if (this.musicMode) {
      this.choreo = null;
      this.lampPreset = 'off';
      this.targetLampIntensity = 0.25;
      this.targetLampColor.setHex(0xffb468);
      this.scenario = 'music';
      this.ringEffect = 'beat';
      this.expression = 'happy';
      this.accentHex = '#c084fc';
      this.statusText = 'MUSIC BEAT · PAN SWAY (92 BPM)';
      this.oled.setBoard({ mode: 'none' });
      this.oled.setNudge('');
      this._log('Mode Musique ELEGNT : balancement rythmé 100 % axe Pan (0 couple Pitch !) + anneau HiLight au tempo');
    } else if (this.scenario === 'music') {
      this.scenario = 'idle_face';
      this.targetLampIntensity = 0;
      this.ringEffect = 'idle';
      this.expression = 'neutral';
      this.accentHex = '#60a5fa';
      this.statusText = 'COMPANION · READY';
    }
  }

  // ===========================================================================
  // SCENARIO 7: EXPRESSIVE MICRO-REACTIONS (Nod / Doubt / Curious / Celebrate / Recoil)
  // ===========================================================================
  triggerReaction(kind) {
    this.oled.setNudge('');
    this.reaction = { kind, t: 0 };
    if (kind === 'nod_yes') {
      this.expression = 'happy';
      this.accentHex = '#34d399';
      this.ringEffect = 'yes';
      this.ringEffectUntil = this.time + 1.5;
      this.sound.playCue('yes');
      this._log('Micro-expression : Hochement OUI doux (≤ 44°/s) + flash émeraude HiLight');
    } else if (kind === 'curious') {
      this.expression = 'curious';
      this.accentHex = '#f59e0b';
      this.ringEffect = 'curious';
      this.ringEffectUntil = this.time + 1.8;
      this.sound.playCue('curious');
      this._log('Micro-expression : Inclinaison curieuse + yeux asymétriques + ambre');
    } else if (kind === 'doubt') {
      this.expression = 'doubt';
      this.accentHex = '#fb7185';
      this.ringEffect = 'no';
      this.ringEffectUntil = this.time + 1.6;
      this.sound.playCue('doubt');
      this._log('Micro-expression : Doute / correction douce (secousse Pan + affaissement léger)');
    } else if (kind === 'celebrate') {
      this.expression = 'happy';
      this.accentHex = '#34d399';
      this.ringEffect = 'celebrate';
      this.ringEffectUntil = this.time + 2.4;
      this.sound.playCue('celebrate');
      // Check off checklist if visible
      if (this.oled.board.mode === 'checklist' && Array.isArray(this.oled.board.items)) {
        this.oled.board.items.forEach((it) => {
          if (typeof it === 'object') it.done = true;
        });
        this.oled.board.note = '4 / 4 TERMINÉS · BRAVO ZACH !';
      }
      this._log('Célébration ELEGNT : double rebond Pan joyeux + arc émeraude HiLight (zéro stress Pitch !)');
    } else if (kind === 'recoil') {
      this.expression = 'surprised';
      this.accentHex = '#60a5fa';
      this.sound.playCue('recoil');
      this._log('Barge-in / Fausse Proxémique : recul expressif OLED (-12 % scale) + écoute instantanée');
    } else if (kind === 'lean_in') {
      this.expression = 'focused';
      this.accentHex = '#60a5fa';
      this._log('Fausse Proxémique : Osmo se penche attentivement (+15 % scale OLED + micro-tilt)');
    }
  }

  // ===========================================================================
  // OPTIONAL LIVE BRIDGE TO REAL OSMO FASTAPI BACKEND (ws://127.0.0.1:8765/ws)
  // ===========================================================================
  toggleLiveBridge(enable) {
    this.liveBridgeEnabled = Boolean(enable);
    if (!this.liveBridgeEnabled) {
      if (this._liveWs) {
        try { this._liveWs.close(); } catch (_) {}
        this._liveWs = null;
      }
      this.liveBridgeStatus = 'OFF';
      this._log('Pont Live Osmo/Gemini déconnecté (retour 100 % simulateur autonome)');
      return;
    }

    const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${wsProto}//${location.host}/ws`;
    this.liveBridgeStatus = 'CONNECTING';
    try {
      const ws = new WebSocket(url);
      this._liveWs = ws;
      ws.onopen = () => {
        this.liveBridgeStatus = 'LIVE';
        this._log('Pont Live connecté à osmo-app (/ws) — les actions Gemini pilotent maintenant le jumeau 3D !');
      };
      ws.onmessage = (ev) => {
        try {
          const s = JSON.parse(ev.data);
          this._applyRealOsmoSnapshot(s);
        } catch (_) {}
      };
      ws.onerror = () => {
        this.liveBridgeStatus = 'OFFLINE';
      };
      ws.onclose = () => {
        if (this.liveBridgeEnabled) {
          this.liveBridgeStatus = 'OFFLINE';
          this.liveBridgeEnabled = false;
          this._log('Serveur osmo-app non détecté sur :8765 — retour en mode simulateur autonome');
        }
      };
    } catch (_) {
      this.liveBridgeStatus = 'OFFLINE';
      this.liveBridgeEnabled = false;
    }
  }

  _applyRealOsmoSnapshot(s) {
    if (s.screen && s.screen.mode) {
      this.oled.setBoard(s.screen);
    }
    if (s.voice) {
      this.speaking = Boolean(s.voice.speaking);
      if (s.voice.busy) {
        this.expression = 'curious';
        this.statusText = `LIVE · ${s.voice.brain_model || 'GEMINI'}`;
      } else if (s.voice.speaking) {
        this.expression = 'happy';
        this.statusText = 'LIVE · SPEAKING';
      }
    }
    if (s.camera && s.camera.facing === 'back') {
      this.scenario = 'desk_lamp';
      this.targetLampIntensity = 1.05;
      const aim = this.rig.computeRearLampAimAngles(this.desk.getActiveDeskTarget());
      this._commandPose(aim.panDeg, aim.pitchDeg, { immediate: true });
    } else if (s.osmo && s.osmo.attitude && s.osmo.connected) {
      this.targetPitch = Math.max(-25, Math.min(58, s.osmo.attitude.pitch || 0));
    }
  }

  // ===========================================================================
  // MAIN 60-120 FPS UPDATE LOOP
  // ===========================================================================
  update(dt) {
    const safeDt = Math.min(0.05, Math.max(0.001, dt));
    this.time += safeDt;

    // 1. Update choreographies & continuous tracking scenarios
    this._updateChoreography(safeDt);

    if (!this.choreo && !this.liveBridgeEnabled) {
      if (this.scenario === 'desk_lamp' && this.followHandsLamp) {
        const aim = this.rig.computeRearLampAimAngles(this.desk.getActiveDeskTarget());
        this.targetPan = aim.panDeg;
        this.targetPitch = aim.pitchDeg;
      } else if (this.scenario === 'social' && this.multiFaceMode) {
        if (this.time >= this._nextSocialSwap) {
          this._socialTarget = this._socialTarget === 'student' ? 'guest' : 'student';
          this._nextSocialSwap = this.time + (2.6 + Math.random() * 1.8);
          const pt =
            this._socialTarget === 'guest'
              ? this.desk.guestMarker.position
              : this.desk.studentMarker.position;
          const aim = this.rig.computeFrontAimAngles(pt);
          this._commandPose(aim.panDeg, aim.pitchDeg);
        }
      } else if (this.scenario === 'music') {
        const aim = this.rig.computeFrontAimAngles(this.desk.studentMarker.position);
        const phase = this.time * (this.sound.bpm / 60) * Math.PI;
        // Pure Pan rhythmic sway (ZERO Pitch stress!)
        this.targetPan = aim.panDeg + Math.sin(phase) * 14.0 * Math.max(0.5, this._amp());
        this.targetPitch = aim.pitchDeg + Math.abs(Math.sin(phase)) * 2.5;
        this.targetRoll = Math.sin(phase) * 3.5;
      } else if (this.scenario === 'idle_face') {
        const aim = this.rig.computeFrontAimAngles(this.desk.studentMarker.position);
        this.targetPan = aim.panDeg;
        this.targetPitch = aim.pitchDeg;
        this.targetRoll = 0;
      }
    }

    // 2. Process 150ms Eye-Lead timer
    if (this.eyeLeadTimer > 0) {
      this.eyeLeadTimer -= safeDt;
      if (this.eyeLeadTimer <= 0 && this.pendingPan !== null) {
        this.targetPan = this.pendingPan;
        this.targetPitch = this.pendingPitch;
        this.pendingPan = null;
        this.pendingPitch = null;
      }
    }

    // 3. Apply additive expressive micro-reactions
    let addPan = 0;
    let addPitch = 0;
    let addRoll = 0;
    let targetScale = 1.0;

    if (this.reaction) {
      this.reaction.t += safeDt;
      const rt = this.reaction.t;
      const amp = Math.max(0.45, this._amp());

      if (this.reaction.kind === 'nod_yes') {
        if (rt < 1.15) {
          addPitch = -Math.sin(rt * Math.PI * 3.5) * 9.0 * amp * (1 - rt / 1.15);
        } else {
          this.reaction = null;
        }
      } else if (this.reaction.kind === 'curious') {
        if (rt < 1.55) {
          const env = Math.sin((rt / 1.55) * Math.PI);
          addPan = 11.0 * env * amp;
          addPitch = 4.5 * env * amp;
          addRoll = 7.0 * env * amp;
          this.eyeLookX = 0.45 * env;
        } else {
          this.reaction = null;
        }
      } else if (this.reaction.kind === 'doubt') {
        if (rt < 1.35) {
          const env = Math.sin((rt / 1.35) * Math.PI);
          addPan = Math.sin(rt * Math.PI * 5.0) * 12.0 * env * amp;
          addPitch = -5.5 * env * amp;
        } else {
          this.reaction = null;
        }
      } else if (this.reaction.kind === 'celebrate') {
        if (rt < 1.9) {
          const env = Math.sin((rt / 1.9) * Math.PI);
          // Joyful horizontal Pan bounce — zero Pitch stress!
          addPan = Math.sin(rt * Math.PI * 4.5) * 18.0 * env * amp;
          addRoll = Math.sin(rt * Math.PI * 4.5) * 5.0 * env * amp;
          targetScale = 1.0 + 0.1 * env;
        } else {
          this.reaction = null;
        }
      } else if (this.reaction.kind === 'recoil') {
        if (rt < 1.1) {
          const env = Math.sin((rt / 1.1) * Math.PI);
          targetScale = 1.0 - 0.14 * env;
          addPitch = 3.5 * env;
        } else {
          this.reaction = null;
          this.expression = 'neutral';
        }
      } else if (this.reaction.kind === 'lean_in') {
        if (rt < 2.2) {
          const env = Math.sin((rt / 2.2) * Math.PI);
          targetScale = 1.0 + 0.16 * env;
          addPitch = -2.5 * env;
        } else {
          this.reaction = null;
          this.expression = 'neutral';
        }
      }
    }

    // Subtle idle life breathing in ELEGNT mode (0 in robotic mode!)
    if (this.movementStyle === 'elegnt' && !this.musicMode) {
      const a = this._amp();
      addPitch += Math.sin(this.time * 1.45) * 0.65 * a;
      addPan += Math.sin(this.time * 0.72) * 0.9 * a;
    }

    const goalPan = this.targetPan + addPan;
    const goalPitch = this.targetPitch + addPitch;
    const goalRoll = this.targetRoll + addRoll;

    // 4. Integrate Pan / Pitch / Roll joints (ELEGNT critically-damped spring vs Robotic linear)
    this.prevVelPitch = this.velPitch;
    if (this.movementStyle === 'robotic') {
      const linStep = (cur, goal, maxSpeed) => {
        const diff = goal - cur;
        const step = Math.sign(diff) * Math.min(Math.abs(diff), maxSpeed * safeDt);
        return { next: cur + step, vel: step / safeDt };
      };
      const pRes = linStep(this.curPan, goalPan, 75);
      const tRes = linStep(this.curPitch, goalPitch, 48);
      const rRes = linStep(this.curRoll, goalRoll, 35);
      this.curPan = pRes.next;
      this.velPan = pRes.vel;
      this.curPitch = tRes.next;
      this.velPitch = tRes.vel;
      this.curRoll = rRes.next;
      this.velRoll = rRes.vel;
      this.eyeLookX = 0;
      this.eyeLookY = 0;
      this.proxScale = 1.0;
    } else {
      // Critically damped Pixar-like spring with realistic DJI OM3 95°/s speed cap!
      const springStep = (cur, vel, goal, stiffness, damping, maxDps) => {
        const acc = (goal - cur) * stiffness - vel * damping;
        let nextVel = vel + acc * safeDt;
        nextVel = Math.max(-maxDps, Math.min(maxDps, nextVel));
        return { next: cur + nextVel * safeDt, vel: nextVel };
      };
      const spPan = springStep(this.curPan, this.velPan, goalPan, 28, 9.4, 95);
      const spPitch = springStep(this.curPitch, this.velPitch, goalPitch, 22, 8.8, 65);
      const spRoll = springStep(this.curRoll, this.velRoll, goalRoll, 26, 9.2, 40);
      this.curPan = spPan.next;
      this.velPan = spPan.vel;
      this.curPitch = spPitch.next;
      this.velPitch = spPitch.vel;
      this.curRoll = spRoll.next;
      this.velRoll = spRoll.vel;

      // Natural eye-follow during motion when not in Eye-Lead anticipation
      if (this.eyeLeadTimer <= 0 && !this.reaction) {
        const leadX = Math.max(-0.85, Math.min(0.85, this.velPan / 55));
        const leadY = Math.max(-0.7, Math.min(0.7, -this.velPitch / 45));
        this.eyeLookX += (leadX - this.eyeLookX) * Math.min(1, safeDt * 10);
        this.eyeLookY += (leadY - this.eyeLookY) * Math.min(1, safeDt * 10);
      }
      this.proxScale += (targetScale - this.proxScale) * Math.min(1, safeDt * 8);
    }

    this.rig.setJoints(this.curPan, this.curPitch, this.curRoll);
    const pitchAccel = (this.velPitch - this.prevVelPitch) / safeDt;
    this.torquePct = this.rig.estimatePitchTorquePct(this.velPitch, pitchAccel);

    // Handle LED reflects OM3 torque / state honestly!
    if (this.torquePct >= 75) {
      this.rig.setHandleLed(0xf59e0b, 2.6);
    } else if (this.lampIntensity > 0.2) {
      this.rig.setHandleLed(0xffcf82, 2.2);
    } else {
      this.rig.setHandleLed(0x34d399, 2.0);
    }

    // 5. Update 8-LED HiLight Ring + Warm Filament Lamp Ramp!
    this._updateHiLight(safeDt);

    // 6. Update Live 1024x512 OLED Front Screen
    this.oled.update(safeDt, {
      eyeLookX: this.eyeLookX,
      eyeLookY: this.eyeLookY,
      proxScale: this.proxScale,
      expression: this.expression,
      accentHex: this.accentHex,
      statusLabel: this.statusText,
      speaking: this.speaking,
      lampIntensity: this.lampIntensity,
    });
  }

  _updateChoreography(dt) {
    if (!this.choreo) return;
    this.choreo.t += dt;
    const t = this.choreo.t;

    if (this.choreo.name === 'show_tell') {
      if (this.choreo.stage === 0) {
        // Stage 0: Swing to notebook + Warm Study Lamp ON (4K scan)
        this.choreo.stage = 1;
        this.oled.setBoard({ mode: 'none' });
        this.lampPreset = 'study';
        this.targetLampColor.setHex(LAMP_PRESETS.study.color);
        this.targetLampIntensity = 1.25;
        this.ringEffect = 'lamp';
        this.expression = 'lamp';
        this.accentHex = '#ffd088';
        this.statusText = 'READING 4K · GEMINI HYDROGEN';
        this.speaking = true;
        this.sound.playCue('lamp_on');
        const aim = this.rig.computeRearLampAimAngles(this.desk.getActiveDeskTarget());
        this._commandPose(aim.panDeg, aim.pitchDeg);
      } else if (this.choreo.stage === 1 && t > 2.9) {
        // Stage 1: Eye-Lead + Pivot back to Student + Push OLED Study Board!
        this.choreo.stage = 2;
        this.targetLampIntensity = 0.0;
        this.lampPreset = 'off';
        this.ringEffect = 'yes';
        this.ringEffectUntil = this.time + 1.8;
        this.expression = 'happy';
        this.accentHex = '#60a5fa';
        this.statusText = 'SHOW & TELL · OLED BOARD';
        this.speaking = true;
        const aim = this.rig.computeFrontAimAngles(this.desk.studentMarker.position);
        this._commandPose(aim.panDeg, aim.pitchDeg);
        this.setBoardPreset(this.choreo.boardMode || 'definition');
      } else if (this.choreo.stage === 2 && t > 7.4) {
        // Stage 2: Student says "Got it, I noted it!" -> Smooth 1-hop return with Warm Lamp!
        this.choreo.stage = 3;
        this.speaking = false;
        this.oled.setBoard({ mode: 'none' });
        this.setLampMode('warm');
        this.choreo = null;
        this._log('Show & Tell terminé : retour 1-hop sur la feuille en mode Lampe Chaude !');
      }
    } else if (this.choreo.name === 'triadic') {
      const faceAim = this.rig.computeFrontAimAngles(this.desk.studentMarker.position);
      if (this.choreo.stage === 0) {
        this.choreo.stage = 1;
        this.expression = 'curious';
        this.accentHex = '#f59e0b';
        this.ringEffect = 'curious';
        this.ringEffectUntil = this.time + 3.0;
        this.statusText = 'THINKING Δ · TRIADIC GAZE';
        this.sound.playCue('curious');
        // Glance subtly down-left toward the notebook while thinking!
        this._commandPose(faceAim.panDeg - 18, faceAim.pitchDeg - 10);
        this.eyeLookX = -0.7;
        this.eyeLookY = 0.55;
      } else if (this.choreo.stage === 1 && t > 2.0) {
        // Insight! Snap eye contact cleanly back to Zach with confident nod!
        this.choreo.stage = 2;
        this.expression = 'happy';
        this.accentHex = '#34d399';
        this.statusText = 'INSIGHT · EXPLAINING';
        this.speaking = true;
        this._commandPose(faceAim.panDeg, faceAim.pitchDeg);
        this.triggerReaction('nod_yes');
      } else if (this.choreo.stage === 2 && t > 4.2) {
        this.speaking = false;
        this.expression = 'neutral';
        this.accentHex = '#60a5fa';
        this.statusText = 'COMPANION · READY';
        this.choreo = null;
      }
    } else if (this.choreo.name === 'nudge') {
      const faceAim = this.rig.computeFrontAimAngles(this.desk.studentMarker.position);
      if (this.choreo.stage === 0) {
        this.choreo.stage = 1;
        this.lampPreset = 'off';
        this.targetLampIntensity = 0.35;
        this.targetLampColor.setHex(0xffb85c);
        this.ringEffect = 'nudge';
        this.expression = 'curious';
        this.accentHex = '#ffcb7d';
        this.statusText = 'SILENT NUDGE · NON-INTRUSIVE';
        this.oled.setBoard({ mode: 'none' });
        this.oled.setNudge('Un indice sur la Q2 ? (ou pause 5 min)');
        this.sound.playCue('nudge');
        this._commandPose(faceAim.panDeg + 9, faceAim.pitchDeg - 3);
        this.targetRoll = 5;
      } else if (this.choreo.stage === 1 && t > 4.6) {
        this.targetLampIntensity = 0.0;
        this.targetRoll = 0;
        this.ringEffect = 'idle';
        this.expression = 'neutral';
        this.accentHex = '#60a5fa';
        this.statusText = 'COMPANION · READY';
        this.oled.setNudge('');
        this.scenario = 'idle_face';
        this.choreo = null;
      }
    }
  }

  _updateHiLight(dt) {
    // Filament ramp speed: organic 0.9s warm fade in ELEGNT mode vs instant harsh snap in Robotic mode!
    const rampRate =
      this.lampPreset === 'sunset'
        ? 0.42
        : this.movementStyle === 'robotic'
          ? 24.0
          : this.targetLampIntensity > this.lampIntensity
            ? 1.85
            : 2.6;
    this.lampIntensity += (this.targetLampIntensity - this.lampIntensity) * Math.min(1, dt * rampRate);
    this.lampColor.lerp(this.targetLampColor, Math.min(1, dt * 3.0));

    // Expire transient ring reactions cleanly
    if (this.ringEffectUntil > 0 && this.time > this.ringEffectUntil) {
      this.ringEffectUntil = 0;
      this.ringEffect = this.lampIntensity > 0.1 ? 'lamp' : this.musicMode ? 'beat' : 'idle';
    }
    this.beatPulse = Math.max(0, this.beatPulse - dt * 3.2);

    const colorsHex = new Array(8);
    const intensities = new Array(8);
    let beamColorHex = this.lampColor.getHex();
    let beamIntensity = this.lampIntensity;

    if (this.ringEffect === 'yes') {
      for (let i = 0; i < 8; i++) {
        colorsHex[i] = 0x34d399;
        intensities[i] = 0.85 + Math.sin(this.time * 10 + i * 0.7) * 0.25;
      }
      beamColorHex = 0x34d399;
      beamIntensity = Math.max(beamIntensity, 0.28);
    } else if (this.ringEffect === 'no') {
      for (let i = 0; i < 8; i++) {
        colorsHex[i] = 0xfb7185;
        intensities[i] = 0.75 + Math.sin(this.time * 14) * 0.25;
      }
      beamColorHex = 0xfb7185;
      beamIntensity = Math.max(beamIntensity, 0.22);
    } else if (this.ringEffect === 'curious' || this.ringEffect === 'nudge') {
      const wave = (this.time * 3.2) % 8;
      for (let i = 0; i < 8; i++) {
        const d = Math.min(Math.abs(i - wave), 8 - Math.abs(i - wave));
        colorsHex[i] = 0xffb84d;
        intensities[i] = 0.3 + Math.max(0, 1 - d * 0.55) * 0.85;
      }
      beamColorHex = 0xffb84d;
      beamIntensity = Math.max(beamIntensity, 0.32);
    } else if (this.ringEffect === 'celebrate') {
      const palette = [0x34d399, 0x60a5fa, 0xffcf82, 0xa78bfa];
      for (let i = 0; i < 8; i++) {
        const idx = (i + Math.floor(this.time * 12)) % palette.length;
        colorsHex[i] = palette[idx];
        intensities[i] = 1.15;
      }
      beamColorHex = 0x34d399;
      beamIntensity = Math.max(beamIntensity, 0.42);
    } else if (this.ringEffect === 'beat') {
      const head = (this.time * 5.0) % 8;
      for (let i = 0; i < 8; i++) {
        const d = Math.min(Math.abs(i - head), 8 - Math.abs(i - head));
        colorsHex[i] = i % 2 === 0 ? 0xffb86c : 0xc084fc;
        intensities[i] = 0.28 + this.beatPulse * 0.95 + Math.max(0, 1 - d * 0.6) * 0.4;
      }
      beamColorHex = 0xffb86c;
      beamIntensity = 0.22 + this.beatPulse * 0.35;
    } else if (this.lampIntensity > 0.03) {
      // Warm tungsten desk lamp with subtle directional LED bias + filament warmth shift!
      const warmShift = new THREE.Color(0xff8e3c).lerp(this.lampColor, Math.min(1, this.lampIntensity));
      beamColorHex = warmShift.getHex();
      const breath = this.movementStyle === 'elegnt' ? 1.0 + Math.sin(this.time * 1.9) * 0.035 : 1.0;
      const dirBias = Math.max(-1, Math.min(1, this.velPan / 45));
      for (let i = 0; i < 8; i++) {
        const ang = (i / 8) * Math.PI * 2;
        const directional = 1.0 + Math.cos(ang) * dirBias * 0.22;
        colorsHex[i] = beamColorHex;
        intensities[i] = this.lampIntensity * breath * directional;
      }
      beamIntensity = this.lampIntensity * breath;
    } else {
      // Subtle living standby breath on rear ring
      const b = this.movementStyle === 'elegnt' ? 0.14 + Math.sin(this.time * 1.8) * 0.06 : 0.04;
      for (let i = 0; i < 8; i++) {
        colorsHex[i] = 0x60a5fa;
        intensities[i] = b;
      }
      beamColorHex = 0x60a5fa;
      beamIntensity = 0.0;
    }

    this.rig.updateHiLight({
      colorsHex,
      intensities,
      beamColorHex,
      beamIntensity,
      time: this.time,
    });
  }
}
