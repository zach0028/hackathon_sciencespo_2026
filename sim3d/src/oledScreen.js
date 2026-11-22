import * as THREE from 'three';

/**
 * Live 1024x512 OLED CanvasTexture rendered directly onto the 3D Pixel 11 Pro front screen.
 * Mirrors Osmo's authentic 120 Hz analytical eyes + Eye-Lead + Fake Proxemics + 6 study boards!
 */
export class OledScreenRenderer {
  constructor() {
    this.width = 1024;
    this.height = 512;
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.width;
    this.canvas.height = this.height;
    this.ctx = this.canvas.getContext('2d');

    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.anisotropy = 8;

    this.blinkPhase = 0;
    this.nextBlinkAt = 2.4;
    this.time = 0;
    this.smoothLookX = 0;
    this.smoothLookY = 0;
    this.smoothScale = 1.0;
    this.smoothSpeak = 0;

    this.board = { mode: 'none', title: '', items: [], note: '' };
    this.nudgeText = '';
    this.statusLabel = 'COMPANION · READY';
    this.accentHex = '#60a5fa';
    this.expression = 'neutral'; // neutral | happy | curious | focused | doubt | lamp | surprised | sleepy
    this.speaking = false;
  }

  setBoard(board) {
    this.board = board || { mode: 'none', title: '', items: [], note: '' };
  }

  setNudge(text) {
    this.nudgeText = text || '';
  }

  update(dt, params = {}) {
    this.time += dt;
    const {
      eyeLookX = 0,
      eyeLookY = 0,
      proxScale = 1.0,
      expression = this.expression,
      accentHex = this.accentHex,
      statusLabel = this.statusLabel,
      speaking = false,
      lampIntensity = 0,
    } = params;

    this.expression = expression;
    this.accentHex = accentHex;
    this.statusLabel = statusLabel;
    this.speaking = speaking;

    const k = Math.min(1, dt * 14);
    this.smoothLookX += (eyeLookX - this.smoothLookX) * k;
    this.smoothLookY += (eyeLookY - this.smoothLookY) * k;
    this.smoothScale += (proxScale - this.smoothScale) * Math.min(1, dt * 7);
    const speakTarget = speaking ? 0.55 + Math.sin(this.time * 18) * 0.35 + Math.cos(this.time * 11) * 0.2 : 0;
    this.smoothSpeak += (speakTarget - this.smoothSpeak) * Math.min(1, dt * 16);

    // Organic blink timer
    if (this.time >= this.nextBlinkAt) {
      this.blinkPhase = 1.0;
      this.nextBlinkAt = this.time + 2.6 + Math.random() * 3.4;
    }
    if (this.blinkPhase > 0) {
      this.blinkPhase = Math.max(0, this.blinkPhase - dt * 6.5);
    }

    this._draw(lampIntensity);
    this.texture.needsUpdate = true;
  }

  _draw(lampIntensity) {
    const ctx = this.ctx;
    const W = this.width;
    const H = this.height;

    // Pure OLED obsidian background
    ctx.fillStyle = '#04060a';
    ctx.fillRect(0, 0, W, H);

    // Subtle CAD technical grid
    ctx.strokeStyle = 'rgba(148, 178, 224, 0.055)';
    ctx.lineWidth = 1;
    for (let x = 32; x < W; x += 48) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    }
    for (let y = 32; y < H; y += 48) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }

    const hasBoard = this.board && this.board.mode && this.board.mode !== 'none';

    if (hasBoard) {
      // Authentic 22% mini-head rail + 78% crisp study whiteboard
      this._drawHead(ctx, 114, H * 0.52, 0.38 * this.smoothScale, lampIntensity);
      this._drawStudyBoard(ctx, 232, 34, W - 262, H - 68);
    } else {
      // Fullscreen 3D analytical face with breathing + Fake Proxemics scale
      const breath = 1 + Math.sin(this.time * 2.1) * 0.014;
      this._drawHead(ctx, W * 0.5, H * 0.5, this.smoothScale * breath, lampIntensity);

      // Discreet ELEGNT silent nudge pill (bottom center)
      if (this.nudgeText) {
        this._drawNudgePill(ctx, W * 0.5, H - 56, this.nudgeText);
      }
    }

    // Minimal CAD corner status bar
    ctx.fillStyle = 'rgba(226, 236, 250, 0.45)';
    ctx.font = '600 16px "SF Mono", Menlo, monospace';
    ctx.fillText('OSMO // OLED 120HZ', 28, 30);

    ctx.textAlign = 'right';
    ctx.fillStyle = this.accentHex;
    ctx.fillText(this.statusLabel.toUpperCase(), W - 28, 30);
    ctx.textAlign = 'left';

    // Corner ticks
    ctx.strokeStyle = 'rgba(226, 236, 250, 0.22)';
    ctx.lineWidth = 2;
    const m = 14;
    const t = 14;
    [
      [m, m, 1, 1],
      [W - m, m, -1, 1],
      [m, H - m, 1, -1],
      [W - m, H - m, -1, -1],
    ].forEach(([cx, cy, dx, dy]) => {
      ctx.beginPath();
      ctx.moveTo(cx + dx * t, cy);
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx, cy + dy * t);
      ctx.stroke();
    });
  }

  _drawHead(ctx, cx, cy, scale, lampIntensity) {
    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(scale, scale);

    // Ambient face aura glow
    const glowColor = lampIntensity > 0.2 ? '#ffcb7d' : this.accentHex;
    const aura = ctx.createRadialGradient(0, 0, 40, 0, 0, 225);
    aura.addColorStop(0, this._alphaHex(glowColor, 0.22 + lampIntensity * 0.14));
    aura.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = aura;
    ctx.beginPath();
    ctx.arc(0, 0, 225, 0, Math.PI * 2);
    ctx.fill();

    // Superellipsoid wireframe CAD shell silhouette
    const headRx = 185;
    const headRy = 132;
    ctx.strokeStyle = this._alphaHex(glowColor, 0.26);
    ctx.lineWidth = 2;
    this._roundedRect(ctx, -headRx, -headRy, headRx * 2, headRy * 2, 68);
    ctx.stroke();

    // Inner contour ring
    ctx.strokeStyle = 'rgba(220, 234, 255, 0.12)';
    ctx.lineWidth = 1.5;
    this._roundedRect(ctx, -headRx + 14, -headRy + 14, (headRx - 14) * 2, (headRy - 14) * 2, 58);
    ctx.stroke();

    // Eye offsets driven by Eye-Lead gaze vector!
    const lookPxX = this.smoothLookX * 48;
    const lookPxY = this.smoothLookY * 32;

    const blinkClose = Math.sin(this.blinkPhase * Math.PI);
    let eyeH = 88 * (1 - blinkClose * 0.92);
    let eyeW = 54;
    let leftTilt = 0;
    let rightTilt = 0;
    let leftScale = 1;
    let rightScale = 1;

    if (this.expression === 'happy') {
      eyeH *= 0.86;
    } else if (this.expression === 'curious') {
      leftScale = 0.88;
      rightScale = 1.14;
      leftTilt = -0.09;
      rightTilt = 0.12;
    } else if (this.expression === 'focused' || this.expression === 'lamp') {
      eyeH *= 0.78;
    } else if (this.expression === 'surprised') {
      eyeH *= 1.18;
      eyeW *= 1.12;
    } else if (this.expression === 'doubt') {
      eyeH *= 0.66;
      leftTilt = 0.14;
      rightTilt = -0.08;
    } else if (this.expression === 'sleepy') {
      eyeH *= 0.42;
    }

    const eyeColor = lampIntensity > 0.25 ? '#ffe3b3' : '#edf4ff';
    const drawEye = (ox, oy, tilt, s) => {
      ctx.save();
      ctx.translate(ox + lookPxX, oy + lookPxY);
      ctx.rotate(tilt);
      ctx.scale(s, s);

      const w = eyeW;
      const h = Math.max(7, eyeH);

      // Soft bloom behind eye
      ctx.shadowColor = glowColor;
      ctx.shadowBlur = 24;
      ctx.fillStyle = eyeColor;

      if (this.expression === 'happy' && blinkClose < 0.2) {
        // Warm smiling crescent arc!
        ctx.strokeStyle = eyeColor;
        ctx.lineWidth = 14;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.arc(0, 10, 30, Math.PI * 1.15, Math.PI * 1.85);
        ctx.stroke();
      } else {
        this._roundedRect(ctx, -w / 2, -h / 2, w, h, Math.min(w, h) * 0.46);
        ctx.fill();

        // Subtle corneal highlight dot
        ctx.shadowBlur = 0;
        ctx.fillStyle = 'rgba(255,255,255,0.85)';
        ctx.beginPath();
        ctx.arc(-w * 0.18 + this.smoothLookX * 5, -h * 0.24 + this.smoothLookY * 4, 6, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    };

    drawEye(-66, -4, leftTilt, leftScale);
    drawEye(+66, -4, rightTilt, rightScale);

    // Speaking voice waveform bar underneath eyes
    if (this.smoothSpeak > 0.04) {
      const bars = 9;
      const gap = 14;
      const startX = -((bars - 1) * gap) / 2;
      ctx.fillStyle = glowColor;
      for (let i = 0; i < bars; i++) {
        const env = Math.sin(((i + 1) / (bars + 1)) * Math.PI);
        const bh = 5 + this.smoothSpeak * env * (18 + Math.sin(this.time * 22 + i * 1.3) * 10);
        this._roundedRect(ctx, startX + i * gap - 3, 78 - bh / 2, 6, bh, 3);
        ctx.fill();
      }
    }

    ctx.restore();
  }

  _drawStudyBoard(ctx, x, y, w, h) {
    const modeColors = {
      definition: '#60a5fa',
      concept: '#818cf8',
      exercise: '#34d399',
      exam: '#f59e0b',
      checklist: '#2dd4bf',
      feedback: '#34d399',
    };
    const accent = modeColors[this.board.mode] || '#60a5fa';

    // Card frame
    ctx.fillStyle = 'rgba(13, 18, 28, 0.96)';
    ctx.strokeStyle = this._alphaHex(accent, 0.5);
    ctx.lineWidth = 2;
    this._roundedRect(ctx, x, y, w, h, 22);
    ctx.fill();
    ctx.stroke();

    // Header pill + title
    ctx.fillStyle = this._alphaHex(accent, 0.2);
    this._roundedRect(ctx, x + 22, y + 20, 128, 30, 8);
    ctx.fill();
    ctx.fillStyle = accent;
    ctx.font = '700 15px "SF Mono", Menlo, monospace';
    ctx.fillText(String(this.board.mode || 'CARD').toUpperCase(), x + 36, y + 40);

    ctx.fillStyle = '#f1f5f9';
    ctx.font = '700 24px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.fillText(this.board.title || 'STUDY BOARD', x + 166, y + 43);

    // Divider
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.18)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x + 22, y + 66);
    ctx.lineTo(x + w - 22, y + 66);
    ctx.stroke();

    // Items
    const items = this.board.items || [];
    let iy = y + 106;
    items.slice(0, 5).forEach((item, idx) => {
      const txt = typeof item === 'string' ? item : item.text || '';
      const done = typeof item === 'object' && item.done;
      const colonIdx = txt.indexOf(':');

      if (this.board.mode === 'checklist') {
        ctx.strokeStyle = done ? '#34d399' : 'rgba(148,163,184,0.5)';
        ctx.lineWidth = 2.5;
        this._roundedRect(ctx, x + 24, iy - 20, 24, 24, 6);
        ctx.stroke();
        if (done) {
          ctx.fillStyle = '#34d399';
          ctx.fillRect(x + 29, iy - 15, 14, 14);
        }
        ctx.fillStyle = done ? 'rgba(203,213,225,0.55)' : '#e2e8f0';
        ctx.font = '500 22px -apple-system, sans-serif';
        ctx.fillText(txt.slice(0, 56), x + 62, iy);
      } else if (colonIdx > 0 && colonIdx <= 14) {
        const tag = txt.slice(0, colonIdx).trim();
        const rest = txt.slice(colonIdx + 1).trim();
        const tagColor =
          tag.toUpperCase() === 'OK'
            ? '#34d399'
            : tag.toUpperCase() === 'FIX' || tag.toUpperCase() === 'TRAP'
              ? '#fb7185'
              : accent;
        ctx.fillStyle = this._alphaHex(tagColor, 0.2);
        const tw = Math.max(82, tag.length * 11 + 24);
        this._roundedRect(ctx, x + 24, iy - 23, tw, 30, 7);
        ctx.fill();
        ctx.fillStyle = tagColor;
        ctx.font = '700 16px "SF Mono", Menlo, monospace';
        ctx.fillText(tag.toUpperCase(), x + 36, iy - 3);

        ctx.fillStyle = '#e6eef9';
        ctx.font = '500 21px -apple-system, sans-serif';
        ctx.fillText(rest.slice(0, 48), x + 36 + tw, iy - 1);
      } else {
        ctx.fillStyle = idx === 0 && this.board.mode === 'definition' ? '#ffffff' : '#d8e3f2';
        ctx.font =
          idx === 0 && this.board.mode === 'definition'
            ? '600 24px -apple-system, sans-serif'
            : '500 21px -apple-system, sans-serif';
        ctx.fillText(txt.slice(0, 58), x + 26, iy);
      }
      iy += 58;
    });

    // Bottom takeaway / hint strip
    if (this.board.note) {
      ctx.fillStyle = 'rgba(30, 41, 59, 0.72)';
      this._roundedRect(ctx, x + 22, y + h - 56, w - 44, 38, 10);
      ctx.fill();
      ctx.fillStyle = '#cbd5e1';
      ctx.font = '600 17px "SF Mono", Menlo, monospace';
      ctx.fillText(this.board.note.slice(0, 66), x + 36, y + h - 31);
    }
  }

  _drawNudgePill(ctx, cx, cy, text) {
    ctx.save();
    ctx.font = '600 21px -apple-system, sans-serif';
    const tw = Math.min(540, ctx.measureText(text).width + 64);
    ctx.fillStyle = 'rgba(255, 196, 110, 0.16)';
    ctx.strokeStyle = 'rgba(255, 206, 128, 0.65)';
    ctx.lineWidth = 2;
    this._roundedRect(ctx, cx - tw / 2, cy - 24, tw, 48, 24);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#ffdfa6';
    ctx.textAlign = 'center';
    ctx.fillText(text, cx, cy + 7);
    ctx.restore();
  }

  _roundedRect(ctx, x, y, w, h, r) {
    const rad = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rad, y);
    ctx.arcTo(x + w, y, x + w, y + h, rad);
    ctx.arcTo(x + w, y + h, x, y + h, rad);
    ctx.arcTo(x, y + h, x, y, rad);
    ctx.arcTo(x, y, x + w, y, rad);
    ctx.closePath();
  }

  _alphaHex(hex, a) {
    const clean = hex.replace('#', '');
    const n = parseInt(clean, 16);
    const r = (n >> 16) & 255;
    const g = (n >> 8) & 255;
    const b = n & 255;
    return `rgba(${r},${g},${b},${a})`;
  }
}
