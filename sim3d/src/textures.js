import * as THREE from 'three';

/**
 * Studio PMREM Environment for photorealistic PBR reflections on brushed aluminum,
 * polished titanium Pixel frame, sapphire AR camera lenses, and MacBook Pro glass.
 */
export function createStudioEnvironment(renderer, mood = 'day') {
  const pmrem = new THREE.PMREMGenerator(renderer);
  pmrem.compileEquirectangularShader();

  const envScene = new THREE.Scene();
  const bgColors = {
    day: 0xdce2ea,
    golden: 0x3b2c25,
    night: 0x101520,
  };
  envScene.background = new THREE.Color(bgColors[mood] ?? 0xdce2ea);

  const addPanel = (w, h, color, intensity, pos, lookAt = [0, 0, 0]) => {
    const geo = new THREE.PlaneGeometry(w, h);
    const mat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(color).multiplyScalar(intensity),
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(...pos);
    mesh.lookAt(...lookAt);
    envScene.add(mesh);
  };

  if (mood === 'day') {
    // Bright sunlit architectural room (like Apple ELEGNT!)
    addPanel(14, 10, 0xfffaf2, 2.8, [-10, 5, 3]);   // Big sunlit window left
    addPanel(16, 12, 0xffffff, 1.5, [0, 10, 0]);    // Bright ceiling bounce
    addPanel(14, 8, 0xf2ece2, 1.1, [10, 4, 2]);     // Warm plaster right wall bounce
    addPanel(14, 6, 0xe8effa, 0.9, [0, 3, 10]);     // Soft sky fill
    addPanel(16, 10, 0x8f775d, 0.55, [0, -6, 0]);   // Warm oak desk bounce
  } else if (mood === 'golden') {
    addPanel(12, 9, 0xffc682, 2.6, [-10, 4.5, 4]);  // Golden sunset window
    addPanel(14, 8, 0x9ab4de, 1.1, [9, 6, -2]);     // Cool twilight sky fill
    addPanel(12, 6, 0xffd7ad, 1.2, [0, 9, 0]);
    addPanel(12, 5, 0x5c4637, 0.5, [0, -5, 0]);
  } else {
    addPanel(10, 5, 0x8db3eb, 1.25, [-8, 6, -4]);
    addPanel(10, 5, 0xffcfa0, 1.1, [8, 5, 5]);
    addPanel(12, 4, 0xd4e4ff, 0.75, [0, 9, 0]);
  }

  const rt = pmrem.fromScene(envScene, 0.035);
  pmrem.dispose();
  return rt.texture;
}

/**
 * Natural Architectural Light Scandinavian Oak Wood (albedo + bump + roughness).
 */
export function createOakWoodTextures() {
  const size = 1024;
  const colorCanvas = document.createElement('canvas');
  colorCanvas.width = size;
  colorCanvas.height = size;
  const ctx = colorCanvas.getContext('2d');

  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = size;
  bumpCanvas.height = size;
  const bctx = bumpCanvas.getContext('2d');

  // Warm natural light-oak base (Apple Store / Scandinavian study desk tone)
  ctx.fillStyle = '#c5a57e';
  ctx.fillRect(0, 0, size, size);
  bctx.fillStyle = '#808080';
  bctx.fillRect(0, 0, size, size);

  for (let y = 0; y < size; y += 2) {
    const wave =
      Math.sin(y * 0.014) * 18 +
      Math.sin(y * 0.048 + 0.9) * 6 +
      Math.cos(y * 0.007) * 26;
    const tone = Math.sin((y + wave) * 0.058);
    const fine = Math.sin(y * 0.42 + wave * 0.14) * 0.5 + 0.5;

    const r = Math.round(186 + tone * 16 + fine * 10);
    const g = Math.round(154 + tone * 14 + fine * 8);
    const b = Math.round(118 + tone * 11 + fine * 6);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(0, y, size, 2);

    const bv = Math.round(124 + tone * 12 + fine * 10);
    bctx.fillStyle = `rgb(${bv},${bv},${bv})`;
    bctx.fillRect(0, y, size, 2);
  }

  // Fine oak pores & subtle medullary rays
  for (let i = 0; i < 220; i++) {
    const x = (i * 211) % size;
    const y = (i * 79) % size;
    const w = 140 + ((i * 61) % 360);
    const h = 1 + (i % 2);
    ctx.fillStyle = i % 3 === 0 ? 'rgba(118, 86, 54, 0.16)' : 'rgba(235, 214, 184, 0.16)';
    ctx.fillRect(x, y, w, h);
    bctx.fillStyle = i % 3 === 0 ? 'rgba(75, 75, 75, 0.2)' : 'rgba(175, 175, 175, 0.12)';
    bctx.fillRect(x, y, w, h);
  }

  // Subtle plank join lines
  ctx.fillStyle = 'rgba(115, 84, 54, 0.22)';
  ctx.fillRect(0, 340, size, 2);
  ctx.fillRect(0, 682, size, 2);

  const map = new THREE.CanvasTexture(colorCanvas);
  map.colorSpace = THREE.SRGBColorSpace;
  map.wrapS = map.wrapT = THREE.RepeatWrapping;
  map.repeat.set(2.0, 1.3);
  map.anisotropy = 8;

  const bumpMap = new THREE.CanvasTexture(bumpCanvas);
  bumpMap.wrapS = bumpMap.wrapT = THREE.RepeatWrapping;
  bumpMap.repeat.set(2.0, 1.3);
  bumpMap.anisotropy = 8;

  return { map, bumpMap };
}

/**
 * Architectural Warm Stone/Linen Felt Desk Mat Texture.
 */
export function createFeltMatTextures() {
  const size = 512;
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d');

  ctx.fillStyle = '#383c44';
  ctx.fillRect(0, 0, size, size);

  let seed = 4242;
  const rand = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
  for (let i = 0; i < 18000; i++) {
    const x = rand() * size;
    const y = rand() * size;
    const v = 48 + Math.floor(rand() * 26);
    ctx.fillStyle = `rgb(${v},${v + 2},${v + 5})`;
    ctx.fillRect(x, y, 1.6, 1.6);
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.03)';
  ctx.lineWidth = 1;
  for (let i = 0; i < 360; i++) {
    const x = rand() * size;
    const y = rand() * size;
    const len = 4 + rand() * 11;
    const ang = rand() * Math.PI;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + Math.cos(ang) * len, y + Math.sin(ang) * len);
    ctx.stroke();
  }

  const map = new THREE.CanvasTexture(c);
  map.colorSpace = THREE.SRGBColorSpace;
  map.wrapS = map.wrapT = THREE.RepeatWrapping;
  map.repeat.set(3.5, 2.2);
  map.anisotropy = 8;
  return { map, bumpMap: map };
}

/**
 * Brushed Anodized Aluminum Bump Map for DJI OM3 motor drums & Pixel titanium rails.
 */
export function createBrushedMetalTexture() {
  const c = document.createElement('canvas');
  c.width = 512;
  c.height = 512;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, 512, 512);
  for (let y = 0; y < 512; y++) {
    const v = 116 + Math.floor((Math.sin(y * 2.1) * 0.5 + 0.5) * 26);
    ctx.fillStyle = `rgb(${v},${v},${v})`;
    ctx.fillRect(0, y, 512, 1);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(2, 2);
  return tex;
}

/**
 * Radial CNC Lathe-Turned Motor Endcap Texture (color + bump) for DJI OM3 brushless drums.
 */
export function createRadialMotorCapTexture() {
  const size = 512;
  const cx = size / 2;
  const cy = size / 2;

  const colCanvas = document.createElement('canvas');
  colCanvas.width = size;
  colCanvas.height = size;
  const ctx = colCanvas.getContext('2d');

  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = size;
  bumpCanvas.height = size;
  const bctx = bumpCanvas.getContext('2d');

  // Anisotropic radial shimmer background
  ctx.fillStyle = '#3f4550';
  ctx.fillRect(0, 0, size, size);
  bctx.fillStyle = '#808080';
  bctx.fillRect(0, 0, size, size);

  // Anisotropic metallic sweep wedge highlights
  for (let a = 0; a < 360; a += 1) {
    const rad = (a * Math.PI) / 180;
    const sheen = Math.pow(Math.abs(Math.sin(rad * 2 + 0.7)), 1.6);
    const v = Math.round(52 + sheen * 42);
    ctx.strokeStyle = `rgb(${v},${v + 4},${v + 10})`;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(rad) * 256, cy + Math.sin(rad) * 256);
    ctx.stroke();
  }

  // Fine concentric CNC lathe grooves
  for (let r = 10; r < 246; r += 3) {
    const wave = Math.sin(r * 0.9) * 0.5 + 0.5;
    ctx.strokeStyle = wave > 0.5 ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.12)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    const bv = Math.round(108 + wave * 42);
    bctx.strokeStyle = `rgb(${bv},${bv},${bv})`;
    bctx.lineWidth = 1.5;
    bctx.beginPath();
    bctx.arc(cx, cy, r, 0, Math.PI * 2);
    bctx.stroke();
  }

  // Outer engraved mechanical index ring + subtle degree ticks
  ctx.strokeStyle = 'rgba(215, 224, 238, 0.24)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, 198, 0, Math.PI * 2);
  ctx.stroke();

  for (let i = 0; i < 24; i++) {
    const rad = (i / 24) * Math.PI * 2;
    const inner = i % 6 === 0 ? 174 : 186;
    ctx.strokeStyle = i % 6 === 0 ? 'rgba(225, 232, 244, 0.42)' : 'rgba(200, 210, 225, 0.2)';
    ctx.lineWidth = i % 6 === 0 ? 2.5 : 1.5;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(rad) * inner, cy + Math.sin(rad) * inner);
    ctx.lineTo(cx + Math.cos(rad) * 198, cy + Math.sin(rad) * 198);
    ctx.stroke();
  }

  // Inner dark recessed rotor center cap
  ctx.fillStyle = '#252931';
  ctx.beginPath();
  ctx.arc(cx, cy, 78, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = 'rgba(220, 228, 240, 0.35)';
  ctx.lineWidth = 3;
  ctx.stroke();

  const map = new THREE.CanvasTexture(colCanvas);
  map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = 8;

  const bumpMap = new THREE.CanvasTexture(bumpCanvas);
  bumpMap.anisotropy = 8;

  return { map, bumpMap };
}

/**
 * Tactile Micro-Diamond Rubber Grip Bump Texture for DJI OM3 palm grip & clamp pads.
 */
export function createRubberGripTexture() {
  const size = 256;
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#555555';
  ctx.fillRect(0, 0, size, size);

  const step = 12;
  for (let y = 0; y < size; y += step) {
    for (let x = 0; x < size; x += step) {
      const off = (Math.floor(y / step) % 2) * (step / 2);
      const px = (x + off) % size;
      ctx.fillStyle = '#b8b8b8';
      ctx.beginPath();
      ctx.moveTo(px, y + 1);
      ctx.lineTo(px + step * 0.42, y + step * 0.5);
      ctx.lineTo(px, y + step - 1);
      ctx.lineTo(px - step * 0.42, y + step * 0.5);
      ctx.closePath();
      ctx.fill();
    }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(5, 8);
  return tex;
}

/**
 * Photorealistic Dual-Tone Fresnel Smartphone LED Flash Texture for Pixel 11 Pro visor.
 */
export function createFlashFresnelTexture() {
  const size = 256;
  const cx = size / 2;
  const cy = size / 2;
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d');

  // Dual-tone phosphor split (warm amber upper + crisp cool white lower)
  const g = ctx.createLinearGradient(0, 0, 0, size);
  g.addColorStop(0.0, '#ffe39a');
  g.addColorStop(0.48, '#ffefb8');
  g.addColorStop(0.52, '#fffdf5');
  g.addColorStop(1.0, '#eef4ff');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);

  // Concentric micro-Fresnel lens rings
  for (let r = 10; r < 126; r += 9) {
    ctx.strokeStyle = 'rgba(130, 108, 72, 0.34)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  // Subtle metallic outer rim shadow
  const rg = ctx.createRadialGradient(cx, cy, 70, cx, cy, 128);
  rg.addColorStop(0, 'rgba(0,0,0,0)');
  rg.addColorStop(1, 'rgba(30,26,20,0.45)');
  ctx.fillStyle = rg;
  ctx.fillRect(0, 0, size, size);

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/**
 * Photorealistic MacBook Pro Backlit Keyboard Deck Texture (keys + speaker grilles + Touch ID).
 */
export function createMacBookKeyboardTexture() {
  const w = 1024;
  const h = 640;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d');

  // Anodized silver/space-gray deck around keyboard well
  ctx.fillStyle = '#adb4be';
  ctx.fillRect(0, 0, w, h);

  // Speaker grille micro-perforations (left & right of keyboard)
  ctx.fillStyle = 'rgba(20, 24, 30, 0.72)';
  for (const sx of [42, w - 102]) {
    for (let x = sx; x < sx + 60; x += 6) {
      for (let y = 52; y < 340; y += 6) {
        ctx.beginPath();
        ctx.arc(x, y, 1.35, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // Recessed black anodized keyboard well
  const kx = 128;
  const ky = 44;
  const kw = w - 256;
  const kh = 310;
  ctx.fillStyle = '#14171c';
  ctx.beginPath();
  ctx.roundRect(kx, ky, kw, kh, 12);
  ctx.fill();

  // Key rows
  const rows = [
    { count: 14, h: 32, y: ky + 12 },
    { count: 14, h: 42, y: ky + 52 },
    { count: 14, h: 42, y: ky + 102 },
    { count: 13, h: 42, y: ky + 152 },
    { count: 12, h: 42, y: ky + 202 },
  ];
  const gap = 7;
  rows.forEach((row, rIdx) => {
    const cellW = (kw - 20 - gap * (row.count - 1)) / row.count;
    for (let i = 0; i < row.count; i++) {
      const x = kx + 10 + i * (cellW + gap);
      ctx.fillStyle = '#1f232a';
      ctx.beginPath();
      ctx.roundRect(x, row.y, cellW, row.h, 5);
      ctx.fill();
      // Subtle top edge highlight + white backlight glyph
      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.lineWidth = 1;
      ctx.stroke();
      if (rIdx === 0 && i === row.count - 1) {
        // Touch ID ring
        ctx.strokeStyle = 'rgba(180,190,205,0.45)';
        ctx.beginPath();
        ctx.arc(x + cellW / 2, row.y + row.h / 2, 9, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        ctx.fillStyle = 'rgba(235, 242, 255, 0.42)';
        ctx.fillRect(x + cellW * 0.35, row.y + row.h * 0.44, cellW * 0.3, 3);
      }
    }
  });

  // Bottom modifier + Spacebar row
  const by = ky + 252;
  const bottomKeys = [0.95, 0.95, 0.95, 1.25, 5.2, 1.25, 0.95, 0.95, 0.95, 0.95];
  const totalU = bottomKeys.reduce((a, b) => a + b, 0);
  const unitW = (kw - 20 - gap * (bottomKeys.length - 1)) / totalU;
  let bx = kx + 10;
  bottomKeys.forEach((u) => {
    const bw = u * unitW;
    ctx.fillStyle = '#1f232a';
    ctx.beginPath();
    ctx.roundRect(bx, by, bw, 46, 6);
    ctx.fill();
    bx += bw + gap;
  });

  // Large Force Touch Glass Trackpad
  ctx.fillStyle = '#9ea6b2';
  ctx.strokeStyle = 'rgba(40, 46, 56, 0.35)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(w * 0.31, 384, w * 0.38, 224, 14);
  ctx.fill();
  ctx.stroke();

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

/**
 * Photorealistic MacBook Pro Liquid Retina Display Texture (clean macOS + Osmo Studio UI).
 */
export function createMacBookScreenTexture() {
  const w = 1024;
  const h = 660;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d');

  // Subtle macOS wallpaper gradient behind clean floating Osmo Studio window
  const bg = ctx.createLinearGradient(0, 0, w, h);
  bg.addColorStop(0, '#101828');
  bg.addColorStop(0.55, '#192234');
  bg.addColorStop(1, '#261e2a');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  // macOS Menu Bar + Camera Notch
  ctx.fillStyle = 'rgba(10, 14, 22, 0.72)';
  ctx.fillRect(0, 0, w, 28);
  ctx.fillStyle = '#05070a';
  ctx.beginPath();
  ctx.roundRect(w / 2 - 58, 0, 116, 26, [0, 0, 10, 10]);
  ctx.fill();

  ctx.fillStyle = '#e2e8f0';
  ctx.font = '600 13px -apple-system, sans-serif';
  ctx.fillText('   Osmo Studio   File   Edit   View', 18, 19);
  ctx.textAlign = 'right';
  ctx.fillText('Gemini 3.8 Live · 16:38', w - 18, 19);
  ctx.textAlign = 'left';

  // Main Studio App Window
  ctx.fillStyle = 'rgba(11, 15, 23, 0.94)';
  ctx.strokeStyle = 'rgba(226, 236, 250, 0.14)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(38, 48, w - 76, h - 104, 16);
  ctx.fill();
  ctx.stroke();

  // Traffic lights
  ['#ff5f56', '#ffbd2e', '#27c93f'].forEach((col, i) => {
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.arc(64 + i * 20, 70, 5.5, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.fillStyle = '#94a3b8';
  ctx.font = '700 13px "SF Mono", Menlo, monospace';
  ctx.fillText('OSMO // DESK STUDIO · AI : GEMINI', 140, 75);

  // Left camera feed mockup card
  ctx.fillStyle = '#151c29';
  ctx.beginPath();
  ctx.roundRect(62, 98, 520, 290, 12);
  ctx.fill();
  ctx.strokeStyle = 'rgba(255, 208, 136, 0.45)';
  ctx.lineWidth = 2;
  ctx.strokeRect(170, 150, 295, 190);
  ctx.fillStyle = '#ffd088';
  ctx.font = '600 13px "SF Mono", Menlo, monospace';
  ctx.fillText('OPTICAL FEED // PIXEL 11 PRO · 4K DESK LOCKED', 80, 124);
  ctx.fillStyle = '#34d399';
  ctx.fillText('SHEET DETECTED · HILIGHT WARM 2900K', 176, 144);

  // Bottom voice waveform bar
  ctx.fillStyle = '#141a26';
  ctx.beginPath();
  ctx.roundRect(62, 404, 520, 174, 12);
  ctx.fill();
  ctx.fillStyle = '#60a5fa';
  ctx.font = '700 13px "SF Mono", Menlo, monospace';
  ctx.fillText('GEMINI 3.8 LIVE · NON_BLOCKING TOOLS ACTIVE', 80, 432);
  for (let i = 0; i < 36; i++) {
    const bh = 14 + Math.abs(Math.sin(i * 0.55)) * 46;
    ctx.fillStyle = i % 3 === 0 ? '#ffd088' : '#60a5fa';
    ctx.fillRect(82 + i * 13, 505 - bh / 2, 6, bh);
  }

  // Right telemetry + notes column
  ctx.fillStyle = '#141a26';
  ctx.beginPath();
  ctx.roundRect(602, 98, 360, 480, 12);
  ctx.fill();
  ctx.fillStyle = '#94a3b8';
  ctx.font = '700 13px "SF Mono", Menlo, monospace';
  ctx.fillText('TELEMETRY // DJI OM3 + ELEGNT', 622, 126);

  const stats = [
    ['MODE', 'ELEGNT COMPANION', '#34d399'],
    ['HILIGHT', 'WARM LAMP · 2900K', '#ffd088'],
    ['VISION', 'APPLE ANE · 11 ms', '#60a5fa'],
    ['MEMORY', 'SQLITE FTS5 · READY', '#cbd5e1'],
  ];
  stats.forEach(([k, v, col], idx) => {
    ctx.fillStyle = 'rgba(255,255,255,0.04)';
    ctx.beginPath();
    ctx.roundRect(622, 150 + idx * 64, 320, 50, 8);
    ctx.fill();
    ctx.fillStyle = '#7c8ca6';
    ctx.font = '600 11px "SF Mono", Menlo, monospace';
    ctx.fillText(k, 636, 170 + idx * 64);
    ctx.fillStyle = col;
    ctx.font = '700 14px "SF Mono", Menlo, monospace';
    ctx.fillText(v, 636, 190 + idx * 64);
  });

  // macOS Dock pill at bottom
  ctx.fillStyle = 'rgba(255, 255, 255, 0.14)';
  ctx.beginPath();
  ctx.roundRect(w / 2 - 140, h - 42, 280, 32, 10);
  ctx.fill();
  const dockCols = ['#38bdf8', '#f59e0b', '#34d399', '#818cf8', '#fb7185', '#e2e8f0'];
  dockCols.forEach((col, i) => {
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.roundRect(w / 2 - 118 + i * 42, h - 36, 22, 22, 6);
    ctx.fill();
  });

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

/**
 * High-DPI Open Study Notebook Page Texture.
 */
export function createNotebookPageTexture(side = 'left') {
  const w = 1024;
  const h = 1320;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d');

  // Warm archival cotton paper
  const grad = ctx.createLinearGradient(side === 'left' ? w : 0, 0, side === 'left' ? 0 : w, 0);
  grad.addColorStop(0.0, '#ddd5c5');
  grad.addColorStop(0.09, '#f5f0e4');
  grad.addColorStop(1.0, '#fbf8f0');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  // Subtle dot-grid engineering paper
  ctx.fillStyle = 'rgba(110, 125, 145, 0.22)';
  for (let x = 72; x < w - 56; x += 36) {
    for (let y = 96; y < h - 72; y += 36) {
      ctx.beginPath();
      ctx.arc(x, y, 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.strokeStyle = 'rgba(65, 82, 108, 0.28)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(76, 118);
  ctx.lineTo(w - 76, 118);
  ctx.stroke();

  if (side === 'left') {
    ctx.fillStyle = '#5a6b82';
    ctx.font = '600 22px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif';
    ctx.fillText('SCIENCES PO · ÉCONOMIE & ANALYSE FINANCIÈRE', 78, 96);
    ctx.fillText('p. 114', w - 140, 96);

    ctx.fillStyle = '#182130';
    ctx.font = '700 40px Georgia, serif';
    ctx.fillText('Chapitre 11 — Analyse du Résultat', 78, 184);

    ctx.fillStyle = 'rgba(255, 214, 102, 0.34)';
    ctx.fillRect(74, 214, 620, 44);
    ctx.fillStyle = '#1b2738';
    ctx.font = '600 27px -apple-system, sans-serif';
    ctx.fillText('1. Ajuster le bénéfice net (True Earnings)', 84, 244);

    ctx.fillStyle = '#2c384a';
    ctx.font = '400 25px Georgia, serif';
    const lines = [
      'Ne pas se fier au bénéfice consolidé brut : déduire',
      'les stock-options, amortissements exceptionnels et',
      'vérifier la soutenabilité du Free Cash Flow (FCF).',
    ];
    lines.forEach((ln, i) => ctx.fillText(ln, 84, 304 + i * 40));

    ctx.strokeStyle = '#33588e';
    ctx.lineWidth = 3;
    ctx.fillStyle = 'rgba(51, 88, 142, 0.06)';
    ctx.fillRect(84, 438, w - 168, 120);
    ctx.strokeRect(84, 438, w - 168, 120);

    ctx.fillStyle = '#1c3e70';
    ctx.font = '700 32px "SF Mono", Menlo, monospace';
    ctx.fillText('FCF = EBIT(1 - t) + D&A - CapEx - ΔBFR', 118, 496);
    ctx.font = '500 22px -apple-system, sans-serif';
    ctx.fillStyle = '#475c78';
    ctx.fillText('Marge de sécurité (Graham) : Prix < 0.67 × Valeur Intrinsèque', 118, 536);

    const gx = 120;
    const gy = 630;
    const gw = 760;
    const gh = 420;
    ctx.strokeStyle = '#253142';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(gx, gy);
    ctx.lineTo(gx, gy + gh);
    ctx.lineTo(gx + gw, gy + gh);
    ctx.stroke();

    ctx.strokeStyle = '#2b6cb0';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(gx + 50, gy + 65);
    ctx.quadraticCurveTo(gx + 280, gy + 260, gx + 680, gy + 375);
    ctx.stroke();

    ctx.strokeStyle = '#c05621';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(gx + 55, gy + 365);
    ctx.quadraticCurveTo(gx + 340, gy + 265, gx + 670, gy + 68);
    ctx.stroke();

    const ex = gx + 340;
    const ey = gy + 236;
    ctx.setLineDash([8, 8]);
    ctx.strokeStyle = 'rgba(37, 49, 66, 0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex, gy + gh);
    ctx.moveTo(ex, ey);
    ctx.lineTo(gx, ey);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(ex, ey, 11, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#182130';
    ctx.font = '700 24px -apple-system, sans-serif';
    ctx.fillText('E* (P*, Q*)', ex + 18, ey - 10);
    ctx.fillText('Prix (P)', gx - 10, gy - 16);
    ctx.fillText('Quantité (Q)', gx + gw - 140, gy + gh + 36);

    ctx.fillStyle = '#1d4ed8';
    ctx.font = 'italic 28px Georgia, serif';
    ctx.fillText('→ Vérifier Q2 avec Osmo !', 120, 1170);
    ctx.fillText('   Attention au signe de ΔBFR', 120, 1212);
  } else {
    ctx.fillStyle = '#5a6b82';
    ctx.font = '600 22px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif';
    ctx.fillText('THERMODYNAMIQUE & EXERCICES PRATIQUES', 78, 96);
    ctx.fillText('p. 115', w - 140, 96);

    ctx.fillStyle = '#182130';
    ctx.font = '700 38px Georgia, serif';
    ctx.fillText('TD n°4 — Transfert Thermique (Qmax)', 78, 184);

    ctx.fillStyle = 'rgba(52, 211, 153, 0.22)';
    ctx.fillRect(76, 218, 560, 42);
    ctx.fillStyle = '#0f3d2e';
    ctx.font = '600 26px -apple-system, sans-serif';
    ctx.fillText('Effusivité thermique & contact peau / matériau', 86, 248);

    ctx.fillStyle = '#202c3d';
    ctx.font = '700 33px "SF Mono", Menlo, monospace';
    ctx.fillText('e = √(k · ρ · c)    [W·s½ / (m²·K)]', 86, 336);
    ctx.fillText('q_max = e · ΔT / √(π · t)', 86, 396);

    ctx.fillStyle = '#1e293b';
    ctx.font = '700 28px -apple-system, sans-serif';
    ctx.fillText('Exercice d\'application :', 84, 494);

    ctx.font = '400 25px Georgia, serif';
    const exLines = [
      'Données : Aluminium k = 237 W/m·K, ρ = 2700 kg/m³,',
      'c = 900 J/kg·K. Bois chêne : e ≈ 450 W·s½/(m²·K).',
      '',
      'Q1. Calculer l\'effusivité thermique de l\'aluminium.',
      'Q2. Pourquoi le métal paraît-il plus froid que le bois',
      '    à la même température ambiante (20 °C) ?',
    ];
    exLines.forEach((ln, i) => ctx.fillText(ln, 84, 546 + i * 42));

    ctx.strokeStyle = 'rgba(29, 78, 216, 0.25)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(84, 835);
    ctx.lineTo(w - 84, 835);
    ctx.stroke();

    ctx.fillStyle = '#1e40af';
    ctx.font = 'italic 30px Georgia, serif';
    ctx.fillText('Ma réponse (brouillon) :', 88, 886);
    ctx.font = 'italic 29px "SF Mono", Menlo, monospace';
    ctx.fillText('e_alu = √(237 × 2700 × 900)', 104, 948);
    ctx.fillText('      = √(575 910 000) ≈ 23 998', 104, 998);
    ctx.font = 'italic 28px Georgia, serif';
    ctx.fillText('→ e_alu >> e_bois, donc le flux de chaleur', 104, 1066);
    ctx.fillText('  extrait de la peau est ~50× plus élevé !', 104, 1110);

    ctx.strokeStyle = '#d97706';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.ellipse(240, 990, 145, 36, -0.05, 0, Math.PI * 2);
    ctx.stroke();
  }

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}
