import * as THREE from 'three';

/**
 * Build a custom Studio Softbox PMREM Environment for crisp photorealistic PBR
 * reflections on brushed titanium, anodized OM3 drums, clearcoat glass & AR lenses.
 */
export function createStudioEnvironment(renderer, mood = 'night') {
  const pmrem = new THREE.PMREMGenerator(renderer);
  pmrem.compileEquirectangularShader();

  const envScene = new THREE.Scene();
  const bgColors = {
    day: 0x262c38,
    golden: 0x1e1918,
    night: 0x0b0e16,
  };
  envScene.background = new THREE.Color(bgColors[mood] ?? 0x0b0e16);

  const addSoftbox = (w, h, color, intensity, pos, lookAt = [0, 0, 0]) => {
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
    addSoftbox(14, 9, 0xfff8ee, 2.4, [0, 8, 2]);
    addSoftbox(10, 6, 0xdceaff, 1.8, [-9, 4, 3]);
    addSoftbox(8, 5, 0xffe8cc, 1.4, [9, 3, -2]);
    addSoftbox(12, 4, 0xffffff, 0.9, [0, 2, -9]);
  } else if (mood === 'golden') {
    addSoftbox(12, 8, 0xffd29a, 2.2, [-8, 5, 4]);
    addSoftbox(10, 6, 0x7aa2d8, 1.1, [8, 6, -3]);
    addSoftbox(12, 4, 0xffaa66, 1.4, [0, 8, 0]);
    addSoftbox(10, 3, 0xffe4c4, 0.7, [0, 1.5, -8]);
  } else {
    // Cozy Night Studio — deep contrast with crisp rim strips for Apple keynote look
    addSoftbox(10, 5, 0x8fb4e8, 1.35, [-7, 6, -5]);
    addSoftbox(9, 4, 0xffcfa0, 1.15, [7, 5, 5]);
    addSoftbox(12, 3, 0xd8e6ff, 0.85, [0, 9, 0]);
    addSoftbox(8, 2.5, 0x4d6b99, 0.65, [0, 2, -9]);
  }

  const rt = pmrem.fromScene(envScene, 0.04);
  pmrem.dispose();
  return rt.texture;
}

/**
 * Procedural Architectural Walnut Wood (albedo + bump + roughness maps).
 */
export function createWalnutWoodTextures() {
  const size = 1024;
  const colorCanvas = document.createElement('canvas');
  colorCanvas.width = size;
  colorCanvas.height = size;
  const ctx = colorCanvas.getContext('2d');

  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = size;
  bumpCanvas.height = size;
  const bctx = bumpCanvas.getContext('2d');

  // Rich satin walnut base
  ctx.fillStyle = '#3b291d';
  ctx.fillRect(0, 0, size, size);
  bctx.fillStyle = '#808080';
  bctx.fillRect(0, 0, size, size);

  // Layered flowing wood fibers
  for (let y = 0; y < size; y += 2) {
    const wave =
      Math.sin(y * 0.017) * 14 +
      Math.sin(y * 0.053 + 1.2) * 5 +
      Math.cos(y * 0.008) * 22;
    const tone = Math.sin((y + wave) * 0.065);
    const fine = Math.sin(y * 0.37 + wave * 0.12) * 0.5 + 0.5;

    const r = Math.round(52 + tone * 15 + fine * 9);
    const g = Math.round(35 + tone * 10 + fine * 6);
    const b = Math.round(24 + tone * 6 + fine * 4);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(0, y, size, 2);

    const bv = Math.round(122 + tone * 14 + fine * 10);
    bctx.fillStyle = `rgb(${bv},${bv},${bv})`;
    bctx.fillRect(0, y, size, 2);
  }

  // Subtle plank seams & natural darker grain streaks
  for (let i = 0; i < 160; i++) {
    const x = (i * 197) % size;
    const y = (i * 83) % size;
    const w = 180 + ((i * 53) % 420);
    const h = 1 + (i % 3);
    ctx.fillStyle = i % 3 === 0 ? 'rgba(22, 13, 8, 0.22)' : 'rgba(96, 68, 46, 0.12)';
    ctx.fillRect(x, y, w, h);
    bctx.fillStyle = i % 3 === 0 ? 'rgba(60, 60, 60, 0.25)' : 'rgba(170, 170, 170, 0.14)';
    bctx.fillRect(x, y, w, h);
  }

  // Subtle horizontal board joints
  ctx.fillStyle = 'rgba(15, 9, 5, 0.38)';
  ctx.fillRect(0, 340, size, 2);
  ctx.fillRect(0, 682, size, 2);
  bctx.fillStyle = '#303030';
  bctx.fillRect(0, 340, size, 2);
  bctx.fillRect(0, 682, size, 2);

  const map = new THREE.CanvasTexture(colorCanvas);
  map.colorSpace = THREE.SRGBColorSpace;
  map.wrapS = map.wrapT = THREE.RepeatWrapping;
  map.repeat.set(2.2, 1.4);
  map.anisotropy = 8;

  const bumpMap = new THREE.CanvasTexture(bumpCanvas);
  bumpMap.wrapS = bumpMap.wrapT = THREE.RepeatWrapping;
  bumpMap.repeat.set(2.2, 1.4);
  bumpMap.anisotropy = 8;

  return { map, bumpMap };
}

/**
 * Procedural Charcoal Wool Felt Desk Mat Texture.
 */
export function createFeltMatTextures() {
  const size = 512;
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d');

  ctx.fillStyle = '#1b1d22';
  ctx.fillRect(0, 0, size, size);

  // Subtle woven fiber micro-noise
  let seed = 1337;
  const rand = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
  for (let i = 0; i < 16000; i++) {
    const x = rand() * size;
    const y = rand() * size;
    const v = 22 + Math.floor(rand() * 24);
    ctx.fillStyle = `rgb(${v},${v + 2},${v + 5})`;
    ctx.fillRect(x, y, 1.6, 1.6);
  }
  // Fine diagonal felt strands
  ctx.strokeStyle = 'rgba(255,255,255,0.025)';
  ctx.lineWidth = 1;
  for (let i = 0; i < 320; i++) {
    const x = rand() * size;
    const y = rand() * size;
    const len = 4 + rand() * 10;
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
 * Brushed Anodized Aluminum Bump Map for DJI OM3 motor housings & Pixel rail.
 */
export function createBrushedMetalTexture() {
  const c = document.createElement('canvas');
  c.width = 512;
  c.height = 512;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, 512, 512);
  for (let y = 0; y < 512; y++) {
    const v = 112 + Math.floor((Math.sin(y * 1.9) * 0.5 + 0.5) * 32);
    ctx.fillStyle = `rgb(${v},${v},${v})`;
    ctx.fillRect(0, y, 512, 1);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(2, 2);
  return tex;
}

/**
 * High-DPI Open Study Notebook Page Texture (crisp math/economics typography + graphs + margin notes).
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
  grad.addColorStop(0.09, '#f3eee2');
  grad.addColorStop(1.0, '#f7f3ea');
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

  // Top header rule
  ctx.strokeStyle = 'rgba(65, 82, 108, 0.28)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(76, 118);
  ctx.lineTo(w - 76, 118);
  ctx.stroke();

  if (side === 'left') {
    // Left page: Economics & Financial Analysis + Supply/Demand Curve
    ctx.fillStyle = '#5a6b82';
    ctx.font = '600 22px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif';
    ctx.fillText('SCIENCES PO · ÉCONOMIE & ANALYSE FINANCIÈRE', 78, 96);
    ctx.fillText('p. 114', w - 140, 96);

    ctx.fillStyle = '#182130';
    ctx.font = '700 40px Georgia, serif';
    ctx.fillText('Chapitre 11 — Analyse du Résultat', 78, 184);

    // Highlight strip
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

    // Formula box
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

    // Diagram: Valuation / Equilibrium Curve
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

    // Curve 1 (blue)
    ctx.strokeStyle = '#2b6cb0';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(gx + 50, gy + 65);
    ctx.quadraticCurveTo(gx + 280, gy + 260, gx + 680, gy + 375);
    ctx.stroke();

    // Curve 2 (terracotta)
    ctx.strokeStyle = '#c05621';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(gx + 55, gy + 365);
    ctx.quadraticCurveTo(gx + 340, gy + 265, gx + 670, gy + 68);
    ctx.stroke();

    // Equilibrium dot & dashed projections
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

    // Handwritten blue ink margin note
    ctx.fillStyle = '#1d4ed8';
    ctx.font = 'italic 28px Georgia, serif';
    ctx.fillText('→ Vérifier Q2 avec Osmo !', 120, 1170);
    ctx.fillText('   Attention au signe de ΔBFR', 120, 1212);
  } else {
    // Right page: Physics & Thermodynamics Exercises (Qmax & Practice Work)
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

    // Exercise section
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

    // Student handwritten draft answer in blue ballpoint pen
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

    // Pencil check circle around Q1
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
