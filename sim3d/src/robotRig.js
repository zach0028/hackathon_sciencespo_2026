import * as THREE from 'three';

/**
 * Create a rounded-rectangle ExtrudeGeometry centered at (0, 0, 0).
 */
export function createRoundedBoxGeo(w, h, d, radius, bevel = 0.012, curveSegments = 20) {
  const hw = w / 2 - bevel;
  const hh = h / 2 - bevel;
  const r = Math.min(radius, hw, hh);
  const shape = new THREE.Shape();
  shape.moveTo(-hw + r, -hh);
  shape.lineTo(hw - r, -hh);
  shape.quadraticCurveTo(hw, -hh, hw, -hh + r);
  shape.lineTo(hw, hh - r);
  shape.quadraticCurveTo(hw, hh, hw - r, hh);
  shape.lineTo(-hw + r, hh);
  shape.quadraticCurveTo(-hw, hh, -hw, hh - r);
  shape.lineTo(-hw, -hh + r);
  shape.quadraticCurveTo(-hw, -hh, -hw + r, -hh);

  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: Math.max(0.004, d - bevel * 2),
    bevelEnabled: true,
    bevelThickness: bevel,
    bevelSize: bevel,
    bevelSegments: 5,
    curveSegments,
  });
  geo.center();
  return geo;
}

/**
 * Create a flat rounded-rectangle ShapeGeometry with exact normalized 0..1 UVs
 * so Pixel OLED screens and MacBook displays have true curved display corners.
 */
export function createRoundedPlaneGeo(w, h, radius, curveSegments = 24) {
  const hw = w / 2;
  const hh = h / 2;
  const r = Math.min(radius, hw, hh);
  const shape = new THREE.Shape();
  shape.moveTo(-hw + r, -hh);
  shape.lineTo(hw - r, -hh);
  shape.quadraticCurveTo(hw, -hh, hw, -hh + r);
  shape.lineTo(hw, hh - r);
  shape.quadraticCurveTo(hw, hh, hw - r, hh);
  shape.lineTo(-hw + r, hh);
  shape.quadraticCurveTo(-hw, hh, -hw, hh - r);
  shape.lineTo(-hw, -hh + r);
  shape.quadraticCurveTo(-hw, -hh, -hw + r, -hh);

  const geo = new THREE.ShapeGeometry(shape, curveSegments);
  const pos = geo.attributes.position;
  const uvs = new Float32Array(pos.count * 2);
  for (let i = 0; i < pos.count; i++) {
    uvs[i * 2] = (pos.getX(i) + hw) / w;
    uvs[i * 2 + 1] = (pos.getY(i) + hh) / h;
  }
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  return geo;
}

/**
 * Procedural frosted obsidian rear glass texture with subtle polished Google 'G' emblem.
 */
function createPixelBackTexture() {
  const w = 1024;
  const h = 512;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d');

  // Satin matte obsidian glass base with delicate silky gradient
  const grad = ctx.createLinearGradient(0, 0, w, h);
  grad.addColorStop(0, '#22262e');
  grad.addColorStop(0.5, '#1b1e25');
  grad.addColorStop(1, '#15181e');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  // Ultra-fine matte glass micro-grain
  let seed = 9191;
  const rand = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
  ctx.fillStyle = 'rgba(255,255,255,0.015)';
  for (let i = 0; i < 4000; i++) {
    ctx.fillRect(rand() * w, rand() * h, 1.5, 1.5);
  }

  // Subtle mirrored titanium 'G' emblem (rotated for landscape Pixel orientation)
  ctx.save();
  ctx.translate(w * 0.56, h * 0.5);
  ctx.rotate(-Math.PI / 2);
  ctx.strokeStyle = 'rgba(198, 206, 220, 0.36)';
  ctx.lineWidth = 11;
  ctx.lineCap = 'butt';
  ctx.beginPath();
  ctx.arc(0, 0, 38, -0.45, Math.PI * 1.72, false);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(4, 0);
  ctx.lineTo(39, 0);
  ctx.stroke();
  ctx.restore();

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

/**
 * Custom GLSL Volumetric Lamp Beam Cone (Fresnel edge fade + longitudinal falloff + warmth).
 */
function createVolumetricBeamMesh() {
  const height = 5.6;
  const topR = 0.13;
  const botR = 1.9;
  const geo = new THREE.CylinderGeometry(topR, botR, height, 48, 24, true);
  geo.translate(0, -height / 2, 0);
  geo.rotateX(-Math.PI / 2);

  const mat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    uniforms: {
      uColor: { value: new THREE.Color(0xffca72) },
      uIntensity: { value: 0.0 },
      uTime: { value: 0.0 },
      uMoodScale: { value: 0.52 },
    },
    vertexShader: `
      varying vec3 vNormalView;
      varying vec3 vViewDir;
      varying vec2 vUv;
      varying float vDist;
      void main() {
        vUv = uv;
        vDist = 1.0 - uv.y;
        vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
        vNormalView = normalize(normalMatrix * normal);
        vViewDir = normalize(-mvPos.xyz);
        gl_Position = projectionMatrix * mvPos;
      }
    `,
    fragmentShader: `
      uniform vec3 uColor;
      uniform float uIntensity;
      uniform float uTime;
      uniform float uMoodScale;
      varying vec3 vNormalView;
      varying vec3 vViewDir;
      varying vec2 vUv;
      varying float vDist;
      void main() {
        float fresnel = pow(abs(dot(normalize(vNormalView), normalize(vViewDir))), 1.75);
        float longFade = smoothstep(0.0, 0.055, vDist) * pow(1.0 - vDist, 1.6);
        float shimmer = 0.95 + 0.05 * sin(vUv.x * 25.13 + uTime * 1.3) * sin(vDist * 8.0 - uTime * 0.8);
        float alpha = fresnel * longFade * shimmer * uIntensity * 0.34 * uMoodScale;
        vec3 col = mix(uColor * 1.2, uColor * 0.88, vDist);
        gl_FragColor = vec4(col, alpha);
      }
    `,
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.renderOrder = 10;
  return { mesh, mat };
}

/**
 * High-Fidelity 1:1 mm Articulated DJI Osmo Mobile 3 + Google Pixel 11 Pro + 8-LED HiLight Rig.
 * Scale: 1.0 Three.js unit = 10 cm (1 mm = 0.01 units).
 */
export class OsmoRobotRig {
  constructor({ oledTexture, brushedMetalTex }) {
    this.root = new THREE.Group();
    this.root.name = 'OsmoRobotRoot';
    this.roomMood = 'day';

    // Authentic Industrial Materials (DJI OM3 Light Titanium Gray + Pixel 11 Pro Titanium & Glass)
    const matOmBody = new THREE.MeshStandardMaterial({
      color: 0xa7adb7,
      roughness: 0.42,
      metalness: 0.18,
    });
    const matOmAccent = new THREE.MeshStandardMaterial({
      color: 0x7b828e,
      roughness: 0.38,
      metalness: 0.32,
    });
    const matGripDark = new THREE.MeshStandardMaterial({
      color: 0x2c3037,
      roughness: 0.86,
      metalness: 0.05,
    });
    const matAnodized = new THREE.MeshStandardMaterial({
      color: 0x454c58,
      roughness: 0.26,
      metalness: 0.84,
      bumpMap: brushedMetalTex,
      bumpScale: 0.0035,
    });
    const matChromeTrim = new THREE.MeshStandardMaterial({
      color: 0xdce2ec,
      roughness: 0.14,
      metalness: 0.96,
    });
    const matRubber = new THREE.MeshStandardMaterial({
      color: 0x1a1d22,
      roughness: 0.92,
      metalness: 0.02,
    });

    // Pixel 11 Pro Premium Materials
    const matPixelRail = new THREE.MeshPhysicalMaterial({
      color: 0x626873,
      roughness: 0.16,
      metalness: 0.94,
      clearcoat: 0.45,
      clearcoatRoughness: 0.12,
      bumpMap: brushedMetalTex,
      bumpScale: 0.0018,
    });
    const matPixelBack = new THREE.MeshPhysicalMaterial({
      map: createPixelBackTexture(),
      roughness: 0.46,
      metalness: 0.06,
      clearcoat: 0.28,
      clearcoatRoughness: 0.34,
    });
    const matGlassVisor = new THREE.MeshPhysicalMaterial({
      color: 0x05070c,
      roughness: 0.03,
      metalness: 0.08,
      clearcoat: 1.0,
      clearcoatRoughness: 0.02,
      reflectivity: 1.0,
    });

    // =========================================================================
    // 1. WEIGHTED FOLDING TRIPOD BASE (y = 0 .. 0.46)
    // =========================================================================
    const tripod = new THREE.Group();
    this.root.add(tripod);

    const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.175, 0.215, 0.18, 36), matGripDark);
    hub.position.y = 0.36;
    hub.castShadow = true;
    hub.receiveShadow = true;
    tripod.add(hub);

    const collar = new THREE.Mesh(new THREE.CylinderGeometry(0.155, 0.172, 0.065, 36), matChromeTrim);
    collar.position.y = 0.47;
    collar.castShadow = true;
    tripod.add(collar);

    for (let i = 0; i < 3; i++) {
      const ang = (i * Math.PI * 2) / 3 + Math.PI / 6;
      const legGroup = new THREE.Group();
      legGroup.rotation.y = ang;
      legGroup.position.y = 0.34;

      const leg = new THREE.Mesh(createRoundedBoxGeo(0.13, 0.095, 0.98, 0.032, 0.012), matGripDark);
      leg.position.set(0, -0.15, 0.46);
      leg.rotation.x = 0.34;
      leg.castShadow = true;
      leg.receiveShadow = true;
      legGroup.add(leg);

      // Sleek metallic hinge pin on each tripod leg
      const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.024, 0.024, 0.145, 16), matAnodized);
      pin.rotation.z = Math.PI / 2;
      pin.position.set(0, -0.02, 0.12);
      legGroup.add(pin);

      const foot = new THREE.Mesh(new THREE.CylinderGeometry(0.064, 0.074, 0.05, 20), matRubber);
      foot.position.set(0, -0.315, 0.88);
      foot.castShadow = true;
      legGroup.add(foot);
      tripod.add(legGroup);
    }

    // =========================================================================
    // 2. ERGONOMIC SCULPTED DJI OM3 HANDLE (authentic 11° ergonomic slant)
    // =========================================================================
    const handleGroup = new THREE.Group();
    handleGroup.position.set(0, 0.48, 0);
    handleGroup.rotation.x = -0.19;
    this.root.add(handleGroup);

    // Tapered sculptural handle body
    const handleLower = new THREE.Mesh(createRoundedBoxGeo(0.35, 1.48, 0.41, 0.145, 0.022, 24), matOmBody);
    handleLower.position.y = 0.74;
    handleLower.castShadow = true;
    handleLower.receiveShadow = true;
    handleGroup.add(handleLower);

    // Textured dark palm grip inlay on rear
    const rearGrip = new THREE.Mesh(createRoundedBoxGeo(0.36, 0.94, 0.23, 0.11, 0.016, 20), matGripDark);
    rearGrip.position.set(0, 0.62, -0.11);
    rearGrip.castShadow = true;
    handleGroup.add(rearGrip);

    // Chamfered thumb control deck on front (+Z)
    const ctrlPanel = new THREE.Mesh(createRoundedBoxGeo(0.26, 0.56, 0.048, 0.085, 0.008, 20), matRubber);
    ctrlPanel.position.set(0, 1.12, 0.198);
    handleGroup.add(ctrlPanel);

    const joyBase = new THREE.Mesh(new THREE.CylinderGeometry(0.068, 0.076, 0.046, 28), matAnodized);
    joyBase.rotation.x = Math.PI / 2;
    joyBase.position.set(0, 1.24, 0.232);
    handleGroup.add(joyBase);

    const joyCap = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.054, 0.026, 28), matGripDark);
    joyCap.rotation.x = Math.PI / 2;
    joyCap.position.set(0, 1.24, 0.26);
    handleGroup.add(joyCap);

    const btnM = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.042, 0.024, 24), matAnodized);
    btnM.rotation.x = Math.PI / 2;
    btnM.position.set(-0.064, 1.05, 0.228);
    handleGroup.add(btnM);

    const btnRec = new THREE.Mesh(
      new THREE.CylinderGeometry(0.038, 0.038, 0.024, 24),
      new THREE.MeshStandardMaterial({ color: 0xe04444, roughness: 0.3, metalness: 0.15 })
    );
    btnRec.rotation.x = Math.PI / 2;
    btnRec.position.set(0.064, 1.05, 0.228);
    handleGroup.add(btnRec);

    this.handleLedMat = new THREE.MeshStandardMaterial({
      color: 0x34d399,
      emissive: 0x34d399,
      emissiveIntensity: 2.0,
      roughness: 0.2,
    });
    const statusLed = new THREE.Mesh(new THREE.SphereGeometry(0.017, 14, 14), this.handleLedMat);
    statusLed.position.set(0, 0.955, 0.225);
    handleGroup.add(statusLed);

    const battLedMat = new THREE.MeshStandardMaterial({
      color: 0xf1f5f9,
      emissive: 0xd8f5e8,
      emissiveIntensity: 1.2,
    });
    for (let i = 0; i < 3; i++) {
      const bLed = new THREE.Mesh(new THREE.SphereGeometry(0.0105, 10, 10), battLedMat);
      bLed.position.set(-0.044 + i * 0.044, 0.898, 0.224);
      handleGroup.add(bLed);
    }

    const zoomRocker = new THREE.Mesh(createRoundedBoxGeo(0.038, 0.21, 0.068, 0.015, 0.005), matAnodized);
    zoomRocker.position.set(-0.185, 1.14, 0.02);
    handleGroup.add(zoomRocker);

    const rearTrigger = new THREE.Mesh(createRoundedBoxGeo(0.108, 0.23, 0.075, 0.03, 0.008), matAnodized);
    rearTrigger.position.set(0, 1.15, -0.215);
    rearTrigger.rotation.x = 0.22;
    handleGroup.add(rearTrigger);

    // Top gimbal neck collar (re-leveled horizontally)
    const neckMount = new THREE.Group();
    neckMount.position.set(0, 1.5, 0);
    neckMount.rotation.x = 0.19;
    handleGroup.add(neckMount);

    const neckBase = new THREE.Mesh(new THREE.CylinderGeometry(0.21, 0.188, 0.14, 40), matOmBody);
    neckBase.position.y = 0.05;
    neckBase.castShadow = true;
    neckMount.add(neckBase);

    // =========================================================================
    // 3. AXIS 1 — PAN MOTOR PIVOT (±160° around vertical Y)
    // =========================================================================
    this.panPivot = new THREE.Group();
    this.panPivot.position.set(0, 0.14, 0);
    neckMount.add(this.panPivot);

    const panDrum = new THREE.Mesh(new THREE.CylinderGeometry(0.232, 0.232, 0.28, 48), matAnodized);
    panDrum.position.y = 0.14;
    panDrum.castShadow = true;
    this.panPivot.add(panDrum);

    const panRing = new THREE.Mesh(new THREE.TorusGeometry(0.234, 0.011, 14, 52), matChromeTrim);
    panRing.rotation.x = Math.PI / 2;
    panRing.position.y = 0.265;
    this.panPivot.add(panRing);

    const armCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0.26, 0),
      new THREE.Vector3(0, 0.44, -0.26),
      new THREE.Vector3(0, 0.78, -0.58),
      new THREE.Vector3(0, 1.18, -0.62),
    ]);
    const sweptArm = new THREE.Mesh(new THREE.TubeGeometry(armCurve, 36, 0.112, 24, false), matOmBody);
    sweptArm.castShadow = true;
    sweptArm.receiveShadow = true;
    this.panPivot.add(sweptArm);

    // Articulated folding elbow hinge
    const knuckle = new THREE.Mesh(new THREE.CylinderGeometry(0.142, 0.142, 0.22, 32), matAnodized);
    knuckle.rotation.z = Math.PI / 2;
    knuckle.position.set(0, 0.42, -0.26);
    knuckle.castShadow = true;
    this.panPivot.add(knuckle);

    const knuckleCapL = new THREE.Mesh(new THREE.CylinderGeometry(0.088, 0.088, 0.235, 24), matOmAccent);
    knuckleCapL.rotation.z = Math.PI / 2;
    knuckleCapL.position.set(0, 0.42, -0.26);
    this.panPivot.add(knuckleCapL);

    // =========================================================================
    // 4. AXIS 2 — ROLL MOTOR PIVOT (rotates around local Z)
    // =========================================================================
    this.rollPivot = new THREE.Group();
    this.rollPivot.position.set(0, 1.18, -0.62);
    this.panPivot.add(this.rollPivot);

    const rollDrum = new THREE.Mesh(new THREE.CylinderGeometry(0.208, 0.208, 0.26, 44), matAnodized);
    rollDrum.rotation.x = Math.PI / 2;
    rollDrum.castShadow = true;
    this.rollPivot.add(rollDrum);

    const rollRing = new THREE.Mesh(new THREE.TorusGeometry(0.21, 0.01, 14, 48), matChromeTrim);
    rollRing.position.z = 0.125;
    this.rollPivot.add(rollRing);

    const crossCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0, 0.08),
      new THREE.Vector3(0.45, 0, 0.14),
      new THREE.Vector3(0.86, 0, 0.36),
      new THREE.Vector3(0.92, 0, 0.62),
    ]);
    const crossArm = new THREE.Mesh(new THREE.TubeGeometry(crossCurve, 32, 0.08, 22, false), matOmBody);
    crossArm.castShadow = true;
    this.rollPivot.add(crossArm);

    // =========================================================================
    // 5. AXIS 3 — PITCH MOTOR PIVOT (centered behind phone)
    // =========================================================================
    this.pitchPivot = new THREE.Group();
    this.pitchPivot.position.set(0, 0, 0.62);
    this.rollPivot.add(this.pitchPivot);

    const pitchDrum = new THREE.Mesh(new THREE.CylinderGeometry(0.184, 0.184, 0.22, 44), matAnodized);
    pitchDrum.rotation.z = Math.PI / 2;
    pitchDrum.position.set(0.92, 0, 0);
    pitchDrum.castShadow = true;
    this.pitchPivot.add(pitchDrum);

    const pitchCapRing = new THREE.Mesh(new THREE.TorusGeometry(0.186, 0.01, 14, 48), matChromeTrim);
    pitchCapRing.rotation.y = Math.PI / 2;
    pitchCapRing.position.set(1.03, 0, 0);
    this.pitchPivot.add(pitchCapRing);

    const clampBridge = new THREE.Mesh(createRoundedBoxGeo(0.94, 0.155, 0.095, 0.032, 0.008), matOmBody);
    clampBridge.position.set(0.46, 0, -0.092);
    clampBridge.castShadow = true;
    this.pitchPivot.add(clampBridge);

    // Spring-loaded Smartphone Clamp
    const clampBack = new THREE.Mesh(createRoundedBoxGeo(0.35, 0.81, 0.075, 0.05, 0.008), matOmAccent);
    clampBack.position.set(0, 0, -0.085);
    clampBack.castShadow = true;
    this.pitchPivot.add(clampBack);

    const jawTop = new THREE.Mesh(createRoundedBoxGeo(0.33, 0.074, 0.19, 0.024, 0.006), matRubber);
    jawTop.position.set(0, 0.39, 0.0);
    jawTop.castShadow = true;
    this.pitchPivot.add(jawTop);

    const jawBot = new THREE.Mesh(createRoundedBoxGeo(0.33, 0.074, 0.19, 0.024, 0.006), matRubber);
    jawBot.position.set(0, -0.39, 0.0);
    jawBot.castShadow = true;
    this.pitchPivot.add(jawBot);

    // =========================================================================
    // 6. HIGH-FIDELITY GOOGLE PIXEL 11 PRO (152.8 x 72.0 x 8.5 mm)
    // =========================================================================
    this.phoneGroup = new THREE.Group();
    this.phoneGroup.position.set(0, 0, 0.01);
    this.pitchPivot.add(this.phoneGroup);

    const PW = 1.528;
    const PH = 0.72;
    const PD = 0.085;

    // Flat-rail polished satin titanium unibody frame (iconic Pixel Pro silhouette)
    const chassis = new THREE.Mesh(createRoundedBoxGeo(PW, PH, PD, 0.118, 0.013, 32), matPixelRail);
    chassis.castShadow = true;
    chassis.receiveShadow = true;
    this.phoneGroup.add(chassis);

    // Subtle antenna break bands on top & bottom rails
    const antMat = new THREE.MeshStandardMaterial({ color: 0x3b4048, roughness: 0.65, metalness: 0.1 });
    for (const ax of [-0.52, 0.52]) {
      const bandT = new THREE.Mesh(new THREE.BoxGeometry(0.016, 0.008, PD * 0.72), antMat);
      bandT.position.set(ax, PH / 2 - 0.001, 0);
      this.phoneGroup.add(bandT);
      const bandB = bandT.clone();
      bandB.position.y = -PH / 2 + 0.001;
      this.phoneGroup.add(bandB);
    }

    // Tactile Power button (slightly above Volume rocker in portrait -> right-of-volume in landscape)
    const powerBtn = new THREE.Mesh(createRoundedBoxGeo(0.105, 0.022, 0.03, 0.008, 0.003), matChromeTrim);
    powerBtn.position.set(-0.24, PH / 2 + 0.007, 0);
    this.phoneGroup.add(powerBtn);

    const volBtn = new THREE.Mesh(createRoundedBoxGeo(0.21, 0.022, 0.03, 0.008, 0.003), matChromeTrim);
    volBtn.position.set(-0.02, PH / 2 + 0.007, 0);
    this.phoneGroup.add(volBtn);

    // Precision USB-C port + speaker grille holes on right edge (bottom of phone)
    const usbPort = new THREE.Mesh(createRoundedBoxGeo(0.014, 0.095, 0.032, 0.012, 0.002), matRubber);
    usbPort.position.set(PW / 2 + 0.001, 0, 0);
    this.phoneGroup.add(usbPort);
    for (const sy of [-0.15, -0.11, 0.11, 0.15]) {
      const hole = new THREE.Mesh(new THREE.CircleGeometry(0.009, 10), matRubber);
      hole.rotation.y = Math.PI / 2;
      hole.position.set(PW / 2 + 0.002, sy, 0);
      this.phoneGroup.add(hole);
    }

    // --- FRONT FACE (+Z): Edge-to-Edge Corning Glass + True Rounded-Corner OLED ---
    const frontGlassBezel = new THREE.Mesh(
      createRoundedBoxGeo(PW - 0.022, PH - 0.022, 0.01, 0.106, 0.004, 32),
      matGlassVisor
    );
    frontGlassBezel.position.z = PD / 2 + 0.001;
    this.phoneGroup.add(frontGlassBezel);

    // True rounded-corner OLED active display panel (zero protruding sharp corners!)
    const oledMat = new THREE.MeshStandardMaterial({
      map: oledTexture,
      emissiveMap: oledTexture,
      emissive: new THREE.Color(0xffffff),
      emissiveIntensity: 1.0,
      roughness: 0.06,
      metalness: 0.04,
    });
    const oledPlane = new THREE.Mesh(
      createRoundedPlaneGeo(PW - 0.068, PH - 0.064, 0.088, 28),
      oledMat
    );
    oledPlane.position.z = PD / 2 + 0.0078;
    this.phoneGroup.add(oledPlane);
    this.oledMat = oledMat;

    // Subtle thin earpiece speaker slit along top edge (left in landscape)
    const earSlit = new THREE.Mesh(createRoundedBoxGeo(0.008, 0.11, 0.004, 0.003, 0.001), matRubber);
    earSlit.position.set(-PW / 2 + 0.022, 0, PD / 2 + 0.0085);
    this.phoneGroup.add(earSlit);

    // Front punch-hole selfie camera with multi-layer AR optical glint
    const frontRing = new THREE.Mesh(new THREE.CircleGeometry(0.021, 24), matRubber);
    frontRing.position.set(-PW / 2 + 0.082, 0, PD / 2 + 0.0088);
    this.phoneGroup.add(frontRing);

    const frontGlass = new THREE.Mesh(
      new THREE.CircleGeometry(0.013, 24),
      new THREE.MeshPhysicalMaterial({
        color: 0x0b1a3d,
        emissive: 0x0d214f,
        emissiveIntensity: 0.4,
        roughness: 0.02,
        metalness: 0.35,
        clearcoat: 1,
      })
    );
    frontGlass.position.set(-PW / 2 + 0.082, 0, PD / 2 + 0.0094);
    this.phoneGroup.add(frontGlass);

    // --- REAR FACE (-Z): Frosted Obsidian Glass + Iconic Pixel Pro Pill Camera Bar ---
    const backGlass = new THREE.Mesh(
      createRoundedPlaneGeo(PW - 0.028, PH - 0.028, 0.102, 28),
      matPixelBack
    );
    backGlass.rotation.y = Math.PI;
    backGlass.position.z = -PD / 2 - 0.0015;
    this.phoneGroup.add(backGlass);

    // Iconic Elevated Pixel Pro Stadium-Pill Camera Bar (landscape left end, clear of center clamp)
    const barGroup = new THREE.Group();
    barGroup.position.set(-0.46, 0, -PD / 2 - 0.022);
    this.phoneGroup.add(barGroup);

    // Polished beveled titanium outer stadium housing
    const barShell = new THREE.Mesh(
      createRoundedBoxGeo(0.265, 0.645, 0.048, 0.132, 0.012, 32),
      matPixelRail
    );
    barShell.castShadow = true;
    barGroup.add(barShell);

    // Inner recessed mirror-black glass optical pill
    const barVisor = new THREE.Mesh(
      createRoundedBoxGeo(0.215, 0.59, 0.014, 0.106, 0.004, 32),
      matGlassVisor
    );
    barVisor.position.z = -0.022;
    barGroup.add(barVisor);

    // 3 Multi-Element Flagship Pixel Pro Lenses + Flash & Temp Sensor
    const lensARMain = new THREE.MeshPhysicalMaterial({
      color: 0x0b1636,
      emissive: 0x0c1e4a,
      emissiveIntensity: 0.38,
      roughness: 0.015,
      metalness: 0.35,
      clearcoat: 1.0,
      clearcoatRoughness: 0.015,
    });
    const lensARTele = new THREE.MeshPhysicalMaterial({
      color: 0x0c2433,
      emissive: 0x0d2b3d,
      emissiveIntensity: 0.35,
      roughness: 0.015,
      metalness: 0.4,
      clearcoat: 1.0,
    });

    const lensSpecs = [
      { y: 0.178, r: 0.056, kind: 'wide' },       // 50MP Main Wide
      { y: 0.028, r: 0.05, kind: 'ultrawide' },   // 48MP Ultra-Wide
      { y: -0.118, r: 0.054, kind: 'periscope' }, // 48MP 5x Periscope Telephoto
    ];
    lensSpecs.forEach(({ y, r, kind }) => {
      // Machined metallic outer trim ring
      const trim = new THREE.Mesh(new THREE.TorusGeometry(r, 0.0065, 14, 32), matChromeTrim);
      trim.position.set(0, y, -0.03);
      barGroup.add(trim);

      // Deep matte black anodized optical barrel
      const barrel = new THREE.Mesh(new THREE.CylinderGeometry(r - 0.005, r - 0.008, 0.016, 28), matRubber);
      barrel.rotation.x = Math.PI / 2;
      barrel.position.set(0, y, -0.025);
      barGroup.add(barrel);

      // Inner aperture step ring
      const innerStep = new THREE.Mesh(new THREE.TorusGeometry(r * 0.64, 0.004, 10, 24), matAnodized);
      innerStep.position.set(0, y, -0.031);
      barGroup.add(innerStep);

      if (kind === 'periscope') {
        const prism = new THREE.Mesh(new THREE.BoxGeometry(0.052, 0.042, 0.006), lensARTele);
        prism.position.set(0, y, -0.032);
        barGroup.add(prism);
      } else {
        const glass = new THREE.Mesh(new THREE.CircleGeometry(r * 0.62, 24), lensARMain);
        glass.rotation.y = Math.PI;
        glass.position.set(0, y, -0.0325);
        barGroup.add(glass);

        // Micro specular AR inner pupil dot
        const pupil = new THREE.Mesh(new THREE.CircleGeometry(r * 0.24, 16), matRubber);
        pupil.rotation.y = Math.PI;
        pupil.position.set(0, y, -0.0332);
        barGroup.add(pupil);
      }
    });

    // Dual-tone LED Flash + Temperature Sensor dots at lower end of Pixel Camera Pill
    const flashMesh = new THREE.Mesh(
      new THREE.CircleGeometry(0.024, 20),
      new THREE.MeshStandardMaterial({ color: 0xfff2d6, roughness: 0.22, metalness: 0.1 })
    );
    flashMesh.rotation.y = Math.PI;
    flashMesh.position.set(0.034, -0.225, -0.031);
    barGroup.add(flashMesh);

    const tempSensor = new THREE.Mesh(
      new THREE.CircleGeometry(0.016, 18),
      new THREE.MeshStandardMaterial({ color: 0x1e222b, roughness: 0.25, metalness: 0.5 })
    );
    tempSensor.rotation.y = Math.PI;
    tempSensor.position.set(-0.034, -0.225, -0.031);
    barGroup.add(tempSensor);

    // =========================================================================
    // 7. 8-LED RGB HILIGHT RING + REAL SHADOW SPOTLIGHT + VOLUMETRIC BEAM
    // =========================================================================
    this.hilightOrigin = new THREE.Group();
    // Sleek precision quartz-diffuser HiLight ring integrated cleanly around the main camera optics
    this.hilightOrigin.position.set(-0.46, 0.028, -PD / 2 - 0.052);
    this.phoneGroup.add(this.hilightOrigin);

    const diffuserHousing = new THREE.Mesh(
      new THREE.TorusGeometry(0.128, 0.014, 18, 52),
      new THREE.MeshPhysicalMaterial({
        color: 0xf4f6fa,
        roughness: 0.22,
        metalness: 0.08,
        transmission: 0.5,
        transparent: true,
        opacity: 0.72,
      })
    );
    this.hilightOrigin.add(diffuserHousing);

    this.ledMats = [];
    this.ledMeshes = [];
    const ringR = 0.128;
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      const mat = new THREE.MeshStandardMaterial({
        color: 0xffe4b5,
        emissive: new THREE.Color(0xffc878),
        emissiveIntensity: 0.2,
        roughness: 0.15,
        toneMapped: false,
      });
      const seg = new THREE.Mesh(new THREE.CapsuleGeometry(0.013, 0.028, 6, 12), mat);
      seg.position.set(Math.cos(a) * ringR, Math.sin(a) * ringR, -0.002);
      seg.rotation.z = a;
      this.hilightOrigin.add(seg);
      this.ledMats.push(mat);
      this.ledMeshes.push(seg);
    }

    // Shadow-Casting Warm Desk Lamp SpotLight (strong enough to paint a warm golden pool even in daylight!)
    this.lampSpot = new THREE.SpotLight(0xffcb78, 0.0, 13.0, Math.PI * 0.23, 0.68, 1.45);
    this.lampSpot.position.set(0, 0, -0.02);
    this.lampSpot.castShadow = true;
    this.lampSpot.shadow.mapSize.set(2048, 2048);
    this.lampSpot.shadow.bias = -0.0004;
    this.lampSpot.shadow.normalBias = 0.02;
    this.lampSpot.shadow.radius = 4;

    this.lampTarget = new THREE.Object3D();
    this.lampTarget.position.set(0, 0, -4.5);
    this.hilightOrigin.add(this.lampTarget);
    this.lampSpot.target = this.lampTarget;
    this.hilightOrigin.add(this.lampSpot);

    // Warm spill PointLight around the rear diffuser
    this.lampSpill = new THREE.PointLight(0xffcb78, 0.0, 2.8, 2.0);
    this.lampSpill.position.set(0, 0, -0.14);
    this.hilightOrigin.add(this.lampSpill);

    // Custom GLSL Volumetric Light Cone Shader
    const vol = createVolumetricBeamMesh();
    this.beamMesh = vol.mesh;
    this.beamMat = vol.mat;
    this.hilightOrigin.add(this.beamMesh);

    // Subtle dust motes illuminated inside the warm lamp cone (prominent in Golden/Night)
    const dustCount = 42;
    const dustPositions = new Float32Array(dustCount * 3);
    for (let i = 0; i < dustCount; i++) {
      const z = -(0.4 + Math.random() * 3.6);
      const maxR = 0.13 + (-z / 5.6) * 1.4;
      const r = Math.sqrt(Math.random()) * maxR;
      const th = Math.random() * Math.PI * 2;
      dustPositions[i * 3] = Math.cos(th) * r;
      dustPositions[i * 3 + 1] = Math.sin(th) * r;
      dustPositions[i * 3 + 2] = z;
    }
    const dustGeo = new THREE.BufferGeometry();
    dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPositions, 3));
    this.dustMat = new THREE.PointsMaterial({
      color: 0xffe4b8,
      size: 0.022,
      transparent: true,
      opacity: 0.0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.dustPoints = new THREE.Points(dustGeo, this.dustMat);
    this.hilightOrigin.add(this.dustPoints);

    this.panDeg = 0;
    this.rollDeg = 0;
    this.pitchDeg = 0;
  }

  setRoomMood(mood) {
    this.roomMood = mood;
    this.beamMat.uniforms.uMoodScale.value = mood === 'day' ? 0.45 : mood === 'golden' ? 0.78 : 1.05;
  }

  setJoints(panDeg, pitchDeg, rollDeg = 0) {
    this.panDeg = Math.max(-165, Math.min(165, panDeg));
    this.pitchDeg = Math.max(-30, Math.min(65, pitchDeg));
    this.rollDeg = Math.max(-25, Math.min(25, rollDeg));

    this.panPivot.rotation.y = THREE.MathUtils.degToRad(this.panDeg);
    this.rollPivot.rotation.z = THREE.MathUtils.degToRad(this.rollDeg);
    this.pitchPivot.rotation.x = THREE.MathUtils.degToRad(-this.pitchDeg);
  }

  getHeadWorldPosition(target = new THREE.Vector3()) {
    this.phoneGroup.getWorldPosition(target);
    return target;
  }

  computeFrontAimAngles(worldTarget) {
    const headPos = new THREE.Vector3(0, 3.28, 0);
    const dx = worldTarget.x - headPos.x;
    const dy = worldTarget.y - headPos.y;
    const dz = worldTarget.z - headPos.z;
    const horiz = Math.max(0.05, Math.hypot(dx, dz));
    const pan = THREE.MathUtils.radToDeg(Math.atan2(dx, dz));
    const pitch = THREE.MathUtils.radToDeg(Math.atan2(dy, horiz));
    return {
      panDeg: Math.max(-155, Math.min(155, pan)),
      pitchDeg: Math.max(-28, Math.min(35, pitch)),
    };
  }

  computeRearLampAimAngles(worldTarget) {
    const headPos = new THREE.Vector3(0, 3.28, 0);
    const dx = worldTarget.x - headPos.x;
    const dy = worldTarget.y - headPos.y;
    const dz = worldTarget.z - headPos.z;
    const horiz = Math.max(0.05, Math.hypot(dx, dz));
    let pan = THREE.MathUtils.radToDeg(Math.atan2(-dx, -dz));
    if (pan > 162) pan = 162;
    if (pan < -162) pan = -162;
    const pitch = THREE.MathUtils.radToDeg(Math.atan2(-dy, horiz));
    return {
      panDeg: pan,
      pitchDeg: Math.max(15, Math.min(62, pitch)),
    };
  }

  updateHiLight({ colorsHex, intensities, beamColorHex, beamIntensity, time }) {
    for (let i = 0; i < 8; i++) {
      const col = new THREE.Color(colorsHex[i] ?? beamColorHex ?? 0xffcb78);
      const k = Math.max(0, intensities[i] ?? 0);
      this.ledMats[i].color.copy(col);
      this.ledMats[i].emissive.copy(col);
      this.ledMats[i].emissiveIntensity = k * 3.8;
    }

    const bCol = new THREE.Color(beamColorHex ?? 0xffcb78);
    const bi = Math.max(0, Math.min(1.5, beamIntensity));
    // Stronger SpotLight gain in bright daylight so the warm golden pool on paper is unmistakable!
    const spotGain = this.roomMood === 'day' ? 42.0 : this.roomMood === 'golden' ? 35.0 : 30.0;
    this.lampSpot.color.copy(bCol);
    this.lampSpot.intensity = bi * spotGain;
    this.lampSpill.color.copy(bCol);
    this.lampSpill.intensity = bi * 1.45;

    this.beamMat.uniforms.uColor.value.copy(bCol);
    this.beamMat.uniforms.uIntensity.value = bi;
    this.beamMat.uniforms.uTime.value = time;

    const dustScale = this.roomMood === 'day' ? 0.14 : this.roomMood === 'golden' ? 0.3 : 0.45;
    this.dustMat.color.copy(bCol);
    this.dustMat.opacity = Math.min(0.5, bi * dustScale);
    this.dustPoints.rotation.z = time * 0.05;
  }

  setHandleLed(hex, intensity = 2.0) {
    this.handleLedMat.color.set(hex);
    this.handleLedMat.emissive.set(hex);
    this.handleLedMat.emissiveIntensity = intensity;
  }

  estimatePitchTorquePct(pitchVelDps, pitchAccelDps2 = 0) {
    const staticFactor = this.pitchDeg < 0 ? Math.abs(this.pitchDeg) * 1.15 : Math.abs(this.pitchDeg) * 0.35;
    const dynamicFactor = Math.abs(pitchVelDps) * 0.48 + Math.abs(pitchAccelDps2) * 0.06;
    return Math.max(8, Math.min(100, Math.round(14 + staticFactor + dynamicFactor)));
  }
}
