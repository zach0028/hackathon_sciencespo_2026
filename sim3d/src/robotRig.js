import * as THREE from 'three';

/**
 * Create a rounded-rectangle ExtrudeGeometry centered at (0, 0, 0).
 */
function createRoundedBoxGeo(w, h, d, radius, bevel = 0.012, curveSegments = 16) {
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
    depth: Math.max(0.005, d - bevel * 2),
    bevelEnabled: true,
    bevelThickness: bevel,
    bevelSize: bevel,
    bevelSegments: 4,
    curveSegments,
  });
  geo.center();
  return geo;
}

/**
 * Custom GLSL Volumetric Lamp Beam Cone (Fresnel edge fade + longitudinal falloff + subtle warmth).
 */
function createVolumetricBeamMesh() {
  const height = 5.4;
  const topR = 0.14;
  const botR = 1.95;
  const geo = new THREE.CylinderGeometry(topR, botR, height, 48, 24, true);
  // Orient cylinder so apex is at z=0 and beam extends along local -Z
  geo.translate(0, -height / 2, 0);
  geo.rotateX(-Math.PI / 2);

  const mat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    uniforms: {
      uColor: { value: new THREE.Color(0xffd088) },
      uIntensity: { value: 0.0 },
      uTime: { value: 0.0 },
    },
    vertexShader: `
      varying vec3 vNormalView;
      varying vec3 vViewDir;
      varying vec2 vUv;
      varying float vDist;
      void main() {
        vUv = uv;
        vDist = 1.0 - uv.y; // 0 at LED ring, 1 at far end
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
      varying vec3 vNormalView;
      varying vec3 vViewDir;
      varying vec2 vUv;
      varying float vDist;
      void main() {
        float fresnel = pow(abs(dot(normalize(vNormalView), normalize(vViewDir))), 1.75);
        float longFade = smoothstep(0.0, 0.06, vDist) * pow(1.0 - vDist, 1.65);
        float shimmer = 0.94 + 0.06 * sin(vUv.x * 25.13 + uTime * 1.4) * sin(vDist * 9.0 - uTime * 0.9);
        float alpha = fresnel * longFade * shimmer * uIntensity * 0.34;
        vec3 col = mix(uColor * 1.25, uColor * 0.85, vDist);
        gl_FragColor = vec4(col, alpha);
      }
    `,
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.renderOrder = 10;
  return { mesh, mat };
}

/**
 * CAD-accurate 1:1 mm articulated DJI Osmo Mobile 3 + Google Pixel 11 Pro + 8-LED HiLight Rig.
 * Scale: 1.0 Three.js unit = 10 cm (1 mm = 0.01 units).
 */
export class OsmoRobotRig {
  constructor({ oledTexture, brushedMetalTex }) {
    this.root = new THREE.Group();
    this.root.name = 'OsmoRobotRoot';

    // Shared high-fidelity PBR materials
    const matGimbalBody = new THREE.MeshStandardMaterial({
      color: 0x7c828c,
      roughness: 0.48,
      metalness: 0.16,
    });
    const matGripDark = new THREE.MeshStandardMaterial({
      color: 0x2a2e35,
      roughness: 0.82,
      metalness: 0.06,
    });
    const matAnodized = new THREE.MeshStandardMaterial({
      color: 0x393f4a,
      roughness: 0.28,
      metalness: 0.82,
      bumpMap: brushedMetalTex,
      bumpScale: 0.004,
    });
    const matChromeTrim = new THREE.MeshStandardMaterial({
      color: 0xc8d0dc,
      roughness: 0.18,
      metalness: 0.94,
    });
    const matRubber = new THREE.MeshStandardMaterial({
      color: 0x181a1f,
      roughness: 0.92,
      metalness: 0.02,
    });
    const matTitanium = new THREE.MeshStandardMaterial({
      color: 0x30353e,
      roughness: 0.24,
      metalness: 0.9,
      bumpMap: brushedMetalTex,
      bumpScale: 0.003,
    });
    const matPixelBack = new THREE.MeshPhysicalMaterial({
      color: 0x161920,
      roughness: 0.42,
      metalness: 0.08,
      clearcoat: 0.38,
      clearcoatRoughness: 0.28,
    });
    const matGlassVisor = new THREE.MeshPhysicalMaterial({
      color: 0x05070c,
      roughness: 0.04,
      metalness: 0.12,
      clearcoat: 1.0,
      clearcoatRoughness: 0.03,
      reflectivity: 1.0,
    });

    // =========================================================================
    // 1. WEIGHTED FOLDING TRIPOD BASE (y = 0 .. 0.46)
    // =========================================================================
    const tripod = new THREE.Group();
    this.root.add(tripod);

    const hubGeo = new THREE.CylinderGeometry(0.18, 0.21, 0.18, 32);
    const hub = new THREE.Mesh(hubGeo, matGripDark);
    hub.position.y = 0.36;
    hub.castShadow = true;
    hub.receiveShadow = true;
    tripod.add(hub);

    const collar = new THREE.Mesh(new THREE.CylinderGeometry(0.155, 0.17, 0.06, 32), matChromeTrim);
    collar.position.y = 0.47;
    collar.castShadow = true;
    tripod.add(collar);

    for (let i = 0; i < 3; i++) {
      const ang = (i * Math.PI * 2) / 3 + Math.PI / 6;
      const legGroup = new THREE.Group();
      legGroup.rotation.y = ang;
      legGroup.position.y = 0.34;

      const leg = new THREE.Mesh(createRoundedBoxGeo(0.13, 0.1, 0.98, 0.03, 0.01), matGripDark);
      leg.position.set(0, -0.15, 0.46);
      leg.rotation.x = 0.34;
      leg.castShadow = true;
      leg.receiveShadow = true;
      legGroup.add(leg);

      // Rubber foot pad resting flush on desk y=0
      const foot = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.075, 0.05, 16), matRubber);
      foot.position.set(0, -0.315, 0.88);
      foot.castShadow = true;
      tripod.add(legGroup);
      legGroup.add(foot);
    }

    // =========================================================================
    // 2. ERGONOMIC SLANTED DJI OM3 HANDLE (y = 0.48 .. 2.02, slanted 11° back)
    // =========================================================================
    const handleGroup = new THREE.Group();
    handleGroup.position.set(0, 0.48, 0);
    handleGroup.rotation.x = -0.19; // authentic ergonomic slant
    this.root.add(handleGroup);

    const handleCore = new THREE.Mesh(createRoundedBoxGeo(0.36, 1.48, 0.42, 0.14, 0.02), matGimbalBody);
    handleCore.position.y = 0.74;
    handleCore.castShadow = true;
    handleCore.receiveShadow = true;
    handleGroup.add(handleCore);

    // Textured dark rear rubber grip
    const rearGrip = new THREE.Mesh(createRoundedBoxGeo(0.37, 0.96, 0.24, 0.11, 0.015), matGripDark);
    rearGrip.position.set(0, 0.64, -0.11);
    rearGrip.castShadow = true;
    handleGroup.add(rearGrip);

    // Control panel bezel on front of handle (+Z)
    const ctrlPanel = new THREE.Mesh(createRoundedBoxGeo(0.27, 0.56, 0.05, 0.08, 0.008), matRubber);
    ctrlPanel.position.set(0, 1.12, 0.2);
    handleGroup.add(ctrlPanel);

    // Analog thumb joystick
    const joyBase = new THREE.Mesh(new THREE.CylinderGeometry(0.068, 0.075, 0.05, 24), matAnodized);
    joyBase.rotation.x = Math.PI / 2;
    joyBase.position.set(0, 1.24, 0.235);
    handleGroup.add(joyBase);

    const joyCap = new THREE.Mesh(new THREE.CylinderGeometry(0.062, 0.055, 0.03, 24), matGripDark);
    joyCap.rotation.x = Math.PI / 2;
    joyCap.position.set(0, 1.24, 0.265);
    handleGroup.add(joyCap);

    // M (Power/Mode) button + red record dot
    const btnM = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.042, 0.025, 20), matAnodized);
    btnM.rotation.x = Math.PI / 2;
    btnM.position.set(-0.065, 1.05, 0.23);
    handleGroup.add(btnM);

    const btnRec = new THREE.Mesh(
      new THREE.CylinderGeometry(0.038, 0.038, 0.025, 20),
      new THREE.MeshStandardMaterial({ color: 0xd9383a, roughness: 0.35, metalness: 0.2 })
    );
    btnRec.rotation.x = Math.PI / 2;
    btnRec.position.set(0.065, 1.05, 0.23);
    handleGroup.add(btnRec);

    // Live OM3 handle status LED + 3 battery micro-LEDs
    this.handleLedMat = new THREE.MeshStandardMaterial({
      color: 0x34d399,
      emissive: 0x34d399,
      emissiveIntensity: 2.2,
      roughness: 0.2,
    });
    const statusLed = new THREE.Mesh(new THREE.SphereGeometry(0.018, 12, 12), this.handleLedMat);
    statusLed.position.set(0, 0.95, 0.228);
    handleGroup.add(statusLed);

    const battLedMat = new THREE.MeshStandardMaterial({
      color: 0xe2e8f0,
      emissive: 0xd8f5e8,
      emissiveIntensity: 1.4,
    });
    for (let i = 0; i < 3; i++) {
      const bLed = new THREE.Mesh(new THREE.SphereGeometry(0.011, 10, 10), battLedMat);
      bLed.position.set(-0.045 + i * 0.045, 0.895, 0.226);
      handleGroup.add(bLed);
    }

    // Left zoom slider + rear trigger
    const zoomRocker = new THREE.Mesh(createRoundedBoxGeo(0.04, 0.22, 0.07, 0.015, 0.005), matAnodized);
    zoomRocker.position.set(-0.19, 1.14, 0.02);
    handleGroup.add(zoomRocker);

    const rearTrigger = new THREE.Mesh(createRoundedBoxGeo(0.11, 0.24, 0.08, 0.03, 0.008), matAnodized);
    rearTrigger.position.set(0, 1.15, -0.22);
    rearTrigger.rotation.x = 0.22;
    handleGroup.add(rearTrigger);

    // Top gimbal neck collar (re-leveled horizontally!)
    const neckMount = new THREE.Group();
    neckMount.position.set(0, 1.5, 0);
    neckMount.rotation.x = 0.19; // cancel handle slant so Pan axis is purely vertical!
    handleGroup.add(neckMount);

    const neckBase = new THREE.Mesh(new THREE.CylinderGeometry(0.21, 0.19, 0.14, 36), matGimbalBody);
    neckBase.position.y = 0.05;
    neckBase.castShadow = true;
    neckMount.add(neckBase);

    // =========================================================================
    // 3. AXIS 1 — PAN MOTOR PIVOT (rotates ±160° around vertical Y)
    // =========================================================================
    this.panPivot = new THREE.Group();
    this.panPivot.position.set(0, 0.14, 0);
    neckMount.add(this.panPivot);

    const panDrum = new THREE.Mesh(new THREE.CylinderGeometry(0.235, 0.235, 0.28, 40), matAnodized);
    panDrum.position.y = 0.14;
    panDrum.castShadow = true;
    this.panPivot.add(panDrum);

    const panRing = new THREE.Mesh(new THREE.TorusGeometry(0.236, 0.012, 12, 48), matChromeTrim);
    panRing.rotation.x = Math.PI / 2;
    panRing.position.y = 0.26;
    this.panPivot.add(panRing);

    // Swept DJI OM3 folding L-arm curving up & back to Roll motor
    const armCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0.26, 0),
      new THREE.Vector3(0, 0.44, -0.26),
      new THREE.Vector3(0, 0.78, -0.58),
      new THREE.Vector3(0, 1.18, -0.62),
    ]);
    const sweptArm = new THREE.Mesh(new THREE.TubeGeometry(armCurve, 32, 0.115, 20, false), matGimbalBody);
    sweptArm.castShadow = true;
    sweptArm.receiveShadow = true;
    this.panPivot.add(sweptArm);

    // Articulated folding knuckle cap on L-arm
    const knuckle = new THREE.Mesh(new THREE.CylinderGeometry(0.145, 0.145, 0.22, 28), matAnodized);
    knuckle.rotation.z = Math.PI / 2;
    knuckle.position.set(0, 0.42, -0.26);
    knuckle.castShadow = true;
    this.panPivot.add(knuckle);

    // =========================================================================
    // 4. AXIS 2 — ROLL MOTOR PIVOT (rotates around local Z)
    // =========================================================================
    this.rollPivot = new THREE.Group();
    this.rollPivot.position.set(0, 1.18, -0.62);
    this.panPivot.add(this.rollPivot);

    const rollDrum = new THREE.Mesh(new THREE.CylinderGeometry(0.21, 0.21, 0.26, 36), matAnodized);
    rollDrum.rotation.x = Math.PI / 2;
    rollDrum.castShadow = true;
    this.rollPivot.add(rollDrum);

    const rollRing = new THREE.Mesh(new THREE.TorusGeometry(0.212, 0.011, 12, 40), matChromeTrim);
    rollRing.position.z = 0.12;
    this.rollPivot.add(rollRing);

    // Horizontal cross-arm extending from Roll drum (+X) to Pitch motor
    const crossCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0, 0.08),
      new THREE.Vector3(0.45, 0, 0.14),
      new THREE.Vector3(0.86, 0, 0.36),
      new THREE.Vector3(0.92, 0, 0.62),
    ]);
    const crossArm = new THREE.Mesh(new THREE.TubeGeometry(crossCurve, 28, 0.082, 18, false), matGimbalBody);
    crossArm.castShadow = true;
    this.rollPivot.add(crossArm);

    // =========================================================================
    // 5. AXIS 3 — PITCH MOTOR PIVOT (rotates around local X, centered behind phone!)
    // =========================================================================
    this.pitchPivot = new THREE.Group();
    this.pitchPivot.position.set(0, 0, 0.62);
    this.rollPivot.add(this.pitchPivot);

    const pitchDrum = new THREE.Mesh(new THREE.CylinderGeometry(0.185, 0.185, 0.22, 36), matAnodized);
    pitchDrum.rotation.z = Math.PI / 2;
    pitchDrum.position.set(0.92, 0, 0);
    pitchDrum.castShadow = true;
    this.pitchPivot.add(pitchDrum);

    const pitchCapRing = new THREE.Mesh(new THREE.TorusGeometry(0.187, 0.01, 12, 40), matChromeTrim);
    pitchCapRing.rotation.y = Math.PI / 2;
    pitchCapRing.position.set(1.03, 0, 0);
    this.pitchPivot.add(pitchCapRing);

    // Clamp bridge connecting right Pitch drum to center smartphone clamp
    const clampBridge = new THREE.Mesh(createRoundedBoxGeo(0.94, 0.16, 0.1, 0.03, 0.008), matGimbalBody);
    clampBridge.position.set(0.46, 0, -0.09);
    clampBridge.castShadow = true;
    this.pitchPivot.add(clampBridge);

    // Spring-loaded Smartphone Clamp (center jaws)
    const clampBack = new THREE.Mesh(createRoundedBoxGeo(0.36, 0.82, 0.08, 0.05, 0.008), matAnodized);
    clampBack.position.set(0, 0, -0.085);
    clampBack.castShadow = true;
    this.pitchPivot.add(clampBack);

    const jawTop = new THREE.Mesh(createRoundedBoxGeo(0.34, 0.08, 0.2, 0.025, 0.006), matRubber);
    jawTop.position.set(0, 0.395, 0.0);
    jawTop.castShadow = true;
    this.pitchPivot.add(jawTop);

    const jawBot = new THREE.Mesh(createRoundedBoxGeo(0.34, 0.08, 0.2, 0.025, 0.006), matRubber);
    jawBot.position.set(0, -0.395, 0.0);
    jawBot.castShadow = true;
    this.pitchPivot.add(jawBot);

    // =========================================================================
    // 6. GOOGLE PIXEL 11 PRO (152.8 x 72.0 x 8.5 mm = 1.528 x 0.720 x 0.085)
    // =========================================================================
    this.phoneGroup = new THREE.Group();
    this.phoneGroup.position.set(0, 0, 0.01);
    this.pitchPivot.add(this.phoneGroup);

    const PW = 1.528;
    const PH = 0.72;
    const PD = 0.086;

    // Brushed titanium unibody chassis
    const chassis = new THREE.Mesh(createRoundedBoxGeo(PW, PH, PD, 0.105, 0.014, 24), matTitanium);
    chassis.castShadow = true;
    chassis.receiveShadow = true;
    this.phoneGroup.add(chassis);

    // Side tactile buttons (Power + Volume rocker on top landscape rail)
    const powerBtn = new THREE.Mesh(createRoundedBoxGeo(0.11, 0.024, 0.032, 0.006, 0.003), matChromeTrim);
    powerBtn.position.set(0.28, PH / 2 + 0.008, 0);
    this.phoneGroup.add(powerBtn);

    const volBtn = new THREE.Mesh(createRoundedBoxGeo(0.22, 0.024, 0.032, 0.006, 0.003), matChromeTrim);
    volBtn.position.set(0.05, PH / 2 + 0.008, 0);
    this.phoneGroup.add(volBtn);

    // USB-C port recess on right edge
    const usbPort = new THREE.Mesh(createRoundedBoxGeo(0.016, 0.09, 0.032, 0.01, 0.002), matRubber);
    usbPort.position.set(PW / 2 + 0.001, 0, 0);
    this.phoneGroup.add(usbPort);

    // --- FRONT FACE (+Z): Corning Glass + Live 120Hz OLED Texture ---
    const bezelMesh = new THREE.Mesh(createRoundedBoxGeo(PW - 0.026, PH - 0.026, 0.012, 0.092, 0.004, 24), matGlassVisor);
    bezelMesh.position.z = PD / 2 + 0.001;
    this.phoneGroup.add(bezelMesh);

    const oledMat = new THREE.MeshStandardMaterial({
      map: oledTexture,
      emissiveMap: oledTexture,
      emissive: new THREE.Color(0xffffff),
      emissiveIntensity: 1.05,
      roughness: 0.08,
      metalness: 0.05,
    });
    const oledPlane = new THREE.Mesh(new THREE.PlaneGeometry(PW - 0.075, PH - 0.068), oledMat);
    oledPlane.position.z = PD / 2 + 0.009;
    this.phoneGroup.add(oledPlane);
    this.oledMat = oledMat;

    // Tiny front selfie punch-hole lens (landscape left-center)
    const frontCam = new THREE.Mesh(
      new THREE.CircleGeometry(0.02, 20),
      new THREE.MeshPhysicalMaterial({ color: 0x091326, roughness: 0.05, metalness: 0.4, clearcoat: 1 })
    );
    frontCam.position.set(-PW / 2 + 0.09, 0, PD / 2 + 0.0105);
    this.phoneGroup.add(frontCam);

    // --- REAR FACE (-Z): Frosted Obsidian Back + Signature Pixel Pill Camera Bar + 8-LED HiLight Ring ---
    const backGlass = new THREE.Mesh(createRoundedBoxGeo(PW - 0.032, PH - 0.032, 0.01, 0.09, 0.004, 24), matPixelBack);
    backGlass.position.z = -PD / 2 - 0.001;
    this.phoneGroup.add(backGlass);

    // Signature elevated Pixel Camera Bar (located on left end in landscape, away from center clamp!)
    const barGroup = new THREE.Group();
    barGroup.position.set(-0.46, 0, -PD / 2 - 0.018);
    this.phoneGroup.add(barGroup);

    const barShell = new THREE.Mesh(createRoundedBoxGeo(0.24, 0.62, 0.044, 0.11, 0.01, 24), matTitanium);
    barShell.castShadow = true;
    barGroup.add(barShell);

    const barVisor = new THREE.Mesh(createRoundedBoxGeo(0.195, 0.565, 0.016, 0.09, 0.005, 24), matGlassVisor);
    barVisor.position.z = -0.022;
    barGroup.add(barVisor);

    // 3 multi-element recessed optical lenses (Ultra-wide 0.51x, Main 50MP 1x, Periscope 5x)
    const lensARMat = new THREE.MeshPhysicalMaterial({
      color: 0x0d1b3a,
      emissive: 0x0b1a40,
      emissiveIntensity: 0.35,
      roughness: 0.02,
      metalness: 0.3,
      clearcoat: 1.0,
      clearcoatRoughness: 0.02,
    });
    const lensYs = [0.165, 0.0, -0.165];
    lensYs.forEach((ly, idx) => {
      const bezel = new THREE.Mesh(new THREE.TorusGeometry(0.056, 0.008, 12, 28), matChromeTrim);
      bezel.position.set(0, ly, -0.03);
      barGroup.add(bezel);

      const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.048, 0.048, 0.016, 24), matRubber);
      barrel.rotation.x = Math.PI / 2;
      barrel.position.set(0, ly, -0.025);
      barGroup.add(barrel);

      if (idx === 2) {
        // Periscope telephoto rectangular prism inside circle!
        const prism = new THREE.Mesh(new THREE.BoxGeometry(0.056, 0.044, 0.006), lensARMat);
        prism.position.set(0, ly, -0.031);
        barGroup.add(prism);
      } else {
        const glass = new THREE.Mesh(new THREE.CircleGeometry(0.036, 24), lensARMat);
        glass.rotation.y = Math.PI;
        glass.position.set(0, ly, -0.032);
        barGroup.add(glass);
      }
    });

    // =========================================================================
    // 7. 8-LED RGB HILIGHT RING + REAL SHADOW SPOTLIGHT + GLSL VOLUMETRIC BEAM!
    // =========================================================================
    this.hilightOrigin = new THREE.Group();
    // Centered cleanly around the rear HiLight emitter module beside the camera bar
    this.hilightOrigin.position.set(-0.46, 0, -PD / 2 - 0.048);
    this.phoneGroup.add(this.hilightOrigin);

    // Subtle metallic diffuser housing ring around the central camera / HiLight ring
    const diffuserRing = new THREE.Mesh(
      new THREE.TorusGeometry(0.125, 0.016, 16, 48),
      new THREE.MeshPhysicalMaterial({
        color: 0xe6e9ef,
        roughness: 0.25,
        metalness: 0.1,
        transmission: 0.45,
        transparent: true,
        opacity: 0.65,
      })
    );
    this.hilightOrigin.add(diffuserRing);

    // 8 individually addressable emissive RGB LED segments (matching Android 17 LightsManager!)
    this.ledMats = [];
    this.ledMeshes = [];
    const ringR = 0.125;
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      const mat = new THREE.MeshStandardMaterial({
        color: 0xffe4b5,
        emissive: new THREE.Color(0xffc878),
        emissiveIntensity: 0.2,
        roughness: 0.15,
        toneMapped: false, // drives crisp selective UnrealBloom!
      });
      const seg = new THREE.Mesh(new THREE.CapsuleGeometry(0.015, 0.028, 6, 12), mat);
      seg.position.set(Math.cos(a) * ringR, Math.sin(a) * ringR, -0.002);
      seg.rotation.z = a;
      this.hilightOrigin.add(seg);
      this.ledMats.push(mat);
      this.ledMeshes.push(seg);
    }

    // Real Shadow-Casting Desk Lamp SpotLight pointing along local -Z
    this.lampSpot = new THREE.SpotLight(0xffcf87, 0.0, 12.0, Math.PI * 0.24, 0.72, 1.55);
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

    // Soft local warm spill PointLight around the rear diffuser (illuminates phone bar & gimbal rim!)
    this.lampSpill = new THREE.PointLight(0xffcf87, 0.0, 2.8, 2.0);
    this.lampSpill.position.set(0, 0, -0.14);
    this.hilightOrigin.add(this.lampSpill);

    // Custom GLSL Volumetric Light Cone Shader!
    const vol = createVolumetricBeamMesh();
    this.beamMesh = vol.mesh;
    this.beamMat = vol.mat;
    this.hilightOrigin.add(this.beamMesh);

    // Subtle floating dust motes illuminated inside the warm lamp cone
    const dustCount = 48;
    const dustPositions = new Float32Array(dustCount * 3);
    for (let i = 0; i < dustCount; i++) {
      const z = -(0.4 + Math.random() * 3.8);
      const maxR = 0.14 + (-z / 5.4) * 1.45;
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
      size: 0.024,
      transparent: true,
      opacity: 0.0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.dustPoints = new THREE.Points(dustGeo, this.dustMat);
    this.hilightOrigin.add(this.dustPoints);

    // Current kinematic state (degrees)
    this.panDeg = 0;
    this.rollDeg = 0;
    this.pitchDeg = 0;
    this._tmpVec = new THREE.Vector3();
  }

  /**
   * Apply real DJI Osmo Mobile 3 joint angles in degrees:
   * - panDeg: -160 .. +160 (0 = front OLED facing +Z student; ±180 = rear camera facing +Z desk)
   * - rollDeg: -25 .. +25 (keep near 0 for real OM3 stability!)
   * - pitchDeg: -30 (upward stop) .. +65 (downward desk stop)
   */
  setJoints(panDeg, pitchDeg, rollDeg = 0) {
    this.panDeg = Math.max(-165, Math.min(165, panDeg));
    this.pitchDeg = Math.max(-30, Math.min(65, pitchDeg));
    this.rollDeg = Math.max(-25, Math.min(25, rollDeg));

    this.panPivot.rotation.y = THREE.MathUtils.degToRad(this.panDeg);
    this.rollPivot.rotation.z = THREE.MathUtils.degToRad(this.rollDeg);
    // Positive pitchDeg tilts front face UP and rear camera DOWN onto the desk (matching real OM3 IMU!)
    this.pitchPivot.rotation.x = THREE.MathUtils.degToRad(-this.pitchDeg);
  }

  /**
   * World position of the Pixel 11 Pro center (for inverse-kinematic aiming).
   */
  getHeadWorldPosition(target = new THREE.Vector3()) {
    this.phoneGroup.getWorldPosition(target);
    return target;
  }

  /**
   * Compute (panDeg, pitchDeg) to aim the FRONT OLED face at a world target (e.g. Student or Guest).
   */
  computeFrontAimAngles(worldTarget) {
    const headPos = new THREE.Vector3(0, 3.28, 0);
    const dx = worldTarget.x - headPos.x;
    const dy = worldTarget.y - headPos.y;
    const dz = worldTarget.z - headPos.z;
    const horiz = Math.max(0.05, Math.hypot(dx, dz));
    const pan = THREE.MathUtils.radToDeg(Math.atan2(dx, dz));
    // Positive pitchDeg tilts front (+Z) UP -> dy > 0 => positive pitchDeg
    const pitch = THREE.MathUtils.radToDeg(Math.atan2(dy, horiz));
    return {
      panDeg: Math.max(-155, Math.min(155, pan)),
      pitchDeg: Math.max(-28, Math.min(35, pitch)),
    };
  }

  /**
   * Compute (panDeg, pitchDeg) to aim the REAR 4K optics + HiLight volumetric lamp at a desk target!
   */
  computeRearLampAimAngles(worldTarget) {
    const headPos = new THREE.Vector3(0, 3.28, 0);
    const dx = worldTarget.x - headPos.x;
    const dy = worldTarget.y - headPos.y; // negative when aiming down at desk (y=0.08)
    const dz = worldTarget.z - headPos.z;
    const horiz = Math.max(0.05, Math.hypot(dx, dz));
    // Rear faces local -Z, so to point -Z toward (dx, dz), rotate +Z toward (-dx, -dz)
    let pan = THREE.MathUtils.radToDeg(Math.atan2(-dx, -dz));
    // Prefer smooth left-side swing (same calibrated direction as real Osmo!)
    if (pan > 162) pan = 162;
    if (pan < -162) pan = -162;
    // When rear points down (dy < 0), pitchDeg is POSITIVE (+40°..+58°!)
    const pitch = THREE.MathUtils.radToDeg(Math.atan2(-dy, horiz));
    return {
      panDeg: pan,
      pitchDeg: Math.max(15, Math.min(62, pitch)),
    };
  }

  /**
   * Drive all 8 individual HiLight LEDs + real shadow SpotLight + volumetric beam shader.
   */
  updateHiLight({ colorsHex, intensities, beamColorHex, beamIntensity, time }) {
    for (let i = 0; i < 8; i++) {
      const col = new THREE.Color(colorsHex[i] ?? beamColorHex ?? 0xffcf87);
      const k = Math.max(0, intensities[i] ?? 0);
      this.ledMats[i].color.copy(col);
      this.ledMats[i].emissive.copy(col);
      this.ledMats[i].emissiveIntensity = k * 3.6;
    }

    const bCol = new THREE.Color(beamColorHex ?? 0xffcf87);
    const bi = Math.max(0, Math.min(1.5, beamIntensity));
    this.lampSpot.color.copy(bCol);
    this.lampSpot.intensity = bi * 28.0;
    this.lampSpill.color.copy(bCol);
    this.lampSpill.intensity = bi * 1.4;

    this.beamMat.uniforms.uColor.value.copy(bCol);
    this.beamMat.uniforms.uIntensity.value = bi;
    this.beamMat.uniforms.uTime.value = time;

    // Gentle upward float of dust motes inside the lamp cone
    this.dustMat.color.copy(bCol);
    this.dustMat.opacity = Math.min(0.55, bi * 0.42);
    this.dustPoints.rotation.z = time * 0.06;
  }

  setHandleLed(hex, intensity = 2.0) {
    this.handleLedMat.color.set(hex);
    this.handleLedMat.emissive.set(hex);
    this.handleLedMat.emissiveIntensity = intensity;
  }

  /**
   * Real-world DJI OM3 pitch motor torque estimator (0..100%)
   * Models Pixel 11 Pro static gravity load + dynamic pitch acceleration so Zach can verify
   * every ELEGNT choreography stays safely below the OM3 thermal protection zone!
   */
  estimatePitchTorquePct(pitchVelDps, pitchAccelDps2 = 0) {
    // Static gravity torque peaks near extreme negative pitch (-30° back-heavy droop)
    const staticFactor = this.pitchDeg < 0 ? Math.abs(this.pitchDeg) * 1.15 : Math.abs(this.pitchDeg) * 0.35;
    const dynamicFactor = Math.abs(pitchVelDps) * 0.48 + Math.abs(pitchAccelDps2) * 0.06;
    return Math.max(8, Math.min(100, Math.round(14 + staticFactor + dynamicFactor)));
  }
}
