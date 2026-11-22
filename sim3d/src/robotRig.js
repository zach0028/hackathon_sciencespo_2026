import * as THREE from 'three';
import {
  createFlashFresnelTexture,
  createRadialMotorCapTexture,
  createRubberGripTexture,
} from './textures.js';

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
 * Create a flat rounded-rectangle ShapeGeometry with exact normalized 0..1 UVs.
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
 * Sculpted CAD Swept Chamfered Structural Beam along any 3D CatmullRomCurve3.
 * Replaces round tube pipes with flat-sided, beveled, tapered CNC/molded gimbal arms!
 */
function createSweptBeamGeo(curve, { halfW, halfH, chamfer = 0.34, steps = 40, upHint = new THREE.Vector3(0, 1, 0) }) {
  // 8-point chamfered octagonal-rectangular architectural section (CCW in local XY)
  const c = Math.max(0.08, Math.min(0.45, chamfer));
  const profile = [
    [1 - c, 1],
    [-1 + c, 1],
    [-1, 1 - c],
    [-1, -1 + c],
    [-1 + c, -1],
    [1 - c, -1],
    [1, -1 + c],
    [1, 1 - c],
  ];
  const ringPts = profile.length;
  const positions = [];
  const uvs = [];
  const indices = [];

  const prevRight = new THREE.Vector3();
  const curTan = new THREE.Vector3();
  const curRight = new THREE.Vector3();
  const curUp = new THREE.Vector3();
  const pt = new THREE.Vector3();

  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    curve.getPointAt(t, pt);
    curve.getTangentAt(t, curTan).normalize();

    if (i === 0) {
      curRight.crossVectors(curTan, upHint);
      if (curRight.lengthSq() < 1e-5) {
        curRight.crossVectors(curTan, new THREE.Vector3(1, 0, 0));
      }
      curRight.normalize();
    } else {
      // Parallel-transport previous right vector to eliminate twist
      curRight.copy(prevRight).addScaledVector(curTan, -prevRight.dot(curTan));
      if (curRight.lengthSq() < 1e-5) {
        curRight.crossVectors(curTan, upHint);
      }
      curRight.normalize();
    }
    prevRight.copy(curRight);
    curUp.crossVectors(curRight, curTan).normalize();

    const w = typeof halfW === 'function' ? halfW(t) : halfW;
    const h = typeof halfH === 'function' ? halfH(t) : halfH;

    for (let j = 0; j < ringPts; j++) {
      const px = profile[j][0] * w;
      const py = profile[j][1] * h;
      const vx = pt.x + curRight.x * px + curUp.x * py;
      const vy = pt.y + curRight.y * px + curUp.y * py;
      const vz = pt.z + curRight.z * px + curUp.z * py;
      positions.push(vx, vy, vz);
      uvs.push(j / ringPts, t * 4);
    }
  }

  // Side faces
  for (let i = 0; i < steps; i++) {
    const row0 = i * ringPts;
    const row1 = (i + 1) * ringPts;
    for (let j = 0; j < ringPts; j++) {
      const jn = (j + 1) % ringPts;
      const a = row0 + j;
      const b = row0 + jn;
      const cIdx = row1 + jn;
      const d = row1 + j;
      indices.push(a, d, cIdx, a, cIdx, b);
    }
  }

  // Start & end caps (outward normals: -curTan at start, +curTan at end)
  const startCenterIdx = positions.length / 3;
  const p0 = curve.getPointAt(0);
  positions.push(p0.x, p0.y, p0.z);
  uvs.push(0.5, 0);
  for (let j = 0; j < ringPts; j++) {
    indices.push(startCenterIdx, (j + 1) % ringPts, j);
  }

  const endCenterIdx = positions.length / 3;
  const p1 = curve.getPointAt(1);
  positions.push(p1.x, p1.y, p1.z);
  uvs.push(0.5, 1);
  const lastRow = steps * ringPts;
  for (let j = 0; j < ringPts; j++) {
    indices.push(endCenterIdx, lastRow + j, lastRow + ((j + 1) % ringPts));
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Sample a point (x, y) around a vertical stadium-pill perimeter for the 8-LED Pixel HiLight ring.
 */
function sampleStadiumPerimeter(u, rx, halfStraightY) {
  const straightLen = halfStraightY * 2;
  const semiLen = Math.PI * rx;
  const total = straightLen * 2 + semiLen * 2;
  let d = (((u % 1) + 1) % 1) * total;

  // 1) Top semicircle (angle pi -> 0, left to right)
  if (d < semiLen) {
    const a = Math.PI - (d / semiLen) * Math.PI;
    return new THREE.Vector2(Math.cos(a) * rx, halfStraightY + Math.sin(a) * rx);
  }
  d -= semiLen;
  // 2) Right straight edge (top to bottom)
  if (d < straightLen) {
    return new THREE.Vector2(rx, halfStraightY - d);
  }
  d -= straightLen;
  // 3) Bottom semicircle (angle 0 -> -pi, right to left)
  if (d < semiLen) {
    const a = -(d / semiLen) * Math.PI;
    return new THREE.Vector2(Math.cos(a) * rx, -halfStraightY + Math.sin(a) * rx);
  }
  d -= semiLen;
  // 4) Left straight edge (bottom to top)
  return new THREE.Vector2(-rx, -halfStraightY + d);
}

/**
 * Create a precision sculpted DJI Osmo Mobile 3 contoured ergonomic handle geometry.
 * Side profile extruded along X with smooth ergonomic palm swell & index-finger recess.
 */
function createErgonomicHandleGeo() {
  // 2D shape in (Z_back_to_front, Y_bottom_to_top)
  const s = new THREE.Shape();
  s.moveTo(-0.155, 0.0);
  s.lineTo(0.155, 0.0);
  // Front profile rising smoothly into angled thumb control deck
  s.bezierCurveTo(0.185, 0.35, 0.2, 0.72, 0.215, 1.06);
  s.bezierCurveTo(0.225, 1.28, 0.21, 1.44, 0.165, 1.5);
  // Top neck collar edge
  s.lineTo(-0.155, 1.5);
  // Rear profile with authentic index-finger trigger recess + lower ergonomic palm swell
  s.bezierCurveTo(-0.18, 1.38, -0.145, 1.24, -0.142, 1.12); // trigger indent
  s.bezierCurveTo(-0.142, 0.96, -0.225, 0.78, -0.222, 0.56); // palm swell
  s.bezierCurveTo(-0.218, 0.32, -0.185, 0.12, -0.155, 0.0);

  const geo = new THREE.ExtrudeGeometry(s, {
    depth: 0.25,
    bevelEnabled: true,
    bevelThickness: 0.055,
    bevelSize: 0.036,
    bevelSegments: 6,
    curveSegments: 24,
  });
  // Extrude was along +Z (which we want to be X width), and shape X was front/back Z!
  // Center Z depth first, then rotate Y so shape X -> +Z (front) and extrusion depth -> X width!
  geo.translate(0, 0, -0.125);
  geo.rotateY(-Math.PI / 2);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Procedural frosted obsidian rear glass texture with subtle mirrored titanium 'G' emblem.
 */
function createPixelBackTexture() {
  const w = 1024;
  const h = 512;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d');

  const grad = ctx.createLinearGradient(0, 0, w, h);
  grad.addColorStop(0, '#232730');
  grad.addColorStop(0.5, '#1b1f27');
  grad.addColorStop(1, '#14171e');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  let seed = 9191;
  const rand = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
  ctx.fillStyle = 'rgba(255,255,255,0.016)';
  for (let i = 0; i < 4500; i++) {
    ctx.fillRect(rand() * w, rand() * h, 1.5, 1.5);
  }

  // Subtle mirrored titanium 'G' logo centered on rear glass
  ctx.save();
  ctx.translate(w * 0.52, h * 0.5);
  ctx.rotate(-Math.PI / 2);
  ctx.strokeStyle = 'rgba(202, 210, 224, 0.34)';
  ctx.lineWidth = 10;
  ctx.beginPath();
  ctx.arc(0, 0, 36, -0.45, Math.PI * 1.72, false);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(4, 0);
  ctx.lineTo(37, 0);
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
  const topR = 0.14;
  const botR = 1.92;
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
      uMoodScale: { value: 0.45 },
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
 * Museum-Grade 1:1 mm Articulated DJI Osmo Mobile 3 + Google Pixel 11 Pro + 8-LED Visor HiLight Rig.
 * Scale: 1.0 Three.js unit = 10 cm (1 mm = 0.01 units).
 */
export class OsmoRobotRig {
  constructor({ oledTexture, brushedMetalTex }) {
    this.root = new THREE.Group();
    this.root.name = 'OsmoRobotRoot';
    this.roomMood = 'day';

    const motorCapTex = createRadialMotorCapTexture();
    const gripBumpTex = createRubberGripTexture();
    const flashTex = createFlashFresnelTexture();

    // =========================================================================
    // SHARED INDUSTRIAL PBR MATERIALS
    // =========================================================================
    const matOmBody = new THREE.MeshPhysicalMaterial({
      color: 0xaab0ba,
      roughness: 0.4,
      metalness: 0.22,
      clearcoat: 0.14,
      clearcoatRoughness: 0.45,
    });
    const matOmRecess = new THREE.MeshStandardMaterial({
      color: 0x6f7682,
      roughness: 0.46,
      metalness: 0.34,
    });
    const matGripDark = new THREE.MeshStandardMaterial({
      color: 0x262a31,
      roughness: 0.88,
      metalness: 0.04,
      bumpMap: gripBumpTex,
      bumpScale: 0.008,
    });
    const matAnodized = new THREE.MeshStandardMaterial({
      color: 0x3b414c,
      roughness: 0.26,
      metalness: 0.86,
      bumpMap: brushedMetalTex,
      bumpScale: 0.003,
    });
    const matMotorCap = new THREE.MeshStandardMaterial({
      map: motorCapTex.map,
      bumpMap: motorCapTex.bumpMap,
      bumpScale: 0.005,
      roughness: 0.24,
      metalness: 0.88,
    });
    const matChromeTrim = new THREE.MeshStandardMaterial({
      color: 0xe2e7f0,
      roughness: 0.12,
      metalness: 0.96,
    });
    const matRubber = new THREE.MeshStandardMaterial({
      color: 0x17191e,
      roughness: 0.92,
      metalness: 0.02,
    });
    const matBaffleBlack = new THREE.MeshStandardMaterial({
      color: 0x07080b,
      roughness: 0.88,
      metalness: 0.08,
    });

    // Pixel 11 Pro Titanium + Optical Glass Materials
    const matPixelRail = new THREE.MeshPhysicalMaterial({
      color: 0x6c737f,
      roughness: 0.16,
      metalness: 0.94,
      clearcoat: 0.48,
      clearcoatRoughness: 0.1,
      bumpMap: brushedMetalTex,
      bumpScale: 0.0016,
    });
    const matPixelPolish = new THREE.MeshPhysicalMaterial({
      color: 0x9aa2b0,
      roughness: 0.08,
      metalness: 0.97,
      clearcoat: 0.7,
    });
    const matPixelBack = new THREE.MeshPhysicalMaterial({
      map: createPixelBackTexture(),
      roughness: 0.46,
      metalness: 0.06,
      clearcoat: 0.28,
      clearcoatRoughness: 0.34,
    });
    const matGlassBezel = new THREE.MeshPhysicalMaterial({
      color: 0x05070c,
      roughness: 0.025,
      metalness: 0.06,
      clearcoat: 1.0,
      clearcoatRoughness: 0.015,
      reflectivity: 1.0,
    });

    // Helper: Multi-stage CNC Brushless Gimbal Motor Drum (axis = 'y' | 'z' | 'x')
    const buildMotorDrum = (radius, length, axis = 'y') => {
      const g = new THREE.Group();

      // Main brushed stator barrel
      const barrel = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius * 0.985, length, 48), matAnodized);
      barrel.castShadow = true;
      barrel.receiveShadow = true;
      g.add(barrel);

      // Recessed rotor/stator precision air-gap ring
      const gap = new THREE.Mesh(new THREE.CylinderGeometry(radius * 0.94, radius * 0.94, 0.022, 40), matRubber);
      gap.position.y = length * 0.14;
      g.add(gap);

      // Upper stepped metallic collar
      const step = new THREE.Mesh(
        new THREE.CylinderGeometry(radius * 0.97, radius * 0.99, length * 0.22, 48),
        matOmRecess
      );
      step.position.y = length * 0.34;
      step.castShadow = true;
      g.add(step);

      // Chamfered diamond-cut bright aluminum bezel rings (both ends)
      const rimTop = new THREE.Mesh(new THREE.TorusGeometry(radius * 0.96, 0.009, 14, 52), matChromeTrim);
      rimTop.rotation.x = Math.PI / 2;
      rimTop.position.y = length * 0.47;
      g.add(rimTop);

      const rimBot = new THREE.Mesh(new THREE.TorusGeometry(radius * 0.97, 0.008, 12, 48), matChromeTrim);
      rimBot.rotation.x = Math.PI / 2;
      rimBot.position.y = -length * 0.46;
      g.add(rimBot);

      // Radial CNC lathe-turned metallic endcaps (top & bottom)
      const capTop = new THREE.Mesh(new THREE.CircleGeometry(radius * 0.94, 44), matMotorCap);
      capTop.rotation.x = -Math.PI / 2;
      capTop.position.y = length * 0.502;
      g.add(capTop);

      const capBot = new THREE.Mesh(new THREE.CircleGeometry(radius * 0.94, 44), matMotorCap);
      capBot.rotation.x = Math.PI / 2;
      capBot.position.y = -length * 0.502;
      g.add(capBot);

      if (axis === 'z') g.rotation.x = Math.PI / 2;
      else if (axis === 'x') g.rotation.z = -Math.PI / 2;
      return g;
    };

    // =========================================================================
    // 1. SCULPTED WEIGHTED FOLDING TRIPOD BASE (y = 0 .. 0.49)
    // =========================================================================
    const tripod = new THREE.Group();
    this.root.add(tripod);

    // Conical stepped tripod hub + knurled 1/4"-20 locking ring
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.225, 0.18, 40), matGripDark);
    hub.position.y = 0.36;
    hub.castShadow = true;
    hub.receiveShadow = true;
    tripod.add(hub);

    const knurlCollar = new THREE.Mesh(new THREE.CylinderGeometry(0.168, 0.175, 0.055, 36), matAnodized);
    knurlCollar.position.y = 0.465;
    knurlCollar.castShadow = true;
    tripod.add(knurlCollar);

    const threadWasher = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.155, 0.022, 32), matChromeTrim);
    threadWasher.position.y = 0.498;
    tripod.add(threadWasher);

    for (let i = 0; i < 3; i++) {
      const ang = (i * Math.PI * 2) / 3 + Math.PI / 6;
      const legGroup = new THREE.Group();
      legGroup.rotation.y = ang;
      legGroup.position.y = 0.34;

      // Tapered sculpted tripod blade (wider at hub, sleekly tapered toward foot)
      const legCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, 0.01, 0.08),
        new THREE.Vector3(0, -0.11, 0.46),
        new THREE.Vector3(0, -0.295, 0.87),
      ]);
      const legGeo = createSweptBeamGeo(legCurve, {
        halfW: (t) => 0.072 - t * 0.024,
        halfH: (t) => 0.056 - t * 0.02,
        chamfer: 0.36,
        steps: 18,
      });
      const leg = new THREE.Mesh(legGeo, matGripDark);
      leg.castShadow = true;
      leg.receiveShadow = true;
      legGroup.add(leg);

      // Subtle central stiffening spine along top of each tripod leg
      const spine = new THREE.Mesh(createRoundedBoxGeo(0.042, 0.022, 0.56, 0.01, 0.004), matAnodized);
      spine.position.set(0, -0.12, 0.45);
      spine.rotation.x = 0.35;
      legGroup.add(spine);

      // Exposed brushed steel folding pivot axle pin
      const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.024, 0.024, 0.16, 18), matChromeTrim);
      pin.rotation.z = Math.PI / 2;
      pin.position.set(0, -0.01, 0.13);
      legGroup.add(pin);

      // Angled wide non-slip rubber foot resting flush on desk y=0
      const foot = new THREE.Mesh(new THREE.CylinderGeometry(0.062, 0.076, 0.048, 22), matRubber);
      foot.position.set(0, -0.316, 0.885);
      foot.castShadow = true;
      legGroup.add(foot);
      tripod.add(legGroup);
    }

    // =========================================================================
    // 2. CONTOURED ERGONOMIC DJI OM3 HANDLE (Authentic 12° Pistol-Grip Slant)
    // =========================================================================
    const handleGroup = new THREE.Group();
    handleGroup.position.set(0, 0.49, 0);
    handleGroup.rotation.x = -0.19;
    this.root.add(handleGroup);

    const handleCore = new THREE.Mesh(createErgonomicHandleGeo(), matOmBody);
    handleCore.castShadow = true;
    handleCore.receiveShadow = true;
    handleGroup.add(handleCore);

    // Textured micro-diamond rubber palm grip inlay hugging the lower rear curve
    const palmInlay = new THREE.Mesh(createRoundedBoxGeo(0.32, 0.86, 0.16, 0.09, 0.022, 20), matGripDark);
    palmInlay.position.set(0, 0.56, -0.162);
    palmInlay.rotation.x = 0.04;
    palmInlay.castShadow = true;
    handleGroup.add(palmInlay);

    // Recessed front thumb control deck (+Z)
    const ctrlBezel = new THREE.Mesh(createRoundedBoxGeo(0.265, 0.57, 0.048, 0.085, 0.01, 20), matRubber);
    ctrlBezel.position.set(0, 1.13, 0.212);
    handleGroup.add(ctrlBezel);

    // Precision concave thumb analog joystick
    const joyCollar = new THREE.Mesh(new THREE.CylinderGeometry(0.076, 0.084, 0.03, 28), matAnodized);
    joyCollar.rotation.x = Math.PI / 2;
    joyCollar.position.set(0, 1.255, 0.236);
    handleGroup.add(joyCollar);

    const joyStem = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.038, 0.04, 20), matRubber);
    joyStem.rotation.x = Math.PI / 2;
    joyStem.position.set(0, 1.255, 0.255);
    handleGroup.add(joyStem);

    const joyHead = new THREE.Mesh(new THREE.CylinderGeometry(0.064, 0.056, 0.024, 28), matGripDark);
    joyHead.rotation.x = Math.PI / 2;
    joyHead.position.set(0, 1.255, 0.274);
    handleGroup.add(joyHead);

    const joyRing = new THREE.Mesh(new THREE.TorusGeometry(0.052, 0.006, 10, 28), matOmRecess);
    joyRing.position.set(0, 1.255, 0.285);
    handleGroup.add(joyRing);

    // Recessed M (Mode/Power) button with satin metallic ring
    const btnM = new THREE.Mesh(new THREE.CylinderGeometry(0.043, 0.043, 0.024, 24), matAnodized);
    btnM.rotation.x = Math.PI / 2;
    btnM.position.set(-0.064, 1.055, 0.238);
    handleGroup.add(btnM);
    const btnMRing = new THREE.Mesh(new THREE.TorusGeometry(0.044, 0.0045, 8, 24), matChromeTrim);
    btnMRing.position.set(-0.064, 1.055, 0.246);
    handleGroup.add(btnMRing);

    // Recessed Shutter/Record button with crisp anodized crimson dot
    const btnRecBase = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.042, 0.024, 24), matAnodized);
    btnRecBase.rotation.x = Math.PI / 2;
    btnRecBase.position.set(0.064, 1.055, 0.238);
    handleGroup.add(btnRecBase);
    const btnRecDot = new THREE.Mesh(
      new THREE.CylinderGeometry(0.024, 0.024, 0.008, 20),
      new THREE.MeshStandardMaterial({ color: 0xe13c3c, roughness: 0.28, metalness: 0.2 })
    );
    btnRecDot.rotation.x = Math.PI / 2;
    btnRecDot.position.set(0.064, 1.055, 0.251);
    handleGroup.add(btnRecDot);

    // Live status LED + 3 micro battery gauge LEDs
    this.handleLedMat = new THREE.MeshStandardMaterial({
      color: 0x34d399,
      emissive: 0x34d399,
      emissiveIntensity: 2.0,
      roughness: 0.2,
    });
    const statusLed = new THREE.Mesh(new THREE.SphereGeometry(0.016, 14, 14), this.handleLedMat);
    statusLed.position.set(0, 0.958, 0.237);
    handleGroup.add(statusLed);

    const battLedMat = new THREE.MeshStandardMaterial({
      color: 0xf1f5f9,
      emissive: 0xd8f5e8,
      emissiveIntensity: 1.2,
    });
    for (let i = 0; i < 3; i++) {
      const bLed = new THREE.Mesh(new THREE.SphereGeometry(0.0105, 10, 10), battLedMat);
      bLed.position.set(-0.042 + i * 0.042, 0.902, 0.235);
      handleGroup.add(bLed);
    }

    // Left recessed T/W Zoom slider track + tactile thumb nub
    const zoomTrack = new THREE.Mesh(createRoundedBoxGeo(0.022, 0.28, 0.072, 0.024, 0.004), matRubber);
    zoomTrack.position.set(-0.184, 1.15, 0.02);
    handleGroup.add(zoomTrack);

    const zoomNub = new THREE.Mesh(createRoundedBoxGeo(0.042, 0.105, 0.056, 0.018, 0.006), matAnodized);
    zoomNub.position.set(-0.198, 1.15, 0.02);
    handleGroup.add(zoomNub);

    // Right recessed USB-C + USB-A port bay
    const portPlate = new THREE.Mesh(createRoundedBoxGeo(0.018, 0.34, 0.095, 0.026, 0.004), matAnodized);
    portPlate.position.set(0.182, 1.06, -0.01);
    handleGroup.add(portPlate);
    const usbCHole = new THREE.Mesh(createRoundedBoxGeo(0.012, 0.082, 0.032, 0.012, 0.002), matRubber);
    usbCHole.position.set(0.19, 1.14, -0.01);
    handleGroup.add(usbCHole);
    const usbAHole = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.135, 0.052), matRubber);
    usbAHole.position.set(0.19, 0.985, -0.01);
    handleGroup.add(usbAHole);

    // Sculpted rear index-finger Trigger nested in the upper rear recess
    const rearTrigger = new THREE.Mesh(createRoundedBoxGeo(0.108, 0.225, 0.085, 0.034, 0.012), matAnodized);
    rearTrigger.position.set(0, 1.14, -0.175);
    rearTrigger.rotation.x = 0.26;
    handleGroup.add(rearTrigger);

    // Top flared gimbal neck collar (re-leveled purely horizontally for vertical Pan axis!)
    const neckMount = new THREE.Group();
    neckMount.position.set(0, 1.5, 0);
    neckMount.rotation.x = 0.19;
    handleGroup.add(neckMount);

    const neckCollar = new THREE.Mesh(new THREE.CylinderGeometry(0.228, 0.195, 0.145, 44), matOmBody);
    neckCollar.position.y = 0.045;
    neckCollar.castShadow = true;
    neckMount.add(neckCollar);

    const neckAccent = new THREE.Mesh(new THREE.TorusGeometry(0.228, 0.008, 12, 48), matOmRecess);
    neckAccent.rotation.x = Math.PI / 2;
    neckAccent.position.y = 0.11;
    neckMount.add(neckAccent);

    // =========================================================================
    // 3. AXIS 1 — PAN BRUSHLESS MOTOR + SCULPTED FOLDING L-ARM
    // =========================================================================
    this.panPivot = new THREE.Group();
    this.panPivot.position.set(0, 0.14, 0);
    neckMount.add(this.panPivot);

    const panMotor = buildMotorDrum(0.236, 0.27, 'y');
    panMotor.position.y = 0.135;
    this.panPivot.add(panMotor);

    // Mechanical shoulder yoke bridging Pan rotor to folding knuckle
    const panYoke = new THREE.Mesh(createRoundedBoxGeo(0.26, 0.14, 0.32, 0.06, 0.014, 16), matOmBody);
    panYoke.position.set(0, 0.28, -0.06);
    panYoke.castShadow = true;
    this.panPivot.add(panYoke);

    // Authentic Sculpted Flat-Sided Tapered DJI OM3 Swept L-Arm (NO round tube pipe!)
    const armCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0.28, -0.04),
      new THREE.Vector3(0, 0.43, -0.27),
      new THREE.Vector3(0, 0.78, -0.57),
      new THREE.Vector3(0, 1.18, -0.62),
    ]);
    const sweptArmGeo = createSweptBeamGeo(armCurve, {
      halfW: (t) => 0.118 - t * 0.016,
      halfH: (t) => 0.125 - t * 0.022,
      chamfer: 0.34,
      steps: 44,
      upHint: new THREE.Vector3(0, 0, -1),
    });
    const sweptArm = new THREE.Mesh(sweptArmGeo, matOmBody);
    sweptArm.castShadow = true;
    sweptArm.receiveShadow = true;
    this.panPivot.add(sweptArm);

    // Recessed anodized structural style channel inlay hugging the outer back of L-arm
    const armInlayCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0.5, -0.37),
      new THREE.Vector3(0, 0.79, -0.635),
      new THREE.Vector3(0, 1.07, -0.695),
    ]);
    const armInlay = new THREE.Mesh(
      createSweptBeamGeo(armInlayCurve, {
        halfW: 0.056,
        halfH: 0.024,
        chamfer: 0.3,
        steps: 20,
        upHint: new THREE.Vector3(0, 0, -1),
      }),
      matAnodized
    );
    this.panPivot.add(armInlay);

    // Precision DJI OM3 Circular Folding Lock Hinge Boss (distinctive mechanical knuckle!)
    const hingeGroup = new THREE.Group();
    hingeGroup.position.set(0, 0.43, -0.27);
    this.panPivot.add(hingeGroup);

    const knuckleDrum = buildMotorDrum(0.155, 0.268, 'x');
    hingeGroup.add(knuckleDrum);

    // Mechanical folding-lock stop collar around knuckle
    const lockCollar = new THREE.Mesh(new THREE.TorusGeometry(0.158, 0.012, 12, 40), matOmRecess);
    lockCollar.rotation.y = Math.PI / 2;
    hingeGroup.add(lockCollar);

    // =========================================================================
    // 4. AXIS 2 — ROLL BRUSHLESS MOTOR + SCULPTED TAPERED CROSS-ARM
    // =========================================================================
    this.rollPivot = new THREE.Group();
    this.rollPivot.position.set(0, 1.18, -0.62);
    this.panPivot.add(this.rollPivot);

    const rollMotor = buildMotorDrum(0.214, 0.27, 'z');
    this.rollPivot.add(rollMotor);

    // Sculpted horizontal tapered CNC cross-arm sweeping from Roll rotor (+X) to Pitch motor!
    const crossCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.0, 0.0, 0.11),
      new THREE.Vector3(0.46, 0.0, 0.14),
      new THREE.Vector3(0.87, 0.0, 0.34),
      new THREE.Vector3(0.92, 0.0, 0.62),
    ]);
    const crossArmGeo = createSweptBeamGeo(crossCurve, {
      halfW: (t) => 0.088 - t * 0.012,
      halfH: (t) => 0.112 - t * 0.024,
      chamfer: 0.35,
      steps: 38,
      upHint: new THREE.Vector3(0, 1, 0),
    });
    const crossArm = new THREE.Mesh(crossArmGeo, matOmBody);
    crossArm.castShadow = true;
    crossArm.receiveShadow = true;
    this.rollPivot.add(crossArm);

    // Subtle matte-dark rear cable channel rib embedded into cross-arm
    const crossRibCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.14, 0.0, 0.062),
      new THREE.Vector3(0.48, 0.0, 0.078),
      new THREE.Vector3(0.84, 0.0, 0.265),
    ]);
    const crossRib = new THREE.Mesh(
      createSweptBeamGeo(crossRibCurve, {
        halfW: 0.024,
        halfH: 0.044,
        chamfer: 0.3,
        steps: 22,
        upHint: new THREE.Vector3(0, 1, 0),
      }),
      matAnodized
    );
    this.rollPivot.add(crossRib);

    // =========================================================================
    // 5. AXIS 3 — PITCH BRUSHLESS MOTOR + PRECISION TELESCOPING CLAMP
    // =========================================================================
    this.pitchPivot = new THREE.Group();
    this.pitchPivot.position.set(0, 0, 0.62);
    this.rollPivot.add(this.pitchPivot);

    const pitchMotor = buildMotorDrum(0.192, 0.23, 'x');
    pitchMotor.position.set(0.92, 0, 0);
    this.pitchPivot.add(pitchMotor);

    // Sculpted structural clamp bridge with recessed dark aluminum pocket
    const clampBridge = new THREE.Mesh(createRoundedBoxGeo(0.94, 0.175, 0.092, 0.042, 0.012, 16), matOmBody);
    clampBridge.position.set(0.45, 0, -0.095);
    clampBridge.castShadow = true;
    this.pitchPivot.add(clampBridge);

    const bridgePocket = new THREE.Mesh(createRoundedBoxGeo(0.56, 0.072, 0.016, 0.022, 0.004, 12), matAnodized);
    bridgePocket.position.set(0.47, 0, -0.14);
    this.pitchPivot.add(bridgePocket);

    // Center Telescoping Smartphone Clamp Hub
    const clampHub = new THREE.Mesh(createRoundedBoxGeo(0.36, 0.44, 0.088, 0.06, 0.012, 18), matOmRecess);
    clampHub.position.set(0, 0, -0.092);
    clampHub.castShadow = true;
    this.pitchPivot.add(clampHub);

    // Center alignment medallion + 4 micro Torx fastener heads on rear of clamp
    const clampMedallion = new THREE.Mesh(new THREE.CylinderGeometry(0.082, 0.088, 0.018, 28), matAnodized);
    clampMedallion.rotation.x = Math.PI / 2;
    clampMedallion.position.set(0, 0, -0.136);
    this.pitchPivot.add(clampMedallion);

    for (const [sx, sy] of [[-0.11, 0.14], [0.11, 0.14], [-0.11, -0.14], [0.11, -0.14]]) {
      const screw = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 0.012, 12), matChromeTrim);
      screw.rotation.x = Math.PI / 2;
      screw.position.set(sx, sy, -0.137);
      this.pitchPivot.add(screw);
    }

    // Dual Exposed Polished Stainless-Steel Telescoping Slide Guide Rails!
    for (const rx of [-0.092, 0.092]) {
      const rail = new THREE.Mesh(new THREE.CylinderGeometry(0.019, 0.019, 0.78, 16), matChromeTrim);
      rail.position.set(rx, 0, -0.084);
      this.pitchPivot.add(rail);
    }

    // Sculpted Upper & Lower Spring-Loaded Rubberized Clamp Jaws
    const buildClampJaw = (signY) => {
      const jg = new THREE.Group();
      jg.position.set(0, signY * 0.39, -0.012);

      // Structural outer shell of jaw
      const jawShell = new THREE.Mesh(createRoundedBoxGeo(0.34, 0.072, 0.21, 0.026, 0.008), matAnodized);
      jawShell.castShadow = true;
      jg.add(jawShell);

      // Front retaining lip that hooks safely onto the phone bezel edge
      const lip = new THREE.Mesh(createRoundedBoxGeo(0.31, 0.056, 0.036, 0.014, 0.005), matGripDark);
      lip.position.set(0, -signY * 0.026, 0.088);
      jg.add(lip);

      // Micro-textured dark silicone inner cushion pad
      const pad = new THREE.Mesh(createRoundedBoxGeo(0.32, 0.024, 0.145, 0.012, 0.003), matGripDark);
      pad.position.set(0, -signY * 0.028, 0.01);
      jg.add(pad);

      return jg;
    };
    this.pitchPivot.add(buildClampJaw(1));
    this.pitchPivot.add(buildClampJaw(-1));

    // =========================================================================
    // 6. HIGH-FIDELITY GOOGLE PIXEL 11 PRO (152.8 x 72.0 x 8.5 mm)
    // =========================================================================
    this.phoneGroup = new THREE.Group();
    this.phoneGroup.position.set(0, 0, 0.01);
    this.pitchPivot.add(this.phoneGroup);

    const PW = 1.528;
    const PH = 0.72;
    const PD = 0.085;

    // Flat-rail polished satin titanium unibody frame
    const chassis = new THREE.Mesh(createRoundedBoxGeo(PW, PH, PD, 0.118, 0.013, 32), matPixelRail);
    chassis.castShadow = true;
    chassis.receiveShadow = true;
    this.phoneGroup.add(chassis);

    // Subtle antenna break bands on top & bottom rails
    const antMat = new THREE.MeshStandardMaterial({ color: 0x3b4048, roughness: 0.65, metalness: 0.1 });
    for (const ax of [-0.52, 0.52]) {
      const bandT = new THREE.Mesh(new THREE.BoxGeometry(0.016, 0.008, PD * 0.74), antMat);
      bandT.position.set(ax, PH / 2 - 0.001, 0);
      this.phoneGroup.add(bandT);
      const bandB = bandT.clone();
      bandB.position.y = -PH / 2 + 0.001;
      this.phoneGroup.add(bandB);
    }

    // Tactile Power button & Volume rocker on upper landscape rail (clear of clamp!)
    const powerBtn = new THREE.Mesh(createRoundedBoxGeo(0.105, 0.022, 0.03, 0.008, 0.003), matPixelPolish);
    powerBtn.position.set(-0.24, PH / 2 + 0.007, 0);
    this.phoneGroup.add(powerBtn);

    const volBtn = new THREE.Mesh(createRoundedBoxGeo(0.21, 0.022, 0.03, 0.008, 0.003), matPixelPolish);
    volBtn.position.set(0.32, PH / 2 + 0.007, 0);
    this.phoneGroup.add(volBtn);

    // USB-C port + speaker grille perforations on bottom phone edge (+X)
    const usbPort = new THREE.Mesh(createRoundedBoxGeo(0.014, 0.095, 0.032, 0.012, 0.002), matRubber);
    usbPort.position.set(PW / 2 + 0.001, 0, 0);
    this.phoneGroup.add(usbPort);
    for (const sy of [-0.15, -0.11, 0.11, 0.15]) {
      const hole = new THREE.Mesh(new THREE.CircleGeometry(0.009, 12), matRubber);
      hole.rotation.y = Math.PI / 2;
      hole.position.set(PW / 2 + 0.002, sy, 0);
      this.phoneGroup.add(hole);
    }

    // --- FRONT FACE (+Z): Edge-to-Edge Corning Glass + True Rounded-Corner OLED ---
    const frontGlassBezel = new THREE.Mesh(
      createRoundedBoxGeo(PW - 0.022, PH - 0.022, 0.01, 0.106, 0.004, 32),
      matGlassBezel
    );
    frontGlassBezel.position.z = PD / 2 + 0.001;
    this.phoneGroup.add(frontGlassBezel);

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

    const earSlit = new THREE.Mesh(createRoundedBoxGeo(0.008, 0.11, 0.004, 0.003, 0.001), matRubber);
    earSlit.position.set(-PW / 2 + 0.022, 0, PD / 2 + 0.0085);
    this.phoneGroup.add(earSlit);

    // Front punch-hole selfie camera with real multi-coat sapphire glass glint
    const frontRing = new THREE.Mesh(new THREE.CircleGeometry(0.021, 24), matBaffleBlack);
    frontRing.position.set(-PW / 2 + 0.082, 0, PD / 2 + 0.0088);
    this.phoneGroup.add(frontRing);
    const frontGlass = new THREE.Mesh(
      new THREE.CircleGeometry(0.013, 24),
      new THREE.MeshPhysicalMaterial({
        color: 0x0b1a3d,
        emissive: 0x0d214f,
        emissiveIntensity: 0.42,
        roughness: 0.02,
        metalness: 0.35,
        clearcoat: 1,
      })
    );
    frontGlass.position.set(-PW / 2 + 0.082, 0, PD / 2 + 0.0094);
    this.phoneGroup.add(frontGlass);

    // --- REAR FACE (-Z): Frosted Obsidian Glass + Museum-Grade Pixel Pro Optical Visor ---
    const backGlass = new THREE.Mesh(
      createRoundedPlaneGeo(PW - 0.028, PH - 0.028, 0.102, 28),
      matPixelBack
    );
    backGlass.rotation.y = Math.PI;
    backGlass.position.z = -PD / 2 - 0.0015;
    this.phoneGroup.add(backGlass);

    // =========================================================================
    // 7. PIXEL 11 PRO MULTI-ELEMENT OPTICAL VISOR BAR + PERIMETER 8-LED HILIGHT
    // =========================================================================
    const barGroup = new THREE.Group();
    barGroup.position.set(-0.455, 0, -PD / 2 - 0.02);
    this.phoneGroup.add(barGroup);

    // 7a. Two-stage machined titanium stadium collar with bright chamfered outer edge
    const barBaseShell = new THREE.Mesh(
      createRoundedBoxGeo(0.268, 0.654, 0.044, 0.134, 0.012, 36),
      matPixelRail
    );
    barBaseShell.castShadow = true;
    barGroup.add(barBaseShell);

    const barPolishLip = new THREE.Mesh(
      createRoundedBoxGeo(0.252, 0.638, 0.012, 0.126, 0.004, 36),
      matPixelPolish
    );
    barPolishLip.position.z = -0.021;
    barGroup.add(barPolishLip);

    // 7b. Dark satin anodized optical chamber plate behind flush visor glass
    const visorPlate = new THREE.Mesh(
      createRoundedPlaneGeo(0.228, 0.614, 0.114, 32),
      new THREE.MeshStandardMaterial({ color: 0x11141a, roughness: 0.28, metalness: 0.65 })
    );
    visorPlate.rotation.y = Math.PI;
    visorPlate.position.z = -0.028;
    barGroup.add(visorPlate);

    // 7c. Signature Pixel Pro recessed deep-black Triple-Lens Pill Cavity (upper 70% of visor!)
    const pillOuterTrim = new THREE.Mesh(
      createRoundedBoxGeo(0.166, 0.442, 0.006, 0.083, 0.002, 32),
      matAnodized
    );
    pillOuterTrim.position.set(0, 0.062, -0.0285);
    barGroup.add(pillOuterTrim);

    const pillCavity = new THREE.Mesh(
      createRoundedPlaneGeo(0.152, 0.428, 0.076, 32),
      matBaffleBlack
    );
    pillCavity.rotation.y = Math.PI;
    pillCavity.position.set(0, 0.062, -0.0322);
    barGroup.add(pillCavity);

    // Multi-coat optical lens glass materials (distinct AR reflection tints per focal element!)
    const matLensWideAR = new THREE.MeshPhysicalMaterial({
      color: 0x0a1638,
      emissive: 0x0e2258,
      emissiveIntensity: 0.36,
      roughness: 0.012,
      metalness: 0.22,
      ior: 1.77,
      clearcoat: 1.0,
      clearcoatRoughness: 0.01,
      reflectivity: 1.0,
    });
    const matLensUltraAR = new THREE.MeshPhysicalMaterial({
      color: 0x160f36,
      emissive: 0x221652,
      emissiveIntensity: 0.34,
      roughness: 0.012,
      metalness: 0.24,
      ior: 1.77,
      clearcoat: 1.0,
      clearcoatRoughness: 0.01,
    });
    const matPeriscopePrism = new THREE.MeshPhysicalMaterial({
      color: 0x072634,
      emissive: 0x0c4257,
      emissiveIntensity: 0.45,
      roughness: 0.01,
      metalness: 0.55,
      clearcoat: 1.0,
      clearcoatRoughness: 0.01,
    });

    // Helper: Build a true multi-element recessed camera lens stack!
    const buildCircularLens = (ly, radius, arMat) => {
      const lg = new THREE.Group();
      lg.position.set(0, ly, -0.0326);

      // Diamond-cut satin metallic outer beauty ring
      const outerRing = new THREE.Mesh(new THREE.TorusGeometry(radius, 0.0052, 14, 36), matPixelPolish);
      lg.add(outerRing);

      // 3 stepped anti-reflective matte-black inner baffle rings (creates real optical depth!)
      const stepRadii = [radius * 0.86, radius * 0.72, radius * 0.54];
      stepRadii.forEach((sr, idx) => {
        const baffle = new THREE.Mesh(new THREE.TorusGeometry(sr, 0.0045, 10, 32), matAnodized);
        baffle.position.z = 0.002 + idx * 0.0018;
        lg.add(baffle);
      });

      // Dark anodized inner lens barrel background
      const well = new THREE.Mesh(new THREE.CircleGeometry(radius * 0.9, 28), matBaffleBlack);
      well.rotation.y = Math.PI;
      well.position.z = 0.001;
      lg.add(well);

      // 6-blade subtle metallic iris diaphragm ring
      const irisRing = new THREE.Mesh(new THREE.RingGeometry(radius * 0.34, radius * 0.5, 6), matAnodized);
      irisRing.rotation.y = Math.PI;
      irisRing.position.z = -0.0005;
      lg.add(irisRing);

      // Deep inner optical sensor pupil + subtle teal specular AR glint
      const pupil = new THREE.Mesh(new THREE.CircleGeometry(radius * 0.32, 24), arMat);
      pupil.rotation.y = Math.PI;
      pupil.position.z = -0.0012;
      lg.add(pupil);

      const glint = new THREE.Mesh(
        new THREE.CircleGeometry(radius * 0.11, 14),
        new THREE.MeshBasicMaterial({ color: 0x80d4ff, transparent: true, opacity: 0.55 })
      );
      glint.rotation.y = Math.PI;
      glint.position.set(radius * 0.16, radius * 0.16, -0.0032);
      lg.add(glint);

      // Real curved convex spherical sapphire front glass element nested cleanly behind visor glass!
      const domeR = radius * 0.86;
      const domeGeo = new THREE.SphereGeometry(domeR, 28, 16, 0, Math.PI * 2, 0, Math.PI * 0.24);
      const rimY = domeR * Math.cos(Math.PI * 0.24);
      domeGeo.translate(0, -rimY, 0); // rim now sits at exact y = 0, apex at y = +(domeR - rimY)
      domeGeo.scale(1, 0.42, 1); // sleek authentic optical element curvature (~0.0045 bulge)
      domeGeo.rotateX(-Math.PI / 2); // bulge points outward toward -Z cleanly behind visorCoverGlass!
      const dome = new THREE.Mesh(
        domeGeo,
        new THREE.MeshPhysicalMaterial({
          color: 0x0c1a3c,
          roughness: 0.015,
          metalness: 0.1,
          transparent: true,
          opacity: 0.36,
          clearcoat: 1.0,
          clearcoatRoughness: 0.01,
          depthWrite: false,
        })
      );
      dome.position.z = -0.0005;
      lg.add(dome);

      barGroup.add(lg);
    };

    // 1) 50MP Main Wide Lens (largest)
    buildCircularLens(0.19, 0.058, matLensWideAR);
    // 2) 48MP Ultra-Wide Macro Lens
    buildCircularLens(0.062, 0.05, matLensUltraAR);

    // 3) 48MP 5× Periscope Telephoto Lens (circular outer trim + rectangular recessed prism mirror!)
    const periGroup = new THREE.Group();
    periGroup.position.set(0, -0.068, -0.0326);
    barGroup.add(periGroup);

    const periTrim = new THREE.Mesh(new THREE.TorusGeometry(0.053, 0.005, 14, 36), matPixelPolish);
    periGroup.add(periTrim);

    const periBg = new THREE.Mesh(new THREE.CircleGeometry(0.049, 28), matBaffleBlack);
    periBg.rotation.y = Math.PI;
    periBg.position.z = 0.001;
    periGroup.add(periBg);

    // Precision machined rectangular periscope aperture bezel
    const rectFrame = new THREE.Mesh(createRoundedBoxGeo(0.064, 0.05, 0.006, 0.009, 0.0015, 12), matAnodized);
    rectFrame.position.z = -0.0005;
    periGroup.add(rectFrame);

    // 45° angled emerald-cyan coated periscope optical prism mirror visible inside the window!
    const prismMirror = new THREE.Mesh(new THREE.PlaneGeometry(0.052, 0.038), matPeriscopePrism);
    prismMirror.rotation.y = Math.PI + 0.22;
    prismMirror.position.z = -0.0038;
    periGroup.add(prismMirror);

    const prismGlint = new THREE.Mesh(
      new THREE.PlaneGeometry(0.046, 0.008),
      new THREE.MeshBasicMaterial({ color: 0x5eead4, transparent: true, opacity: 0.45 })
    );
    prismGlint.rotation.y = Math.PI;
    prismGlint.position.set(0, 0.01, -0.0044);
    periGroup.add(prismGlint);

    // 7d. Side Optical Cluster (outside triple-lens pill: Dual-Tone Fresnel Flash + IR Temp + Zoom Mic)
    // Dual-tone Fresnel LED Flash
    const flashRim = new THREE.Mesh(new THREE.TorusGeometry(0.026, 0.0035, 10, 24), matPixelPolish);
    flashRim.position.set(0.038, -0.214, -0.0305);
    barGroup.add(flashRim);

    const flashLens = new THREE.Mesh(
      new THREE.CircleGeometry(0.024, 24),
      new THREE.MeshPhysicalMaterial({
        map: flashTex,
        roughness: 0.14,
        metalness: 0.05,
        clearcoat: 1.0,
      })
    );
    flashLens.rotation.y = Math.PI;
    flashLens.position.set(0.038, -0.214, -0.0308);
    barGroup.add(flashLens);

    // Infrared Temperature + Laser AF Sensor
    const irRim = new THREE.Mesh(new THREE.TorusGeometry(0.019, 0.003, 10, 24), matAnodized);
    irRim.position.set(-0.038, -0.214, -0.0305);
    barGroup.add(irRim);

    const irGlass = new THREE.Mesh(
      new THREE.CircleGeometry(0.017, 20),
      new THREE.MeshPhysicalMaterial({
        color: 0x1a080d,
        emissive: 0x260610,
        emissiveIntensity: 0.35,
        roughness: 0.03,
        metalness: 0.3,
        clearcoat: 1.0,
      })
    );
    irGlass.rotation.y = Math.PI;
    irGlass.position.set(-0.038, -0.214, -0.0308);
    barGroup.add(irGlass);

    // Rear Acoustic Zoom Microphone pinhole
    const micHole = new THREE.Mesh(new THREE.CircleGeometry(0.0085, 16), matBaffleBlack);
    micHole.rotation.y = Math.PI;
    micHole.position.set(0, -0.266, -0.0306);
    barGroup.add(micHole);

    // 7e. Continuous Flush Sapphire Visor Cover Glass over the entire camera chamber!
    const visorCoverGlass = new THREE.Mesh(
      createRoundedPlaneGeo(0.228, 0.614, 0.114, 32),
      new THREE.MeshPhysicalMaterial({
        color: 0xffffff,
        roughness: 0.015,
        metalness: 0.05,
        transparent: true,
        opacity: 0.12,
        clearcoat: 1.0,
        clearcoatRoughness: 0.01,
        reflectivity: 1.0,
        depthWrite: false,
      })
    );
    visorCoverGlass.rotation.y = Math.PI;
    visorCoverGlass.position.z = -0.038;
    barGroup.add(visorCoverGlass);

    // =========================================================================
    // 8. INTEGRATED 8-LED PERIMETER VISOR HILIGHT HALO + SHADOW SPOTLIGHT + BEAM
    //    Wraps cleanly around the outer perimeter bevel of the Pixel Pro camera
    //    visor so ALL 3 optical lenses remain 100% crisp & unobstructed!
    // =========================================================================
    this.hilightOrigin = new THREE.Group();
    this.hilightOrigin.position.set(-0.455, 0, -PD / 2 - 0.052);
    this.phoneGroup.add(this.hilightOrigin);

    this.ledMats = [];
    this.ledMeshes = [];
    const haloRx = 0.136;
    const haloSy = 0.193;

    // 8 individually addressable curved perimeter light-guide arcs (matching Android 17 LightsManager!)
    for (let i = 0; i < 8; i++) {
      const u0 = i / 8 + 0.006;
      const u1 = (i + 1) / 8 - 0.006;
      const pts = [];
      const segSub = 10;
      for (let s = 0; s <= segSub; s++) {
        const u = u0 + (u1 - u0) * (s / segSub);
        const p2 = sampleStadiumPerimeter(u, haloRx, haloSy);
        pts.push(new THREE.Vector3(p2.x, p2.y, 0.008));
      }
      const arcCurve = new THREE.CatmullRomCurve3(pts);
      const arcGeo = new THREE.TubeGeometry(arcCurve, 14, 0.0095, 10, false);

      const mat = new THREE.MeshStandardMaterial({
        color: 0xf4efe6,
        emissive: new THREE.Color(0xffc878),
        emissiveIntensity: 0.15,
        roughness: 0.18,
        toneMapped: false,
      });
      const seg = new THREE.Mesh(arcGeo, mat);
      this.hilightOrigin.add(seg);
      this.ledMats.push(mat);
      this.ledMeshes.push(seg);
    }

    // Shadow-Casting Warm Desk Lamp SpotLight
    this.lampSpot = new THREE.SpotLight(0xffcb78, 0.0, 13.5, Math.PI * 0.23, 0.68, 1.45);
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

    this.lampSpill = new THREE.PointLight(0xffcb78, 0.0, 2.8, 2.0);
    this.lampSpill.position.set(0, 0, -0.14);
    this.hilightOrigin.add(this.lampSpill);

    const vol = createVolumetricBeamMesh();
    this.beamMesh = vol.mesh;
    this.beamMat = vol.mat;
    this.hilightOrigin.add(this.beamMesh);

    const dustCount = 42;
    const dustPositions = new Float32Array(dustCount * 3);
    for (let i = 0; i < dustCount; i++) {
      const z = -(0.4 + Math.random() * 3.6);
      const maxR = 0.14 + (-z / 5.6) * 1.4;
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

  /**
   * Compute a live macro camera pose framing the Pixel 11 Pro rear optical visor + DJI OM3 arm
   * cleanly regardless of the current Pan/Pitch orientation!
   */
  getOpticsCameraPose() {
    const visorPos = new THREE.Vector3();
    const rearDir = new THREE.Vector3(0, 0, -1);
    const rightDir = new THREE.Vector3(1, 0, 0);
    const upDir = new THREE.Vector3(0, 1, 0);
    const q = new THREE.Quaternion();

    this.hilightOrigin.getWorldPosition(visorPos);
    this.hilightOrigin.getWorldQuaternion(q);
    rearDir.applyQuaternion(q).normalize();
    rightDir.applyQuaternion(q).normalize();
    upDir.applyQuaternion(q).normalize();

    const target = visorPos.clone().addScaledVector(rightDir, 0.35);
    const pos = target
      .clone()
      .addScaledVector(rearDir, 1.85)
      .addScaledVector(rightDir, 0.72)
      .addScaledVector(upDir, 0.35);
    // Keep camera comfortably above the desk surface!
    pos.y = Math.max(1.15, pos.y);
    return { pos, target };
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
      this.ledMats[i].color.copy(k > 0.05 ? col : new THREE.Color(0xd7dce6));
      this.ledMats[i].emissive.copy(col);
      this.ledMats[i].emissiveIntensity = k * 3.8;
    }

    const bCol = new THREE.Color(beamColorHex ?? 0xffcb78);
    const bi = Math.max(0, Math.min(1.5, beamIntensity));
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
