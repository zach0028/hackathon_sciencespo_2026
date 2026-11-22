import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

import { createBrushedMetalTexture, createStudioEnvironment } from './textures.js';
import { OledScreenRenderer } from './oledScreen.js';
import { OsmoRobotRig } from './robotRig.js';
import { DeskScene } from './deskScene.js';
import { SoundEngine } from './soundEngine.js';
import { ElegntEngine } from './elegntEngine.js';

// =============================================================================
// 1. STUDIO WEBGL2 RENDERER + ACES FILMIC + SELECTIVE HDR BLOOM
// =============================================================================
const container = document.getElementById('viewport');
const renderer = new THREE.WebGLRenderer({
  antialias: true,
  powerPreference: 'high-performance',
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.12;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
let envTex = createStudioEnvironment(renderer, 'night');
scene.environment = envTex;
scene.environmentIntensity = 0.85;

const camera = new THREE.PerspectiveCamera(36, window.innerWidth / window.innerHeight, 0.1, 60);
camera.position.set(3.4, 3.65, 5.75);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 1.6;
controls.maxDistance = 14.5;
controls.maxPolarAngle = Math.PI * 0.485;
controls.target.set(-0.2, 1.55, 0.85);
controls.update();

// Post-processing pipeline: Crisp selective bloom for the 8 HiLight LEDs + OLED eyes
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  0.38, // subtle optical halation — never washes out typography!
  0.55,
  0.88
);
composer.addPass(bloomPass);
composer.addPass(new OutputPass());

// =============================================================================
// 2. BUILD STUDIO DESK + CAD-ACCURATE OSMO ROBOT + ELEGNT BEHAVIOR ENGINE
// =============================================================================
const brushedTex = createBrushedMetalTexture();
const oled = new OledScreenRenderer();
const rig = new OsmoRobotRig({ oledTexture: oled.texture, brushedMetalTex: brushedTex });
scene.add(rig.root);

const desk = new DeskScene(scene);
const sound = new SoundEngine();
const engine = new ElegntEngine(rig, desk, oled, sound);

const logEl = document.getElementById('lbl-log');
engine.onLog = (msg) => {
  if (logEl) logEl.textContent = msg;
};

// Boot into a stunning initial Warm Desk Lamp demo pose after 0.6s so Zach sees the HiLight magic immediately!
setTimeout(() => {
  engine.setLampMode('warm');
  syncLampButtons('warm');
}, 600);

// =============================================================================
// 3. SMOOTH CINEMATIC CAMERA PRESETS
// =============================================================================
const CAM_PRESETS = {
  hero: { pos: new THREE.Vector3(3.35, 3.55, 5.65), target: new THREE.Vector3(-0.2, 1.55, 0.95) },
  lamp: { pos: new THREE.Vector3(2.35, 2.75, 4.15), target: new THREE.Vector3(-0.45, 0.95, 1.65) },
  pov: { pos: new THREE.Vector3(0.0, 3.45, 4.55), target: new THREE.Vector3(0.0, 2.85, 0.0) },
  rear: { pos: new THREE.Vector3(-1.85, 3.55, -2.55), target: new THREE.Vector3(-0.15, 2.9, 0.0) },
  top: { pos: new THREE.Vector3(0.0, 8.8, 2.4), target: new THREE.Vector3(-0.2, 0.2, 1.35) },
};
let camLerp = null;

function goToCamera(key) {
  const p = CAM_PRESETS[key];
  if (!p) return;
  camLerp = {
    t: 0,
    dur: 0.75,
    fromPos: camera.position.clone(),
    toPos: p.pos.clone(),
    fromTarget: controls.target.clone(),
    toTarget: p.target.clone(),
  };
  document.querySelectorAll('[data-cam]').forEach((b) => {
    b.classList.toggle('active', b.dataset.cam === key);
  });
}

// =============================================================================
// 4. INTERACTIVE 3D RAYCAST DRAGGING (Notebook, Finger Pointer, Student, Guest)
// =============================================================================
const raycaster = new THREE.Raycaster();
const pointerNdc = new THREE.Vector2();
const dragPlane = new THREE.Plane();
const dragHit = new THREE.Vector3();
const dragOffset = new THREE.Vector3();
let activeDrag = null; // { kind: 'notebook'|'student'|'guest', obj, moved: false }

function updatePointerNdc(ev) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointerNdc.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointerNdc.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
}

renderer.domElement.addEventListener('pointerdown', (ev) => {
  if (ev.button !== 0) return;
  updatePointerNdc(ev);
  raycaster.setFromCamera(pointerNdc, camera);
  const hits = raycaster.intersectObjects(desk.draggables, false);
  if (!hits.length) return;

  const hit = hits[0];
  const kind = hit.object.userData.dragKind;
  if (!kind) return;

  controls.enabled = false;
  container.classList.add('dragging');
  camLerp = null;

  if (kind === 'notebook') {
    dragPlane.set(new THREE.Vector3(0, 1, 0), -desk.notebook.position.y);
    raycaster.ray.intersectPlane(dragPlane, dragHit);
    dragOffset.subVectors(desk.notebook.position, dragHit);
    activeDrag = { kind, obj: desk.notebook, startX: ev.clientX, startY: ev.clientY, moved: false, hitPoint: hit.point.clone() };
  } else if (kind === 'student' || kind === 'guest') {
    const marker = kind === 'student' ? desk.studentMarker : desk.guestMarker;
    // Drag on vertical plane parallel to camera XY so Zach can move left/right AND up/down!
    const camDir = new THREE.Vector3();
    camera.getWorldDirection(camDir);
    dragPlane.setFromNormalAndCoplanarPoint(camDir.negate(), marker.position);
    raycaster.ray.intersectPlane(dragPlane, dragHit);
    dragOffset.subVectors(marker.position, dragHit);
    activeDrag = { kind, obj: marker, startX: ev.clientX, startY: ev.clientY, moved: false };
  }
});

window.addEventListener('pointermove', (ev) => {
  if (!activeDrag) return;
  const distPx = Math.hypot(ev.clientX - activeDrag.startX, ev.clientY - activeDrag.startY);
  if (distPx > 5) activeDrag.moved = true;

  updatePointerNdc(ev);
  raycaster.setFromCamera(pointerNdc, camera);
  if (!raycaster.ray.intersectPlane(dragPlane, dragHit)) return;

  const next = dragHit.add(dragOffset);
  if (activeDrag.kind === 'notebook') {
    activeDrag.obj.position.x = Math.max(-2.6, Math.min(2.6, next.x));
    activeDrag.obj.position.z = Math.max(1.15, Math.min(3.15, next.z));
    // If finger pointer was placed on page, hide it while moving whole book
    if (activeDrag.moved && desk.pointerGroup.visible) {
      desk.pointerGroup.position.x = activeDrag.obj.position.x;
      desk.pointerGroup.position.z = activeDrag.obj.position.z;
    }
  } else {
    activeDrag.obj.position.x = Math.max(-3.6, Math.min(3.6, next.x));
    activeDrag.obj.position.y = Math.max(2.1, Math.min(4.8, next.y));
    activeDrag.obj.position.z = Math.max(2.4, Math.min(5.2, next.z));
  }
});

window.addEventListener('pointerup', () => {
  if (!activeDrag) return;
  // Short click on a notebook page -> snap the interactive pointing finger there!
  if (activeDrag.kind === 'notebook' && !activeDrag.moved && activeDrag.hitPoint) {
    desk.pointerGroup.visible = true;
    desk.pointerGroup.position.set(activeDrag.hitPoint.x, activeDrag.hitPoint.y + 0.03, activeDrag.hitPoint.z);
    engine.sound.playCue('curious');
    if (engine.scenario !== 'desk_lamp') {
      engine.setLampMode('warm');
      syncLampButtons('warm');
    }
    engine._log('Doigt pointeur placé sur la page — Osmo ajuste son faisceau chaud HiLight pile sur ce point !');
  }
  activeDrag = null;
  controls.enabled = true;
  container.classList.remove('dragging');
});

renderer.domElement.addEventListener('dblclick', () => {
  if (desk.pointerGroup.visible) {
    desk.pointerGroup.visible = false;
    engine._log('Doigt pointeur retiré — faisceau recentré sur le cahier entier');
  }
});

// =============================================================================
// 5. UI WIRING
// =============================================================================
function syncLampButtons(activePreset) {
  document.querySelectorAll('[data-lamp]').forEach((b) => {
    b.classList.remove('warm-on', 'active');
    if (b.dataset.lamp === activePreset && activePreset !== 'off') {
      b.classList.add('warm-on');
    }
  });
}

document.querySelectorAll('[data-cam]').forEach((btn) => {
  btn.addEventListener('click', () => goToCamera(btn.dataset.cam));
});

document.querySelectorAll('[data-mood]').forEach((btn) => {
  btn.addEventListener('click', () => {
    const m = btn.dataset.mood;
    desk.setMood(m);
    if (envTex) envTex.dispose();
    envTex = createStudioEnvironment(renderer, m);
    scene.environment = envTex;
    scene.environmentIntensity = m === 'day' ? 1.1 : m === 'golden' ? 0.95 : 0.85;
    bloomPass.strength = m === 'night' ? 0.4 : m === 'golden' ? 0.3 : 0.2;
    document.querySelectorAll('[data-mood]').forEach((b) => b.classList.remove('warm-on', 'active'));
    btn.classList.add(m === 'day' ? 'active' : 'warm-on');
  });
});

document.getElementById('btn-style-elegnt').addEventListener('click', (e) => {
  engine.setMovementStyle('elegnt');
  e.currentTarget.classList.add('active');
  document.getElementById('btn-style-robotic').classList.remove('active');
  document.getElementById('lbl-style-badge').textContent = 'EXPRESSIF';
});

document.getElementById('btn-style-robotic').addEventListener('click', (e) => {
  engine.setMovementStyle('robotic');
  e.currentTarget.classList.add('active');
  document.getElementById('btn-style-elegnt').classList.remove('active');
  document.getElementById('lbl-style-badge').textContent = 'ROBOTIQUE';
});

document.querySelectorAll('[data-exp]').forEach((btn) => {
  btn.addEventListener('click', () => {
    engine.setExpressiveness(btn.dataset.exp);
    document.querySelectorAll('[data-exp]').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
  });
});

document.getElementById('btn-follow-hands').addEventListener('click', (e) => {
  engine.followHandsLamp = !engine.followHandsLamp;
  e.currentTarget.textContent = engine.followHandsLamp ? 'SUIVI MAIN : ON' : 'SUIVI MAIN : OFF';
  e.currentTarget.className = engine.followHandsLamp ? 'mini emerald-on' : 'mini';
});

document.querySelectorAll('[data-lamp]').forEach((btn) => {
  btn.addEventListener('click', () => {
    engine.setLampMode(btn.dataset.lamp);
    syncLampButtons(btn.dataset.lamp);
  });
});

document.getElementById('btn-sc-showtell').addEventListener('click', () => {
  engine.runShowAndTell('definition');
  syncLampButtons('study');
});
document.getElementById('btn-sc-triadic').addEventListener('click', () => {
  engine.runTriadicThinking();
  syncLampButtons('off');
});
document.getElementById('btn-sc-nudge').addEventListener('click', () => {
  engine.runSilentNudge();
  syncLampButtons('off');
});
document.getElementById('btn-sc-social').addEventListener('click', (e) => {
  const on = engine.toggleMultiFace();
  e.currentTarget.classList.toggle('emerald-on', on);
  if (on) goToCamera('hero');
});
document.getElementById('btn-sc-music').addEventListener('click', (e) => {
  const next = !engine.musicMode;
  engine.setMusicMode(next);
  e.currentTarget.classList.toggle('emerald-on', next);
  syncLampButtons('off');
});

document.querySelectorAll('[data-react]').forEach((btn) => {
  btn.addEventListener('click', () => engine.triggerReaction(btn.dataset.react));
});

document.querySelectorAll('[data-board]').forEach((btn) => {
  btn.addEventListener('click', () => {
    engine.setBoardPreset(btn.dataset.board);
    document.querySelectorAll('[data-board]').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    if (btn.dataset.board !== 'none' && engine.scenario === 'desk_lamp') {
      engine.setLampMode('off');
      syncLampButtons('off');
    }
  });
});

document.getElementById('btn-sound').addEventListener('click', (e) => {
  sound.enabled = !sound.enabled;
  if (!sound.enabled && engine.musicMode) engine.setMusicMode(false);
  e.currentTarget.textContent = sound.enabled ? 'SON : ON' : 'SON : OFF';
  e.currentTarget.classList.toggle('active', sound.enabled);
});

document.getElementById('btn-bridge').addEventListener('click', (e) => {
  const next = !engine.liveBridgeEnabled;
  engine.toggleLiveBridge(next);
  e.currentTarget.textContent = next ? 'SYNC LIVE : ON' : 'SYNC LIVE : OFF';
  e.currentTarget.classList.toggle('emerald-on', next);
});

const hudEl = document.getElementById('hud');
const unhideBtn = document.getElementById('btn-unhide');
function toggleHud() {
  const hide = !hudEl.classList.contains('hidden');
  hudEl.classList.toggle('hidden', hide);
  unhideBtn.classList.toggle('visible', hide);
}
document.getElementById('btn-hide').addEventListener('click', toggleHud);
unhideBtn.addEventListener('click', toggleHud);

window.addEventListener('keydown', (ev) => {
  if (ev.key === '1') goToCamera('hero');
  else if (ev.key === '2') goToCamera('lamp');
  else if (ev.key === '3') goToCamera('pov');
  else if (ev.key === '4') goToCamera('rear');
  else if (ev.key === '5') goToCamera('top');
  else if (ev.key.toLowerCase() === 'h') toggleHud();
});

window.addEventListener('resize', () => {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
});

// =============================================================================
// 6. MAIN 60–120 FPS ANIMATION & TELEMETRY LOOP
// =============================================================================
const tPan = document.getElementById('t-pan');
const tPitch = document.getElementById('t-pitch');
const tRoll = document.getElementById('t-roll');
const tVPan = document.getElementById('t-vpan');
const tVPitch = document.getElementById('t-vpitch');
const tFps = document.getElementById('t-fps');
const tTorque = document.getElementById('t-torque');
const tTorqueFill = document.getElementById('t-torque-fill');

let prevTs = performance.now();
let fpsFrames = 0;
let fpsLast = prevTs;
let curFps = 60;
let hudTick = 0;

function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min(0.05, (now - prevTs) / 1000);
  prevTs = now;

  fpsFrames += 1;
  if (now - fpsLast >= 500) {
    curFps = Math.round((fpsFrames * 1000) / (now - fpsLast));
    fpsFrames = 0;
    fpsLast = now;
  }

  // Smooth camera preset transition
  if (camLerp) {
    camLerp.t += dt;
    const u = Math.min(1, camLerp.t / camLerp.dur);
    const ease = 1 - Math.pow(1 - u, 3);
    camera.position.lerpVectors(camLerp.fromPos, camLerp.toPos, ease);
    controls.target.lerpVectors(camLerp.fromTarget, camLerp.toTarget, ease);
    if (u >= 1) camLerp = null;
  }

  // Subtle pulse on pointing finger reticle
  if (desk.pointerGroup.visible) {
    const s = 1 + Math.sin(now * 0.006) * 0.12;
    desk.pointerGroup.children[0].scale.set(s, s, 1);
  }

  engine.update(dt);
  controls.update();
  composer.render();

  // Refresh telemetry HUD at ~15 Hz
  hudTick += 1;
  if (hudTick % 4 === 0) {
    tPan.textContent = `${engine.curPan >= 0 ? '+' : ''}${engine.curPan.toFixed(1)}°`;
    tPitch.textContent = `${engine.curPitch >= 0 ? '+' : ''}${engine.curPitch.toFixed(1)}°`;
    tRoll.textContent = `${engine.curRoll >= 0 ? '+' : ''}${engine.curRoll.toFixed(1)}°`;
    tVPan.textContent = `${Math.round(engine.velPan)}°/s`;
    tVPitch.textContent = `${Math.round(engine.velPitch)}°/s`;
    tFps.textContent = String(curFps);

    const tq = engine.torquePct;
    const label = tq >= 78 ? 'HIGH' : tq >= 55 ? 'MED' : 'SAFE';
    tTorque.textContent = `${tq}% · ${label}`;
    tTorqueFill.style.width = `${tq}%`;
    tTorqueFill.style.backgroundColor = tq >= 78 ? '#fb7185' : tq >= 55 ? '#f59e0b' : '#34d399';
  }
}

requestAnimationFrame(animate);
