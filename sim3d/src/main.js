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
// 1. WEBGL2 STUDIO RENDERER + ACES FILMIC + SELECTIVE BLOOM
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
renderer.toneMappingExposure = 1.05;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
let envTex = createStudioEnvironment(renderer, 'day');
scene.environment = envTex;
scene.environmentIntensity = 0.95;

const camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.1, 60);
camera.position.set(3.45, 3.25, 5.65);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 1.5;
controls.maxDistance = 11.5;
controls.maxPolarAngle = Math.PI * 0.485;
controls.target.set(-0.35, 1.65, 0.45);
controls.update();

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  0.22, // clean optical glow without washing out daylight plaster walls
  0.5,
  0.92
);
composer.addPass(bloomPass);
composer.addPass(new OutputPass());

// =============================================================================
// 2. SUNLIT ARCHITECTURAL ROOM + PIXEL 11 PRO / DJI OM3 + ELEGNT ENGINE
// =============================================================================
const brushedTex = createBrushedMetalTexture();
const oled = new OledScreenRenderer();
const rig = new OsmoRobotRig({ oledTexture: oled.texture, brushedMetalTex: brushedTex });
rig.setRoomMood('day');
scene.add(rig.root);

const desk = new DeskScene(scene);
const sound = new SoundEngine();
const engine = new ElegntEngine(rig, desk, oled, sound);

const logEl = document.getElementById('lbl-log');
const brandDot = document.getElementById('brand-dot');
engine.onLog = (msg) => {
  if (logEl) logEl.textContent = msg;
};

// Boot smoothly into Warm Desk Lamp mode so Zach immediately sees the warm beam on the notebook
setTimeout(() => {
  engine.setLampMode('warm');
  syncLampUi('warm');
}, 550);

// =============================================================================
// 3. CINEMATIC CAMERA PRESETS (Keys 1-5)
// =============================================================================
const CAM_PRESETS = {
  hero: { pos: new THREE.Vector3(3.45, 3.25, 5.65), target: new THREE.Vector3(-0.35, 1.65, 0.45) },
  lamp: { pos: new THREE.Vector3(2.25, 2.55, 4.05), target: new THREE.Vector3(-0.48, 0.95, 1.55) },
  pov: { pos: new THREE.Vector3(0.0, 3.35, 4.25), target: new THREE.Vector3(0.0, 2.85, 0.0) },
  rear: { pos: new THREE.Vector3(-1.75, 3.35, -2.35), target: new THREE.Vector3(-0.18, 2.9, 0.0) },
  top: { pos: new THREE.Vector3(0.0, 8.2, 2.2), target: new THREE.Vector3(-0.25, 0.2, 1.15) },
};
let camLerp = null;

function goToCamera(key) {
  const p = key === 'rear' ? rig.getOpticsCameraPose() : CAM_PRESETS[key];
  if (!p) return;
  camLerp = {
    t: 0,
    dur: 0.72,
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
// 4. DIRECT 3D INTERACTION (Drag Notebook, Click Page = Finger, Drag Gaze Orbs)
// =============================================================================
const raycaster = new THREE.Raycaster();
const pointerNdc = new THREE.Vector2();
const dragPlane = new THREE.Plane();
const dragHit = new THREE.Vector3();
const dragOffset = new THREE.Vector3();
let activeDrag = null;

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
    activeDrag = {
      kind,
      obj: desk.notebook,
      startX: ev.clientX,
      startY: ev.clientY,
      moved: false,
      hitPoint: hit.point.clone(),
    };
  } else if (kind === 'student' || kind === 'guest') {
    const marker = kind === 'student' ? desk.studentMarker : desk.guestMarker;
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
  if (Math.hypot(ev.clientX - activeDrag.startX, ev.clientY - activeDrag.startY) > 5) {
    activeDrag.moved = true;
  }

  updatePointerNdc(ev);
  raycaster.setFromCamera(pointerNdc, camera);
  if (!raycaster.ray.intersectPlane(dragPlane, dragHit)) return;

  const next = dragHit.add(dragOffset);
  if (activeDrag.kind === 'notebook') {
    activeDrag.obj.position.x = Math.max(-2.35, Math.min(2.35, next.x));
    activeDrag.obj.position.z = Math.max(1.15, Math.min(3.05, next.z));
    if (activeDrag.moved && desk.pointerGroup.visible) {
      desk.pointerGroup.position.x = activeDrag.obj.position.x;
      desk.pointerGroup.position.z = activeDrag.obj.position.z;
    }
  } else {
    activeDrag.obj.position.x = Math.max(-3.5, Math.min(3.5, next.x));
    activeDrag.obj.position.y = Math.max(2.1, Math.min(4.8, next.y));
    activeDrag.obj.position.z = Math.max(2.4, Math.min(5.2, next.z));
  }
});

window.addEventListener('pointerup', () => {
  if (!activeDrag) return;
  if (activeDrag.kind === 'notebook' && !activeDrag.moved && activeDrag.hitPoint) {
    desk.pointerGroup.visible = true;
    desk.pointerGroup.position.set(activeDrag.hitPoint.x, activeDrag.hitPoint.y + 0.03, activeDrag.hitPoint.z);
    engine.sound.playCue('curious');
    if (engine.scenario !== 'desk_lamp') {
      engine.setLampMode('warm');
      syncLampUi('warm');
    }
    engine._log('Doigt pointeur placé — faisceau chaud HiLight centré sur ce point');
  }
  activeDrag = null;
  controls.enabled = true;
  container.classList.remove('dragging');
});

renderer.domElement.addEventListener('dblclick', () => {
  if (desk.pointerGroup.visible) {
    desk.pointerGroup.visible = false;
    engine._log('Doigt pointeur retiré — faisceau recentré sur le cahier');
  }
});

// =============================================================================
// 5. MINIMAL FLOATING DOCK & COLLAPSIBLE STUDIO DRAWER WIRING
// =============================================================================
const btnDockLamp = document.getElementById('btn-dock-lamp');
const lampSubbar = document.getElementById('lamp-subbar');
const btnSocial = document.getElementById('btn-sc-social');
const btnMusic = document.getElementById('btn-sc-music');
const btnShowTell = document.getElementById('btn-sc-showtell');
const btnTriadic = document.getElementById('btn-sc-triadic');
const btnStyleToggle = document.getElementById('btn-style-toggle');
const btnDrawer = document.getElementById('btn-drawer');
const studioDrawer = document.getElementById('studio-drawer');

function syncLampUi(preset) {
  const isOn = preset !== 'off' && preset !== 'sunset';
  btnDockLamp.classList.toggle('warm-on', isOn);
  lampSubbar.classList.toggle('visible', isOn);
  document.querySelectorAll('[data-lamp]').forEach((b) => {
    b.classList.toggle('active', b.dataset.lamp === preset);
  });
  if (isOn) {
    btnSocial.classList.remove('emerald-on');
    btnMusic.classList.remove('emerald-on');
  }
}

// 1) Main Lamp Toggle (clicking toggles Warm Lamp ON/OFF; swatches adjust 2900K/Study/Cozy/Sunset)
btnDockLamp.addEventListener('click', () => {
  const next = engine.scenario === 'desk_lamp' && engine.lampPreset !== 'off' ? 'off' : 'warm';
  engine.setLampMode(next);
  syncLampUi(next);
});

document.querySelectorAll('[data-lamp]').forEach((btn) => {
  btn.addEventListener('click', () => {
    const p = btn.dataset.lamp;
    engine.setLampMode(p);
    syncLampUi(p);
  });
});

// 2) Core Scenarios in Floating Dock
btnShowTell.addEventListener('click', () => {
  engine.runShowAndTell('definition');
  syncLampUi('study');
});

btnTriadic.addEventListener('click', () => {
  engine.runTriadicThinking();
  syncLampUi('off');
});

btnSocial.addEventListener('click', () => {
  const on = engine.toggleMultiFace();
  btnSocial.classList.toggle('emerald-on', on);
  if (on) {
    syncLampUi('off');
    btnMusic.classList.remove('emerald-on');
    goToCamera('hero');
  }
});

btnMusic.addEventListener('click', () => {
  const next = !engine.musicMode;
  engine.setMusicMode(next);
  btnMusic.classList.toggle('emerald-on', next);
  if (next) {
    syncLampUi('off');
    btnSocial.classList.remove('emerald-on');
  }
});

// 3) 1-Click ELEGNT vs Robotic Toggle
btnStyleToggle.addEventListener('click', () => {
  const next = engine.movementStyle === 'elegnt' ? 'robotic' : 'elegnt';
  engine.setMovementStyle(next);
  btnStyleToggle.textContent = next === 'elegnt' ? '✦ ELEGNT' : '▤ Robot';
  btnStyleToggle.classList.toggle('active', next === 'elegnt');
});

// 4) Ambient Lighting Mood (Jour / Soirée / Nuit) + Adaptive Glass Theme!
document.querySelectorAll('[data-mood]').forEach((btn) => {
  btn.addEventListener('click', () => {
    const m = btn.dataset.mood;
    desk.setMood(m);
    rig.setRoomMood(m);
    if (envTex) envTex.dispose();
    envTex = createStudioEnvironment(renderer, m);
    scene.environment = envTex;
    scene.environmentIntensity = m === 'day' ? 0.95 : m === 'golden' ? 0.9 : 0.8;
    renderer.toneMappingExposure = m === 'day' ? 1.05 : m === 'golden' ? 1.08 : 1.12;
    bloomPass.strength = m === 'day' ? 0.22 : m === 'golden' ? 0.32 : 0.42;
    document.body.classList.toggle('theme-light', m === 'day');

    document.querySelectorAll('[data-mood]').forEach((b) => b.classList.remove('active', 'warm-on'));
    btn.classList.add(m === 'day' ? 'active' : 'warm-on');
  });
});

// 5) Collapsible Studio Drawer
function toggleDrawer(force) {
  const open = force !== undefined ? force : !studioDrawer.classList.contains('open');
  studioDrawer.classList.toggle('open', open);
  studioDrawer.setAttribute('aria-hidden', String(!open));
  btnDrawer.classList.toggle('open', open);
}
btnDrawer.addEventListener('click', () => toggleDrawer());

document.querySelectorAll('[data-cam]').forEach((btn) => {
  btn.addEventListener('click', () => goToCamera(btn.dataset.cam));
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
  e.currentTarget.textContent = engine.followHandsLamp ? 'Suivi Main : ON' : 'Suivi Main : OFF';
  e.currentTarget.classList.toggle('on', engine.followHandsLamp);
});

document.querySelectorAll('[data-react]').forEach((btn) => {
  btn.addEventListener('click', () => engine.triggerReaction(btn.dataset.react));
});

document.getElementById('btn-sc-nudge').addEventListener('click', () => {
  engine.runSilentNudge();
  syncLampUi('off');
});

document.querySelectorAll('[data-board]').forEach((btn) => {
  btn.addEventListener('click', () => {
    engine.setBoardPreset(btn.dataset.board);
    document.querySelectorAll('[data-board]').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    if (btn.dataset.board !== 'none' && engine.scenario === 'desk_lamp') {
      engine.setLampMode('off');
      syncLampUi('off');
    }
  });
});

document.getElementById('btn-sound').addEventListener('click', (e) => {
  sound.enabled = !sound.enabled;
  if (!sound.enabled && engine.musicMode) {
    engine.setMusicMode(false);
    btnMusic.classList.remove('emerald-on');
  }
  e.currentTarget.textContent = sound.enabled ? 'Son : ON' : 'Son : OFF';
  e.currentTarget.classList.toggle('active', sound.enabled);
});

document.getElementById('btn-bridge').addEventListener('click', (e) => {
  const next = !engine.liveBridgeEnabled;
  engine.toggleLiveBridge(next);
  e.currentTarget.textContent = next ? 'Sync Live : ON' : 'Sync Live : OFF';
  e.currentTarget.classList.toggle('emerald-on', next);
});

const hudEl = document.getElementById('hud');
const unhideBtn = document.getElementById('btn-unhide');
function toggleHud() {
  const hide = !hudEl.classList.contains('hidden');
  hudEl.classList.toggle('hidden', hide);
  unhideBtn.classList.toggle('visible', hide);
  if (hide) toggleDrawer(false);
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
  else if (ev.key.toLowerCase() === 's') toggleDrawer();
  else if (ev.key === 'Escape') toggleDrawer(false);
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
// 6. MAIN 60-120 FPS RENDER & TELEMETRY LOOP
// =============================================================================
const tPan = document.getElementById('t-pan');
const tPitch = document.getElementById('t-pitch');
const tFps = document.getElementById('t-fps');
const tTorque = document.getElementById('t-torque');
const tTorqueFill = document.getElementById('t-torque-fill');

let prevTs = performance.now();
let fpsFrames = 0;
let fpsLast = prevTs;
let curFps = 60;
let hudTick = 0;
let lastSyncedLamp = null;

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

  if (camLerp) {
    camLerp.t += dt;
    const u = Math.min(1, camLerp.t / camLerp.dur);
    const ease = 1 - Math.pow(1 - u, 3);
    camera.position.lerpVectors(camLerp.fromPos, camLerp.toPos, ease);
    controls.target.lerpVectors(camLerp.fromTarget, camLerp.toTarget, ease);
    if (u >= 1) camLerp = null;
  }

  if (desk.pointerGroup.visible) {
    const s = 1 + Math.sin(now * 0.006) * 0.12;
    desk.pointerGroup.children[0].scale.set(s, s, 1);
  }

  engine.update(dt);
  controls.update();
  composer.render();

  hudTick += 1;
  if (hudTick % 4 === 0) {
    if (lastSyncedLamp !== engine.lampPreset) {
      lastSyncedLamp = engine.lampPreset;
      syncLampUi(engine.lampPreset);
    }
    if (brandDot) {
      brandDot.style.background = engine.accentHex || '#f59e0b';
      brandDot.style.boxShadow = `0 0 10px ${engine.accentHex || '#f59e0b'}`;
    }
    tPan.textContent = `${engine.curPan >= 0 ? '+' : ''}${engine.curPan.toFixed(1)}°`;
    tPitch.textContent = `${engine.curPitch >= 0 ? '+' : ''}${engine.curPitch.toFixed(1)}°`;
    tFps.textContent = String(curFps);

    const tq = engine.torquePct;
    const label = tq >= 78 ? 'HIGH' : tq >= 55 ? 'MED' : 'SAFE';
    tTorque.textContent = `${tq}% · ${label}`;
    tTorqueFill.style.width = `${tq}%`;
    tTorqueFill.style.backgroundColor = tq >= 78 ? '#ef4444' : tq >= 55 ? '#f59e0b' : '#10b981';
  }
}

requestAnimationFrame(animate);
