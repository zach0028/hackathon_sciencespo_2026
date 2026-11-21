/* Studio telemetry, Canvas binary WebSocket video feed, radar & hardware controls */

let ws = null;
let state = null;
let jogTimer = null;

function escapeHtml(s) {
  return String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (ev) => {
    state = JSON.parse(ev.data);
    render(state);
  };
  ws.onclose = () => {
    if (typeof stopLiveCall === 'function' && typeof liveActive !== 'undefined' && liveActive) {
      stopLiveCall();
    }
    setTimeout(connect, 1000);
  };
}
connect();

function connectVideo() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const vws = new WebSocket(`${proto}://${location.host}/ws_video`);
  vws.binaryType = 'blob';
  const canvas = document.getElementById('feed-canvas');
  const ctx = canvas.getContext('2d');
  let busy = false;
  vws.onmessage = async (ev) => {
    if (busy) return;
    busy = true;
    try {
      const bmp = await createImageBitmap(ev.data);
      if (canvas.width !== bmp.width || canvas.height !== bmp.height) {
        canvas.width = bmp.width;
        canvas.height = bmp.height;
      }
      ctx.drawImage(bmp, 0, 0);
      bmp.close();
    } catch (_) {
    } finally {
      busy = false;
    }
  };
  vws.onclose = () => setTimeout(connectVideo, 1000);
}
connectVideo();

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function toggleLiveMotors() {
  if (!state) return;
  send({ action: 'set_live_motors', enabled: !state.osmo.live_motors });
}
function rotateNext() {
  if (!state) return;
  send({ action: 'camera', rotate_deg: (state.camera.rotate_deg + 90) % 360 });
}
function toggleMirror() {
  if (!state) return;
  send({ action: 'camera', mirror: !state.camera.mirror });
}
function toggleInv(axis) {
  if (!state) return;
  if (axis === 'swap') send({ action: 'pid', swap_axes: !state.pid.swap_axes });
  else if (axis === 'yaw') send({ action: 'pid', invert_yaw: !state.pid.invert_yaw });
  else send({ action: 'pid', invert_pitch: !state.pid.invert_pitch });
}
function pushPid() {
  const kp = parseFloat(document.getElementById('sl-kp').value);
  const deadzone = parseFloat(document.getElementById('sl-dz').value);
  const max_dps = parseFloat(document.getElementById('sl-max').value);
  document.getElementById('v-kp').textContent = kp;
  document.getElementById('v-dz').textContent = deadzone.toFixed(2);
  document.getElementById('v-max').textContent = max_dps;
  send({ action: 'pid', kp, deadzone, max_dps });
}
function jog(yaw, pitch) {
  jogStop();
  send({ action: 'manual', yaw, pitch });
  jogTimer = setInterval(() => send({ action: 'manual', yaw, pitch }), 200);
}
function jogStop() {
  if (jogTimer) { clearInterval(jogTimer); jogTimer = null; }
  send({ action: 'manual', yaw: 0, pitch: 0 });
}

function drawRadar(v, dz) {
  const c = document.getElementById('radar');
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height, cx = w / 2, cy = h / 2;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = '#243044';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.moveTo(0, cy); ctx.lineTo(w, cy); ctx.stroke();

  const dzPx = (dz / 2) * w;
  ctx.strokeStyle = 'rgba(245, 185, 66, 0.7)';
  ctx.strokeRect(cx - dzPx, cy - dzPx, dzPx * 2, dzPx * 2);

  if (v.face_detected) {
    const fx = cx + v.cx * (w / 2 - 8);
    const fy = cy + v.cy * (h / 2 - 8);
    ctx.strokeStyle = '#38c9f0';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(fx, fy); ctx.stroke();
    ctx.fillStyle = '#38e58c';
    ctx.beginPath(); ctx.arc(fx, fy, 6, 0, Math.PI * 2); ctx.fill();
  }
}

function render(s) {
  const camOk = s.camera.status.startsWith('live_');
  document.getElementById('dot-cam').className = 'dot ' + (camOk ? 'ok' : 'warn');
  document.getElementById('lbl-cam').textContent = `Pixel 11 Pro : ${s.camera.status} (${(s.camera.zoom || 1).toFixed(1)}×)`;
  document.getElementById('lbl-fps').textContent = `${s.camera.fps} fps`;
  document.getElementById('lbl-rot').textContent = `${s.camera.rotate_deg}°`;
  const lblRes = document.getElementById('lbl-res');
  if (lblRes && s.camera.width) {
    const sh = s.camera.sheet && s.camera.sheet.found ? ` · 📄 ${s.camera.sheet.lines_count}L` : '';
    lblRes.textContent = `${s.camera.width}×${s.camera.height} · ${(s.camera.zoom || 1).toFixed(1)}×${sh}`;
  }
  const z = Number(s.camera.zoom || 1.0);
  const zMap = [['btn-z-05', 0.51], ['btn-z-1', 1.0], ['btn-z-2', 2.0], ['btn-z-3', 3.0], ['btn-z-5', 5.0]];
  for (const [id, val] of zMap) {
    const el = document.getElementById(id);
    if (el) el.className = Math.abs(z - val) < 0.25 ? 'active' : '';
  }
  document.getElementById('btn-front').className = s.camera.facing === 'front' ? 'active' : '';
  document.getElementById('btn-back').className = s.camera.facing === 'back' ? 'active' : '';
  document.getElementById('btn-mirror').className = s.camera.mirror ? 'active' : '';

  if (s.hilight) {
    const dotHl = document.getElementById('dot-hl');
    dotHl.className = 'dot ' + (s.hilight.connected ? 'ok' : 'warn');
    if (s.hilight.connected && s.hilight.color && s.hilight.color !== '#000000') {
      dotHl.style.background = s.hilight.color;
      dotHl.style.boxShadow = `0 0 10px ${s.hilight.color}`;
    } else {
      dotHl.style.background = '';
      dotHl.style.boxShadow = '';
    }
    document.getElementById('lbl-hl').textContent = s.hilight.connected
      ? `HiLight (${s.hilight.num_lights} LEDs) : ${s.hilight.effect}`
      : 'HiLight : attente...';
  }

  document.getElementById('dot-osmo').className = 'dot ' + (s.osmo.connected ? 'ok' : 'err');
  document.getElementById('lbl-osmo').textContent = s.osmo.connected ? `Osmo : ${s.osmo.name}` : 'Osmo BLE : déconnecté';

  const btnLive = document.getElementById('btn-live');
  btnLive.textContent = s.osmo.live_motors ? '🟢 Moteurs : LIVE' : '⏸ Moteurs : DRY-RUN';
  btnLive.className = s.osmo.live_motors ? 'live-on' : '';

  document.getElementById('btn-mode-speed').className = s.osmo.control_mode === 'speed' ? 'active' : '';
  document.getElementById('btn-mode-rel').className = s.osmo.control_mode === 'relative' ? 'active' : '';
  document.getElementById('btn-mode-joy').className = s.osmo.control_mode === 'joystick' ? 'active' : '';
  document.getElementById('btn-swap').className = s.pid.swap_axes ? 'active' : '';
  document.getElementById('btn-inv-yaw').className = s.pid.invert_yaw ? 'active' : '';
  document.getElementById('btn-inv-pitch').className = s.pid.invert_pitch ? 'active' : '';
  document.getElementById('btn-pol-search').className = s.pid.lost_policy === 'search' ? 'active' : '';
  document.getElementById('btn-pol-home').className = s.pid.lost_policy === 'home' ? 'active' : '';
  document.getElementById('btn-pol-last').className = s.pid.lost_policy === 'last_pos' ? 'active' : '';

  const v = s.vision;
  document.getElementById('lbl-summary').textContent = v.summary;
  const st = document.getElementById('lbl-state');
  st.textContent = v.state;
  st.className = 'badge ' + v.state;

  document.getElementById('m-xy').textContent = v.face_detected ? `${v.cx.toFixed(2)}, ${v.cy.toFixed(2)}` : '—';
  document.getElementById('m-ms').textContent = `${v.vision_ms} ms`;
  document.getElementById('m-vyaw').textContent = `${v.cmd_yaw_dps >= 0 ? '+' : ''}${v.cmd_yaw_dps}°/s`;
  document.getElementById('m-vpitch').textContent = `${v.cmd_pitch_dps >= 0 ? '+' : ''}${v.cmd_pitch_dps}°/s`;

  document.getElementById('m-pkts').textContent = `RX ${s.osmo.packets} | TX ${s.osmo.tx_packets || 0}`;
  if (s.osmo.attitude) {
    document.getElementById('m-pitch').textContent = `${s.osmo.attitude.pitch}°`;
    document.getElementById('m-roll').textContent = `${s.osmo.attitude.roll}°`;
    document.getElementById('m-yaw').textContent = `${s.osmo.attitude.yaw}°`;
  }

  drawRadar(v, s.pid.deadzone);

  const logEl = document.getElementById('event-log');
  logEl.innerHTML = s.events.map(e =>
    `<div class="log-line"><span class="log-ts">${e.ts}</span><span class="log-tag">[${e.tag}]</span>${e.msg}</div>`
  ).join('');

  if (s.voice && typeof renderVoiceState === 'function') {
    renderVoiceState(s.voice);
  }
}
