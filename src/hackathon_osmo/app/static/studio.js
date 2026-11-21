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
async function launchPixelFace() {
  try {
    const res = await fetch('/api/pixel/launch_face', { method: 'POST' });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      alert('Impossible de lancer /face sur le Pixel : ' + (data.error || res.status));
    }
  } catch (err) {
    alert('Erreur réseau : ' + err.message);
  }
}
async function addMemoryItem() {
  const inp = document.getElementById('txt-mem');
  const cat = document.getElementById('sel-mem-cat')?.value || 'fact';
  const content = (inp?.value || '').trim();
  if (!content) return;
  inp.value = '';
  await fetch('/api/memory', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, category: cat }),
  }).catch(() => {});
}
async function deleteMemoryItem(id) {
  await fetch(`/api/memory/${id}`, { method: 'DELETE' }).catch(() => {});
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

  // CAD dashed crosshairs + concentric range ring
  ctx.save();
  ctx.strokeStyle = 'rgba(232, 237, 245, 0.14)';
  ctx.lineWidth = 1;
  ctx.setLineDash([2.5, 3.5]);
  ctx.beginPath();
  ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
  ctx.moveTo(0, cy); ctx.lineTo(w, cy);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(cx, cy, w * 0.36, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  // Deadzone framing square
  const dzPx = (dz / 2) * w;
  ctx.strokeStyle = 'rgba(96, 165, 250, 0.45)';
  ctx.strokeRect(cx - dzPx, cy - dzPx, dzPx * 2, dzPx * 2);

  // Center origin hollow node
  ctx.beginPath();
  ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
  ctx.fillStyle = '#07080b';
  ctx.fill();
  ctx.strokeStyle = 'rgba(232, 237, 245, 0.55)';
  ctx.stroke();

  if (v.face_detected) {
    const fx = cx + v.cx * (w / 2 - 8);
    const fy = cy + v.cy * (h / 2 - 8);
    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(fx, fy); ctx.stroke();
    ctx.fillStyle = '#07080b';
    ctx.beginPath(); ctx.arc(fx, fy, 4.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#34d399';
    ctx.lineWidth = 1.6;
    ctx.stroke();
  }
  ctx.restore();
}

function render(s) {
  const camOk = s.camera.status.startsWith('live_');
  document.getElementById('dot-cam').className = 'dot ' + (camOk ? 'ok' : 'warn');
  document.getElementById('lbl-cam').textContent = `PIXEL · ${s.camera.status.toUpperCase()} (${(s.camera.zoom || 1).toFixed(1)}×)`;
  document.getElementById('lbl-fps').textContent = `${s.camera.fps} fps`;
  document.getElementById('lbl-rot').textContent = `${s.camera.rotate_deg}°`;
  const lblRes = document.getElementById('lbl-res');
  if (lblRes && s.camera.width) {
    const sh = s.camera.sheet && s.camera.sheet.found ? ` · SHEET ${s.camera.sheet.lines_count}L` : '';
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
      ? `HILIGHT · ${String(s.hilight.effect || 'idle').toUpperCase()}`
      : 'HILIGHT · WAIT';
  }

  document.getElementById('dot-osmo').className = 'dot ' + (s.osmo.connected ? 'ok' : 'err');
  document.getElementById('lbl-osmo').textContent = s.osmo.connected ? `OSMO · ${s.osmo.name}` : 'OSMO · OFFLINE';

  const btnLive = document.getElementById('btn-live');
  btnLive.textContent = s.osmo.live_motors ? 'MOTORS : LIVE' : 'MOTORS : DRY-RUN';
  btnLive.className = s.osmo.live_motors ? 'live-on' : '';

  const lblLat = document.getElementById('lbl-latency');
  if (lblLat && s.telemetry && s.telemetry.latencies_ms) {
    const p50 = Number(s.telemetry.latencies_ms.turn_total_p50 || 0);
    lblLat.textContent = p50 > 0 ? `P50 ${(p50 / 1000).toFixed(1)}s` : 'P50 —';
  }

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

  if (s.memory && s.memory.stats) {
    const mc = document.getElementById('m-mem-count');
    if (mc) mc.textContent = `${s.memory.stats.memories_count} MEM · ${s.memory.stats.episodes_count} EP`;
    const ml = document.getElementById('mem-list');
    if (ml) {
      const items = s.memory.items || [];
      ml.innerHTML = items.length
        ? items.map(m => `
            <div class="mem-item">
              <span class="mem-cat">${escapeHtml(m.category)}</span>
              <span class="mem-txt">${escapeHtml(m.content)}</span>
              <button class="mem-del" onclick="deleteMemoryItem(${m.id})" title="Supprimer">×</button>
            </div>
          `).join('')
        : '<div class="chat-empty">Aucun souvenir enregistré.</div>';
    }
  }

  if (s.voice && typeof renderVoiceState === 'function') {
    renderVoiceState(s.voice);
  }
}
