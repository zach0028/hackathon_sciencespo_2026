/* OpenAI GPT-Live 1.1 WebRTC Full-Duplex Client + Delegated GPT-6.2 Astra Brain UI */

let peerConn = null;
let dataChannel = null;
let micStream = null;
let audioCtx = null;
let analyser = null;
let vuLoopId = null;
let liveActive = false;
let micMuted = false;
let textBusy = false;

function renderVoiceState(voice) {
  const selModel = document.getElementById('sel-model');
  const selEffort = document.getElementById('sel-effort');
  const selVoice = document.getElementById('sel-voice');
  if (selModel && voice.brain_model) selModel.value = voice.brain_model;
  if (selEffort && voice.reasoning_effort) selEffort.value = voice.reasoning_effort;
  if (selVoice && voice.voice) selVoice.value = voice.voice;

  const btnVision = document.getElementById('btn-vision');
  if (btnVision) btnVision.className = voice.use_vision ? 'active' : '';

  const lbl = document.getElementById('lbl-voice-status');
  if (!textBusy) {
    if (!voice.ready) {
      lbl.textContent = 'NO API KEY';
    } else if (voice.busy) {
      lbl.textContent = `THINKING · ${voice.brain_model}`;
    } else if (voice.speaking) {
      lbl.textContent = 'SPEAKING · FULL-DUPLEX';
    } else if (liveActive && voice.sideband_connected) {
      lbl.textContent = micMuted ? 'LIVE · MUTED' : 'LIVE · ACTIVE';
    } else if (liveActive) {
      lbl.textContent = 'WEBRTC · LINKING';
    } else {
      lbl.textContent = 'READY';
    }
  }

  if (voice.transcript && voice.transcript.length > 0) {
    document.getElementById('chat-box').innerHTML = voice.transcript.map(t => {
      const toolsStr = (t.tools && t.tools.length) ? t.tools.join(', ') : '—';
      const msStr = t.brain_ms ? `${t.brain_ms}ms` : 'live';
      return `
        <div class="chat-turn">
          <div class="chat-user"><b>Zach</b> (${t.ts}) : ${escapeHtml(t.user || '')}</div>
          <div class="chat-bot"><b>Osmo</b> : ${escapeHtml(t.reply || '…')}</div>
          <div class="chat-meta">${escapeHtml(t.model || '')} · outils=${escapeHtml(toolsStr)} · ${msStr}</div>
        </div>
      `;
    }).join('');
  }
}

function updateVoiceCfg() {
  send({
    action: 'voice_config',
    brain_model: document.getElementById('sel-model').value,
    reasoning_effort: document.getElementById('sel-effort').value,
    voice: document.getElementById('sel-voice').value,
  });
}

function toggleVision() {
  if (!state || !state.voice) return;
  send({ action: 'voice_config', use_vision: !state.voice.use_vision });
}

function startVuLoop(stream) {
  if (vuLoopId) cancelAnimationFrame(vuLoopId);
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const src = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  src.connect(analyser);

  const buf = new Uint8Array(analyser.fftSize);
  const tick = () => {
    if (!analyser || !liveActive) {
      document.getElementById('vu-fill').style.width = '0%';
      return;
    }
    if (micMuted) {
      document.getElementById('vu-fill').style.width = '0%';
    } else {
      analyser.getByteTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) {
        const v = (buf[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / buf.length);
      document.getElementById('vu-fill').style.width = Math.min(100, Math.round(rms * 420)) + '%';
    }
    vuLoopId = requestAnimationFrame(tick);
  };
  vuLoopId = requestAnimationFrame(tick);
}

function waitIceComplete(pc, timeoutMs = 1500) {
  if (pc.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, timeoutMs);
    const onState = () => {
      if (pc.iceGatheringState === 'complete') {
        clearTimeout(timer);
        pc.removeEventListener('icegatheringstatechange', onState);
        resolve();
      }
    };
    pc.addEventListener('icegatheringstatechange', onState);
  });
}

async function toggleLiveCall() {
  if (liveActive) {
    await stopLiveCall();
  } else {
    await startLiveCall();
  }
}

async function startLiveCall() {
  const btn = document.getElementById('btn-gpt-live');
  const btnMute = document.getElementById('btn-mute');
  const lbl = document.getElementById('lbl-voice-status');

  btn.disabled = true;
  lbl.textContent = '⏳ Connexion WebRTC GPT-Live 1.1...';

  try {
    // 1. Capture microphone with browser hardware Echo Cancellation (essential for full-duplex barge-in!)
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    micMuted = false;

    // 2. Create WebRTC peer connection + audio track + events data channel
    peerConn = new RTCPeerConnection();
    const audioEl = document.getElementById('live-audio');
    peerConn.ontrack = (ev) => {
      audioEl.srcObject = ev.streams[0];
      audioEl.play().catch(() => {});
    };
    peerConn.onconnectionstatechange = () => {
      if (!peerConn) return;
      if (peerConn.connectionState === 'failed' || peerConn.connectionState === 'disconnected') {
        lbl.textContent = '⚠️ WebRTC déconnecté';
      }
    };

    for (const track of micStream.getAudioTracks()) {
      peerConn.addTrack(track, micStream);
    }

    dataChannel = peerConn.createDataChannel('oai-events');
    dataChannel.onmessage = (ev) => {
      // Primary event stream (sideband on Python server handles delegation + tools securely)
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'error') {
          lbl.textContent = '❌ Live: ' + (msg.error?.message || 'erreur');
        }
      } catch (_) {}
    };

    // 3. Generate SDP offer & negotiate via local FastAPI backend (keeps OPENAI_API_KEY server-side)
    const offer = await peerConn.createOffer();
    await peerConn.setLocalDescription(offer);
    await waitIceComplete(peerConn, 1200);

    const resp = await fetch('/api/live/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sdp: peerConn.localDescription.sdp }),
    });
    const data = await resp.json();
    if (!resp.ok || data.error || !data.sdp) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }

    await peerConn.setRemoteDescription({ type: 'answer', sdp: data.sdp });

    liveActive = true;
    startVuLoop(micStream);
    btn.classList.add('recording');
    btn.textContent = 'STOP LIVE SESSION';
    btnMute.disabled = false;
    btnMute.textContent = 'MIC ON';
    btnMute.className = 'active';
    lbl.textContent = 'LIVE · CONNECTED';
  } catch (err) {
    lbl.textContent = 'ERR · ' + (err.message || err);
    await stopLiveCall();
  } finally {
    btn.disabled = false;
  }
}

async function stopLiveCall() {
  liveActive = false;
  micMuted = false;

  if (vuLoopId) {
    cancelAnimationFrame(vuLoopId);
    vuLoopId = null;
  }
  if (dataChannel) {
    try { dataChannel.close(); } catch (_) {}
    dataChannel = null;
  }
  if (peerConn) {
    try { peerConn.close(); } catch (_) {}
    peerConn = null;
  }
  if (micStream) {
    for (const t of micStream.getTracks()) t.stop();
    micStream = null;
  }
  if (audioCtx) {
    try { await audioCtx.close(); } catch (_) {}
    audioCtx = null;
    analyser = null;
  }

  const audioEl = document.getElementById('live-audio');
  if (audioEl) audioEl.srcObject = null;

  fetch('/api/live/stop', { method: 'POST' }).catch(() => {});

  const btn = document.getElementById('btn-gpt-live');
  const btnMute = document.getElementById('btn-mute');
  btn.classList.remove('recording');
  btn.textContent = 'START LIVE SESSION';
  btnMute.disabled = true;
  btnMute.textContent = 'MIC ON';
  btnMute.className = '';
  document.getElementById('vu-fill').style.width = '0%';
}

function toggleMicMute() {
  if (!micStream || !liveActive) return;
  micMuted = !micMuted;
  for (const track of micStream.getAudioTracks()) {
    track.enabled = !micMuted;
  }
  const btnMute = document.getElementById('btn-mute');
  btnMute.textContent = micMuted ? 'MIC MUTED' : 'MIC ON';
  btnMute.className = micMuted ? 'danger' : 'active';
}

async function sendTextTurn() {
  const inp = document.getElementById('txt-prompt');
  const txt = inp.value.trim();
  if (!txt || textBusy) return;
  inp.value = '';
  textBusy = true;
  const lbl = document.getElementById('lbl-voice-status');
  lbl.textContent = '🧠 Cerveau OpenAI + outils natifs...';
  try {
    const resp = await fetch('/api/voice/text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: txt }),
    });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      lbl.textContent = '❌ ' + (data.error || `HTTP ${resp.status}`);
    } else {
      lbl.textContent = `✅ Répondu en ${data.brain_ms || 0} ms`;
    }
  } catch (e) {
    lbl.textContent = '❌ ' + e.message;
  } finally {
    textBusy = false;
  }
}

window.addEventListener('keydown', (e) => {
  if (e.key.toLowerCase() === 'm' && liveActive && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'SELECT') {
    e.preventDefault();
    toggleMicMute();
  }
});
window.addEventListener('beforeunload', () => {
  if (liveActive) fetch('/api/live/stop', { method: 'POST', keepalive: true }).catch(() => {});
});
