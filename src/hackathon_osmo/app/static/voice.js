/* Dual-Provider Full-Duplex Voice Client:
 *  - GEMINI : Gemini 3.8 Live (WebSocket PCM16@16kHz in / 24kHz out + direct NON_BLOCKING tools) + Gemini 4 Neon 4K Brain
 *  - OPENAI : GPT-Live 1.1 (WebRTC Full-Duplex) + Delegated GPT-6.2 Vision Brain
 */

let peerConn = null;
let dataChannel = null;
let geminiWs = null;
let geminiProc = null;
let geminiSink = null;
let playCtx = null;
let playDestNode = null;
let nextPlayTime = 0;
let activePlayNodes = new Set();

let micStream = null;
let audioCtx = null;
let analyser = null;
let vuLoopId = null;
let liveActive = false;
let liveProvider = null; // 'gemini' | 'openai'
let micMuted = false;
let pttLockIn = false;   // Optional noisy-room mode: hold SPACE to transmit mic (default: false = Full-Duplex)
let pttSpaceHeld = false;
let textBusy = false;
let _lastProviderKey = '';
let _wasTransmitting = false;

function currentProvider() {
  return (state && state.voice && state.voice.provider) ? state.voice.provider : 'gemini';
}

function sendGeminiStreamEnd() {
  if (geminiWs && geminiWs.readyState === WebSocket.OPEN) {
    try { geminiWs.send(JSON.stringify({ type: 'audio_stream_end' })); } catch (_) {}
  }
}

function applyMicTrackState() {
  if (!micStream) return;
  const shouldTransmit = pttLockIn ? pttSpaceHeld : !micMuted;
  for (const track of micStream.getAudioTracks()) {
    track.enabled = shouldTransmit;
  }
  // Hybrid VAD cue for Gemini 3.8 Live when mic closes or PTT key releases
  if (_wasTransmitting && !shouldTransmit && liveProvider === 'gemini') {
    sendGeminiStreamEnd();
  }
  _wasTransmitting = shouldTransmit;

  const btnMute = document.getElementById('btn-mute');
  if (btnMute && liveActive) {
    if (pttLockIn) {
      btnMute.textContent = pttSpaceHeld ? 'TX · SPACE' : 'HOLD SPACE';
      btnMute.className = pttSpaceHeld ? 'live-on' : 'danger';
    } else {
      btnMute.textContent = micMuted ? 'MIC MUTED' : 'MIC ON';
      btnMute.className = micMuted ? 'danger' : 'active';
    }
  }
}

function togglePttMode() {
  pttLockIn = !pttLockIn;
  pttSpaceHeld = false;
  const btn = document.getElementById('btn-ptt-mode');
  if (btn) {
    btn.textContent = pttLockIn ? 'PTT · SPACE' : 'PTT : OFF';
    btn.className = pttLockIn ? 'active' : '';
  }
  applyMicTrackState();
}

function _syncSelectOptions(selEl, items, prefix, currentVal) {
  if (!selEl || !Array.isArray(items) || items.length === 0) return;
  const html = items.map(it => `<option value="${escapeHtml(it.id)}">${prefix} · ${escapeHtml(it.label)}</option>`).join('');
  selEl.innerHTML = html;
  if (currentVal) selEl.value = currentVal;
}

function renderVoiceState(voice) {
  const prov = voice.provider || 'gemini';
  const btnProv = document.getElementById('btn-provider');
  if (btnProv) {
    const isGem = prov === 'gemini';
    btnProv.textContent = isGem ? 'AI : GEMINI' : 'AI : OPENAI';
    btnProv.className = isGem ? 'live-on' : 'active';
  }

  const titleEl = document.getElementById('lbl-voice-title');
  if (titleEl) {
    titleEl.textContent = prov === 'gemini'
      ? 'VOICE ENGINE // GEMINI 3.8 LIVE + GEMINI 4 NEON'
      : 'VOICE ENGINE // GPT-LIVE 1.1 + VISION BRAIN';
  }

  const selModel = document.getElementById('sel-model');
  const selEffort = document.getElementById('sel-effort');
  const selVoice = document.getElementById('sel-voice');

  const provKey = `${prov}:${(voice.models || []).length}:${(voice.voices || []).length}`;
  if (provKey !== _lastProviderKey) {
    _lastProviderKey = provKey;
    _syncSelectOptions(selModel, voice.models, 'MODEL', voice.brain_model);
    _syncSelectOptions(selVoice, voice.voices, 'VOICE', voice.voice);
  } else {
    if (selModel && voice.brain_model && document.activeElement !== selModel) selModel.value = voice.brain_model;
    if (selVoice && voice.voice && document.activeElement !== selVoice) selVoice.value = voice.voice;
  }
  if (selEffort && voice.reasoning_effort && document.activeElement !== selEffort) {
    selEffort.value = voice.reasoning_effort;
  }

  const btnVision = document.getElementById('btn-vision');
  if (btnVision) btnVision.className = voice.use_vision ? 'active' : '';

  const btnLang = document.getElementById('btn-lang');
  if (btnLang && voice.language) {
    const isFr = voice.language === 'fr';
    btnLang.textContent = isFr ? 'LANG : FR' : 'LANG : EN';
    btnLang.className = isFr ? 'live-on' : 'active';
  }

  const lbl = document.getElementById('lbl-voice-status');
  if (!textBusy) {
    if (!voice.ready) {
      lbl.textContent = prov === 'gemini' ? 'NO GEMINI KEY' : 'NO OPENAI KEY';
    } else if (voice.busy) {
      lbl.textContent = `THINKING · ${voice.brain_model}`;
    } else if (voice.speaking) {
      lbl.textContent = 'SPEAKING · FULL-DUPLEX';
    } else if (liveActive && voice.sideband_connected) {
      if (pttLockIn) {
        lbl.textContent = pttSpaceHeld ? 'PTT · TRANSMITTING' : 'LOCK-IN · HOLD SPACE';
      } else {
        lbl.textContent = micMuted ? 'LIVE · MUTED' : `LIVE · ${prov.toUpperCase()}`;
      }
    } else if (liveActive) {
      lbl.textContent = prov === 'gemini' ? 'GEMINI WS · LINKING' : 'WEBRTC · LINKING';
    } else {
      lbl.textContent = `READY · ${prov.toUpperCase()}`;
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

async function toggleProvider() {
  const cur = currentProvider();
  const next = cur === 'gemini' ? 'openai' : 'gemini';
  const wasLive = liveActive;
  if (wasLive) {
    await stopLiveCall();
  }
  send({ action: 'voice_config', provider: next });
  _lastProviderKey = '';
  const btnProv = document.getElementById('btn-provider');
  if (btnProv) {
    btnProv.textContent = next === 'gemini' ? 'AI : GEMINI' : 'AI : OPENAI';
    btnProv.className = next === 'gemini' ? 'live-on' : 'active';
  }
  if (wasLive) {
    setTimeout(() => { startLiveCall(next); }, 260);
  }
}

async function toggleLanguage() {
  const cur = (state && state.voice && state.voice.language) ? state.voice.language : 'en';
  const next = cur === 'fr' ? 'en' : 'fr';
  send({ action: 'voice_config', language: next });
  const btnLang = document.getElementById('btn-lang');
  if (btnLang) {
    btnLang.textContent = next === 'fr' ? 'LANG : FR' : 'LANG : EN';
    btnLang.className = next === 'fr' ? 'live-on' : 'active';
  }
  if (liveActive) {
    await stopLiveCall();
    setTimeout(() => { startLiveCall(); }, 220);
  }
}

function startVuLoop(stream, existingCtx = null) {
  if (vuLoopId) cancelAnimationFrame(vuLoopId);
  audioCtx = existingCtx || new (window.AudioContext || window.webkitAudioContext)();
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
    const transmitting = pttLockIn ? pttSpaceHeld : !micMuted;
    if (!transmitting) {
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
  return src;
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

function _downsampleFloat32ToInt16(float32Buf, inputRate, targetRate = 16000) {
  if (inputRate === targetRate) {
    const out = new Int16Array(float32Buf.length);
    for (let i = 0; i < float32Buf.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Buf[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }
  const ratio = inputRate / targetRate;
  const outLen = Math.max(1, Math.floor(float32Buf.length / ratio));
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const s0 = float32Buf[idx] || 0;
    const s1 = float32Buf[Math.min(idx + 1, float32Buf.length - 1)] || s0;
    const s = Math.max(-1, Math.min(1, s0 + (s1 - s0) * frac));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function _flushGeminiPlayback() {
  for (const node of activePlayNodes) {
    try { node.stop(); } catch (_) {}
  }
  activePlayNodes.clear();
  nextPlayTime = 0;
}

function _playGeminiPcm24k(arrayBuffer) {
  if (!playCtx || playCtx.state === 'closed') return;
  if (playCtx.state === 'suspended') {
    playCtx.resume().catch(() => {});
  }
  const int16 = new Int16Array(arrayBuffer);
  if (int16.length === 0) return;
  const audioBuf = playCtx.createBuffer(1, int16.length, 24000);
  const chan = audioBuf.getChannelData(0);
  for (let i = 0; i < int16.length; i++) {
    chan[i] = int16[i] / 32768.0;
  }
  const src = playCtx.createBufferSource();
  src.buffer = audioBuf;
  src.connect(playDestNode || playCtx.destination);
  const now = playCtx.currentTime;
  const startAt = Math.max(now + 0.015, nextPlayTime);
  src.start(startAt);
  nextPlayTime = startAt + audioBuf.duration;
  activePlayNodes.add(src);
  src.onended = () => { activePlayNodes.delete(src); };
}

async function toggleLiveCall() {
  if (liveActive) {
    await stopLiveCall();
  } else {
    await startLiveCall();
  }
}

async function startLiveCall(forceProvider = null) {
  const prov = forceProvider || currentProvider();
  if (prov === 'gemini') {
    await startGeminiLiveCall();
  } else {
    await startOpenAILiveCall();
  }
}

async function startGeminiLiveCall() {
  const btn = document.getElementById('btn-gpt-live');
  const btnMute = document.getElementById('btn-mute');
  const lbl = document.getElementById('lbl-voice-status');

  btn.disabled = true;
  lbl.textContent = '⏳ Connexion Gemini 3.8 Live...';

  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    micMuted = false;

    playCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    await playCtx.resume().catch(() => {});
    // Route playback through <audio id="live-audio"> via MediaStreamDestination when available
    // so Chrome/macOS Acoustic Echo Cancellation (AEC) suppresses speaker bleed into the mic!
    const audioEl = document.getElementById('live-audio');
    if (audioEl && typeof playCtx.createMediaStreamDestination === 'function') {
      const msd = playCtx.createMediaStreamDestination();
      playDestNode = msd;
      audioEl.srcObject = msd.stream;
      audioEl.play().catch(() => {});
    } else {
      playDestNode = null;
    }
    nextPlayTime = 0;
    activePlayNodes = new Set();

    const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    geminiWs = new WebSocket(`${wsProto}//${location.host}/ws/gemini_live`);
    geminiWs.binaryType = 'arraybuffer';

    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Timeout connexion Gemini 3.8 Live')), 12000);
      geminiWs.onmessage = (ev) => {
        if (typeof ev.data === 'string') {
          try {
            const msg = JSON.parse(ev.data);
            if (msg.type === 'ready') {
              clearTimeout(timer);
              resolve();
            } else if (msg.type === 'error') {
              clearTimeout(timer);
              reject(new Error(msg.message || 'Erreur Gemini Live'));
            }
          } catch (_) {}
        }
      };
      geminiWs.onerror = () => {
        clearTimeout(timer);
        reject(new Error('WebSocket Gemini Live échoué'));
      };
      geminiWs.onclose = () => {
        clearTimeout(timer);
        if (!liveActive) reject(new Error('Session Gemini Live fermée'));
      };
    });

    // Attach permanent Gemini WS handlers after ready
    geminiWs.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        _playGeminiPcm24k(ev.data);
      } else if (typeof ev.data === 'string') {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'interrupted') {
            _flushGeminiPlayback();
          } else if (msg.type === 'error') {
            lbl.textContent = '❌ Gemini: ' + (msg.message || 'erreur');
          }
        } catch (_) {}
      }
    };
    geminiWs.onclose = () => {
      if (liveActive && liveProvider === 'gemini') {
        stopLiveCall();
      }
    };

    liveActive = true;
    liveProvider = 'gemini';
    pttSpaceHeld = false;
    _wasTransmitting = false;

    const micSrc = startVuLoop(micStream);
    if (audioCtx && audioCtx.state === 'suspended') {
      await audioCtx.resume().catch(() => {});
    }
    geminiProc = audioCtx.createScriptProcessor(2048, 1, 1);
    geminiSink = audioCtx.createGain();
    geminiSink.gain.value = 0;
    const inRate = audioCtx.sampleRate;

    geminiProc.onaudioprocess = (ev) => {
      if (!liveActive || !geminiWs || geminiWs.readyState !== WebSocket.OPEN) return;
      const transmitting = pttLockIn ? pttSpaceHeld : !micMuted;
      if (!transmitting) return;
      const input = ev.inputBuffer.getChannelData(0);
      const pcm16 = _downsampleFloat32ToInt16(input, inRate, 16000);
      geminiWs.send(pcm16.buffer);
    };

    micSrc.connect(geminiProc);
    geminiProc.connect(geminiSink);
    geminiSink.connect(audioCtx.destination);

    btn.classList.add('recording');
    btn.textContent = 'STOP LIVE SESSION';
    btnMute.disabled = false;
    applyMicTrackState();
    lbl.textContent = pttLockIn ? 'LOCK-IN · HOLD SPACE' : 'LIVE · GEMINI 3.8';
  } catch (err) {
    lbl.textContent = 'ERR · ' + (err.message || err);
    await stopLiveCall();
  } finally {
    btn.disabled = false;
  }
}

async function startOpenAILiveCall() {
  const btn = document.getElementById('btn-gpt-live');
  const btnMute = document.getElementById('btn-mute');
  const lbl = document.getElementById('lbl-voice-status');

  btn.disabled = true;
  lbl.textContent = '⏳ Connexion WebRTC GPT-Live 1.1...';

  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    micMuted = false;

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
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'error') {
          lbl.textContent = '❌ Live: ' + (msg.error?.message || 'erreur');
        }
      } catch (_) {}
    };

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
    liveProvider = 'openai';
    pttSpaceHeld = false;
    startVuLoop(micStream);
    btn.classList.add('recording');
    btn.textContent = 'STOP LIVE SESSION';
    btnMute.disabled = false;
    applyMicTrackState();
    lbl.textContent = pttLockIn ? 'LOCK-IN · HOLD SPACE' : 'LIVE · OPENAI';
  } catch (err) {
    lbl.textContent = 'ERR · ' + (err.message || err);
    await stopLiveCall();
  } finally {
    btn.disabled = false;
  }
}

async function stopLiveCall() {
  liveActive = false;
  liveProvider = null;
  micMuted = false;
  _wasTransmitting = false;

  if (vuLoopId) {
    cancelAnimationFrame(vuLoopId);
    vuLoopId = null;
  }
  if (geminiProc) {
    try { geminiProc.disconnect(); } catch (_) {}
    geminiProc.onaudioprocess = null;
    geminiProc = null;
  }
  if (geminiSink) {
    try { geminiSink.disconnect(); } catch (_) {}
    geminiSink = null;
  }
  if (geminiWs) {
    try {
      if (geminiWs.readyState === WebSocket.OPEN) {
        geminiWs.send(JSON.stringify({ type: 'stop' }));
      }
      geminiWs.close();
    } catch (_) {}
    geminiWs = null;
  }
  _flushGeminiPlayback();
  playDestNode = null;
  if (playCtx) {
    try { await playCtx.close(); } catch (_) {}
    playCtx = null;
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
  if (pttLockIn) {
    togglePttMode();
    return;
  }
  micMuted = !micMuted;
  applyMicTrackState();
}

async function sendTextTurn() {
  const inp = document.getElementById('txt-prompt');
  const txt = inp.value.trim();
  if (!txt || textBusy) return;
  inp.value = '';
  textBusy = true;
  const prov = currentProvider().toUpperCase();
  const lbl = document.getElementById('lbl-voice-status');
  lbl.textContent = `🧠 Cerveau ${prov} + outils natifs...`;
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

function _isTypingTarget(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || el.isContentEditable;
}

window.addEventListener('keydown', (e) => {
  if (_isTypingTarget(document.activeElement)) return;
  if (e.code === 'Space' && pttLockIn && liveActive) {
    e.preventDefault();
    if (!pttSpaceHeld) {
      pttSpaceHeld = true;
      applyMicTrackState();
    }
    return;
  }
  if (e.key.toLowerCase() === 'm' && liveActive && !pttLockIn) {
    e.preventDefault();
    toggleMicMute();
  }
});

window.addEventListener('keyup', (e) => {
  if (e.code === 'Space' && pttLockIn && pttSpaceHeld) {
    e.preventDefault();
    pttSpaceHeld = false;
    applyMicTrackState();
  }
});

window.addEventListener('blur', () => {
  if (pttLockIn && pttSpaceHeld) {
    pttSpaceHeld = false;
    applyMicTrackState();
  }
});
window.addEventListener('beforeunload', () => {
  if (liveActive) fetch('/api/live/stop', { method: 'POST', keepalive: true }).catch(() => {});
});
