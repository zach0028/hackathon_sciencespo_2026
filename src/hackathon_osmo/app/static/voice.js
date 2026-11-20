/* Voice Companion: Native 16 kHz PCM WAV Capture (Safari/Chrome/Arc safe) + Push-to-Talk + Auto-VAD */

let micStream = null;
let audioCtx = null;
let analyser = null;
let pcmNode = null;
let isRecording = false;
let pcmBuffers = [];
let pttActive = false;
let handsFree = false;
let vadSpeaking = false;
let vadSilenceSince = 0;
let voiceBusyLocal = false;
let robotSpeaking = false;
let serverSpeaking = false;
let vadLoopId = null;

function renderVoiceState(voice) {
  document.getElementById('sel-model').value = voice.model;
  document.getElementById('sel-voice').value = voice.voice_id;
  document.getElementById('btn-vision').className = voice.use_vision ? 'active' : '';
  const btnOut = document.getElementById('btn-out');
  btnOut.textContent = voice.audio_output === 'mac' ? '🔈 Sortie : MacBook' : '🔈 Sortie : Navigateur';
  btnOut.className = 'active';
  serverSpeaking = !!voice.speaking;

  if (!voiceBusyLocal) {
    document.getElementById('lbl-voice-status').textContent = voice.ready
      ? (voice.speaking ? '🗣️ Osmo parle...' : (voice.busy ? '🧠 Réflexion...' : (handsFree ? '🔊 Écoute auto active' : '✅ Prêt')))
      : '⚠️ Clés API manquantes';
  }
  if (voice.transcript && voice.transcript.length > 0) {
    document.getElementById('chat-box').innerHTML = voice.transcript.map(t => `
      <div class="chat-turn">
        <div class="chat-user"><b>Zach</b> (${t.ts}) : ${escapeHtml(t.user)}</div>
        <div class="chat-bot"><b>Osmo</b> : ${escapeHtml(t.reply)}</div>
        <div class="chat-meta">geste=${t.gesture || '—'} · led=${t.led || '—'} · STT ${t.stt_ms}ms · LLM ${t.llm_ms}ms · TTS ${t.tts_ms}ms</div>
      </div>
    `).join('');
  }
}

function updateVoiceCfg() {
  send({
    action: 'voice_config',
    model: document.getElementById('sel-model').value,
    voice_id: document.getElementById('sel-voice').value,
  });
}

function toggleVision() {
  if (!state || !state.voice) return;
  send({ action: 'voice_config', use_vision: !state.voice.use_vision });
}

function toggleAudioOut() {
  if (!state || !state.voice) return;
  const next = state.voice.audio_output === 'mac' ? 'browser' : 'mac';
  send({ action: 'voice_config', audio_output: next });
}

function encodeWav16k(chunks, inputRate) {
  let totalLen = 0;
  for (const c of chunks) totalLen += c.length;
  const merged = new Float32Array(totalLen);
  let off = 0;
  for (const c of chunks) { merged.set(c, off); off += c.length; }

  const targetRate = 16000;
  const ratio = inputRate / targetRate;
  const outLen = Math.max(1, Math.floor(merged.length / ratio));
  const pcm = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const idx = Math.min(merged.length - 1, Math.floor(i * ratio));
    const s = Math.max(-1, Math.min(1, merged[idx]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }

  const buf = new ArrayBuffer(44 + pcm.byteLength);
  const v = new DataView(buf);
  const writeStr = (pos, str) => { for (let i = 0; i < str.length; i++) v.setUint8(pos + i, str.charCodeAt(i)); };
  writeStr(0, 'RIFF');
  v.setUint32(4, 36 + pcm.byteLength, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, targetRate, true);
  v.setUint32(28, targetRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  writeStr(36, 'data');
  v.setUint32(40, pcm.byteLength, true);
  new Int16Array(buf, 44).set(pcm);
  return new Blob([buf], { type: 'audio/wav' });
}

async function ensureMic() {
  if (micStream && audioCtx) {
    if (audioCtx.state === 'suspended') await audioCtx.resume();
    return micStream;
  }
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
  });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const src = audioCtx.createMediaStreamSource(micStream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  src.connect(analyser);

  pcmNode = audioCtx.createScriptProcessor(4096, 1, 1);
  pcmNode.onaudioprocess = (ev) => {
    if (!isRecording) return;
    pcmBuffers.push(new Float32Array(ev.inputBuffer.getChannelData(0)));
  };
  src.connect(pcmNode);
  pcmNode.connect(audioCtx.destination);
  startVuLoop();
  return micStream;
}

function getMicRms() {
  if (!analyser) return 0;
  const buf = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(buf);
  let sum = 0;
  for (let i = 0; i < buf.length; i++) {
    const v = (buf[i] - 128) / 128;
    sum += v * v;
  }
  return Math.sqrt(sum / buf.length);
}

function startVuLoop() {
  if (vadLoopId) return;
  const tick = () => {
    const rms = getMicRms();
    document.getElementById('vu-fill').style.width = Math.min(100, Math.round(rms * 420)) + '%';

    if (handsFree && !voiceBusyLocal && !robotSpeaking && !serverSpeaking) {
      const now = performance.now();
      const THRESH = 0.030;
      if (rms > THRESH) {
        vadSilenceSince = 0;
        if (!vadSpeaking) {
          vadSpeaking = true;
          beginRecording();
          document.getElementById('lbl-voice-status').textContent = '🎙️ Voix détectée...';
        }
      } else if (vadSpeaking) {
        if (!vadSilenceSince) vadSilenceSince = now;
        if (now - vadSilenceSince > 850) {
          vadSpeaking = false;
          vadSilenceSince = 0;
          finishRecordingAndSend();
        }
      }
    }
    vadLoopId = requestAnimationFrame(tick);
  };
  vadLoopId = requestAnimationFrame(tick);
}

function beginRecording() {
  if (!audioCtx || isRecording) return;
  pcmBuffers = [];
  isRecording = true;
  document.getElementById('btn-ptt').classList.add('recording');
}

async function finishRecordingAndSend() {
  document.getElementById('btn-ptt').classList.remove('recording');
  if (!isRecording) return;
  isRecording = false;
  if (!pcmBuffers.length || !audioCtx) return;

  const wavBlob = encodeWav16k(pcmBuffers, audioCtx.sampleRate);
  pcmBuffers = [];
  if (wavBlob.size < 4000) return;

  voiceBusyLocal = true;
  document.getElementById('lbl-voice-status').textContent = '⏳ Voxtral STT → LLM → TTS...';
  try {
    const fd = new FormData();
    fd.append('audio', wavBlob, 'speech.wav');
    const resp = await fetch('/api/voice/audio', { method: 'POST', body: fd });
    const data = await resp.json();
    await playVoiceResponse(data);
  } catch (e) {
    document.getElementById('lbl-voice-status').textContent = '❌ Erreur : ' + e.message;
  } finally {
    voiceBusyLocal = false;
  }
}

async function startPTT() {
  if (pttActive || voiceBusyLocal) return;
  pttActive = true;
  try {
    await ensureMic();
    document.getElementById('lbl-voice-status').textContent = '🎙️ Enregistrement...';
    beginRecording();
  } catch (e) {
    pttActive = false;
    document.getElementById('lbl-voice-status').textContent = '❌ Micro refusé';
  }
}

async function stopPTT() {
  if (!pttActive) return;
  pttActive = false;
  await finishRecordingAndSend();
}

async function toggleHandsFree() {
  handsFree = !handsFree;
  const btn = document.getElementById('btn-vad');
  btn.className = handsFree ? 'live-on' : '';
  btn.textContent = handsFree ? '🔊 Mains libres : ACTIF' : '🔊 Mains libres (Auto)';
  if (handsFree) {
    try {
      await ensureMic();
      document.getElementById('lbl-voice-status').textContent = '🔊 Écoute auto active';
    } catch (e) {
      handsFree = false;
      btn.className = '';
      btn.textContent = '🔊 Mains libres (Auto)';
    }
  } else {
    vadSpeaking = false;
    isRecording = false;
  }
}

async function sendTextTurn() {
  const inp = document.getElementById('txt-prompt');
  const txt = inp.value.trim();
  if (!txt || voiceBusyLocal) return;
  inp.value = '';
  voiceBusyLocal = true;
  document.getElementById('lbl-voice-status').textContent = '🧠 LLM → Voxtral TTS...';
  try {
    const resp = await fetch('/api/voice/text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: txt }),
    });
    const data = await resp.json();
    await playVoiceResponse(data);
  } finally {
    voiceBusyLocal = false;
  }
}

async function playVoiceResponse(data) {
  if (!data || data.error) {
    document.getElementById('lbl-voice-status').textContent = data?.error ? '❌ ' + data.error : '✅ Prêt';
    return;
  }
  if (data.played_on_mac) {
    document.getElementById('lbl-voice-status').textContent = '🗣️ Osmo parle (MacBook)...';
    return;
  }
  if (!data.audio_b64) {
    document.getElementById('lbl-voice-status').textContent = '✅ Prêt';
    return;
  }
  robotSpeaking = true;
  document.getElementById('lbl-voice-status').textContent = '🗣️ Osmo parle...';
  try {
    const audio = new Audio(`data:${data.audio_mime || 'audio/wav'};base64,${data.audio_b64}`);
    await new Promise((resolve) => {
      audio.onended = resolve;
      audio.onerror = resolve;
      audio.play().catch(resolve);
    });
  } finally {
    setTimeout(() => { robotSpeaking = false; }, 350);
  }
}

window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !e.repeat && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'SELECT') {
    e.preventDefault();
    startPTT();
  }
});
window.addEventListener('keyup', (e) => {
  if (e.code === 'Space' && pttActive) {
    e.preventDefault();
    stopPTT();
  }
});
