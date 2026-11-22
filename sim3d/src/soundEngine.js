/**
 * Subtle Wall-E / Pixar-style procedural WebAudio micro-sounds + chill lo-fi BPM synth
 * for ELEGNT non-verbal acoustic feedback (zero external audio files required).
 */
export class SoundEngine {
  constructor() {
    this.enabled = true;
    this.ctx = null;
    this.musicActive = false;
    this.bpm = 92;
    this._beatTimer = null;
    this._step = 0;
    this.onBeat = null;
  }

  _ensureCtx() {
    if (!this.enabled) return null;
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume().catch(() => {});
    }
    return this.ctx;
  }

  playCue(kind) {
    const ctx = this._ensureCtx();
    if (!ctx) return;
    const now = ctx.currentTime;

    const tone = (f0, f1, dur, delay = 0, vol = 0.08, type = 'sine') => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(f0, now + delay);
      osc.frequency.exponentialRampToValueAtTime(Math.max(40, f1), now + delay + dur);
      gain.gain.setValueAtTime(0.0001, now + delay);
      gain.gain.exponentialRampToValueAtTime(vol, now + delay + 0.018);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + delay + dur);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now + delay);
      osc.stop(now + delay + dur + 0.02);
    };

    if (kind === 'curious') {
      // Gentle rising "hmm?"
      tone(340, 520, 0.22, 0, 0.065, 'sine');
    } else if (kind === 'yes') {
      // Warm double affirmative chirp
      tone(440, 587, 0.11, 0, 0.06, 'sine');
      tone(587, 659, 0.14, 0.11, 0.055, 'sine');
    } else if (kind === 'doubt') {
      // Soft downward "ohh?"
      tone(420, 310, 0.24, 0, 0.055, 'triangle');
    } else if (kind === 'lamp_on') {
      // Warm tungsten glow swell
      tone(220, 330, 0.35, 0, 0.05, 'sine');
      tone(440, 660, 0.3, 0.06, 0.035, 'triangle');
    } else if (kind === 'lamp_off') {
      tone(392, 246, 0.28, 0, 0.045, 'sine');
    } else if (kind === 'card') {
      // Crystalline OLED board chime
      tone(659, 659, 0.12, 0, 0.055, 'sine');
      tone(988, 988, 0.22, 0.08, 0.05, 'sine');
    } else if (kind === 'nudge') {
      // Discreet gentle tap
      tone(523, 587, 0.15, 0, 0.04, 'sine');
    } else if (kind === 'celebrate') {
      // Playful 3-note Pixar arpeggio
      tone(523, 523, 0.1, 0.0, 0.06, 'sine');
      tone(659, 659, 0.1, 0.09, 0.065, 'sine');
      tone(784, 1046, 0.24, 0.18, 0.07, 'sine');
    } else if (kind === 'recoil') {
      tone(540, 380, 0.12, 0, 0.045, 'sine');
    }
  }

  toggleMusic(enable) {
    if (enable && !this.musicActive) {
      const ctx = this._ensureCtx();
      if (!ctx) return;
      this.musicActive = true;
      this._step = 0;
      const stepMs = (60 / this.bpm / 2) * 1000; // 8th notes
      const chords = [
        [261.63, 329.63, 392.0, 493.88], // Cmaj7
        [220.0, 261.63, 329.63, 392.0],  // Am7
        [174.61, 220.0, 261.63, 329.63], // Fmaj7
        [196.0, 246.94, 293.66, 392.0],  // G6
      ];
      const tick = () => {
        if (!this.musicActive || !this.ctx) return;
        const now = this.ctx.currentTime;
        const beat = this._step % 8;
        if (beat === 0 || beat === 4) {
          // Soft kick
          const osc = this.ctx.createOscillator();
          const g = this.ctx.createGain();
          osc.frequency.setValueAtTime(115, now);
          osc.frequency.exponentialRampToValueAtTime(42, now + 0.14);
          g.gain.setValueAtTime(0.11, now);
          g.gain.exponentialRampToValueAtTime(0.001, now + 0.16);
          osc.connect(g);
          g.connect(this.ctx.destination);
          osc.start(now);
          osc.stop(now + 0.17);
          if (this.onBeat) this.onBeat(beat === 0 ? 1.0 : 0.7);
        }
        if (beat === 0) {
          // Warm Rhodes-like pad chord
          const ch = chords[Math.floor(this._step / 8) % chords.length];
          ch.forEach((f) => {
            const osc = this.ctx.createOscillator();
            const g = this.ctx.createGain();
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(f, now);
            g.gain.setValueAtTime(0.0001, now);
            g.gain.exponentialRampToValueAtTime(0.022, now + 0.04);
            g.gain.exponentialRampToValueAtTime(0.0001, now + 1.1);
            osc.connect(g);
            g.connect(this.ctx.destination);
            osc.start(now);
            osc.stop(now + 1.15);
          });
        } else if (beat === 2 || beat === 6) {
          // Soft brushed hi-hat click
          const osc = this.ctx.createOscillator();
          const g = this.ctx.createGain();
          osc.type = 'square';
          osc.frequency.setValueAtTime(1400, now);
          g.gain.setValueAtTime(0.012, now);
          g.gain.exponentialRampToValueAtTime(0.0001, now + 0.035);
          osc.connect(g);
          g.connect(this.ctx.destination);
          osc.start(now);
          osc.stop(now + 0.04);
        }
        this._step += 1;
      };
      tick();
      this._beatTimer = setInterval(tick, stepMs);
    } else if (!enable && this.musicActive) {
      this.musicActive = false;
      if (this._beatTimer) {
        clearInterval(this._beatTimer);
        this._beatTimer = null;
      }
    }
  }
}
