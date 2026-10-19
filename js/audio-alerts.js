/**
 * Audio & Voice Alert Engine
 * Posture Detection System Using PoseNet
 * Author: Aryan Raj Pandey (B.Tech CSE - AI & ML)
 */

class AudioAlertManager {
  constructor() {
    this.audioCtx = null;
    this.voiceEnabled = true;
    this.soundEnabled = true;
    this.lastSpeechTime = 0;
    this.speechCooldown = 5000; // 5 seconds default
    this.lastSpokenText = '';
    this.synth = window.speechSynthesis || null;
    
    // Track alert state with time-based duration
    this.badPostureStartTime = 0;
    this.primaryAlertTriggered = false;
    this.thresholdAlertMs = 3000; // 3 seconds of continuous deviation before voice/tone alert
  }

  initAudioContext() {
    if (!this.audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (AudioContext) {
        this.audioCtx = new AudioContext();
      }
    }
    if (this.audioCtx && this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
  }

  setVoiceEnabled(enabled) {
    this.voiceEnabled = enabled;
  }

  setSoundEnabled(enabled) {
    this.soundEnabled = enabled;
  }

  setCooldown(seconds) {
    this.speechCooldown = seconds * 1000;
  }

  // Play synthetic tone using Web Audio API (smooth chimes, no external assets required)
  playTone(type = 'warning') {
    if (!this.soundEnabled) return;
    try {
      this.initAudioContext();
      if (!this.audioCtx) return;

      const now = this.audioCtx.currentTime;
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();

      osc.connect(gain);
      gain.connect(this.audioCtx.destination);

      if (type === 'good') {
        // High harmonic double-pip
        osc.type = 'sine';
        osc.frequency.setValueAtTime(587.33, now); // D5
        osc.frequency.exponentialRampToValueAtTime(880.00, now + 0.15); // A5
        gain.gain.setValueAtTime(0.08, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);
        osc.start(now);
        osc.stop(now + 0.25);
      } else if (type === 'warning') {
        // Mild reminder blip
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(440, now); // A4
        osc.frequency.setValueAtTime(392, now + 0.1); // G4
        gain.gain.setValueAtTime(0.12, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
        osc.start(now);
        osc.stop(now + 0.3);
      } else if (type === 'alert') {
        // Attention low double tone
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(260, now);
        osc.frequency.setValueAtTime(220, now + 0.15);
        gain.gain.setValueAtTime(0.15, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
        osc.start(now);
        osc.stop(now + 0.35);
      }
    } catch (e) {
      console.warn('Audio tone error:', e);
    }
  }

  // Speak alert using Web Speech API
  speak(text) {
    if (!this.voiceEnabled || !this.synth) return;

    const now = Date.now();
    // Prevent repetitive nagging within cooldown
    if (now - this.lastSpeechTime < this.speechCooldown && this.lastSpokenText === text) {
      return;
    }
    if (now - this.lastSpeechTime < 2500) {
      return;
    }

    try {
      this.synth.cancel(); // Cancel any lingering utterances
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.05;
      utterance.pitch = 1.0;
      utterance.volume = 0.85;

      // Select natural English voice if present
      const voices = this.synth.getVoices();
      const engVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha')));
      if (engVoice) {
        utterance.voice = engVoice;
      }

      this.synth.speak(utterance);
      this.lastSpeechTime = now;
      this.lastSpokenText = text;
    } catch (e) {
      console.warn('Speech synthesis error:', e);
    }
  }

  // Handle posture frame evaluation with time-based debouncing
  processPostureAlert(classification) {
    if (!classification) return;

    const now = Date.now();

    if (classification.status === 'good') {
      if (this.badPostureStartTime > 0 && this.primaryAlertTriggered) {
        this.playTone('good');
      }
      this.badPostureStartTime = 0;
      this.primaryAlertTriggered = false;
    } else {
      if (this.badPostureStartTime === 0) {
        this.badPostureStartTime = now;
        this.primaryAlertTriggered = false;
      }

      const elapsed = now - this.badPostureStartTime;

      if (elapsed >= this.thresholdAlertMs && !this.primaryAlertTriggered) {
        this.primaryAlertTriggered = true;
        this.playTone('warning');
        
        // Choose primary voice warning message
        if (classification.issues.length > 0) {
          const primaryIssue = classification.issues[0];
          this.speak(primaryIssue.speechMessage || 'Please adjust your posture');
        }
      } else if (this.primaryAlertTriggered && elapsed > this.thresholdAlertMs * 3) {
        // Continuous reminder every 10 seconds if user continues bad posture
        if (now - this.lastSpeechTime >= this.speechCooldown) {
          this.playTone('alert');
          if (classification.issues.length > 0) {
            this.speak(classification.issues[0].speechMessage || 'Check your posture');
          }
        }
      }
    }
  }
}

// Global audio alert manager singleton
window.audioAlertManager = new AudioAlertManager();
