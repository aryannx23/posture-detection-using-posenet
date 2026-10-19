/**
 * Main Application Orchestrator & Session Manager
 * Posture Detection System Using PoseNet
 * Author: Aryan Raj Pandey (B.Tech CSE - AI & ML)
 */

document.addEventListener('DOMContentLoaded', async () => {
  // 1. Initialize DOM elements
  const webcamVideo = document.getElementById('webcam');
  const outputCanvas = document.getElementById('outputCanvas');
  const toggleCamBtn = document.getElementById('toggleCamBtn');
  const startCamBtn = document.getElementById('startCamBtn');
  const loadSampleBtn = document.getElementById('loadSampleBtn');
  const calibrateBtn = document.getElementById('calibrateBtn');
  const flipCamBtn = document.getElementById('flipCamBtn');
  const voiceAlertToggle = document.getElementById('voiceAlertToggle');
  const soundToneToggle = document.getElementById('soundToneToggle');
  const guideToggle = document.getElementById('guideToggle');
  const sensitivitySelect = document.getElementById('sensitivitySelect');
  const modelMultiplier = document.getElementById('modelMultiplier');
  const inputResolutionSelect = document.getElementById('inputResolutionSelect');
  const alertDelay = document.getElementById('alertDelay');
  const resetStatsBtn = document.getElementById('resetStatsBtn');
  const exportReportBtn = document.getElementById('exportReportBtn');

  // Mode buttons
  const modeSittingBtn = document.getElementById('modeSittingBtn');
  const modeStandingBtn = document.getElementById('modeStandingBtn');
  const modeDemoBtn = document.getElementById('modeDemoBtn');

  // Modal elements
  const helpModalBtn = document.getElementById('helpModalBtn');
  const docModal = document.getElementById('docModal');
  const closeDocModalBtn = document.getElementById('closeDocModalBtn');
  const closeDocModalBtn2 = document.getElementById('closeDocModalBtn2');

  // 2. Initialize Subsystems
  const audioManager = window.audioAlertManager;
  const classifier = new PostureClassifier();
  const uiManager = new UIManager(outputCanvas, webcamVideo);
  const poseEngine = new PoseNetEngine(webcamVideo, outputCanvas);

  // 3. Session Statistics State
  const sessionStats = {
    startTime: null,
    totalSeconds: 0,
    goodSeconds: 0,
    warningSeconds: 0,
    badSeconds: 0,
    currentStreak: 0,
    longestStreak: 0,
    deviations: 0,
    isInBadPosture: false,
    timerInterval: null
  };

  function startSessionTimer() {
    if (sessionStats.timerInterval) clearInterval(sessionStats.timerInterval);
    sessionStats.startTime = Date.now();
    
    sessionStats.timerInterval = setInterval(() => {
      if (!poseEngine.isRunning) return;
      sessionStats.totalSeconds++;

      // Check current posture for streak
      if (lastClassification && lastClassification.detected) {
        if (lastClassification.status === 'good') {
          sessionStats.goodSeconds++;
          sessionStats.currentStreak++;
          if (sessionStats.currentStreak > sessionStats.longestStreak) {
            sessionStats.longestStreak = sessionStats.currentStreak;
          }
          if (sessionStats.isInBadPosture) {
            sessionStats.isInBadPosture = false;
          }
        } else {
          sessionStats.currentStreak = 0;
          if (lastClassification.status === 'bad') {
            sessionStats.badSeconds++;
          } else {
            sessionStats.warningSeconds++;
          }
          if (!sessionStats.isInBadPosture) {
            sessionStats.isInBadPosture = true;
            sessionStats.deviations++;
          }
        }
      }
    }, 1000);
  }

  function getFormattedSessionData() {
    const mins = Math.floor(sessionStats.totalSeconds / 60).toString().padStart(2, '0');
    const secs = (sessionStats.totalSeconds % 60).toString().padStart(2, '0');
    const goodPercent = sessionStats.totalSeconds > 0 
      ? Math.round((sessionStats.goodSeconds / sessionStats.totalSeconds) * 100) 
      : 100;

    const longestStreakEl = document.getElementById('longestStreakVal');
    if (longestStreakEl) {
      longestStreakEl.textContent = `${sessionStats.longestStreak}s`;
    }

    return {
      formattedTime: `${mins}:${secs}`,
      goodPercent,
      currentStreak: sessionStats.currentStreak,
      longestStreak: sessionStats.longestStreak,
      deviations: sessionStats.deviations,
      totalSeconds: sessionStats.totalSeconds
    };
  }

  // 4. Calibration Management
  let isCalibrating = false;
  let calibrationSamples = [];

  function triggerCalibration() {
    if (!poseEngine.isRunning) {
      alert('Please start the camera or simulation first before calibrating.');
      return;
    }

    isCalibrating = true;
    calibrationSamples = [];
    let count = 3;
    uiManager.showCalibration(count);
    audioManager.speak('Sit upright and hold your ideal posture.');

    const interval = setInterval(() => {
      count--;
      if (count > 0) {
        uiManager.showCalibration(count);
      } else {
        clearInterval(interval);
        // Multi-frame average to finalize rock-solid baseline
        if (calibrationSamples.length > 0) {
          const success = classifier.calibrate(calibrationSamples);
          if (success) {
            audioManager.playTone('good');
            audioManager.speak('Baseline calibrated successfully!');
          }
        }
        uiManager.hideCalibration();
        isCalibrating = false;
      }
    }, 1000);
  }

  // 5. Connect PoseNet Frame & Render Callbacks
  let lastClassification = null;

  // 60 FPS Visual Frame Renderer (renders video & interpolated skeleton smoothly)
  poseEngine.onRenderFrame = (renderData) => {
    uiManager.drawPose(
      renderData.displayKeypoints,
      lastClassification,
      renderData.isSimulationMode,
      renderData.isFlipped,
      renderData.imageElement
    );
  };

  // AI Inference Result Callback (evaluates posture and updates metrics asynchronously)
  poseEngine.onPoseDetected = (poseData) => {
    // Collect calibration sample if active
    if (isCalibrating) {
      calibrationSamples.push(poseData.keypointsMap);
    }

    // Evaluate Posture
    const classification = classifier.evaluate(poseData.keypointsMap);
    lastClassification = classification;

    // Update Telemetry metrics & gauges
    const stats = getFormattedSessionData();
    uiManager.updateTelemetry(classification, stats, {
      isPhotoMode: poseData.isPhotoMode,
      photoName: poseData.photoName
    });

    // Audio & Voice alerts dispatch
    audioManager.processPostureAlert(classification);
  };

  poseEngine.onFpsUpdate = (fps) => {
    uiManager.updateFps(fps);
  };

  poseEngine.onStatusChange = (msg, state) => {
    if (state === 'error') {
      uiManager.setGlobalStatus(msg, 'bad');
    } else if (state === 'loading') {
      uiManager.setGlobalStatus(msg, 'standby');
    } else if (state === 'active' || state === 'simulation') {
      uiManager.setPlaceholderVisible(false);
      uiManager.setGlobalStatus('System Active & Monitoring', 'good');
    } else if (state === 'standby') {
      uiManager.setPlaceholderVisible(true);
      uiManager.setGlobalStatus('System Ready (Camera Off)', 'standby');
      toggleCamBtn.innerHTML = '<i class="fa-solid fa-video"></i> <span>Start Camera</span>';
      toggleCamBtn.className = 'btn btn-primary';
    }
  };

  // 6. Event Listeners for UI Controls
  async function handleStartCamera() {
    toggleCamBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>Starting...</span>';
    const success = await poseEngine.startCamera();
    if (success) {
      toggleCamBtn.innerHTML = '<i class="fa-solid fa-video-slash"></i> <span>Stop Camera</span>';
      toggleCamBtn.className = 'btn btn-danger';
      startSessionTimer();
    } else {
      toggleCamBtn.innerHTML = '<i class="fa-solid fa-video"></i> <span>Start Camera</span>';
      toggleCamBtn.className = 'btn btn-primary';
    }
  }

  function handleStopCamera() {
    poseEngine.stopCamera();
    toggleCamBtn.innerHTML = '<i class="fa-solid fa-video"></i> <span>Start Camera</span>';
    toggleCamBtn.className = 'btn btn-primary';
  }

  toggleCamBtn.addEventListener('click', () => {
    if (poseEngine.isRunning && !poseEngine.isPhotoMode) {
      handleStopCamera();
    } else {
      handleStartCamera();
    }
  });

  startCamBtn.addEventListener('click', handleStartCamera);

  loadSampleBtn.addEventListener('click', () => {
    setMode('demo');
    poseEngine.startSimulationMode();
    startSessionTimer();
  });

  calibrateBtn.addEventListener('click', triggerCalibration);

  // Photo Upload & Drag-and-Drop Image Analysis
  const uploadPhotoBtn = document.getElementById('uploadPhotoBtn');
  const placeholderUploadBtn = document.getElementById('placeholderUploadBtn');
  const photoFileInput = document.getElementById('photoFileInput');
  const viewportWrapper = document.getElementById('viewportWrapper');

  async function handlePhotoFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      alert('Please upload a valid image file (PNG, JPG, JPEG, WebP).');
      return;
    }

    const origUploadBtnHtml = uploadPhotoBtn ? uploadPhotoBtn.innerHTML : '';
    if (uploadPhotoBtn) {
      uploadPhotoBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>Analyzing...</span>';
    }

    try {
      const result = await poseEngine.analyzeImage(file, file.name);
      if (result.success) {
        audioManager.playTone('good');
        audioManager.speak('Photo posture analyzed successfully.');
        uiManager.setPlaceholderVisible(false);
        toggleCamBtn.innerHTML = '<i class="fa-solid fa-video"></i> <span>Live Camera</span>';
        toggleCamBtn.className = 'btn btn-primary';
      }
    } catch (err) {
      alert('Could not analyze photo: ' + err.message);
    } finally {
      if (uploadPhotoBtn) {
        uploadPhotoBtn.innerHTML = origUploadBtnHtml;
      }
      if (photoFileInput) {
        photoFileInput.value = '';
      }
    }
  }

  if (uploadPhotoBtn && photoFileInput) {
    uploadPhotoBtn.addEventListener('click', () => photoFileInput.click());
  }

  if (placeholderUploadBtn && photoFileInput) {
    placeholderUploadBtn.addEventListener('click', () => photoFileInput.click());
  }

  const loadSamplePhotoBtn = document.getElementById('loadSamplePhotoBtn');
  if (loadSamplePhotoBtn) {
    loadSamplePhotoBtn.addEventListener('click', async () => {
      try {
        const result = await poseEngine.analyzeImage('assets/samples/sample-posture.png', 'sample-posture.png');
        if (result.success) {
          audioManager.playTone('good');
          audioManager.speak('Sample posture photo analyzed.');
          uiManager.setPlaceholderVisible(false);
          toggleCamBtn.innerHTML = '<i class="fa-solid fa-video"></i> <span>Live Camera</span>';
          toggleCamBtn.className = 'btn btn-primary';
        }
      } catch (err) {
        alert('Could not load sample photo: ' + err.message);
      }
    });
  }

  if (photoFileInput) {
    photoFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handlePhotoFile(e.target.files[0]);
      }
    });
  }

  // Stage Drag & Drop Upload Handlers
  if (viewportWrapper) {
    ['dragenter', 'dragover'].forEach(evt => {
      viewportWrapper.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        viewportWrapper.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(evt => {
      viewportWrapper.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        viewportWrapper.classList.remove('dragover');
      });
    });

    viewportWrapper.addEventListener('drop', (e) => {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
        handlePhotoFile(e.dataTransfer.files[0]);
      }
    });
  }

  // Initialize Mirror Button state
  if (poseEngine.isFlipped) {
    flipCamBtn.className = 'btn btn-primary';
    flipCamBtn.innerHTML = '<i class="fa-solid fa-repeat"></i> <span>Mirror: ON</span>';
  }

  flipCamBtn.addEventListener('click', () => {
    const isFlipped = poseEngine.toggleFlip();
    if (isFlipped) {
      flipCamBtn.className = 'btn btn-primary';
      flipCamBtn.innerHTML = '<i class="fa-solid fa-repeat"></i> <span>Mirror: ON</span>';
      flipCamBtn.title = 'Mirror view active (Click for normal view)';
    } else {
      flipCamBtn.className = 'btn btn-secondary';
      flipCamBtn.innerHTML = '<i class="fa-solid fa-repeat"></i> <span>Mirror: OFF</span>';
      flipCamBtn.title = 'Normal view active (Click for mirror view)';
    }
  });

  // Toggles
  voiceAlertToggle.addEventListener('change', (e) => {
    audioManager.setVoiceEnabled(e.target.checked);
  });

  soundToneToggle.addEventListener('change', (e) => {
    audioManager.setSoundEnabled(e.target.checked);
  });

  guideToggle.addEventListener('change', (e) => {
    uiManager.setGuideVisible(e.target.checked);
  });

  // Settings
  sensitivitySelect.addEventListener('change', (e) => {
    classifier.setSensitivity(e.target.value);
  });

  modelMultiplier.addEventListener('change', async (e) => {
    const wasRunning = poseEngine.isRunning;
    const wasSim = poseEngine.isSimulationMode;
    if (wasRunning) {
      poseEngine.stopCamera();
    }
    await poseEngine.loadModel(e.target.value, poseEngine.inputResolution);
    if (wasRunning && !wasSim) {
      handleStartCamera();
    } else if (wasSim) {
      poseEngine.startSimulationMode();
    }
  });

  if (inputResolutionSelect) {
    inputResolutionSelect.addEventListener('change', async (e) => {
      const wasRunning = poseEngine.isRunning;
      const wasSim = poseEngine.isSimulationMode;
      if (wasRunning) {
        poseEngine.stopCamera();
      }
      await poseEngine.loadModel(poseEngine.multiplier, e.target.value);
      if (wasRunning && !wasSim) {
        handleStartCamera();
      } else if (wasSim) {
        poseEngine.startSimulationMode();
      }
    });
  }

  alertDelay.addEventListener('change', (e) => {
    audioManager.setCooldown(parseInt(e.target.value, 10));
  });

  // Modes
  function setMode(mode) {
    modeSittingBtn.classList.remove('active');
    modeStandingBtn.classList.remove('active');
    modeDemoBtn.classList.remove('active');

    if (mode === 'sitting') {
      modeSittingBtn.classList.add('active');
      classifier.setMode('sitting');
      document.getElementById('hudModeTag').innerHTML = '<i class="fa-solid fa-chair"></i> Mode: Sitting Ergonomics';
    } else if (mode === 'standing') {
      modeStandingBtn.classList.add('active');
      classifier.setMode('standing');
      document.getElementById('hudModeTag').innerHTML = '<i class="fa-solid fa-person"></i> Mode: Standing Form';
    } else if (mode === 'demo') {
      modeDemoBtn.classList.add('active');
      document.getElementById('hudModeTag').innerHTML = '<i class="fa-solid fa-circle-play"></i> Mode: Simulated Test';
    }

    // If currently inspecting a photo, re-evaluate posture for the selected ergonomics mode immediately
    if (poseEngine.isPhotoMode && poseEngine.smoothedKeypoints && Object.keys(poseEngine.smoothedKeypoints).length > 0) {
      const classification = classifier.evaluate(poseEngine.smoothedKeypoints);
      lastClassification = classification;
      const stats = getFormattedSessionData();
      uiManager.updateTelemetry(classification, stats, {
        isPhotoMode: true,
        photoName: poseEngine.currentPhotoName
      });
    }
  }

  modeSittingBtn.addEventListener('click', () => {
    setMode('sitting');
    if (poseEngine.isSimulationMode) {
      handleStartCamera();
    }
  });

  modeStandingBtn.addEventListener('click', () => {
    setMode('standing');
    if (poseEngine.isSimulationMode) {
      handleStartCamera();
    }
  });

  modeDemoBtn.addEventListener('click', () => {
    setMode('demo');
    poseEngine.startSimulationMode();
    startSessionTimer();
  });

  // Comprehensive Session Reset Functionality
  function handleResetStats() {
    sessionStats.totalSeconds = 0;
    sessionStats.goodSeconds = 0;
    sessionStats.warningSeconds = 0;
    sessionStats.badSeconds = 0;
    sessionStats.currentStreak = 0;
    sessionStats.longestStreak = 0;
    sessionStats.deviations = 0;
    sessionStats.startTime = Date.now();

    if (uiManager.goodPosturePercent) uiManager.goodPosturePercent.textContent = '100%';
    if (uiManager.currentStreak) uiManager.currentStreak.textContent = '0s';
    if (uiManager.deviationCount) uiManager.deviationCount.textContent = '0';
    if (uiManager.sessionTimeBadge) uiManager.sessionTimeBadge.textContent = '00:00';
    const longestStreakEl = document.getElementById('longestStreakVal');
    if (longestStreakEl) longestStreakEl.textContent = '0s';

    // Reset chart history data
    if (uiManager.chart) {
      uiManager.chart.data.datasets[0].data = Array(20).fill(98);
      uiManager.chart.data.datasets[0].borderColor = '#38bdf8';
      uiManager.chart.update();
    }

    audioManager.playTone('good');

    // Visual button feedback animation
    const resetBtns = [
      document.getElementById('resetStatsBtnTop'),
      document.getElementById('resetStatsBtnCard'),
      document.getElementById('resetStatsBtn')
    ].filter(Boolean);

    resetBtns.forEach(btn => {
      const origHtml = btn.innerHTML;
      btn.innerHTML = '<i class="fa-solid fa-check"></i> <span>Reset!</span>';
      setTimeout(() => { btn.innerHTML = origHtml; }, 1200);
    });
  }

  // Wire all Reset buttons
  const resetStatsBtnTop = document.getElementById('resetStatsBtnTop');
  const resetStatsBtnCard = document.getElementById('resetStatsBtnCard');
  const calibrateBtnCard = document.getElementById('calibrateBtnCard');

  if (resetStatsBtnTop) resetStatsBtnTop.addEventListener('click', handleResetStats);
  if (resetStatsBtnCard) resetStatsBtnCard.addEventListener('click', handleResetStats);
  if (resetStatsBtn) resetStatsBtn.addEventListener('click', handleResetStats);
  if (calibrateBtnCard) calibrateBtnCard.addEventListener('click', triggerCalibration);

  // Export Session Report
  exportReportBtn.addEventListener('click', () => {
    const data = getFormattedSessionData();
    const report = {
      project: "Posture Detection System Using PoseNet",
      author: "Aryan Raj Pandey (B.Tech CSE - AI & ML)",
      timestamp: new Date().toISOString(),
      sessionSummary: {
        totalDuration: data.formattedTime,
        totalSeconds: data.totalSeconds,
        goodPosturePercentage: `${data.goodPercent}%`,
        deviationsCount: data.deviations,
        longestGoodStreakSeconds: data.longestStreak
      },
      currentAngles: lastClassification ? lastClassification.angles : {},
      sensitivity: classifier.sensitivity,
      mode: classifier.mode
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `posture-report-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  // Modal Dialog
  if (helpModalBtn) {
    helpModalBtn.addEventListener('click', () => {
      docModal.classList.add('active');
    });
  }

  const closeModal = () => docModal.classList.remove('active');
  closeDocModalBtn.addEventListener('click', closeModal);
  closeDocModalBtn2.addEventListener('click', closeModal);
  docModal.addEventListener('click', (e) => {
    if (e.target === docModal) closeModal();
  });

  // Quick Alert Control Buttons in Dark Action Card
  const voiceToggleBtn = document.getElementById('voiceToggleBtn');
  const soundToggleBtn = document.getElementById('soundToggleBtn');
  const guideToggleBtn = document.getElementById('guideToggleBtn');

  if (voiceToggleBtn) {
    voiceToggleBtn.addEventListener('click', () => {
      const isEnabled = !audioManager.voiceEnabled;
      audioManager.setVoiceEnabled(isEnabled);
      voiceAlertToggle.checked = isEnabled;
      voiceToggleBtn.classList.toggle('active', isEnabled);
    });
  }

  if (soundToggleBtn) {
    soundToggleBtn.addEventListener('click', () => {
      const isEnabled = !audioManager.soundEnabled;
      audioManager.setSoundEnabled(isEnabled);
      soundToneToggle.checked = isEnabled;
      soundToggleBtn.classList.toggle('active', isEnabled);
    });
  }

  if (guideToggleBtn) {
    guideToggleBtn.addEventListener('click', () => {
      const isVisible = silhouetteGuide && silhouetteGuide.style.display !== 'none';
      uiManager.setGuideVisible(!isVisible);
      guideToggle.checked = !isVisible;
      guideToggleBtn.classList.toggle('active', !isVisible);
    });
  }

  // Sidebar navigation actions
  const sidebarCamBtn = document.getElementById('sidebarCamBtn');
  if (sidebarCamBtn) {
    sidebarCamBtn.addEventListener('click', () => {
      if (poseEngine.isRunning) {
        handleStopCamera();
      } else {
        handleStartCamera();
      }
    });
  }

  const sidebarStatsBtn = document.getElementById('sidebarStatsBtn');
  if (sidebarStatsBtn) {
    sidebarStatsBtn.addEventListener('click', () => {
      document.querySelector('.table-bento-card')?.scrollIntoView({ behavior: 'smooth' });
    });
  }

  const sidebarUserBtn = document.getElementById('sidebarUserBtn');
  if (sidebarUserBtn) {
    sidebarUserBtn.addEventListener('click', () => {
      docModal.classList.add('active');
    });
  }

  const sidebarModelBtn = document.getElementById('sidebarModelBtn');
  if (sidebarModelBtn) {
    sidebarModelBtn.addEventListener('click', () => {
      docModal.classList.add('active');
    });
  }

  const sidebarExitBtn = document.getElementById('sidebarExitBtn');
  if (sidebarExitBtn) {
    sidebarExitBtn.addEventListener('click', () => {
      if (poseEngine.isRunning) {
        handleStopCamera();
      }
    });
  }

  // Timeline pills interaction
  document.querySelectorAll('.time-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.time-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  // Collapsible Tuning Drawer Toggle
  const tuningHeaderToggle = document.getElementById('tuningHeaderToggle');
  const tuningBody = document.getElementById('tuningBody');
  const tuningChevron = document.getElementById('tuningChevron');
  if (tuningHeaderToggle && tuningBody) {
    tuningHeaderToggle.addEventListener('click', () => {
      const isCollapsed = tuningBody.classList.toggle('collapsed');
      if (tuningChevron) {
        tuningChevron.style.transform = isCollapsed ? 'rotate(0deg)' : 'rotate(180deg)';
      }
    });
  }

  // ---------------------------------------------------------
  // Theme Management (Dark & Light Theme)
  // ---------------------------------------------------------
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const sidebarThemeBtn = document.getElementById('sidebarThemeBtn');
  const themeBtnText = document.getElementById('themeBtnText');

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('postureguard_theme', theme);
    } catch (e) {}

    if (themeBtnText) {
      themeBtnText.textContent = theme === 'dark' ? 'Light Mode' : 'Dark Mode';
    }
    if (themeToggleBtn) {
      themeToggleBtn.setAttribute('title', theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode');
    }
    if (sidebarThemeBtn) {
      sidebarThemeBtn.setAttribute('title', theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode');
    }
  }

  function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(nextTheme);
  }

  // Load saved theme or fall back to system preference (or default light)
  let initialTheme = 'light';
  try {
    const savedTheme = localStorage.getItem('postureguard_theme');
    if (savedTheme === 'dark' || savedTheme === 'light') {
      initialTheme = savedTheme;
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      initialTheme = 'dark';
    }
  } catch (e) {}

  applyTheme(initialTheme);

  if (themeToggleBtn) themeToggleBtn.addEventListener('click', toggleTheme);
  if (sidebarThemeBtn) sidebarThemeBtn.addEventListener('click', toggleTheme);

  // Pre-load PoseNet in background so it's instantly ready when user clicks Start Camera
  poseEngine.loadModel(0.75, 257).catch(err => console.log('Warmup info:', err));
});
