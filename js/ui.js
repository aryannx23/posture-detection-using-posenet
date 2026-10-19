/**
 * UI Renderer, Canvas Skeleton Visualizer, and Telemetry Engine
 * Posture Detection System Using PoseNet
 * Author: Aryan Raj Pandey (B.Tech CSE - AI & ML)
 */

class UIManager {
  constructor(canvasElement, videoElement) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.video = videoElement;

    // DOM Elements
    this.globalStatusPill = document.getElementById('globalStatusPill');
    this.globalStatusText = document.getElementById('globalStatusText');
    this.fpsDisplay = document.getElementById('fpsDisplay');
    this.hudScore = document.getElementById('hudScore');
    this.hudWarningList = document.getElementById('hudWarningList');
    this.hudConfidenceText = document.getElementById('hudConfidenceText');
    this.hudModeTag = document.getElementById('hudModeTag');
    this.mainScoreNum = document.getElementById('mainScoreNum');
    this.scoreRingFill = document.getElementById('scoreRingFill');
    this.postureStateBadge = document.getElementById('postureStateBadge');
    this.goodPosturePercent = document.getElementById('goodPosturePercent');
    this.currentStreak = document.getElementById('currentStreak');
    this.deviationCount = document.getElementById('deviationCount');

    // Angle boxes
    this.shoulderAngleVal = document.getElementById('shoulderAngleVal');
    this.shoulderCardSub = document.getElementById('shoulderCardSub');
    this.shoulderEval = document.getElementById('shoulderEval');
    this.shoulderBarFill = document.getElementById('shoulderBarFill');

    this.headAngleVal = document.getElementById('headAngleVal');
    this.headCardSub = document.getElementById('headCardSub');
    this.headEval = document.getElementById('headEval');
    this.headBarFill = document.getElementById('headBarFill');

    this.spineAngleVal = document.getElementById('spineAngleVal');
    this.spineCardSub = document.getElementById('spineCardSub');
    this.spineEval = document.getElementById('spineEval');
    this.spineBarFill = document.getElementById('spineBarFill');

    this.slouchRatioVal = document.getElementById('slouchRatioVal');
    this.slouchEval = document.getElementById('slouchEval');
    this.slouchBarFill = document.getElementById('slouchBarFill');

    // Table elements
    this.tableShoulderVal = document.getElementById('tableShoulderVal');
    this.tableShoulderPill = document.getElementById('tableShoulderPill');
    this.tableHeadVal = document.getElementById('tableHeadVal');
    this.tableHeadPill = document.getElementById('tableHeadPill');
    this.tableSpineVal = document.getElementById('tableSpineVal');
    this.tableSpinePill = document.getElementById('tableSpinePill');

    this.adviceContent = document.getElementById('adviceContent');
    this.cameraPlaceholder = document.getElementById('cameraPlaceholder');
    this.calibrationOverlay = document.getElementById('calibrationOverlay');
    this.calibCountdown = document.getElementById('calibCountdown');
    this.silhouetteGuide = document.getElementById('silhouetteGuide');
    this.sessionTimeBadge = document.getElementById('sessionTimeBadge');

    // Chart.js instance
    this.chart = null;
    this.initChart();

    // Standard PoseNet Skeleton Connection Map
    this.skeletonConnections = [
      ['leftEar', 'leftEye'],
      ['leftEye', 'nose'],
      ['nose', 'rightEye'],
      ['rightEye', 'rightEar'],
      ['leftShoulder', 'rightShoulder'],
      ['leftShoulder', 'leftElbow'],
      ['leftElbow', 'leftWrist'],
      ['rightShoulder', 'rightElbow'],
      ['rightElbow', 'rightWrist'],
      ['leftShoulder', 'leftHip'],
      ['rightShoulder', 'rightHip'],
      ['leftHip', 'rightHip'],
      ['leftHip', 'leftKnee'],
      ['leftKnee', 'leftAnkle'],
      ['rightHip', 'rightKnee'],
      ['rightKnee', 'rightAnkle']
    ];
  }

  initChart() {
    const chartCtx = document.getElementById('scoreChart');
    if (!chartCtx || typeof Chart === 'undefined') return;

    const initialLabels = Array(20).fill('');
    const initialData = Array(20).fill(95);

    this.chart = new Chart(chartCtx, {
      type: 'line',
      data: {
        labels: initialLabels,
        datasets: [{
          label: 'Posture Score',
          data: initialData,
          borderColor: '#38bdf8',
          borderWidth: 2.2,
          tension: 0.42,
          pointRadius: 0,
          fill: true,
          backgroundColor: (context) => {
            const chart = context.chart;
            const { ctx, chartArea } = chart;
            if (!chartArea) return null;
            const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
            gradient.addColorStop(0, 'rgba(56, 189, 248, 0.28)');
            gradient.addColorStop(1, 'rgba(56, 189, 248, 0.0)');
            return gradient;
          }
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
          x: { display: false },
          y: {
            display: false,
            min: 0,
            max: 100
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false }
        }
      }
    });
  }

  updateChart(score, status) {
    if (!this.chart) return;
    const dataset = this.chart.data.datasets[0];
    dataset.data.shift();
    dataset.data.push(score);

    // Color gradient shift based on score
    if (status === 'bad') {
      dataset.borderColor = '#ef4444';
    } else if (status === 'warning') {
      dataset.borderColor = '#f59e0b';
    } else {
      dataset.borderColor = '#10b981';
    }

    this.chart.update();
  }

  /**
   * Render video frame and skeleton visualization with smooth sub-frame interpolation
   * Supports native horizontal mirror flipping without inverting text or HUD badges
   * Also supports static photo analysis rendering
   */
  drawPose(keypointsOrPoseData, classification, isSimulation = false, isFlipped = false, imageElement = null) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // 1. Draw uploaded image if in photo mode, or live webcam feed if active and not simulated
    if (imageElement) {
      ctx.drawImage(imageElement, 0, 0, w, h);
    } else if (!isSimulation && this.video.readyState >= 2) {
      if (isFlipped) {
        ctx.save();
        ctx.translate(w, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(this.video, 0, 0, w, h);
        ctx.restore();
      } else {
        ctx.drawImage(this.video, 0, 0, w, h);
      }
    }

    if (!keypointsOrPoseData || !classification || !classification.detected) {
      return;
    }

    // Extract keypoints map
    const keypoints = keypointsOrPoseData.keypointsMap || keypointsOrPoseData;
    if (!keypoints) return;

    // Coordinate mapping helper: cleanly maps X to mirrored canvas if active
    const getX = (kp) => (isFlipped ? (w - kp.x) : kp.x);
    const getY = (kp) => kp.y;

    const status = classification.status;

    // Palette based on posture health
    let themeColor = '#10b981'; // Good (Emerald)
    let glowColor = 'rgba(16, 185, 129, 0.6)';
    if (status === 'warning') {
      themeColor = '#f59e0b'; // Amber
      glowColor = 'rgba(245, 158, 11, 0.7)';
    } else if (status === 'bad') {
      themeColor = '#ef4444'; // Red
      glowColor = 'rgba(239, 68, 68, 0.8)';
    }

    // 2. Draw Skeleton Lines
    ctx.lineWidth = 3.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (const [p1Name, p2Name] of this.skeletonConnections) {
      const p1 = keypoints[p1Name];
      const p2 = keypoints[p2Name];

      if (p1 && p2 && p1.score > 0.28 && p2.score > 0.28) {
        ctx.beginPath();
        ctx.moveTo(getX(p1), getY(p1));
        ctx.lineTo(getX(p2), getY(p2));
        ctx.strokeStyle = themeColor;
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = 10;
        ctx.stroke();
        ctx.shadowBlur = 0; // reset
      }
    }

    // 3. Draw Shoulder Alignment Horizon Guide
    const lSh = keypoints.leftShoulder;
    const rSh = keypoints.rightShoulder;
    if (lSh && rSh && lSh.score > 0.28 && rSh.score > 0.28) {
      const lX = getX(lSh);
      const lY = getY(lSh);
      const rX = getX(rSh);
      const rY = getY(rSh);

      ctx.beginPath();
      ctx.moveTo(lX, lY);
      ctx.lineTo(rX, rY);
      ctx.lineWidth = 4;
      ctx.strokeStyle = classification.states.shoulder === 'optimal' ? '#10b981' : '#ef4444';
      ctx.stroke();

      // Horizontal reference dashed line
      const midY = (lY + rY) / 2;
      const minX = Math.min(lX, rX) - 30;
      const maxX = Math.max(lX, rX) + 30;
      ctx.beginPath();
      ctx.setLineDash([5, 5]);
      ctx.moveTo(minX, midY);
      ctx.lineTo(maxX, midY);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.setLineDash([]); // clear dash

      // Angle label bubble on canvas (rendered upright in un-flipped space)
      const angle = classification.angles.shoulder;
      const midX = (lX + rX) / 2;
      const shBadge = classification.diagnostics && classification.diagnostics.shoulder ? classification.diagnostics.shoulder.badgeText : (classification.states.shoulder === 'optimal' ? 'Balanced' : 'Uneven');
      this.drawBadge(ctx, `${shBadge} (${angle.toFixed(1)}°)`, midX, midY - 14, classification.states.shoulder);
    }

    // 4. Draw Spine Vector & Head-to-Shoulder line
    const nose = keypoints.nose;
    if (nose && lSh && rSh && nose.score > 0.28) {
      const lX = getX(lSh);
      const lY = getY(lSh);
      const rX = getX(rSh);
      const rY = getY(rSh);
      const nX = getX(nose);
      const nY = getY(nose);
      const midShoulder = { x: (lX + rX) / 2, y: (lY + rY) / 2 };

      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.moveTo(nX, nY);
      ctx.lineTo(midShoulder.x, midShoulder.y);
      ctx.strokeStyle = classification.states.slouch === 'optimal' ? 'rgba(56, 189, 248, 0.8)' : '#ef4444';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 5. Draw Keypoint Glowing Nodes
    for (const name in keypoints) {
      const kp = keypoints[name];
      if (kp.score > 0.28) {
        const kX = getX(kp);
        const kY = getY(kp);

        // Outer glow
        ctx.beginPath();
        ctx.arc(kX, kY, 6.5, 0, 2 * Math.PI);
        ctx.fillStyle = themeColor;
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = 12;
        ctx.fill();
        ctx.shadowBlur = 0;

        // Inner white nucleus
        ctx.beginPath();
        ctx.arc(kX, kY, 2.8, 0, 2 * Math.PI);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
      }
    }
  }

  drawBadge(ctx, text, x, y, state) {
    ctx.save();
    ctx.font = '600 11px "JetBrains Mono", monospace';
    const textWidth = ctx.measureText(text).width;
    const padding = 6;
    const bg = state === 'optimal' ? 'rgba(16, 185, 129, 0.90)' : (state === 'warning' ? 'rgba(245, 158, 11, 0.90)' : 'rgba(239, 68, 68, 0.90)');
    
    ctx.fillStyle = bg;
    ctx.roundRect(x - textWidth / 2 - padding, y - 10, textWidth + padding * 2, 20, 5);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x, y);
    ctx.restore();
  }

  /**
   * Update all DOM metrics, gauges, and cards
   */
  updateTelemetry(classification, sessionStats, photoInfo = null) {
    if (!classification.detected) {
      this.hudConfidenceText.textContent = 'Searching for Person...';
      this.hudScore.textContent = '--';
      this.mainScoreNum.textContent = '--';
      return;
    }

    if (photoInfo && photoInfo.isPhotoMode) {
      this.hudConfidenceText.textContent = 'Photo Analyzed (17 Keypoints Tracked)';
      if (this.hudModeTag) {
        this.hudModeTag.innerHTML = `<i class="fa-solid fa-image"></i> Photo: ${photoInfo.photoName || 'Analyzed'}`;
      }
    } else {
      this.hudConfidenceText.textContent = 'Person Tracked (Confidence: High)';
    }

    const score = classification.score;

    // HUD Score & Main Ring
    this.hudScore.textContent = score;
    this.mainScoreNum.textContent = score;

    // Update SVG Ring (dashoffset 0 to 314)
    // 100 score -> 0 offset, 0 score -> 314 offset
    const offset = 314 - (314 * (score / 100));
    this.scoreRingFill.style.strokeDashoffset = offset;

    if (score >= 85) {
      this.scoreRingFill.style.stroke = 'var(--status-good)';
      this.mainScoreNum.style.color = 'var(--status-good)';
    } else if (score >= 60) {
      this.scoreRingFill.style.stroke = 'var(--status-warning)';
      this.mainScoreNum.style.color = 'var(--status-warning)';
    } else {
      this.scoreRingFill.style.stroke = 'var(--status-bad)';
      this.mainScoreNum.style.color = 'var(--status-bad)';
    }

    // Posture State Badge
    if (classification.status === 'good') {
      this.postureStateBadge.textContent = 'Optimal Alignment';
      this.postureStateBadge.className = 'stat-value text-success';
      this.setGlobalStatus('Optimal Posture (Monitoring)', 'good');
    } else if (classification.status === 'warning') {
      this.postureStateBadge.textContent = 'Minor Deviation';
      this.postureStateBadge.className = 'stat-value text-warning';
      this.setGlobalStatus('Warning: Posture Drift Detected', 'warning');
    } else {
      this.postureStateBadge.textContent = 'Poor Posture Alert';
      this.postureStateBadge.className = 'stat-value text-danger';
      this.setGlobalStatus('Alert: Please Adjust Posture', 'bad');
    }

    const d = classification.diagnostics || {};

    // 1. Shoulder Box
    const shAngle = classification.angles.shoulder;
    this.shoulderAngleVal.textContent = `${shAngle.toFixed(1)}°`;
    if (this.shoulderCardSub && d.shoulder) {
      this.shoulderCardSub.textContent = d.shoulder.summary;
    }
    const shBadge = d.shoulder ? d.shoulder.badgeText : (classification.states.shoulder === 'optimal' ? 'Balanced' : 'Uneven');
    this.applyMetricState(this.shoulderEval, this.shoulderBarFill, classification.states.shoulder, Math.min(100, Math.max(10, 100 - (shAngle / 12) * 100)), shBadge);

    // 2. Head Box
    const hdAngle = classification.angles.head;
    this.headAngleVal.textContent = `${hdAngle.toFixed(1)}°`;
    if (this.headCardSub && d.head) {
      this.headCardSub.textContent = d.head.summary;
    }
    const hdBadge = d.head ? d.head.badgeText : (classification.states.head === 'optimal' ? 'Upright' : 'Tilted');
    this.applyMetricState(this.headEval, this.headBarFill, classification.states.head, Math.min(100, Math.max(10, 100 - (hdAngle / 15) * 100)), hdBadge);

    // 3. Spine Box
    const spAngle = classification.angles.spine;
    this.spineAngleVal.textContent = `${spAngle.toFixed(1)}°`;
    if (this.spineCardSub && d.spine) {
      this.spineCardSub.textContent = d.spine.summary;
    }
    const spBadge = d.spine ? d.spine.badgeText : (classification.states.spine === 'optimal' ? 'Straight' : 'Leaning');
    const spineDev = Math.abs(90 - spAngle);
    this.applyMetricState(this.spineEval, this.spineBarFill, classification.states.spine, Math.min(100, Math.max(10, 100 - (spineDev / 16) * 100)), spBadge);

    // 4. Slouch Box
    const slPercentage = classification.angles.slouchPercentage || Math.round(classification.angles.slouchRatio * 100);
    if (this.slouchRatioVal) {
      this.slouchRatioVal.textContent = `${slPercentage}%`;
    }
    const slBadge = d.slouch ? d.slouch.badgeText : (classification.states.slouch === 'optimal' ? 'Tall Form' : 'Slouching');
    this.applyMetricState(this.slouchEval, this.slouchBarFill, classification.states.slouch, slPercentage, slBadge);

    // Update bottom table metrics with human-friendly descriptions
    if (this.tableShoulderVal) {
      this.tableShoulderVal.textContent = `${shAngle.toFixed(1)}°`;
      this.updateTablePill(this.tableShoulderPill, classification.states.shoulder, shBadge);
    }
    if (this.tableHeadVal) {
      this.tableHeadVal.textContent = `${hdAngle.toFixed(1)}°`;
      this.updateTablePill(this.tableHeadPill, classification.states.head, hdBadge);
    }
    if (this.tableSpineVal) {
      this.tableSpineVal.textContent = `${spAngle.toFixed(1)}°`;
      this.updateTablePill(this.tableSpinePill, classification.states.spine, spBadge);
    }

    // HUD In-Canvas Warnings
    this.renderHudWarnings(classification.issues);

    // Ergonomic Advice Banner
    this.renderErgonomicAdvice(classification);

    // Session Statistics
    if (sessionStats) {
      if (this.goodPosturePercent) this.goodPosturePercent.textContent = `${sessionStats.goodPercent}%`;
      if (this.currentStreak) this.currentStreak.textContent = `${sessionStats.currentStreak}s`;
      if (this.deviationCount) this.deviationCount.textContent = sessionStats.deviations;
      if (this.sessionTimeBadge) this.sessionTimeBadge.textContent = sessionStats.formattedTime;
      const longestEl = document.getElementById('longestStreakVal');
      if (longestEl) longestEl.textContent = `${sessionStats.longestStreak}s`;
    }

    // Chart update
    this.updateChart(score, classification.status);
  }

  updateTablePill(pillEl, state, badgeText) {
    if (!pillEl) return;
    if (state === 'optimal') {
      pillEl.className = 'joint-change-pill optimal';
      pillEl.textContent = badgeText || 'Balanced';
    } else if (state === 'warning') {
      pillEl.className = 'joint-change-pill warning';
      pillEl.textContent = badgeText || 'Deviation';
    } else {
      pillEl.className = 'joint-change-pill bad';
      pillEl.textContent = badgeText || 'Poor Posture';
    }
  }

  applyMetricState(evalEl, barEl, state, percentage, customText = null) {
    if (evalEl) {
      if (state === 'optimal') {
        evalEl.textContent = customText || 'Optimal';
        evalEl.className = 'pastel-gain-pill';
      } else if (state === 'warning') {
        evalEl.textContent = customText || 'Warning';
        evalEl.className = 'pastel-gain-pill warning-pill';
      } else {
        evalEl.textContent = customText || 'Poor';
        evalEl.className = 'pastel-gain-pill bad-pill';
      }
    }
    if (barEl) {
      barEl.style.width = `${Math.max(10, Math.min(100, percentage))}%`;
      if (state === 'optimal') {
        barEl.style.background = '#10b981';
      } else if (state === 'warning') {
        barEl.style.background = '#f59e0b';
      } else {
        barEl.style.background = '#ef4444';
      }
    }
  }

  renderHudWarnings(issues) {
    if (!issues || issues.length === 0) {
      this.hudWarningList.innerHTML = '';
      return;
    }

    this.hudWarningList.innerHTML = issues.slice(0, 2).map(issue => `
      <div class="hud-warning-item ${issue.severity === 'warning' ? 'warning-caution' : ''}">
        <i class="fa-solid fa-triangle-exclamation"></i>
        <span>${issue.title}: ${issue.description}</span>
      </div>
    `).join('');
  }

  renderErgonomicAdvice(classification) {
    if (!this.adviceContent) return;
    if (classification.advice) {
      const adv = classification.advice;
      this.adviceContent.className = 'advice-banner';
      this.adviceContent.innerHTML = `
        <i class="fa-solid ${adv.icon}"></i>
        <span><strong>${adv.primaryTitle}:</strong> ${adv.primaryText} <span class="advice-action-hint">→ <em>${adv.actionStep}</em></span></span>
      `;
      return;
    }

    if (!classification.issues || classification.issues.length === 0) {
      this.adviceContent.className = 'advice-banner';
      this.adviceContent.innerHTML = `
        <i class="fa-solid fa-circle-check text-success"></i>
        <span><strong>Optimal Posture:</strong> Shoulders balanced and head upright. Maintain relaxed breathing.</span>
      `;
      return;
    }

    const primaryIssue = classification.issues[0];
    const icon = primaryIssue.severity === 'bad' ? 'fa-circle-xmark text-danger' : 'fa-triangle-exclamation text-amber';
    this.adviceContent.className = 'advice-banner';
    this.adviceContent.innerHTML = `
      <i class="fa-solid ${icon}"></i>
      <span><strong>${primaryIssue.title}:</strong> ${primaryIssue.description}</span>
    `;
  }

  setGlobalStatus(text, state) {
    this.globalStatusText.textContent = text;
    this.globalStatusPill.className = `status-pill status-${state}-state`;
  }

  updateFps(fpsInfo) {
    if (typeof fpsInfo === 'object' && fpsInfo !== null) {
      if (fpsInfo.latencyMs > 0) {
        this.fpsDisplay.textContent = `${fpsInfo.renderFps} FPS · ${fpsInfo.latencyMs}ms AI`;
      } else {
        this.fpsDisplay.textContent = `${fpsInfo.renderFps} FPS`;
      }
    } else {
      this.fpsDisplay.textContent = `${fpsInfo} FPS`;
    }
  }

  showCalibration(secondsLeft) {
    this.calibrationOverlay.classList.add('active');
    this.calibCountdown.textContent = secondsLeft;
  }

  hideCalibration() {
    this.calibrationOverlay.classList.remove('active');
  }

  setGuideVisible(visible) {
    this.silhouetteGuide.style.display = visible ? 'block' : 'none';
  }

  setPlaceholderVisible(visible) {
    if (visible) {
      this.cameraPlaceholder.classList.remove('hidden');
    } else {
      this.cameraPlaceholder.classList.add('hidden');
    }
  }
}

// Attach to window
window.UIManager = UIManager;
