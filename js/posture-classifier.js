/**
 * Posture Classification Engine
 * Posture Detection System Using PoseNet
 * Author: Aryan Raj Pandey (B.Tech CSE - AI & ML)
 *
 * Implements:
 * 1. Biomechanically precise angle calculations:
 *    - Shoulder horizon tilt with dropped-side detection
 *    - Craniovertebral angle & forward head ("tech-neck") detection
 *    - Coronal thoracic spine verticality (with desk-occlusion fallback)
 *    - Perspective-invariant slouch & sitting height extension
 * 2. Keypoint temporal smoothing (EMA) to eliminate frame-to-frame jitter
 * 3. Human-centered plain-English diagnostics and real-time coaching advice
 * 4. Multi-frame calibration baseline averaging
 */

class PostureClassifier {
  constructor() {
    this.mode = 'sitting'; // 'sitting' | 'standing'
    this.sensitivity = 'normal'; // 'relaxed' | 'normal' | 'strict'
    
    // Calibrated baseline values (locked upon calibration to avoid baseline drift)
    this.baseline = {
      isCalibrated: false,
      cranialRatio: 0.50,       // Anthropometric standard: cranial height is ~50% of shoulder span
      shoulderWidth: null,      // Distance between left and right shoulders
      spineAngle: 90.0,         // Nominal vertical plumb line
      shoulderAngle: 0.0,       // Nominal horizontal angle
      headAngle: 0.0,
      timestamp: null
    };

    // Preset thresholds by sensitivity level
    this.thresholds = {
      relaxed: {
        shoulderTilt: 8.5,     // degrees deviation from horizontal
        headTilt: 11.0,        // degrees lateral head tilt
        spineDev: 12.0,        // degrees deviation from upright
        slouchRatio: 0.78      // sitting height compression ratio
      },
      normal: {
        shoulderTilt: 5.5,
        headTilt: 8.0,
        spineDev: 8.5,
        slouchRatio: 0.84
      },
      strict: {
        shoulderTilt: 3.8,
        headTilt: 5.5,
        spineDev: 6.0,
        slouchRatio: 0.89
      }
    };

    // Smoothing filters for rock-solid stability without perceptual lag
    this.smoothedScore = 100;
    this.smoothedKeypoints = {};
  }

  setMode(mode) {
    this.mode = mode;
  }

  setSensitivity(level) {
    if (this.thresholds[level]) {
      this.sensitivity = level;
    }
  }

  /**
   * Standard Angle Calculation: atan2(dy, dx) in degrees
   */
  static calculateAngle(p1, p2) {
    if (!p1 || !p2) return 0;
    const dy = p2.y - p1.y;
    const dx = p2.x - p1.x;
    return Math.atan2(dy, dx) * (180 / Math.PI);
  }

  /**
   * Euclidean distance between two 2D points
   */
  static distance(p1, p2) {
    if (!p1 || !p2) return 0;
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  /**
   * Midpoint between two keypoints
   */
  static midpoint(p1, p2) {
    if (!p1 && !p2) return { x: 0, y: 0 };
    if (!p1) return { x: p2.x, y: p2.y };
    if (!p2) return { x: p1.x, y: p1.y };
    return {
      x: (p1.x + p2.x) / 2,
      y: (p1.y + p2.y) / 2
    };
  }

  /**
   * Reset calibration baseline to default anthropometric values
   */
  resetBaseline() {
    this.baseline = {
      isCalibrated: false,
      cranialRatio: 0.50,
      shoulderWidth: null,
      spineAngle: 90.0,
      shoulderAngle: 0.0,
      headAngle: 0.0,
      timestamp: null
    };
  }

  /**
   * Calibrate neutral baseline using reliable keypoints.
   * Supports either a single keypointsMap or an array of sample frames collected during countdown.
   */
  calibrate(keypointsInput) {
    let sample = keypointsInput;

    // If an array of sample frames was collected, average keypoints across all valid samples
    if (Array.isArray(keypointsInput) && keypointsInput.length > 0) {
      const averaged = {};
      const keys = ['leftShoulder', 'rightShoulder', 'nose', 'leftEar', 'rightEar', 'leftEye', 'rightEye', 'leftHip', 'rightHip'];
      
      for (const k of keys) {
        let sumX = 0;
        let sumY = 0;
        let count = 0;
        for (const frame of keypointsInput) {
          const pt = frame[k];
          if (pt && pt.score > 0.3) {
            sumX += pt.x;
            sumY += pt.y;
            count++;
          }
        }
        if (count > 0) {
          averaged[k] = { x: sumX / count, y: sumY / count, score: 0.9 };
        }
      }
      sample = averaged;
    }

    if (!sample) return false;

    const lSh = sample.leftShoulder;
    const rSh = sample.rightShoulder;
    const nose = sample.nose;
    const lEar = sample.leftEar;
    const rEar = sample.rightEar;
    const lHip = sample.leftHip;
    const rHip = sample.rightHip;

    if (!lSh || !rSh) return false;

    const midShoulder = PostureClassifier.midpoint(lSh, rSh);
    const shoulderDist = PostureClassifier.distance(lSh, rSh);
    if (shoulderDist < 20) return false;

    // Head center
    let headCenter = nose;
    if (lEar && rEar) {
      headCenter = PostureClassifier.midpoint(lEar, rEar);
    }

    const verticalClearance = headCenter ? Math.max(0, midShoulder.y - headCenter.y) : (shoulderDist * 0.50);
    const cranialRatio = verticalClearance / shoulderDist;

    // Shoulder baseline tilt from horizontal
    const dy = Math.abs(rSh.y - lSh.y);
    const dx = Math.abs(rSh.x - lSh.x);
    const shoulderAngle = Math.atan2(dy, Math.max(1, dx)) * (180 / Math.PI);

    // Spine baseline
    let spineAngle = 90.0;
    if (lHip && rHip) {
      const midHip = PostureClassifier.midpoint(lHip, rHip);
      const spineDx = midShoulder.x - midHip.x;
      const spineDy = midHip.y - midShoulder.y;
      spineAngle = Math.abs(Math.atan2(spineDy, spineDx) * (180 / Math.PI));
    }

    this.baseline = {
      isCalibrated: true,
      cranialRatio: Math.max(0.35, Math.min(0.70, cranialRatio)),
      shoulderWidth: shoulderDist,
      spineAngle: spineAngle,
      shoulderAngle: shoulderAngle,
      headAngle: 0.0,
      timestamp: Date.now()
    };

    return true;
  }

  /**
   * Main Posture Evaluation Pipeline
   * Calculates precise biomechanical metrics and translates them into plain-English diagnostics
   */
  evaluate(keypointsMap) {
    const minConfidence = 0.28;
    const lSh = keypointsMap.leftShoulder;
    const rSh = keypointsMap.rightShoulder;
    const nose = keypointsMap.nose;
    const lEar = keypointsMap.leftEar;
    const rEar = keypointsMap.rightEar;
    const lEye = keypointsMap.leftEye;
    const rEye = keypointsMap.rightEye;
    const lHip = keypointsMap.leftHip;
    const rHip = keypointsMap.rightHip;

    // Check if shoulders are visible
    if (!lSh || !rSh || lSh.score < minConfidence || rSh.score < minConfidence) {
      return {
        detected: false,
        status: 'searching',
        message: 'Searching for Person...',
        score: 0,
        rawScore: 0,
        angles: { shoulder: 0, head: 0, spine: 90, slouchRatio: 1.0, slouchPercentage: 100 },
        states: { shoulder: 'optimal', head: 'optimal', spine: 'optimal', slouch: 'optimal' },
        diagnostics: {
          shoulder: { title: 'Shoulders', badgeText: 'Searching', summary: 'Searching for shoulders', state: 'optimal' },
          head: { title: 'Head & Neck', badgeText: 'Searching', summary: 'Searching for head', state: 'optimal' },
          spine: { title: 'Spine', badgeText: 'Searching', summary: 'Searching for spine', state: 'optimal' },
          slouch: { title: 'Sitting Form', badgeText: 'Searching', summary: 'Position in camera frame', state: 'optimal' }
        },
        advice: {
          primaryTitle: 'Position Yourself in View',
          primaryText: 'Ensure your upper body and head are clearly visible in the camera frame.',
          actionStep: 'Center yourself in the webcam view',
          icon: 'fa-circle-question text-cyan'
        },
        issues: []
      };
    }

    // ---------------------------------------------------------
    // 0. SUB-FRAME KEYPOINT SMOOTHING (EMA FILTER)
    // ---------------------------------------------------------
    // Eliminates PoseNet frame-to-frame micro-jitter without adding perceptual lag
    const alpha = 0.65; // 65% new frame, 35% history
    const allKeys = ['leftShoulder', 'rightShoulder', 'nose', 'leftEar', 'rightEar', 'leftEye', 'rightEye', 'leftHip', 'rightHip'];
    const smoothed = {};

    for (const k of allKeys) {
      const raw = keypointsMap[k];
      if (raw && raw.score >= minConfidence) {
        if (!this.smoothedKeypoints[k]) {
          this.smoothedKeypoints[k] = { x: raw.x, y: raw.y };
        } else {
          this.smoothedKeypoints[k].x = this.smoothedKeypoints[k].x * (1 - alpha) + raw.x * alpha;
          this.smoothedKeypoints[k].y = this.smoothedKeypoints[k].y * (1 - alpha) + raw.y * alpha;
        }
        smoothed[k] = { x: this.smoothedKeypoints[k].x, y: this.smoothedKeypoints[k].y, score: raw.score };
      } else {
        smoothed[k] = raw;
      }
    }

    const smLSh = smoothed.leftShoulder || lSh;
    const smRSh = smoothed.rightShoulder || rSh;
    const smNose = smoothed.nose || nose;
    const smLEye = smoothed.leftEye || lEye;
    const smREye = smoothed.rightEye || rEye;
    const smLEar = smoothed.leftEar || lEar;
    const smREar = smoothed.rightEar || rEar;
    const smLHip = smoothed.leftHip || lHip;
    const smRHip = smoothed.rightHip || rHip;

    const midShoulder = PostureClassifier.midpoint(smLSh, smRSh);
    const shoulderWidth = Math.max(30, PostureClassifier.distance(smLSh, smRSh));
    const cfg = this.thresholds[this.sensitivity];

    const issues = [];
    let score = 100;

    // ---------------------------------------------------------
    // 1. SHOULDER BALANCE & HORIZON ANALYSIS
    // ---------------------------------------------------------
    // Acute angle of shoulder axis relative to horizontal
    const shDy = Math.abs(smRSh.y - smLSh.y);
    const shDx = Math.abs(smRSh.x - smLSh.x);
    const rawShoulderTilt = Math.atan2(shDy, Math.max(1, shDx)) * (180 / Math.PI);
    
    // Deviation from calibrated neutral (or 0 if uncalibrated)
    const netShoulderTilt = Math.abs(rawShoulderTilt - (this.baseline.isCalibrated ? this.baseline.shoulderAngle : 0));

    // Determine which shoulder is dropped
    // Note: in front-facing camera coordinates, canvas Y is downward.
    let droppedShoulder = 'none';
    if (Math.abs(smLSh.y - smRSh.y) > 4) {
      droppedShoulder = smLSh.y > smRSh.y ? 'Left' : 'Right';
    }

    let shoulderState = 'optimal';
    let shoulderBadge = 'Balanced';
    let shoulderSummary = `Balanced (${netShoulderTilt.toFixed(1)}°)`;
    let shoulderTip = 'Shoulders are level and relaxed.';

    if (netShoulderTilt > cfg.shoulderTilt) {
      shoulderState = netShoulderTilt > (cfg.shoulderTilt * 1.5) ? 'bad' : 'warning';
      const sideName = droppedShoulder !== 'none' ? `${droppedShoulder} Dropped` : 'Uneven';
      shoulderBadge = sideName;
      shoulderSummary = `${droppedShoulder !== 'none' ? droppedShoulder : 'One'} shoulder dropped ${netShoulderTilt.toFixed(1)}°`;
      shoulderTip = `Gently raise your ${droppedShoulder !== 'none' ? droppedShoulder.toLowerCase() : 'dropped'} shoulder and roll both back.`;

      issues.push({
        type: 'shoulder_tilt',
        severity: shoulderState,
        title: 'Uneven Shoulders',
        description: `${droppedShoulder !== 'none' ? droppedShoulder : 'One'} shoulder is dropped by ${netShoulderTilt.toFixed(1)}°.`,
        speechMessage: `Lift your ${droppedShoulder !== 'none' ? droppedShoulder.toLowerCase() : ''} shoulder to keep level`
      });

      const penalty = (netShoulderTilt - cfg.shoulderTilt) * 3.2;
      score -= Math.min(25, penalty);
    }

    // ---------------------------------------------------------
    // 2. HEAD & NECK ALIGNMENT (Lateral Tilt & Forward Head)
    // ---------------------------------------------------------
    let headTiltAngle = 0;
    let headTiltSide = 'none';

    // Prioritize eyes for lateral head roll; fallback to ears
    if (smLEye && smREye && smLEye.score > minConfidence && smREye.score > minConfidence) {
      const eDy = Math.abs(smREye.y - smLEye.y);
      const eDx = Math.abs(smREye.x - smLEye.x);
      headTiltAngle = Math.atan2(eDy, Math.max(1, eDx)) * (180 / Math.PI);
      if (Math.abs(smLEye.y - smREye.y) > 3) {
        headTiltSide = smLEye.y > smREye.y ? 'Left' : 'Right';
      }
    } else if (smLEar && smREar && smLEar.score > minConfidence && smREar.score > minConfidence) {
      const eDy = Math.abs(smREar.y - smLEar.y);
      const eDx = Math.abs(smREar.x - smLEar.x);
      headTiltAngle = Math.atan2(eDy, Math.max(1, eDx)) * (180 / Math.PI);
      if (Math.abs(smLEar.y - smREar.y) > 3) {
        headTiltSide = smLEar.y > smREar.y ? 'Left' : 'Right';
      }
    }

    const netHeadTilt = Math.abs(headTiltAngle);

    // Head Center Reference
    let headCenter = smNose;
    if (smLEar && smREar && smLEar.score > minConfidence && smREar.score > minConfidence) {
      headCenter = PostureClassifier.midpoint(smLEar, smREar);
    }

    // Forward Head / Craniovertebral Clearance
    const verticalCranialClearance = headCenter ? Math.max(0, midShoulder.y - headCenter.y) : (shoulderWidth * 0.50);
    const currentCranialRatio = verticalCranialClearance / shoulderWidth;
    const baseCranialRatio = this.baseline.isCalibrated ? this.baseline.cranialRatio : 0.50;
    
    // Slouch / Sitting Height extension factor (1.0 = ideal, <0.82 = hunched)
    const slouchRatio = Math.max(0.3, Math.min(1.2, currentCranialRatio / baseCranialRatio));
    const slouchPercentage = Math.round(Math.min(100, Math.max(40, slouchRatio * 100)));

    let headState = 'optimal';
    let headBadge = 'Upright';
    let headSummary = `Upright & Centered (${netHeadTilt.toFixed(1)}°)`;
    let headTip = 'Head is centered with chin tucked naturally.';

    // Check Lateral Head Tilt
    if (netHeadTilt > cfg.headTilt) {
      headState = netHeadTilt > (cfg.headTilt * 1.5) ? 'bad' : 'warning';
      headBadge = headTiltSide !== 'none' ? `Tilt ${headTiltSide}` : 'Tilted';
      headSummary = `Head tilted ${headTiltSide !== 'none' ? headTiltSide.toLowerCase() : ''} by ${netHeadTilt.toFixed(1)}°`;
      headTip = 'Straighten your neck and keep ears level above shoulders.';

      issues.push({
        type: 'head_tilt',
        severity: headState,
        title: 'Head Tilted',
        description: `Head is tilted laterally by ${netHeadTilt.toFixed(1)}°.`,
        speechMessage: 'Straighten your head'
      });

      const penalty = (netHeadTilt - cfg.headTilt) * 2.5;
      score -= Math.min(20, penalty);
    }

    // ---------------------------------------------------------
    // 3. SITTING FORM & SLOUCH / TECH-NECK COMPRESSION
    // ---------------------------------------------------------
    let slouchState = 'optimal';
    let slouchBadge = 'Tall Form';
    let slouchSummary = `${slouchPercentage}% Upright Height`;
    let slouchTip = 'Spine is elongated with ribs lifted.';

    if (slouchRatio < cfg.slouchRatio) {
      slouchState = slouchRatio < (cfg.slouchRatio * 0.88) ? 'bad' : 'warning';
      slouchBadge = slouchState === 'bad' ? 'Hunched' : 'Mild Slouch';
      slouchSummary = `Torso collapsed to ${slouchPercentage}% height`;
      slouchTip = 'Lengthen your spine, lift your chest, and bring chin back.';

      // Also adjust head state if forward head / tech-neck is the culprit
      if (headState === 'optimal') {
        headState = slouchState;
        headBadge = 'Tech-Neck';
        headSummary = 'Forward head / chin drop detected';
        headTip = 'Tuck chin slightly and bring head back over shoulders.';
      }

      issues.push({
        type: 'slouch',
        severity: slouchState,
        title: 'Forward Slouch / Tech-Neck',
        description: `Sitting height compressed (${slouchPercentage}%). Sit tall and tuck chin.`,
        speechMessage: 'Sit tall and pull your shoulders back'
      });

      const penalty = (cfg.slouchRatio - slouchRatio) * 115;
      score -= Math.min(30, penalty);
    }

    // ---------------------------------------------------------
    // 4. SPINE VERTICALITY & TORSO PLUMB LINE
    // ---------------------------------------------------------
    let spineAngle = 90.0;
    let spineLeanSide = 'none';
    let spineState = 'optimal';
    let spineBadge = 'Straight';
    let spineSummary = 'Straight Vertical (90.0°)';
    let spineTip = 'Back is straight and evenly supported.';

    const hasHips = (smLHip && smLHip.score > minConfidence) || (smRHip && smRHip.score > minConfidence);

    if (hasHips) {
      // Direct full-torso spine vector from hip midpoint to shoulder midpoint
      const hipPt1 = (smLHip && smLHip.score > minConfidence) ? smLHip : smRHip;
      const hipPt2 = (smRHip && smRHip.score > minConfidence) ? smRHip : smLHip;
      const midHip = PostureClassifier.midpoint(hipPt1, hipPt2);

      const sDx = midShoulder.x - midHip.x;
      const sDy = Math.max(1, midHip.y - midShoulder.y); // positive upright
      spineAngle = Math.atan2(sDy, sDx) * (180 / Math.PI);
      if (Math.abs(sDx) > 8) {
        spineLeanSide = sDx > 0 ? 'Right' : 'Left';
      }
    } else {
      // Fallback for desk-level webcam view (where hips are occluded by desk)
      // Estimates upper thoracic verticality via shoulder horizon and cervical plumb line
      const cervicalDx = headCenter ? (headCenter.x - midShoulder.x) : 0;
      const cervicalDy = headCenter ? Math.max(1, midShoulder.y - headCenter.y) : shoulderWidth * 0.5;
      const plumbAngle = Math.atan2(cervicalDx, cervicalDy) * (180 / Math.PI);
      
      const compositeDev = Math.max(Math.abs(plumbAngle * 0.75), netShoulderTilt * 0.65);
      if (cervicalDx > 6) {
        spineAngle = 90.0 - compositeDev;
        spineLeanSide = 'Right';
      } else if (cervicalDx < -6) {
        spineAngle = 90.0 + compositeDev;
        spineLeanSide = 'Left';
      } else {
        spineAngle = 90.0;
      }
    }

    const spineDeviation = Math.abs(90.0 - spineAngle);

    if (spineDeviation > cfg.spineDev) {
      spineState = spineDeviation > (cfg.spineDev * 1.5) ? 'bad' : 'warning';
      const sideText = spineLeanSide !== 'none' ? `Leaning ${spineLeanSide}` : 'Leaning';
      spineBadge = sideText;
      spineSummary = `Spine angled ${spineDeviation.toFixed(1)}° from vertical`;
      spineTip = `Re-center your torso — avoid leaning to the ${spineLeanSide !== 'none' ? spineLeanSide.toLowerCase() : 'side'}.`;

      issues.push({
        type: 'spine_lean',
        severity: spineState,
        title: 'Spine Leaning',
        description: `Torso is leaning ${spineLeanSide !== 'none' ? spineLeanSide : 'off-center'} by ${spineDeviation.toFixed(1)}°.`,
        speechMessage: 'Align your spine vertically'
      });

      const penalty = (spineDeviation - cfg.spineDev) * 2.8;
      score -= Math.min(25, penalty);
    } else {
      spineSummary = `Straight & Vertical (${spineAngle.toFixed(1)}°)`;
    }

    // ---------------------------------------------------------
    // 5. SCORE CLAMPING & RESPONSIVE FILTERING
    // ---------------------------------------------------------
    score = Math.max(0, Math.min(100, Math.round(score)));
    this.smoothedScore = Math.round(this.smoothedScore * 0.65 + score * 0.35);

    // Overall classification state
    let overallStatus = 'good';
    if (this.smoothedScore < 60 || issues.some(i => i.severity === 'bad')) {
      overallStatus = 'bad';
    } else if (this.smoothedScore < 85 || issues.length > 0) {
      overallStatus = 'warning';
    }

    // ---------------------------------------------------------
    // 6. HUMAN-FRIENDLY COACHING ADVICE GENERATOR
    // ---------------------------------------------------------
    let adviceTitle = 'Optimal Spinal Alignment';
    let adviceText = 'Shoulders balanced, neck upright, and spine centered. Keep breathing relaxed!';
    let adviceAction = 'Maintain this comfortable posture';
    let adviceIcon = 'fa-circle-check text-success';

    if (issues.length > 0) {
      // Pick the most severe issue
      const primaryIssue = issues.find(i => i.severity === 'bad') || issues[0];
      adviceIcon = primaryIssue.severity === 'bad' ? 'fa-circle-xmark text-danger' : 'fa-triangle-exclamation text-amber';
      adviceTitle = primaryIssue.title;

      if (primaryIssue.type === 'shoulder_tilt') {
        adviceText = `${droppedShoulder !== 'none' ? droppedShoulder : 'One'} shoulder is dropped.`;
        adviceAction = `Lift your ${droppedShoulder !== 'none' ? droppedShoulder.toLowerCase() : 'lower'} shoulder and level your back`;
      } else if (primaryIssue.type === 'slouch') {
        adviceText = `Torso compressed to ${slouchPercentage}% height (Forward head / hunched shoulders).`;
        adviceAction = 'Sit tall, lift your chest, and bring chin back';
      } else if (primaryIssue.type === 'head_tilt') {
        adviceText = `Neck tilted by ${netHeadTilt.toFixed(1)}°.`;
        adviceAction = 'Straighten your neck and center your gaze forward';
      } else if (primaryIssue.type === 'spine_lean') {
        adviceText = `Spine leaning ${spineLeanSide !== 'none' ? spineLeanSide.toLowerCase() : 'off-center'} by ${spineDeviation.toFixed(1)}°.`;
        adviceAction = 'Center your weight evenly across your chair';
      }
    }

    return {
      detected: true,
      status: overallStatus,
      score: this.smoothedScore,
      rawScore: score,
      angles: {
        shoulder: Math.round(netShoulderTilt * 10) / 10,
        head: Math.round(netHeadTilt * 10) / 10,
        spine: Math.round(spineAngle * 10) / 10,
        slouchRatio: Math.round(slouchRatio * 100) / 100,
        slouchPercentage: slouchPercentage
      },
      states: {
        shoulder: shoulderState,
        head: headState,
        spine: spineState,
        slouch: slouchState
      },
      diagnostics: {
        shoulder: {
          title: 'Shoulders',
          state: shoulderState,
          angle: netShoulderTilt,
          droppedSide: droppedShoulder,
          badgeText: shoulderBadge,
          summary: shoulderSummary,
          tip: shoulderTip
        },
        head: {
          title: 'Head & Neck',
          state: headState,
          angle: netHeadTilt,
          badgeText: headBadge,
          summary: headSummary,
          tip: headTip
        },
        spine: {
          title: 'Spine',
          state: spineState,
          angle: spineAngle,
          leanSide: spineLeanSide,
          badgeText: spineBadge,
          summary: spineSummary,
          tip: spineTip
        },
        slouch: {
          title: 'Sitting Form',
          state: slouchState,
          percentage: slouchPercentage,
          badgeText: slouchBadge,
          summary: slouchSummary,
          tip: slouchTip
        }
      },
      advice: {
        primaryTitle: adviceTitle,
        primaryText: adviceText,
        actionStep: adviceAction,
        icon: adviceIcon
      },
      issues: issues,
      points: {
        midShoulder,
        headCenter
      }
    };
  }
}

// Attach to window
window.PostureClassifier = PostureClassifier;
