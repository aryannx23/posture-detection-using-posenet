/**
 * PoseNet Inference & Camera Management Engine
 * Posture Detection System Using PoseNet
 * Author: Aryan Raj Pandey (B.Tech CSE - AI & ML)
 * 
 * Performance & Stability Upgrades:
 * 1. Decoupled 60 FPS Render Loop from Async PoseNet Inference Loop (Zero video lag/stutter)
 * 2. MobileNetV1 optimized input resolution (257x257 default) with WebGL GPU acceleration & shader warmup
 * 3. Adaptive Velocity-Aware Anti-Jitter Filter (deadband for sensor noise + responsive fast tracking)
 * 4. Sub-frame Lerp Keypoint Interpolation for buttery smooth 60 FPS skeleton movements
 * 5. Native Canvas Mirroring with upright, un-flipped text badges & guide horizon
 * 6. Dynamic Container Aspect-Ratio Synchronization (Zero video stretching or jitter)
 */

class PoseNetEngine {
  constructor(videoElement, canvasElement) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d', { alpha: false });
    this.viewportWrapper = document.getElementById('viewportWrapper');
    
    this.net = null;
    this.isRunning = false;
    this.isModelLoading = false;
    this.isInferring = false;
    this.multiplier = 0.75;
    this.inputResolution = 257; // 257x257 provides optimal 30-60 FPS real-time throughput
    this.stream = null;
    this.isFlipped = true; // Mirror view by default for natural webcam interaction
    
    // Animation loop handles
    this.renderAnimId = null;
    this.inferenceAnimId = null;

    // Dual Performance Metrics (Render FPS & AI Inference Latency)
    this.lastRenderTime = performance.now();
    this.renderFrameCount = 0;
    this.aiFrameCount = 0;
    this.renderFps = 0;
    this.aiFps = 0;
    this.lastLatencyMs = 0;
    this.fpsUpdateInterval = 500;
    this.lastFpsUpdate = performance.now();

    // Keypoint Stabilization States
    this.rawKeypoints = {};
    this.smoothedKeypoints = {};
    this.displayKeypoints = {};

    // Simulation & Photo Mode State
    this.isSimulationMode = false;
    this.simulatedTime = 0;
    this.isPhotoMode = false;
    this.currentImage = null;
    this.currentPhotoName = '';

    // Event callbacks
    this.onPoseDetected = null;
    this.onRenderFrame = null;
    this.onStatusChange = null;
    this.onFpsUpdate = null;
  }

  /**
   * Load PoseNet Deep Learning Model with WebGL hardware acceleration & warmup
   */
  async loadModel(multiplier = 0.75, resolution = 257) {
    this.isModelLoading = true;
    this.multiplier = parseFloat(multiplier);
    this.inputResolution = parseInt(resolution, 10) || 257;
    this.triggerStatus('Optimizing TensorFlow WebGL & loading PoseNet...', 'loading');

    try {
      if (typeof posenet === 'undefined') {
        throw new Error('PoseNet library not found. Please verify CDN connection.');
      }

      // Configure high-performance WebGL backend flags
      if (window.tf) {
        try {
          if (tf.getBackend() !== 'webgl') {
            await tf.setBackend('webgl');
          }
          await tf.ready();
          tf.env().set('WEBGL_PACK', true);
          tf.env().set('WEBGL_CPU_FORWARD', false);
          tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
        } catch (backendErr) {
          console.warn('TFJS WebGL optimization note:', backendErr);
        }
      }

      this.net = await posenet.load({
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: { width: this.inputResolution, height: this.inputResolution },
        multiplier: this.multiplier,
        quantBytes: 2
      });

      // Warmup model with dummy canvas to compile WebGL shader pipelines upfront
      // Prevents the common 2-3 second video freeze when starting the camera
      try {
        const dummyCanvas = document.createElement('canvas');
        dummyCanvas.width = this.inputResolution;
        dummyCanvas.height = this.inputResolution;
        await this.net.estimateSinglePose(dummyCanvas, {
          flipHorizontal: false,
          decodingMethod: 'single-person'
        });
      } catch (warmErr) {
        console.warn('Model warmup notice:', warmErr);
      }

      this.isModelLoading = false;
      this.triggerStatus('PoseNet Deep Learning Engine Ready', 'ready');
      return true;
    } catch (err) {
      console.error('Error loading PoseNet:', err);
      this.isModelLoading = false;
      this.triggerStatus(`Failed to load model: ${err.message}`, 'error');
      return false;
    }
  }

  /**
   * Start Live Webcam with Smooth Video Pipeline
   */
  async startCamera() {
    this.isSimulationMode = false;
    try {
      if (!this.net) {
        await this.loadModel(this.multiplier, this.inputResolution);
      }

      this.triggerStatus('Accessing webcam feed...', 'loading');
      
      const constraints = {
        audio: false,
        video: {
          width: { ideal: 1280, min: 640 },
          height: { ideal: 720, min: 480 },
          aspectRatio: { ideal: 1.7777777778 },
          frameRate: { ideal: 30, max: 60 },
          facingMode: 'user'
        }
      };

      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.video.srcObject = this.stream;
      this.video.playsInline = true;
      this.video.muted = true;
      this.video.setAttribute('playsinline', '');
      this.video.setAttribute('autoplay', '');
      this.video.setAttribute('muted', '');

      // Wait until video metadata and initial frame data are ready
      await new Promise((resolve) => {
        if (this.video.readyState >= 2 && this.video.videoWidth > 0) {
          this.video.play().then(resolve).catch(resolve);
        } else {
          this.video.onloadeddata = () => {
            this.video.play().then(resolve).catch(resolve);
          };
        }
      });

      // Synchronize canvas dimensions with actual camera feed
      const vWidth = this.video.videoWidth || 1280;
      const vHeight = this.video.videoHeight || 720;
      this.canvas.width = vWidth;
      this.canvas.height = vHeight;

      this.isRunning = true;
      this.isInferring = false;
      this.triggerStatus('Live Vision Active (60 FPS Stable)', 'active');

      // Start dual decoupled execution loops
      this.startRenderLoop();
      this.startInferenceLoop();
      return true;
    } catch (err) {
      console.warn('Camera access unavailable:', err);
      this.triggerStatus(`Camera Error: ${err.message}. You can switch to Simulation Mode.`, 'error');
      return false;
    }
  }

  /**
   * Gracefully stop camera and reset active loops
   */
  stopCamera() {
    this.isRunning = false;
    this.isInferring = false;

    if (this.renderAnimId) {
      cancelAnimationFrame(this.renderAnimId);
      this.renderAnimId = null;
    }
    if (this.inferenceAnimId) {
      cancelAnimationFrame(this.inferenceAnimId);
      this.inferenceAnimId = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    if (this.video) {
      this.video.srcObject = null;
    }

    this.smoothedKeypoints = {};
    this.displayKeypoints = {};
    this.isPhotoMode = false;
    this.currentImage = null;
    this.currentPhotoName = '';
    this.triggerStatus('Camera Stopped (Standby)', 'standby');
  }

  /**
   * Toggle Mirror (Horizontal Flip) Mode
   */
  toggleFlip() {
    this.isFlipped = !this.isFlipped;
    return this.isFlipped;
  }

  /**
   * Launch Synthetic Posture Simulation Mode
   */
  startSimulationMode() {
    this.stopCamera();
    this.isSimulationMode = true;
    this.isRunning = true;
    this.canvas.width = 640;
    this.canvas.height = 360;

    this.triggerStatus('Simulation Mode Active', 'simulation');
    this.startRenderLoop();
  }

  /**
   * Load and Analyze Static Photo/Image for Ergonomics
   */
  async analyzeImage(fileOrUrl, filename = 'Uploaded Photo') {
    this.stopCamera();
    this.isSimulationMode = false;
    this.isPhotoMode = true;
    this.currentPhotoName = filename;

    if (!this.net) {
      await this.loadModel(this.multiplier, this.inputResolution);
    }

    this.triggerStatus(`Analyzing photo: ${filename}...`, 'loading');

    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';

      img.onload = async () => {
        try {
          this.currentImage = img;

          // Determine optimal canvas dimensions preserving aspect ratio
          let w = img.naturalWidth || img.width;
          let h = img.naturalHeight || img.height;

          // Scale down if image is huge (e.g. 4000x3000) for fast inference & display
          const maxDim = 1280;
          if (w > maxDim || h > maxDim) {
            if (w > h) {
              h = Math.round((h * maxDim) / w);
              w = maxDim;
            } else {
              w = Math.round((w * maxDim) / h);
              h = maxDim;
            }
          }

          this.canvas.width = w;
          this.canvas.height = h;

          const t0 = performance.now();
          const pose = await this.net.estimateSinglePose(img, {
            flipHorizontal: false,
            decodingMethod: 'single-person'
          });
          const latency = Math.round(performance.now() - t0);
          this.lastLatencyMs = latency;

          if (pose && pose.keypoints) {
            const keypointsMap = {};
            for (const kp of pose.keypoints) {
              keypointsMap[kp.part] = { x: kp.position.x, y: kp.position.y, score: kp.score };
            }
            this.smoothedKeypoints = keypointsMap;
            this.displayKeypoints = keypointsMap;

            this.isRunning = true;
            this.triggerStatus(`Photo Analyzed (${filename})`, 'active');

            if (this.onPoseDetected) {
              this.onPoseDetected({
                rawPose: pose,
                keypointsMap: keypointsMap,
                latencyMs: latency,
                canvasWidth: w,
                canvasHeight: h,
                isPhotoMode: true,
                photoName: filename
              });
            }

            // Start photo render loop so canvas stays visible with HUD
            this.startRenderLoop();
            resolve({ success: true, pose, keypointsMap });
          } else {
            this.triggerStatus('No person detected in the photo', 'warning');
            resolve({ success: false, reason: 'No person detected' });
          }
        } catch (err) {
          console.error('Error analyzing image:', err);
          this.triggerStatus(`Analysis failed: ${err.message}`, 'error');
          reject(err);
        }
      };

      img.onerror = (err) => {
        this.triggerStatus('Failed to load image file', 'error');
        reject(err);
      };

      if (typeof fileOrUrl === 'string') {
        img.src = fileOrUrl;
      } else if (fileOrUrl instanceof File || fileOrUrl instanceof Blob) {
        img.src = URL.createObjectURL(fileOrUrl);
      }
    });
  }

  /**
   * High-Performance 60 FPS Render Loop
   * Renders the video feed and interpolated skeleton at full screen refresh rate
   */
  startRenderLoop() {
    const render = () => {
      if (!this.isRunning) return;

      const now = performance.now();
      this.renderFrameCount++;

      // Update dual FPS telemetry every 500ms
      if (now - this.lastFpsUpdate >= this.fpsUpdateInterval) {
        const elapsed = now - this.lastFpsUpdate;
        this.renderFps = Math.round((this.renderFrameCount * 1000) / elapsed);
        this.aiFps = Math.round((this.aiFrameCount * 1000) / elapsed);
        this.renderFrameCount = 0;
        this.aiFrameCount = 0;
        this.lastFpsUpdate = now;

        if (this.onFpsUpdate) {
          this.onFpsUpdate({
            renderFps: this.renderFps,
            aiFps: this.aiFps,
            latencyMs: this.lastLatencyMs
          });
        }
      }

      if (this.isSimulationMode) {
        // Generate simulated cycle
        this.simulatedTime += 0.035;
        const simPose = this.generateSimulatedPose(this.simulatedTime, this.canvas.width, this.canvas.height);
        this.renderSimulationCanvas();
        this.smoothKeypoints(simPose.keypoints);
        if (this.onPoseDetected) {
          this.onPoseDetected({
            rawPose: simPose,
            keypointsMap: this.smoothedKeypoints,
            latencyMs: 1,
            canvasWidth: this.canvas.width,
            canvasHeight: this.canvas.height
          });
        }
      }

      // Smoothly interpolate display keypoints toward target positions (sub-frame lerp)
      this.updateDisplayKeypoints();

      // Trigger frame render callback to paint video and skeleton overlay
      if (this.onRenderFrame) {
        this.onRenderFrame({
          displayKeypoints: this.displayKeypoints,
          canvasWidth: this.canvas.width,
          canvasHeight: this.canvas.height,
          isFlipped: this.isPhotoMode ? false : this.isFlipped,
          isSimulationMode: this.isSimulationMode,
          isPhotoMode: this.isPhotoMode,
          imageElement: this.isPhotoMode ? this.currentImage : null
        });
      }

      if (this.isRunning) {
        this.renderAnimId = requestAnimationFrame(render);
      }
    };

    this.renderAnimId = requestAnimationFrame(render);
  }

  /**
   * Asynchronous Non-Blocking PoseNet Inference Loop
   * Executes as fast as the model allows without ever blocking the 60 FPS render loop
   */
  startInferenceLoop() {
    const runInference = async () => {
      if (!this.isRunning || this.isSimulationMode || this.isPhotoMode) return;

      if (!this.isInferring && this.video.readyState >= 2 && this.net) {
        this.isInferring = true;
        const t0 = performance.now();

        try {
          const pose = await this.net.estimateSinglePose(this.video, {
            flipHorizontal: false,
            decodingMethod: 'single-person'
          });

          const t1 = performance.now();
          this.lastLatencyMs = Math.round(t1 - t0);
          this.aiFrameCount++;

          if (pose && pose.keypoints) {
            // Apply Adaptive Anti-Jitter Filter
            const smoothedMap = this.smoothKeypoints(pose.keypoints);
            if (this.onPoseDetected) {
              this.onPoseDetected({
                rawPose: pose,
                keypointsMap: smoothedMap,
                latencyMs: this.lastLatencyMs,
                canvasWidth: this.canvas.width,
                canvasHeight: this.canvas.height
              });
            }
          }
        } catch (err) {
          console.error('Inference step error:', err);
        } finally {
          this.isInferring = false;
        }
      }

      if (this.isRunning && !this.isSimulationMode) {
        // Yield to next microtask so the browser UI thread remains fluid
        this.inferenceAnimId = requestAnimationFrame(runInference);
      }
    };

    runInference();
  }

  /**
   * Adaptive Velocity-Aware Anti-Jitter Filter
   * Eliminates webcam sensor trembling while maintaining crisp responsiveness for voluntary movements
   */
  smoothKeypoints(keypointsList) {
    const map = {};
    for (const kp of keypointsList) {
      const name = kp.part;
      const prev = this.smoothedKeypoints[name];
      const current = { x: kp.position.x, y: kp.position.y, score: kp.score };

      if (!prev) {
        this.smoothedKeypoints[name] = current;
      } else {
        // Low confidence gating: hold position with slow confidence decay to prevent jumping
        if (current.score < 0.25) {
          this.smoothedKeypoints[name] = {
            x: prev.x,
            y: prev.y,
            score: (prev.score || 0) * 0.92
          };
        } else {
          const dx = current.x - prev.x;
          const dy = current.y - prev.y;
          const dist = Math.hypot(dx, dy);

          // Adaptive alpha formulation:
          // - Small jitter (< 2.5px): heavy dampening (alpha = 0.12) -> Rock-solid posture horizon
          // - Moderate shifts (2.5 - 12px): smooth progressive blend
          // - Large movements (> 12px): fast snap (alpha = 0.82) -> Zero trailing lag
          let alpha;
          if (dist < 2.5) {
            alpha = 0.12;
          } else if (dist <= 12.0) {
            alpha = 0.12 + ((dist - 2.5) / 9.5) * 0.45;
          } else {
            alpha = 0.82;
          }

          this.smoothedKeypoints[name] = {
            x: prev.x + alpha * dx,
            y: prev.y + alpha * dy,
            score: 0.35 * current.score + 0.65 * (prev.score || current.score)
          };
        }
      }
      map[name] = this.smoothedKeypoints[name];
    }
    return map;
  }

  /**
   * Sub-frame Linear Interpolation (Lerp)
   * Connects asynchronous AI updates to the 60 FPS visual renderer
   */
  updateDisplayKeypoints() {
    const lerpFactor = 0.42;
    for (const name in this.smoothedKeypoints) {
      const target = this.smoothedKeypoints[name];
      const current = this.displayKeypoints[name];
      if (!current) {
        this.displayKeypoints[name] = { x: target.x, y: target.y, score: target.score };
      } else {
        this.displayKeypoints[name] = {
          x: current.x + (target.x - current.x) * lerpFactor,
          y: current.y + (target.y - current.y) * lerpFactor,
          score: target.score
        };
      }
    }
  }

  /**
   * Synthetic pose generator simulating human posture transitions
   */
  generateSimulatedPose(t, w, h) {
    const cx = w / 2;
    const cy = h * 0.45;

    // Posture state cycles every 24 seconds
    const cycle = (t % 24);
    let slouchOffset = 0;
    let shoulderTiltAngle = 0;
    let headTiltAngle = 0;

    if (cycle < 6) {
      // Optimal posture
      slouchOffset = 0;
      shoulderTiltAngle = Math.sin(t * 0.5) * 1.5;
    } else if (cycle < 12) {
      // Slouching starts and deepens
      const progress = (cycle - 6) / 6;
      slouchOffset = progress * 38;
    } else if (cycle < 18) {
      // Uneven shoulders / leaning to side
      slouchOffset = 25;
      shoulderTiltAngle = 10.5 * Math.sin(t * 0.8);
      headTiltAngle = 8.0 * Math.sin(t * 0.8);
    } else {
      // Recovering to good posture
      const progress = (24 - cycle) / 6;
      slouchOffset = progress * 20;
      shoulderTiltAngle = progress * 4;
    }

    const radShoulder = shoulderTiltAngle * (Math.PI / 180);
    const radHead = headTiltAngle * (Math.PI / 180);
    const shHalfWidth = 110;

    const midShX = cx;
    const midShY = cy + slouchOffset * 0.3;

    const leftShX = midShX - shHalfWidth * Math.cos(radShoulder);
    const leftShY = midShY - shHalfWidth * Math.sin(radShoulder);

    const rightShX = midShX + shHalfWidth * Math.cos(radShoulder);
    const rightShY = midShY + shHalfWidth * Math.sin(radShoulder);

    const noseX = cx + Math.sin(radHead) * 20;
    const noseY = cy - 90 + slouchOffset;

    const leftEyeX = noseX - 18;
    const leftEyeY = noseY - 12 - radHead * 15;
    const rightEyeX = noseX + 18;
    const rightEyeY = noseY - 12 + radHead * 15;

    const leftEarX = noseX - 42;
    const leftEarY = noseY - 6;
    const rightEarX = noseX + 42;
    const rightEarY = noseY - 6;

    const leftElbowX = leftShX - 45;
    const leftElbowY = leftShY + 110;
    const rightElbowX = rightShX + 45;
    const rightElbowY = rightShY + 110;

    const leftWristX = leftElbowX + 30;
    const leftWristY = leftElbowY + 80;
    const rightWristX = rightElbowX - 30;
    const rightWristY = rightElbowY + 80;

    const midHipY = cy + 220;
    const hipHalfWidth = 85;
    const leftHipX = cx - hipHalfWidth;
    const leftHipY = midHipY;
    const rightHipX = cx + hipHalfWidth;
    const rightHipY = midHipY;

    const keypoints = [
      { part: 'nose', position: { x: noseX, y: noseY }, score: 0.98 },
      { part: 'leftEye', position: { x: leftEyeX, y: leftEyeY }, score: 0.95 },
      { part: 'rightEye', position: { x: rightEyeX, y: rightEyeY }, score: 0.95 },
      { part: 'leftEar', position: { x: leftEarX, y: leftEarY }, score: 0.92 },
      { part: 'rightEar', position: { x: rightEarX, y: rightEarY }, score: 0.92 },
      { part: 'leftShoulder', position: { x: leftShX, y: leftShY }, score: 0.97 },
      { part: 'rightShoulder', position: { x: rightShX, y: rightShY }, score: 0.97 },
      { part: 'leftElbow', position: { x: leftElbowX, y: leftElbowY }, score: 0.88 },
      { part: 'rightElbow', position: { x: rightElbowX, y: rightElbowY }, score: 0.88 },
      { part: 'leftWrist', position: { x: leftWristX, y: leftWristY }, score: 0.84 },
      { part: 'rightWrist', position: { x: rightWristX, y: rightWristY }, score: 0.84 },
      { part: 'leftHip', position: { x: leftHipX, y: leftHipY }, score: 0.89 },
      { part: 'rightHip', position: { x: rightHipX, y: rightHipY }, score: 0.89 }
    ];

    return { keypoints, score: 0.95 };
  }

  renderSimulationCanvas() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // Dark cybernetic grid background for simulated mode
    ctx.fillStyle = '#0a0f1d';
    ctx.fillRect(0, 0, w, h);

    // Subtle office/desk silhouettes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 30) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Chair backrest outline
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.2)';
    ctx.lineWidth = 3;
    ctx.strokeRect(w * 0.32, h * 0.25, w * 0.36, h * 0.6);
  }

  triggerStatus(message, state) {
    if (this.onStatusChange) {
      this.onStatusChange(message, state);
    }
  }
}

// Export to window
window.PoseNetEngine = PoseNetEngine;
