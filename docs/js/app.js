/* Student Engagement Analysis - in-browser demo
 * Mirrors the desktop pipeline in demo.py:
 *   BlazeFace -> largest face + 30% padding -> 224x224 -> resnet_v2 [-1,1]
 *   -> ResNet50V2 softmax -> P(Engaged) -> 12-frame rolling mean.
 * Everything runs client-side; no frames leave the device.
 */
(() => {
  "use strict";

  // ---- config (kept in sync with config.py) ----
  const IMG_SIZE = 224;
  const PADDING_RATIO = 0.3;
  const SMOOTH_WINDOW = 12;
  const LABELS = ["Not Engaged", "Engaged"];
  const MODEL_URL = "model/model.json";
  const DETECT_INTERVAL_MS = 100; // ~10 detections/sec, render stays at full rate

  // ---- elements ----
  const video = document.getElementById("video");
  const canvas = document.getElementById("overlay");
  const ctx = canvas.getContext("2d");
  const stageMsg = document.getElementById("stage-msg");
  const banner = document.getElementById("model-banner");

  const btnStart = document.getElementById("btn-start");
  const btnStop = document.getElementById("btn-stop");
  const btnPause = document.getElementById("btn-pause");
  const btnSnap = document.getElementById("btn-snap");

  const bigScore = document.getElementById("big-score");
  const meterFill = document.getElementById("meter-fill");
  const readoutState = document.getElementById("readout-state");
  const statStatus = document.getElementById("stat-status");
  const statFaces = document.getElementById("stat-faces");
  const statFps = document.getElementById("stat-fps");
  const statModel = document.getElementById("stat-model");

  // ---- state ----
  let stream = null;
  let faceModel = null;     // BlazeFace
  let engagementModel = null; // TF.js LayersModel (ResNet50V2 head)
  let modelLoaded = false;
  let running = false;
  let paused = false;
  let detecting = false;
  let detectTimer = null;
  let rafId = null;

  let lastResult = null;    // { box:[x1,y1,x2,y2], engaged: number|null }
  const recentProbs = [];   // rolling window for smoothing
  let smoothedFps = 0;
  let prevTick = 0;

  // ---- colour ramp: red(0) -> amber(0.5) -> green(1), matching demo.py get_colour ----
  function getColour(p) {
    p = Math.max(0, Math.min(1, p));
    let r, g, b;
    if (p < 0.5) {
      const t = p / 0.5;
      r = 220; g = Math.round(165 * t); b = 0;
    } else {
      const t = (p - 0.5) / 0.5;
      r = Math.round(220 * (1 - t)); g = Math.round(165 + 35 * t); b = 0;
    }
    return `rgb(${r}, ${g}, ${b})`;
  }

  // ---- model loading (graceful: site works for detection even without it) ----
  async function loadEngagementModel() {
    statModel.textContent = "loading…";
    try {
      engagementModel = await tf.loadLayersModel(MODEL_URL);
      // warm up so the first real prediction isn't slow
      tf.tidy(() => engagementModel.predict(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3])));
      modelLoaded = true;
      statModel.textContent = "ready";
      hideBanner();
    } catch (err) {
      modelLoaded = false;
      statModel.textContent = "not loaded";
      showBanner(
        "warn",
        'Engagement model not found at <code>model/model.json</code>. ' +
        "Face detection still works below, but scoring is disabled until the model is added. " +
        "See <code>docs/CONVERT_MODEL.md</code> for the one-command conversion step."
      );
      console.warn("Engagement model failed to load:", err);
    }
  }

  function showBanner(kind, html) {
    banner.className = "banner " + (kind === "warn" ? "banner-warn" : "banner-info");
    banner.innerHTML = html;
    banner.hidden = false;
  }
  function hideBanner() { banner.hidden = true; }

  // ---- camera ----
  async function startCamera() {
    if (running) return;
    btnStart.disabled = true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
    } catch (err) {
      btnStart.disabled = false;
      showBanner("warn", "Could not access the camera: " + err.message +
        ". Check browser permissions and that the page is served over HTTPS (or localhost).");
      return;
    }

    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    if (!faceModel) {
      statStatus.textContent = "loading detector…";
      faceModel = await blazeface.load();
    }

    running = true;
    paused = false;
    stageMsg.style.display = "none";
    btnStop.disabled = false;
    btnPause.disabled = false;
    btnSnap.disabled = false;
    btnPause.textContent = "Pause";
    statStatus.textContent = "running";

    prevTick = performance.now();
    renderLoop();
    detectLoop();
  }

  function stopCamera() {
    running = false;
    paused = false;
    if (detectTimer) clearTimeout(detectTimer);
    if (rafId) cancelAnimationFrame(rafId);
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    video.srcObject = null;
    lastResult = null;
    recentProbs.length = 0;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    stageMsg.style.display = "flex";
    btnStart.disabled = false;
    btnStop.disabled = true;
    btnPause.disabled = true;
    btnSnap.disabled = true;
    statStatus.textContent = "stopped";
    statFaces.textContent = "0";
    statFps.textContent = "0.0";
    bigScore.textContent = "—";
    meterFill.style.width = "0%";
    meterFill.style.background = "var(--red)";
    readoutState.textContent = "Idle";
  }

  function togglePause() {
    if (!running) return;
    paused = !paused;
    btnPause.textContent = paused ? "Resume" : "Pause";
    statStatus.textContent = paused ? "paused" : "running";
  }

  // ---- detection + classification loop (async, throttled) ----
  async function detectLoop() {
    if (!running) return;
    if (!paused && !detecting) {
      detecting = true;
      try {
        await detectOnce();
      } catch (e) {
        console.error(e);
      }
      detecting = false;
    }
    detectTimer = setTimeout(detectLoop, DETECT_INTERVAL_MS);
  }

  async function detectOnce() {
    const preds = await faceModel.estimateFaces(video, false);
    statFaces.textContent = String(preds.length);
    if (!preds.length) {
      lastResult = null;
      recentProbs.length = 0;
      return;
    }

    // pick the largest face
    let best = preds[0], bestArea = -1;
    for (const p of preds) {
      const w = p.bottomRight[0] - p.topLeft[0];
      const h = p.bottomRight[1] - p.topLeft[1];
      const area = w * h;
      if (area > bestArea) { bestArea = area; best = p; }
    }

    const W = canvas.width, H = canvas.height;
    let x1 = best.topLeft[0], y1 = best.topLeft[1];
    let x2 = best.bottomRight[0], y2 = best.bottomRight[1];

    // 30% padding (width-based, as in demo.py)
    const pad = PADDING_RATIO * (x2 - x1);
    x1 = Math.max(0, Math.round(x1 - pad));
    y1 = Math.max(0, Math.round(y1 - pad));
    x2 = Math.min(W, Math.round(x2 + pad));
    y2 = Math.min(H, Math.round(y2 + pad));

    const boxW = x2 - x1, boxH = y2 - y1;
    if (boxW < 2 || boxH < 2) { lastResult = null; return; }

    let engaged = null;
    if (modelLoaded) {
      engaged = tf.tidy(() => {
        const frame = tf.browser.fromPixels(video);           // [H,W,3] RGB 0..255
        const crop = frame.slice([y1, x1, 0], [boxH, boxW, 3]);
        const resized = tf.image.resizeBilinear(crop, [IMG_SIZE, IMG_SIZE]);
        // resnet_v2.preprocess_input -> scale to [-1, 1]
        const x = resized.toFloat().div(127.5).sub(1).expandDims(0);
        const out = engagementModel.predict(x);
        return out.dataSync()[1]; // P(Engaged) = softmax index 1
      });
    }

    lastResult = { box: [x1, y1, x2, y2], engaged };
  }

  // ---- render loop (full frame rate) ----
  function renderLoop() {
    if (!running) return;
    rafId = requestAnimationFrame(renderLoop);
    if (paused) return; // freeze last drawn frame

    const W = canvas.width, H = canvas.height;

    // draw the mirrored webcam frame
    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-W, 0);
    ctx.drawImage(video, 0, 0, W, H);
    ctx.restore();

    // smoothed engagement value for this frame
    let smooth = null;
    if (lastResult && lastResult.engaged !== null) {
      recentProbs.push(lastResult.engaged);
      while (recentProbs.length > SMOOTH_WINDOW) recentProbs.shift();
      smooth = recentProbs.reduce((a, b) => a + b, 0) / recentProbs.length;
    }

    if (lastResult) {
      drawFaceBox(lastResult.box, smooth);
    } else {
      ctx.fillStyle = "rgba(150,150,150,0.9)";
      ctx.font = "600 20px Inter, sans-serif";
      ctx.fillText("no face detected", 20, 36);
    }

    if (smooth !== null) drawEngagementBar(smooth);
    updateReadout(smooth);
    drawFps();
  }

  function drawFaceBox(box, smooth) {
    const W = canvas.width;
    const [x1, y1, x2, y2] = box;
    // mirror x coordinates so the box lines up with the mirrored video
    const mx1 = W - x2, mx2 = W - x1;
    const colour = smooth === null ? "rgb(120,140,200)" : getColour(smooth);

    ctx.lineWidth = 3;
    ctx.strokeStyle = colour;
    ctx.strokeRect(mx1, y1, mx2 - mx1, y2 - y1);

    const text = smooth === null
      ? "Face (model off)"
      : `${LABELS[smooth >= 0.5 ? 1 : 0]} ${Math.round(smooth * 100)}%`;
    ctx.font = "600 18px Inter, sans-serif";
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = colour;
    ctx.fillRect(mx1, y1 - 26, tw + 14, 24);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, mx1 + 7, y1 - 8);
  }

  function drawEngagementBar(p) {
    const barX = 20, barH = 16, barW = 220;
    const barY = canvas.height - 40;
    const clip = Math.max(0, Math.min(1, p));
    ctx.fillStyle = "rgba(40,40,40,0.85)";
    ctx.fillRect(barX, barY, barW, barH);
    ctx.fillStyle = getColour(clip);
    ctx.fillRect(barX, barY, Math.round(barW * clip), barH);
    ctx.strokeStyle = "rgba(220,220,220,0.9)";
    ctx.lineWidth = 1;
    ctx.strokeRect(barX, barY, barW, barH);
    ctx.fillStyle = "rgba(235,235,235,0.95)";
    ctx.font = "500 13px Inter, sans-serif";
    ctx.fillText(`engagement ${Math.round(clip * 100)}%`, barX, barY - 6);
  }

  function drawFps() {
    const now = performance.now();
    const dt = Math.max(1e-3, (now - prevTick) / 1000);
    const inst = 1 / dt;
    smoothedFps = smoothedFps === 0 ? inst : 0.9 * smoothedFps + 0.1 * inst;
    prevTick = now;
    statFps.textContent = smoothedFps.toFixed(1);
  }

  function updateReadout(smooth) {
    if (smooth === null) {
      if (!modelLoaded) {
        bigScore.textContent = "—";
        readoutState.textContent = "Model not loaded";
        meterFill.style.width = "0%";
      } else {
        bigScore.textContent = "—";
        readoutState.textContent = lastResult ? "Reading…" : "No face";
        meterFill.style.width = "0%";
      }
      return;
    }
    const pct = Math.round(smooth * 100);
    bigScore.textContent = pct + "%";
    bigScore.style.color = getColour(smooth);
    meterFill.style.width = pct + "%";
    meterFill.style.background = getColour(smooth);
    readoutState.textContent = LABELS[smooth >= 0.5 ? 1 : 0];
    readoutState.style.color = getColour(smooth);
  }

  // ---- snapshot ----
  function saveSnapshot() {
    if (!running) return;
    const url = canvas.toDataURL("image/jpeg", 0.92);
    const a = document.createElement("a");
    const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    a.href = url;
    a.download = `engagement_${ts}.jpg`;
    a.click();
  }

  // ---- wire up ----
  btnStart.addEventListener("click", startCamera);
  btnStop.addEventListener("click", stopCamera);
  btnPause.addEventListener("click", togglePause);
  btnSnap.addEventListener("click", saveSnapshot);
  document.addEventListener("keydown", (e) => {
    if (!running) return;
    if (e.key === "p") togglePause();
    else if (e.key === "s") saveSnapshot();
    else if (e.key === "q") stopCamera();
  });

  // sanity check that the CDN libs loaded
  if (typeof tf === "undefined" || typeof blazeface === "undefined") {
    showBanner("warn", "TensorFlow.js failed to load from the CDN. Check your internet connection and reload.");
    btnStart.disabled = true;
  } else {
    loadEngagementModel();
  }
})();
