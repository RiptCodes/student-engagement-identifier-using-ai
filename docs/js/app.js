(() => {
  "use strict";

  const IMG_SIZE       = 224;
  const PADDING_RATIO  = 0.3;
  const SMOOTH_WINDOW  = 12;
  const LABELS         = ["Not Engaged", "Engaged"];
  const MODEL_URL      = "model/model.json";
  const DETECT_MS      = 100;

  const video       = document.getElementById("video");
  const canvas      = document.getElementById("overlay");
  const ctx         = canvas.getContext("2d");
  const startCover  = document.getElementById("start-cover");
  const btnStart    = document.getElementById("btn-start");
  const btnStop     = document.getElementById("btn-stop");
  const btnPause    = document.getElementById("btn-pause");
  const btnSnap     = document.getElementById("btn-snap");
  const controls    = document.getElementById("controls");
  const scoreLabel  = document.getElementById("score-label");
  const scorePct    = document.getElementById("score-pct");

  let stream = null, faceModel = null, engagementModel = null;
  let modelLoaded = false, running = false, paused = false, detecting = false;
  let detectTimer = null, rafId = null, lastResult = null;
  const recentProbs = [];
  let smoothedFps = 0, prevTick = 0;

  // red(0) -> amber(0.5) -> green(1)  matching demo.py
  function getColour(p) {
    p = Math.max(0, Math.min(1, p));
    let r, g;
    if (p < 0.5) { const t = p / 0.5;  r = 220; g = Math.round(165 * t); }
    else          { const t = (p - 0.5) / 0.5; r = Math.round(220 * (1 - t)); g = Math.round(165 + 35 * t); }
    return `rgb(${r},${g},0)`;
  }

  async function loadEngagementModel() {
    try {
      engagementModel = await tf.loadGraphModel(MODEL_URL);
      tf.tidy(() => {
        const w = engagementModel.predict({ input_layer: tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]) });
        w.dispose();
      });
      modelLoaded = true;
    } catch (e) {
      console.warn("Engagement model not loaded:", e);
    }
  }

  async function startCamera() {
    btnStart.disabled = true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
    } catch (e) {
      btnStart.disabled = false;
      scoreLabel.textContent = "Camera denied";
      return;
    }

    video.srcObject = stream;
    await video.play();
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;

    if (!faceModel) faceModel = await blazeface.load();

    running = true;
    paused  = false;
    startCover.style.display = "none";
    controls.hidden = false;
    btnPause.textContent = "Pause";

    prevTick = performance.now();
    renderLoop();
    detectLoop();
  }

  function stopCamera() {
    running = false;
    paused  = false;
    clearTimeout(detectTimer);
    cancelAnimationFrame(rafId);
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null; video.srcObject = null; lastResult = null; recentProbs.length = 0;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    startCover.style.display = "flex";
    controls.hidden = true;
    btnStart.disabled = false;
    scoreLabel.textContent = "—";
    scorePct.textContent   = "";
    scoreLabel.style.color = "";
  }

  function togglePause() {
    paused = !paused;
    btnPause.textContent = paused ? "Resume" : "Pause";
  }

  // ---- detection loop ----
  async function detectLoop() {
    if (!running) return;
    if (!paused && !detecting) {
      detecting = true;
      try { await detectOnce(); } catch (e) { console.error(e); }
      detecting = false;
    }
    detectTimer = setTimeout(detectLoop, DETECT_MS);
  }

  async function detectOnce() {
    const preds = await faceModel.estimateFaces(video, false);
    if (!preds.length) { lastResult = null; recentProbs.length = 0; return; }

    let best = preds[0], bestArea = -1;
    for (const p of preds) {
      const area = (p.bottomRight[0] - p.topLeft[0]) * (p.bottomRight[1] - p.topLeft[1]);
      if (area > bestArea) { bestArea = area; best = p; }
    }

    const W = canvas.width, H = canvas.height;
    const pad = PADDING_RATIO * (best.bottomRight[0] - best.topLeft[0]);
    let x1 = Math.max(0, Math.round(best.topLeft[0]     - pad));
    let y1 = Math.max(0, Math.round(best.topLeft[1]     - pad));
    let x2 = Math.min(W, Math.round(best.bottomRight[0] + pad));
    let y2 = Math.min(H, Math.round(best.bottomRight[1] + pad));

    const bw = x2 - x1, bh = y2 - y1;
    if (bw < 2 || bh < 2) { lastResult = null; return; }

    let engaged = null;
    if (modelLoaded) {
      engaged = tf.tidy(() => {
        const frame   = tf.browser.fromPixels(video);
        const crop    = frame.slice([y1, x1, 0], [bh, bw, 3]);
        const resized = tf.image.resizeBilinear(crop, [IMG_SIZE, IMG_SIZE]);
        const x       = resized.toFloat().div(127.5).sub(1).expandDims(0);
        return engagementModel.predict({ input_layer: x }).dataSync()[1];
      });
    }

    lastResult = { box: [x1, y1, x2, y2], engaged };
  }

  // ---- render loop ----
  function renderLoop() {
    if (!running) return;
    rafId = requestAnimationFrame(renderLoop);
    if (paused) return;

    const W = canvas.width, H = canvas.height;
    ctx.save(); ctx.scale(-1, 1); ctx.translate(-W, 0);
    ctx.drawImage(video, 0, 0, W, H);
    ctx.restore();

    let smooth = null;
    if (lastResult && lastResult.engaged !== null) {
      recentProbs.push(lastResult.engaged);
      while (recentProbs.length > SMOOTH_WINDOW) recentProbs.shift();
      smooth = recentProbs.reduce((a, b) => a + b, 0) / recentProbs.length;
    }

    if (lastResult) drawFaceBox(lastResult.box, smooth);
    updateScore(smooth);

    // fps
    const now = performance.now();
    smoothedFps = smoothedFps === 0 ? 60 : 0.9 * smoothedFps + 0.1 * (1000 / Math.max(1, now - prevTick));
    prevTick = now;
  }

  function drawFaceBox(box, smooth) {
    const W = canvas.width;
    const [x1, y1, x2, y2] = box;
    const mx1 = W - x2, mx2 = W - x1;
    const colour = smooth === null ? "rgba(140,155,200,0.8)" : getColour(smooth);

    ctx.lineWidth = 3;
    ctx.strokeStyle = colour;
    ctx.strokeRect(mx1, y1, mx2 - mx1, y2 - y1);

    if (smooth !== null) {
      const label = `${LABELS[smooth >= 0.5 ? 1 : 0]}  ${Math.round(smooth * 100)}%`;
      ctx.font = "bold 17px Inter, sans-serif";
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = colour;
      ctx.fillRect(mx1, y1 - 26, tw + 12, 24);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, mx1 + 6, y1 - 8);
    }
  }

  function updateScore(smooth) {
    if (smooth === null) {
      scoreLabel.textContent = lastResult ? "Reading…" : "No face";
      scoreLabel.style.color = "";
      scorePct.textContent   = modelLoaded ? "" : "model loading…";
      return;
    }
    const pct = Math.round(smooth * 100);
    const label = LABELS[smooth >= 0.5 ? 1 : 0];
    scoreLabel.textContent = label;
    scoreLabel.style.color = getColour(smooth);
    scorePct.textContent   = pct + "%";
    scorePct.style.color   = getColour(smooth);
  }

  function saveSnapshot() {
    if (!running) return;
    const a = document.createElement("a");
    a.href = canvas.toDataURL("image/jpeg", 0.92);
    a.download = `engagement_${new Date().toISOString().slice(0,19).replace(/[:.]/g,"-")}.jpg`;
    a.click();
  }

  btnStart.addEventListener("click", startCamera);
  btnStop .addEventListener("click", stopCamera);
  btnPause.addEventListener("click", togglePause);
  btnSnap .addEventListener("click", saveSnapshot);
  document.addEventListener("keydown", e => {
    if (!running) return;
    if (e.key === "p") togglePause();
    if (e.key === "s") saveSnapshot();
    if (e.key === "q") stopCamera();
  });

  loadEngagementModel();
})();
