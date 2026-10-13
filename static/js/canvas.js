/**
 * canvas.js
 * =========
 * Handles all canvas drawing interactions and communicates
 * with the Flask backend to get digit predictions.
 */

(function () {
  "use strict";

  // ── DOM references ──────────────────────────────────────────────────────
  const canvas      = document.getElementById("drawing-canvas");
  const ctx         = canvas.getContext("2d");
  const predictBtn  = document.getElementById("btn-predict");
  const clearBtn    = document.getElementById("btn-clear");
  const resultBox   = document.getElementById("result-display");
  const loader      = document.getElementById("loader");

  // ── Drawing state ────────────────────────────────────────────────────────
  let isDrawing  = false;
  let lastX      = 0;
  let lastY      = 0;

  // ── Canvas setup ─────────────────────────────────────────────────────────
  function setupCanvas() {
    ctx.fillStyle = "#ffffff";          // white background
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#000000";        // black ink
    // Thinner strokes align better with EMNIST-style training / binarization.
    ctx.lineWidth   = 12;
    ctx.lineCap     = "round";
    ctx.lineJoin    = "round";
    ctx.imageSmoothingEnabled = false;
  }

  setupCanvas();

  // ── Helper: canvas coordinates from mouse/touch event ───────────────────
  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    // Support both mouse and touch events
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
  }

  // ── Drawing handlers ─────────────────────────────────────────────────────
  function startDraw(e) {
    e.preventDefault();
    isDrawing = true;
    const { x, y } = getPos(e);
    lastX = x;
    lastY = y;
    // Draw a dot for single clicks/taps
    ctx.beginPath();
    ctx.arc(x, y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = "#000000";
    ctx.fill();
  }

  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const { x, y } = getPos(e);

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();

    lastX = x;
    lastY = y;
  }

  function stopDraw(e) {
    if (isDrawing) {
      e.preventDefault();
      isDrawing = false;
    }
  }

  // ── Mouse events ─────────────────────────────────────────────────────────
  canvas.addEventListener("mousedown",  startDraw);
  canvas.addEventListener("mousemove",  draw);
  canvas.addEventListener("mouseup",    stopDraw);
  canvas.addEventListener("mouseleave", stopDraw);

  // ── Touch events (mobile support) ────────────────────────────────────────
  canvas.addEventListener("touchstart", startDraw, { passive: false });
  canvas.addEventListener("touchmove",  draw,      { passive: false });
  canvas.addEventListener("touchend",   stopDraw,  { passive: false });

  // ── Clear canvas ─────────────────────────────────────────────────────────
  function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setupCanvas();
    showResult(null);    // reset result area
  }

  // ── Send image to backend & display result ────────────────────────────────
  async function sendPrediction() {
    const imageData = canvas.toDataURL("image/png");

    // Show loading state
    setLoading(true);
    showResult(null);

    try {
      const response = await fetch("/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ image: imageData }),
      });

      let data;
      try {
        data = await response.json();
      } catch (_) {
        showError("Invalid response from server.");
        return;
      }

      renderPredictionPanel(data);
    } catch (err) {
      showError("Network error – is the Flask server running?");
    } finally {
      setLoading(false);
    }
  }

  // ── UI helpers ────────────────────────────────────────────────────────────

  /** Toggle the loading spinner visibility. */
  function setLoading(show) {
    loader.style.display = show ? "flex" : "none";
    predictBtn.disabled  = show;
  }

  function escapeHtml(str) {
    return String(str)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  /** Build server-provided “Output analysis” block for the browser. */
  function analysisHtml(analysis) {
    if (!analysis || !Array.isArray(analysis.items) || analysis.items.length === 0) return "";
    const rows = analysis.items
      .map((row) => {
        const tone = row.tone === "bad" ? " analysis-row--bad" : row.tone === "good" ? " analysis-row--good" : row.tone === "tip" ? " analysis-row--tip" : "";
        return `
          <div class="analysis-row${tone}">
            <span class="analysis-label">${escapeHtml(row.label)}</span>
            <span class="analysis-value">${escapeHtml(row.value)}</span>
          </div>`;
      })
      .join("");
    return `
      <div class="analysis-panel">
        <div class="analysis-heading">Output analysis</div>
        ${rows}
      </div>`;
  }

  /**
   * Show prediction + confidence + structured analysis (works for HTTP 200 and error JSON with analysis).
   */
  function renderPredictionPanel(data) {
    if (!data) {
      resultBox.innerHTML = `
        <p class="placeholder-text">Draw digits/letters and click <strong>Predict</strong></p>`;
      return;
    }

    const err = data.error;
    const analysisBlock = analysisHtml(data.analysis);

    if (err) {
      resultBox.innerHTML = `
        <p class="error-banner">⚠ ${escapeHtml(err)}</p>
        ${analysisBlock || `<p class="error-text">No analysis available.</p>`}`;
      return;
    }

    const lines = data.lines || [];
    const predictionText = data.prediction_text ?? data.prediction ?? "";
    const predictionLines = data.prediction_lines ?? lines.map((l) => l.text);

    if (String(predictionText).trim() === "" && (!predictionLines || predictionLines.length === 0)) {
      resultBox.innerHTML = `
        <p class="error-text">No text recognized yet. Try thicker strokes or a clearer line.</p>
        ${analysisBlock}`;
      return;
    }

    const avgConf = lines.length
      ? lines.reduce((sum, l) => sum + (Number.isFinite(l.confidence) ? l.confidence : 0), 0) / lines.length
      : 0;

    const chips = (predictionLines.length ? predictionLines : []).map((text, idx) => {
      const line = lines[idx];
      const conf = line && Number.isFinite(line.confidence) ? line.confidence : 0;
      return `
        <span class="digit-chip">
          <span class="chip-digit">${escapeHtml(text)}</span>
          <span class="chip-conf">${Number(conf).toFixed(1)}%</span>
        </span>
      `;
    }).join("");

    resultBox.innerHTML = `
      <div class="result-main" style="white-space: pre-line;">${escapeHtml(predictionText)}</div>
      <div class="digit-chips">${chips}</div>
      <div class="conf-bar-wrap">
        <div class="conf-bar" style="width:${Math.max(0, Math.min(100, avgConf))}%"></div>
      </div>
      <div class="conf-label">Avg confidence: <strong>${Number(avgConf).toFixed(1)}%</strong></div>
      ${analysisBlock}
    `;
  }

  /** Reset placeholder (no data). */
  function showResult(data) {
    renderPredictionPanel(data);
  }

  /** Network / hard errors without JSON analysis. */
  function showError(msg) {
    resultBox.innerHTML = `
      <p class="error-banner">⚠ ${escapeHtml(msg)}</p>
      <div class="analysis-panel">
        <div class="analysis-heading">Output analysis</div>
        <div class="analysis-row analysis-row--tip">
          <span class="analysis-label">Suggestion</span>
          <span class="analysis-value">Confirm the Flask server is running and you are logged in, then try again.</span>
        </div>
      </div>`;
  }

  // ── Button listeners ──────────────────────────────────────────────────────
  predictBtn.addEventListener("click", sendPrediction);
  clearBtn.addEventListener("click",   clearCanvas);

  // ── Keyboard shortcut: Enter = Predict, Escape = Clear ──────────────────
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter")  sendPrediction();
    if (e.key === "Escape") clearCanvas();
  });

})();
