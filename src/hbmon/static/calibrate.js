// src/hbmon/static/calibrate.js
//
// Handles interactive ROI selection on the calibration page.
// User drags a rectangle over the latest snapshot; coordinates
// are saved as normalized (0–1) values relative to image size.

(function () {
  const img = document.getElementById("frame");
  const box = document.getElementById("box");
  const refreshBtn = document.getElementById("refresh");

  if (!img || !box) {
    // Page not loaded or missing expected elements
    return;
  }

  let start = null;   // {x, y} in normalized coords
  let rect = null;    // {x, y, w, h} normalized

  function clamp01(v) {
    return Math.max(0, Math.min(1, v));
  }

  function imgRect() {
    return img.getBoundingClientRect();
  }

  function setHiddenInputs(x1, y1, x2, y2) {
    const elx1 = document.getElementById("x1");
    const ely1 = document.getElementById("y1");
    const elx2 = document.getElementById("x2");
    const ely2 = document.getElementById("y2");
    if (!elx1 || !ely1 || !elx2 || !ely2) return;

    elx1.value = x1.toFixed(6);
    ely1.value = y1.toFixed(6);
    elx2.value = x2.toFixed(6);
    ely2.value = y2.toFixed(6);
  }

  function drawBox(r) {
    box.style.display = "block";
    box.style.left = (r.x * 100) + "%";
    box.style.top = (r.y * 100) + "%";
    box.style.width = (r.w * 100) + "%";
    box.style.height = (r.h * 100) + "%";
  }

  img.addEventListener("pointerdown", (e) => {
    const b = imgRect();
    const x = clamp01((e.clientX - b.left) / b.width);
    const y = clamp01((e.clientY - b.top) / b.height);

    start = { x, y };
    rect = { x, y, w: 0, h: 0 };
    drawBox(rect);

    try {
      img.setPointerCapture(e.pointerId);
    } catch (_) {}
  });

  img.addEventListener("pointermove", (e) => {
    if (!start) return;

    const b = imgRect();
    const x = clamp01((e.clientX - b.left) / b.width);
    const y = clamp01((e.clientY - b.top) / b.height);

    const x1 = Math.min(start.x, x);
    const y1 = Math.min(start.y, y);
    const x2 = Math.max(start.x, x);
    const y2 = Math.max(start.y, y);

    rect = {
      x: x1,
      y: y1,
      w: x2 - x1,
      h: y2 - y1,
    };

    drawBox(rect);
    setHiddenInputs(x1, y1, x2, y2);
  });

  img.addEventListener("pointerup", (e) => {
    start = null;
    try {
      img.releasePointerCapture(e.pointerId);
    } catch (_) {}
  });

  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => {
      // Bust cache to force latest frame
      img.src = "/api/frame.jpg?ts=" + Date.now();
    });
  }
})();
