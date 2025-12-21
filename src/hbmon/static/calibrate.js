// src/hbmon/static/calibrate.js
//
// Handles interactive ROI selection on the calibration page.
// User drags a rectangle over the latest snapshot; coordinates
// are saved as normalized (0–1) values relative to image size.
//
// Features:
// - Red dashed box shows current ROI (if set)
// - Green dashed box shows proposed new ROI
// - Image dragging is disabled
// - Proposed ROI persists until Save or Clear Selection

(function () {
  const img = document.getElementById("frame");
  const currentRoiBox = document.getElementById("current-roi-box");
  const proposedRoiBox = document.getElementById("proposed-roi-box");
  const refreshBtn = document.getElementById("refresh");
  const clearBtn = document.getElementById("clear-selection");

  if (!img || !currentRoiBox || !proposedRoiBox) {
    // Page not loaded or missing expected elements
    return;
  }

  let start = null;   // {x, y} in normalized coords during drawing
  let proposedRect = null;    // {x, y, w, h} normalized for proposed ROI

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

  function drawBox(element, r) {
    element.style.display = "block";
    element.style.left = (r.x * 100) + "%";
    element.style.top = (r.y * 100) + "%";
    element.style.width = (r.w * 100) + "%";
    element.style.height = (r.h * 100) + "%";
  }

  function hideBox(element) {
    element.style.display = "none";
  }

  function clearProposedRoi() {
    proposedRect = null;
    hideBox(proposedRoiBox);
    // Clear hidden inputs
    setHiddenInputs(0, 0, 0, 0);
  }

  // Draw current ROI on page load (red dashed box)
  function drawCurrentRoi() {
    const elx1 = document.getElementById("x1");
    const ely1 = document.getElementById("y1");
    const elx2 = document.getElementById("x2");
    const ely2 = document.getElementById("y2");
    
    if (!elx1 || !ely1 || !elx2 || !ely2) return;
    
    const x1 = parseFloat(elx1.value);
    const y1 = parseFloat(ely1.value);
    const x2 = parseFloat(elx2.value);
    const y2 = parseFloat(ely2.value);
    
    // Only draw if we have valid ROI coordinates
    if (!isNaN(x1) && !isNaN(y1) && !isNaN(x2) && !isNaN(y2) && 
        x1 > 0 && y1 > 0 && x2 > 0 && y2 > 0) {
      const r = {
        x: x1,
        y: y1,
        w: x2 - x1,
        h: y2 - y1
      };
      drawBox(currentRoiBox, r);
    }
  }

  // Prevent default drag behavior on image
  img.addEventListener("dragstart", (e) => {
    e.preventDefault();
  });

  img.addEventListener("pointerdown", (e) => {
    // Clear any previous proposed ROI when starting a new selection
    clearProposedRoi();

    const b = imgRect();
    const x = clamp01((e.clientX - b.left) / b.width);
    const y = clamp01((e.clientY - b.top) / b.height);

    start = { x, y };
    proposedRect = { x, y, w: 0, h: 0 };
    drawBox(proposedRoiBox, proposedRect);

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

    proposedRect = {
      x: x1,
      y: y1,
      w: x2 - x1,
      h: y2 - y1,
    };

    drawBox(proposedRoiBox, proposedRect);
    setHiddenInputs(x1, y1, x2, y2);
  });

  img.addEventListener("pointerup", (e) => {
    start = null;
    // Keep proposedRect and the green box visible
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

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      clearProposedRoi();
    });
  }

  // Draw current ROI on page load
  drawCurrentRoi();
})();
