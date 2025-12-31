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
  if (!img || !currentRoiBox || !proposedRoiBox) {
    // Page not loaded or missing expected elements
    return;
  }

  const refreshBtn = document.getElementById("refresh");
  const clearBtn = document.getElementById("clear-selection");
  const liveSrc = img.dataset.liveSrc || "";
  const fallbackSrc = img.dataset.fallbackSrc || "";
  let usingFallback = !liveSrc;

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
    const label = document.getElementById("proposed-roi-label");
    if (!elx1 || !ely1 || !elx2 || !ely2) return;

    elx1.value = x1.toFixed(6);
    ely1.value = y1.toFixed(6);
    elx2.value = x2.toFixed(6);
    ely2.value = y2.toFixed(6);

    if (label) {
      label.textContent = `New ROI: ${elx1.value}, ${ely1.value}, ${elx2.value}, ${ely2.value}`;
    }
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
    // Clear hidden inputs by setting them to empty strings
    const elx1 = document.getElementById("x1");
    const ely1 = document.getElementById("y1");
    const elx2 = document.getElementById("x2");
    const ely2 = document.getElementById("y2");
    const label = document.getElementById("proposed-roi-label");
    if (elx1 && ely1 && elx2 && ely2) {
      elx1.value = "";
      ely1.value = "";
      elx2.value = "";
      ely2.value = "";
    }
    if (label) {
      label.textContent = "";
    }
  }

  function readRoiFromDom() {
    const elx1 = document.getElementById("x1");
    const ely1 = document.getElementById("y1");
    const elx2 = document.getElementById("x2");
    const ely2 = document.getElementById("y2");

    if (!elx1 || !ely1 || !elx2 || !ely2) return null;

    const x1 = parseFloat(elx1.value);
    const y1 = parseFloat(ely1.value);
    const x2 = parseFloat(elx2.value);
    const y2 = parseFloat(ely2.value);

    if (
      !isNaN(x1) && !isNaN(y1) && !isNaN(x2) && !isNaN(y2) &&
      x2 > x1 && y2 > y1
    ) {
      return {
        x: x1,
        y: y1,
        w: x2 - x1,
        h: y2 - y1
      };
    }
    return null;
  }

  // Draw current ROI on page load (red dashed box)
  function drawCurrentRoi() {
    const r = readRoiFromDom();
    if (r) {
      drawBox(currentRoiBox, r);
    }
  }

  function stripQuery(src) {
    return src ? src.split("?")[0] : "";
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
    } catch (_) { }
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
    } catch (_) { }
  });

  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => {
      const base = liveSrc || fallbackSrc;
      if (base) {
        usingFallback = !liveSrc || base === fallbackSrc;
        img.src = base + "?ts=" + Date.now();
      }
    });
  }

  img.addEventListener("error", () => {
    if (usingFallback || !fallbackSrc || stripQuery(img.src) === fallbackSrc) {
      return;
    }
    usingFallback = true;
    img.src = fallbackSrc + "?ts=" + Date.now();
  });

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      clearProposedRoi();
    });
  }

  // ---------------------------------------------------------
  // Resize Handles Logic
  // ---------------------------------------------------------
  let resizeState = null;

  const handles = proposedRoiBox.querySelectorAll(".resize-handle");
  handles.forEach((h) => {
    h.addEventListener("pointerdown", (e) => {
      if (!proposedRect) return;
      e.preventDefault();
      // Stop propagation so we don't trigger a new box draw on the underlying image
      e.stopPropagation();

      const edge = h.dataset.edge;
      resizeState = {
        edge,
        // Snapshot current known coords
        x1: proposedRect.x,
        y1: proposedRect.y,
        x2: proposedRect.x + proposedRect.w,
        y2: proposedRect.y + proposedRect.h,
      };

      h.classList.add("active");
      try {
        h.setPointerCapture(e.pointerId);
      } catch (_) { }
    });

    h.addEventListener("pointermove", (e) => {
      if (!resizeState) return;
      e.preventDefault();
      e.stopPropagation();

      const b = imgRect();
      const x = clamp01((e.clientX - b.left) / b.width);
      const y = clamp01((e.clientY - b.top) / b.height);

      let { x1, y1, x2, y2 } = resizeState;

      // Adjust the relevant boundary, preventing crossover
      // (min/max checks ensure we don't invert the box)
      const minSize = 0.005; // minimum size constraint

      if (resizeState.edge === "n") {
        y1 = Math.min(y, y2 - minSize);
      } else if (resizeState.edge === "s") {
        y2 = Math.max(y, y1 + minSize);
      } else if (resizeState.edge === "w") {
        x1 = Math.min(x, x2 - minSize);
      } else if (resizeState.edge === "e") {
        x2 = Math.max(x, x1 + minSize);
      }

      // Update global proposedRect
      proposedRect = {
        x: x1,
        y: y1,
        w: x2 - x1,
        h: y2 - y1,
      };

      drawBox(proposedRoiBox, proposedRect);
      setHiddenInputs(x1, y1, x2, y2);
    });

    h.addEventListener("pointerup", (e) => {
      if (!resizeState) return;

      const hEl = e.target;
      hEl.classList.remove("active");
      try {
        hEl.releasePointerCapture(e.pointerId);
      } catch (_) { }

      resizeState = null;
    });
  });

  // Initialize proposed ROI from current ROI (if any) so it can be tweaked immediately
  function initProposedRoi() {
    const r = readRoiFromDom();
    if (r) {
      proposedRect = r;
      drawBox(proposedRoiBox, proposedRect);
    }
  }

  // Draw current ROI on page load (red dashed box)
  drawCurrentRoi();
  // Also populate the green editable box
  initProposedRoi();
})();
