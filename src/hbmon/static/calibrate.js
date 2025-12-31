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

  function snapToStride32(normVal, totalPx) {
    if (!totalPx || totalPx <= 0) return normVal;
    // Calculate raw pixels
    const rawPx = Math.abs(normVal * totalPx);
    // Round up to nearest 32
    // Math.ceil(0) is 0, so we handle that naturally
    let snappedPx = Math.ceil(rawPx / 32) * 32;

    // Optional: if user is trying to make it very small but non-zero,
    // ensure at least 32px? The Math.ceil handles anything > 0 to become 32.
    // However, if rawPx was 0 (start point), snappedPx is 0.

    // Constraint: don't exceed image dimension
    if (snappedPx > totalPx) {
      snappedPx = Math.floor(totalPx / 32) * 32;
    }
    return snappedPx / totalPx;
  }

  function imgRect() {
    return img.getBoundingClientRect();
  }

  function getRoiInputs() {
    return {
      x1: document.getElementById("x1"),
      y1: document.getElementById("y1"),
      x2: document.getElementById("x2"),
      y2: document.getElementById("y2"),
      label: document.getElementById("proposed-roi-label"),
    };
  }

  function setHiddenInputs(x1, y1, x2, y2) {
    const inputs = getRoiInputs();
    if (!inputs.x1 || !inputs.y1 || !inputs.x2 || !inputs.y2) return;

    inputs.x1.value = x1.toFixed(6);
    inputs.y1.value = y1.toFixed(6);
    inputs.x2.value = x2.toFixed(6);
    inputs.y2.value = y2.toFixed(6);

    if (inputs.label) {
      inputs.label.textContent = `New ROI: ${inputs.x1.value}, ${inputs.y1.value}, ${inputs.x2.value}, ${inputs.y2.value}`;
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
    const inputs = getRoiInputs();
    if (inputs.x1 && inputs.y1 && inputs.x2 && inputs.y2) {
      inputs.x1.value = "";
      inputs.y1.value = "";
      inputs.x2.value = "";
      inputs.y2.value = "";
    }
    if (inputs.label) {
      inputs.label.textContent = "";
    }
  }

  function readRoiFromDom() {
    const inputs = getRoiInputs();
    if (!inputs.x1 || !inputs.y1 || !inputs.x2 || !inputs.y2) return null;

    const x1 = parseFloat(inputs.x1.value);
    const y1 = parseFloat(inputs.y1.value);
    const x2 = parseFloat(inputs.x2.value);
    const y2 = parseFloat(inputs.y2.value);

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

    const nw = img.naturalWidth || 1000;
    const nh = img.naturalHeight || 1000;

    let x1, x2, y1, y2;

    // Calculate width relative to start x
    const rawW = Math.abs(x - start.x);
    const snappedW = snapToStride32(rawW, nw);

    if (x < start.x) {
      // Dragging left
      x2 = start.x;
      x1 = Math.max(0, x2 - snappedW);
    } else {
      // Dragging right
      x1 = start.x;
      x2 = Math.min(1, x1 + snappedW);
    }

    // Calculate height relative to start y
    const rawH = Math.abs(y - start.y);
    const snappedH = snapToStride32(rawH, nh);

    if (y < start.y) {
      // Dragging up
      y2 = start.y;
      y1 = Math.max(0, y2 - snappedH);
    } else {
      // Dragging down
      y1 = start.y;
      y2 = Math.min(1, y1 + snappedH);
    }

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
      const nw = img.naturalWidth || 1000;
      const nh = img.naturalHeight || 1000;

      // Determine which edge is moving and snap the resulting dimension
      if (resizeState.edge === "n") {
        // Changing top edge (y1). Fixed bottom is y2.
        // New height = y2 - y
        let newH = y2 - y;
        if (newH < 0) newH = 0;
        let snappedH = snapToStride32(newH, nh);
        y1 = Math.max(0, y2 - snappedH);

      } else if (resizeState.edge === "s") {
        // Changing bottom edge (y2). Fixed top is y1.
        let newH = y - y1;
        let snappedH = snapToStride32(newH, nh);
        y2 = Math.min(1, y1 + snappedH);

      } else if (resizeState.edge === "w") {
        // Changing left edge (x1). Fixed right is x2.
        let newW = x2 - x;
        let snappedW = snapToStride32(newW, nw);
        x1 = Math.max(0, x2 - snappedW);

      } else if (resizeState.edge === "e") {
        // Changing right edge (x2). Fixed left is x1.
        let newW = x - x1;
        let snappedW = snapToStride32(newW, nw);
        x2 = Math.min(1, x1 + snappedW);
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
