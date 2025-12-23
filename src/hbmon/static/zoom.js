// src/hbmon/static/zoom.js
//
// Lightweight zoom controls for snapshot images.
// Adds zoom in/out/reset buttons for any container marked with data-zoom-container.
(function () {
  const MIN_SCALE = 0.5;
  const MAX_SCALE = 5;

  function clampScale(val) {
    if (!Number.isFinite(val)) return 1;
    return Math.min(MAX_SCALE, Math.max(MIN_SCALE, val));
  }

  function setupZoom(container) {
    const target = container.querySelector("[data-zoom-target]");
    if (!target) return;

    let scale = 1;

    function apply() {
      target.style.transform = `scale(${scale})`;
    }

    const zoomIn = () => {
      scale = clampScale(scale * 1.25);
      apply();
    };

    const zoomOut = () => {
      scale = clampScale(scale * 0.8);
      apply();
    };

    const reset = () => {
      scale = 1;
      apply();
    };

    container.querySelector("[data-zoom-in]")?.addEventListener("click", zoomIn);
    container.querySelector("[data-zoom-out]")?.addEventListener("click", zoomOut);
    container.querySelector("[data-zoom-reset]")?.addEventListener("click", reset);

    apply();
  }

  function init() {
    document.querySelectorAll("[data-zoom-container]").forEach(setupZoom);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
