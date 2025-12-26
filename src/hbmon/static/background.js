// src/hbmon/static/background.js
//
// Handles live background preview refresh and fallback behavior
// on the background configuration page.

(function () {
  const img = document.getElementById("live-background-frame");
  if (!img) {
    return;
  }

  const refreshBtn = document.getElementById("background-refresh");
  const captureBtn = document.getElementById("background-capture");
  const statusEl = document.getElementById("background-live-status");
  const liveSrc = img.dataset.liveSrc || "";
  const fallbackSrc = img.dataset.fallbackSrc || "";
  const rtspConfigured = img.dataset.rtspConfigured === "1";
  let usingFallback = !liveSrc;

  function setStatus(text) {
    if (!statusEl) {
      return;
    }
    statusEl.textContent = text;
  }

  function setCaptureEnabled(enabled) {
    if (!captureBtn) {
      return;
    }
    captureBtn.disabled = !enabled;
    if (captureBtn.disabled) {
      captureBtn.title = "Live snapshot capture is disabled until the preview is live.";
      captureBtn.setAttribute("aria-disabled", "true");
    } else {
      captureBtn.title = "";
      captureBtn.removeAttribute("aria-disabled");
    }
  }

  function stripQuery(src) {
    return src ? src.split("?")[0] : "";
  }

  function refreshFrame() {
    const base = liveSrc || fallbackSrc;
    if (!base) {
      return;
    }
    usingFallback = !liveSrc || base === fallbackSrc;
    if (!rtspConfigured || !liveSrc) {
      setCaptureEnabled(false);
      setStatus("Live preview unavailable until RTSP is configured.");
    }
    img.src = base + "?ts=" + Date.now();
  }

  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => {
      refreshFrame();
    });
  }

  if (captureBtn && captureBtn.form) {
    captureBtn.form.addEventListener("submit", (event) => {
      if (captureBtn.disabled) {
        event.preventDefault();
      }
    });
  }

  img.addEventListener("load", () => {
    if (!liveSrc || !rtspConfigured) {
      return;
    }
    if (!usingFallback) {
      setCaptureEnabled(true);
      setStatus("");
    }
  });

  img.addEventListener("error", () => {
    if (usingFallback || !fallbackSrc || stripQuery(img.src) === fallbackSrc) {
      return;
    }
    usingFallback = true;
    setCaptureEnabled(false);
    if (liveSrc) {
      setStatus("Live preview unavailable. Showing latest snapshot instead.");
    }
    img.src = fallbackSrc + "?ts=" + Date.now();
  });

  if (rtspConfigured && liveSrc) {
    setCaptureEnabled(false);
    setStatus("Loading live preview...");
  } else {
    setCaptureEnabled(false);
    if (!rtspConfigured) {
      setStatus("Live preview unavailable until RTSP is configured.");
    }
  }
})();
