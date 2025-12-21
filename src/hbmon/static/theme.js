// src/hbmon/static/theme.js
//
// Light/Dark mode toggle persisted in localStorage.
// Applies theme by setting <html data-theme="dark|light">.
//
// Default is dark unless user has explicitly chosen.

(function () {
  const KEY = "hbmon_theme"; // "dark" | "light"

  function getSavedTheme() {
    try {
      return localStorage.getItem(KEY);
    } catch (_) {
      return null;
    }
  }

  function setSavedTheme(theme) {
    try {
      localStorage.setItem(KEY, theme);
    } catch (_) {}
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    const btn = document.getElementById("themeToggle");
    if (btn) {
      btn.textContent = theme === "dark" ? "Light mode" : "Dark mode";
      btn.setAttribute("aria-label", theme === "dark" ? "Switch to light mode" : "Switch to dark mode");
    }
  }

  function init() {
    const saved = getSavedTheme();
    const theme = (saved === "light" || saved === "dark") ? saved : "dark";
    applyTheme(theme);

    const btn = document.getElementById("themeToggle");
    if (btn) {
      btn.addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme") || "dark";
        const next = current === "dark" ? "light" : "dark";
        setSavedTheme(next);
        applyTheme(next);
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
