// src/hbmon/static/timestamps.js
//
// Convert UTC ISO timestamps in data-utc-ts attributes to the user's local time.
(function () {
  const body = document.body || document.documentElement;

  function getConfiguredTimezone() {
    const tz = (body?.dataset?.hbmonTz || "").trim();
    if (!tz || tz.toLowerCase() === "local") return null;
    try {
      // Validate timezone; throws RangeError if invalid
      new Intl.DateTimeFormat(undefined, { timeZone: tz });
      return tz;
    } catch (err) {
      return null;
    }
  }

  function formatWithOptions(date, options) {
    try {
      return date.toLocaleString(undefined, options);
    } catch (err) {
      if (options && options.timeZone) {
        const fallback = { ...options };
        delete fallback.timeZone;
        try {
          return date.toLocaleString(undefined, fallback);
        } catch (err2) {
          return date.toString();
        }
      }
      return date.toString();
    }
  }

  function formatLocal(iso) {
    if (!iso) return "";
    let txt = String(iso).trim();
    if (!/[zZ]$|[+-]\d{2}:\d{2}$|[+-]\d{4}$/.test(txt)) {
      txt = txt.replace(" ", "T") + "Z";
    }
    const d = new Date(txt);
    if (Number.isNaN(d.getTime())) {
      return iso;
    }
    const tz = getConfiguredTimezone();
    return formatWithOptions(d, {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      timeZoneName: "short",
      timeZone: tz || undefined,
    });
  }

  function applyLocalTimestamps() {
    const nodes = document.querySelectorAll("[data-utc-ts]");
    nodes.forEach((el) => {
      const iso = el.getAttribute("data-utc-ts");
      if (!iso || iso === "") return;
      el.textContent = formatLocal(iso);
    });
  }

  function updateCurrentTime() {
    const node = document.getElementById("footer-current-time");
    if (!node) return;
    const tz = getConfiguredTimezone();
    const now = new Date();
    node.textContent = formatWithOptions(now, {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      timeZoneName: "short",
      timeZone: tz || undefined,
    });
  }

  function onReady(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
    } else {
      fn();
    }
  }

  onReady(applyLocalTimestamps);
  onReady(updateCurrentTime);
  setInterval(updateCurrentTime, 1000);
})();
