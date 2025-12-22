// src/hbmon/static/timestamps.js
//
// Convert UTC ISO timestamps in data-utc-ts attributes to the user's local time.
(function () {
  function formatLocal(iso) {
    if (!iso) return "";
    let txt = String(iso).trim();
    if (!/[zZ]|[+-]\d{2}:?\d{2}$/.test(txt)) {
      txt = txt.replace(" ", "T") + "Z";
    }
    const d = new Date(txt);
    if (Number.isNaN(d.getTime())) {
      return iso;
    }
    return d.toLocaleString(undefined, {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      timeZoneName: "short",
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

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", applyLocalTimestamps);
  } else {
    applyLocalTimestamps();
  }
})();
