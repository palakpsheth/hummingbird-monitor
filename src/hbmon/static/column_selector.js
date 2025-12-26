// src/hbmon/static/column_selector.js
//
// Column visibility controls for observation tables.
(function () {
  const STORAGE_PREFIX = "hbmon:columns:";

  function getStorageKey(tableId) {
    return `${STORAGE_PREFIX}${tableId}`;
  }

  function safeParse(value) {
    try {
      return JSON.parse(value);
    } catch (error) {
      return null;
    }
  }

  function setVisibility(table, columnKey, isVisible) {
    table.querySelectorAll(`[data-col-key="${columnKey}"]`).forEach((cell) => {
      cell.classList.toggle("col-hidden", !isVisible);
    });
  }

  function readStoredState(tableId) {
    const raw = window.localStorage.getItem(getStorageKey(tableId));
    if (!raw) return null;
    const parsed = safeParse(raw);
    return parsed && typeof parsed === "object" ? parsed : null;
  }

  function persistState(tableId, state) {
    window.localStorage.setItem(getStorageKey(tableId), JSON.stringify(state));
  }

  function setupColumnControls(control) {
    const tableId = control.dataset.tableId;
    if (!tableId) return;
    const table = document.querySelector(`[data-table-id="${tableId}"]`);
    if (!table) return;
    const storedState = readStoredState(tableId) || {};
    const checkboxes = Array.from(control.querySelectorAll("input[type=\"checkbox\"][data-column-key]"));
    if (!checkboxes.length) return;

    checkboxes.forEach((checkbox) => {
      const columnKey = checkbox.dataset.columnKey;
      if (!columnKey) return;
      if (Object.prototype.hasOwnProperty.call(storedState, columnKey)) {
        checkbox.checked = Boolean(storedState[columnKey]);
      }
      setVisibility(table, columnKey, checkbox.checked);
      checkbox.addEventListener("change", () => {
        const nextState = readStoredState(tableId) || {};
        nextState[columnKey] = checkbox.checked;
        persistState(tableId, nextState);
        setVisibility(table, columnKey, checkbox.checked);
      });
    });
  }

  function onReady(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
    } else {
      fn();
    }
  }

  onReady(() => {
    document.querySelectorAll("[data-column-controls]").forEach((control) => setupColumnControls(control));
  });
})();
