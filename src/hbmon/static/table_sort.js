// src/hbmon/static/table_sort.js
//
// Lightweight sortable table behavior for hbmon grid tables.
(function () {
  function parseValue(value, type) {
    if (value === null || value === undefined) return null;
    const text = String(value).trim();
    if (!text) return null;
    if (type === "number") {
      const num = Number.parseFloat(text);
      return Number.isNaN(num) ? null : num;
    }
    if (type === "date") {
      const parsed = Date.parse(text);
      return Number.isNaN(parsed) ? null : parsed;
    }
    return text.toLowerCase();
  }

  function compareValues(aVal, bVal, direction) {
    if (aVal === null && bVal === null) return 0;
    if (aVal === null) return 1;
    if (bVal === null) return -1;
    if (aVal < bVal) return direction === "asc" ? -1 : 1;
    if (aVal > bVal) return direction === "asc" ? 1 : -1;
    return 0;
  }

  function setSortIndicators(headers, activeIndex, direction) {
    headers.forEach((header, index) => {
      if (!header.dataset.sortType) return;
      if (index === activeIndex) {
        header.dataset.sortDirection = direction;
      } else {
        delete header.dataset.sortDirection;
      }
    });
  }

  function sortTable(table, columnIndex, sortType, direction) {
    const rows = Array.from(table.querySelectorAll(".row")).filter((row) => !row.classList.contains("head"));
    rows.sort((a, b) => {
      const aCell = a.children[columnIndex];
      const bCell = b.children[columnIndex];
      const aValue = parseValue(aCell?.dataset?.sortValue ?? aCell?.textContent ?? "", sortType);
      const bValue = parseValue(bCell?.dataset?.sortValue ?? bCell?.textContent ?? "", sortType);
      return compareValues(aValue, bValue, direction);
    });
    rows.forEach((row) => table.appendChild(row));
  }

  function setupTable(table) {
    const head = table.querySelector(".row.head");
    if (!head) return;
    const headers = Array.from(head.children);
    headers.forEach((header, index) => {
      const sortType = header.dataset.sortType;
      if (!sortType) return;
      header.classList.add("sortable-header");
      header.addEventListener("click", () => {
        const currentIndex = Number.parseInt(table.dataset.sortIndex || "-1", 10);
        const isSameColumn = currentIndex === index;
        const defaultDirection = header.dataset.sortDefault || "asc";
        const direction = isSameColumn
          ? (table.dataset.sortDirection === "asc" ? "desc" : "asc")
          : defaultDirection;
        table.dataset.sortIndex = String(index);
        table.dataset.sortDirection = direction;
        setSortIndicators(headers, index, direction);
        sortTable(table, index, sortType, direction);
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
    document.querySelectorAll("[data-sortable-table]").forEach((table) => setupTable(table));
  });
})();
