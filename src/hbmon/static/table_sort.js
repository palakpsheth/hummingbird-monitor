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

  function findHeaderIndex(headers, columnKey) {
    if (!columnKey) return -1;
    return headers.findIndex((header) => header.dataset.colKey === columnKey);
  }

  function sortTable(table, columnIndex, sortType, direction) {
    const tbody = table.querySelector("tbody");
    const rows = tbody
      ? Array.from(tbody.querySelectorAll("tr[data-sort-row]"))
      : Array.from(table.querySelectorAll(".row")).filter((row) => !row.classList.contains("head"));
    rows.sort((a, b) => {
      const aCell = a.children[columnIndex];
      const bCell = b.children[columnIndex];
      const aValue = parseValue(aCell?.dataset?.sortValue ?? aCell?.textContent ?? "", sortType);
      const bValue = parseValue(bCell?.dataset?.sortValue ?? bCell?.textContent ?? "", sortType);
      return compareValues(aValue, bValue, direction);
    });
    const container = tbody || table;
    rows.forEach((row) => container.appendChild(row));
  }

  function applySort(table, headers, columnIndex, direction) {
    const header = headers[columnIndex];
    if (!header) return;
    const sortType = header.dataset.sortType;
    if (!sortType) return;
    table.dataset.sortIndex = String(columnIndex);
    table.dataset.sortDirection = direction;
    setSortIndicators(headers, columnIndex, direction);
    sortTable(table, columnIndex, sortType, direction);
    const columnKey = header.dataset.colKey;
    if (columnKey) {
      table.dispatchEvent(
        new CustomEvent("table:sorted", { detail: { columnKey, direction } })
      );
    }
  }

  function setupSortControls(table, headers) {
    const tableId = table.dataset.tableId;
    if (!tableId) return;
    const controls = document.querySelectorAll(`[data-sort-controls][data-table-id=\"${tableId}\"]`);
    if (!controls.length) return;

    const sortIndex = Number.parseInt(table.dataset.sortIndex || "-1", 10);
    const activeIndex = Number.isNaN(sortIndex) ? -1 : sortIndex;
    const defaultIndex = headers.findIndex((header) => header.dataset.sortDefault);
    const initialIndex = activeIndex >= 0 ? activeIndex : defaultIndex >= 0 ? defaultIndex : -1;
    const initialDirection = table.dataset.sortDirection || "asc";

    const updateControls = (columnKey, direction) => {
      controls.forEach((control) => {
        const columnSelect = control.querySelector("[data-sort-column]");
        const directionSelect = control.querySelector("[data-sort-direction]");
        if (columnSelect && columnKey) {
          columnSelect.value = columnKey;
        }
        if (directionSelect && direction) {
          directionSelect.value = direction;
        }
      });
    };

    table.addEventListener("table:sorted", (event) => {
      const detail = event.detail || {};
      updateControls(detail.columnKey, detail.direction);
    });

    controls.forEach((control) => {
      const columnSelect = control.querySelector("[data-sort-column]");
      const directionSelect = control.querySelector("[data-sort-direction]");
      const applyButton = control.querySelector("[data-sort-apply]");

      if (columnSelect && initialIndex >= 0) {
        const key = headers[initialIndex]?.dataset?.colKey;
        if (key) {
          columnSelect.value = key;
        }
      }

      if (directionSelect) {
        directionSelect.value = initialDirection;
      }

      const apply = () => {
        const columnKey = columnSelect?.value;
        const columnIndex = findHeaderIndex(headers, columnKey);
        if (columnIndex < 0) return;
        const direction = directionSelect?.value || "asc";
        applySort(table, headers, columnIndex, direction);
      };

      if (columnSelect) {
        columnSelect.addEventListener("change", apply);
      }
      if (directionSelect) {
        directionSelect.addEventListener("change", apply);
      }
      if (applyButton) {
        applyButton.addEventListener("click", apply);
      }
    });

    if (initialIndex >= 0) {
      const key = headers[initialIndex]?.dataset?.colKey;
      updateControls(key, initialDirection);
    }
  }

  function setupTable(table) {
    const headRow = table.querySelector("thead tr");
    const gridHead = table.querySelector(".row.head");
    const headers = headRow
      ? Array.from(headRow.children)
      : gridHead
        ? Array.from(gridHead.children)
        : [];
    if (!headers.length) return;
    const defaultIndex = headers.findIndex((header) => header.dataset.sortDefault);
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
        applySort(table, headers, index, direction);
      });
    });
    if (defaultIndex >= 0) {
      const header = headers[defaultIndex];
      const sortType = header.dataset.sortType;
      if (sortType) {
        const direction = header.dataset.sortDefault || "asc";
        applySort(table, headers, defaultIndex, direction);
      }
    }
    setupSortControls(table, headers);
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
