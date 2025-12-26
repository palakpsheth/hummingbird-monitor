// src/hbmon/static/observations_bulk.js
//
// Bulk selection controls for observation tables.
(function () {
  function updateSelectionState(form, table) {
    const checkboxes = Array.from(table.querySelectorAll("input[type=\"checkbox\"][data-select-row]"));
    const selectAll = table.querySelector("input[type=\"checkbox\"][data-select-all]");
    const deleteButton = form.querySelector("[data-bulk-delete-button]");
    const countLabel = form.querySelector("[data-bulk-count]");
    const selectedCount = checkboxes.filter((checkbox) => checkbox.checked).length;

    if (deleteButton) {
      deleteButton.disabled = selectedCount === 0;
    }
    if (countLabel) {
      countLabel.textContent = String(selectedCount);
    }

    if (selectAll) {
      if (selectedCount === 0) {
        selectAll.checked = false;
        selectAll.indeterminate = false;
      } else if (selectedCount === checkboxes.length) {
        selectAll.checked = true;
        selectAll.indeterminate = false;
      } else {
        selectAll.checked = false;
        selectAll.indeterminate = true;
      }
    }
  }

  function setupBulkControls(form) {
    const tableId = form.dataset.tableId;
    if (!tableId) return;
    const table = document.querySelector(`[data-table-id="${tableId}"]`);
    if (!table) return;

    const checkboxes = Array.from(table.querySelectorAll("input[type=\"checkbox\"][data-select-row]"));
    const selectAll = table.querySelector("input[type=\"checkbox\"][data-select-all]");

    if (selectAll) {
      selectAll.addEventListener("change", () => {
        checkboxes.forEach((checkbox) => {
          checkbox.checked = selectAll.checked;
        });
        updateSelectionState(form, table);
      });
    }

    checkboxes.forEach((checkbox) => {
      checkbox.addEventListener("change", () => updateSelectionState(form, table));
    });

    updateSelectionState(form, table);
  }

  function onReady(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
    } else {
      fn();
    }
  }

  onReady(() => {
    document.querySelectorAll("[data-bulk-delete]").forEach((form) => setupBulkControls(form));
  });
})();
