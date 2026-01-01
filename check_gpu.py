from __future__ import annotations

"""
Legacy GPU/debugging helper (now intentionally a no-op).

This module previously contained experimental monkeypatching of
``openvino.runtime.Core.compile_model`` to force GPU usage when
running Ultralytics YOLO models exported to OpenVINO.

That behavior is not appropriate for production code because:

* It globally modified OpenVINO runtime behavior via monkeypatching.
* It forced device selection (CPU → GPU) without configuration.
* It relied on hard-coded paths and debug-oriented print statements.

To keep the codebase safe and predictable, all such behavior has been
removed. The module is retained only as a stub to avoid import errors
in case any external tooling still references ``check_gpu.py``.

If you need GPU diagnostics or benchmarking, create a separate,
clearly development-only script outside the main application package
and avoid global monkeypatching of OpenVINO or other libraries.
"""

if __name__ == "__main__":
    print(
        "check_gpu.py is a deprecated development-only helper and no longer "
        "performs any OpenVINO/YOLO monkeypatching or GPU checks."
    )
