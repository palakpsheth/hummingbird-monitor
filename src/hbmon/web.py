# src/hbmon/web.py
"""
FastAPI + Jinja2 web UI for hbmon (LAN-only, no auth).

Routes:
- /                      Dashboard
- /observations           Gallery + filters
- /observations/{id}      Observation detail
- /individuals            Individuals list
- /individuals/{id}       Individual detail + heatmap + rename
- /individuals/{id}/split_review  Suggest A/B split & review UI
- /individuals/{id}/split_apply   Apply split assignments
- /calibrate              ROI calibration page

API:
- /api/health
- /api/frame.jpg          Latest snapshot (or placeholder)
- /api/roi  (GET/POST)    Get/set ROI (POST accepts form)

Exports:
- /export/observations.csv
- /export/individuals.csv
- /export/media_bundle.tar.gz

Notes:
- Media is served from HBMON_MEDIA_DIR (default /media)
- DB is served from HBMON_DATA_DIR (default /data)
"""

from __future__ import annotations

import csv
import json
from json import JSONDecodeError
import io
import math
import os
import subprocess
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

"""
FastAPI web application for hbmon.

This module defines the web routes and API for the hummingbird monitor.  It
attempts to import FastAPI and SQLAlchemy at runtime.  If either of those
dependencies is missing, the :func:`make_app` function will raise a
``RuntimeError`` when called.  This allows the rest of the package to be
imported in minimal environments without installing heavy dependencies.
"""

try:
    from fastapi import Depends, FastAPI, Form, HTTPException, Request  # type: ignore
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse  # type: ignore
    from fastapi.staticfiles import StaticFiles  # type: ignore
    from fastapi.templating import Jinja2Templates  # type: ignore
    _FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover
    # FastAPI is not available; define stubs to allow import but not use.
    class _StubExc(Exception):
        pass

    Depends = FastAPI = Form = Request = object  # type: ignore
    HTTPException = _StubExc  # type: ignore
    FileResponse = HTMLResponse = RedirectResponse = StreamingResponse = object  # type: ignore
    StaticFiles = object  # type: ignore
    Jinja2Templates = object  # type: ignore
    _FASTAPI_AVAILABLE = False

try:
    from sqlalchemy import delete, desc, func, select  # type: ignore
    from sqlalchemy.exc import OperationalError  # type: ignore
    from sqlalchemy.orm import Session  # type: ignore
    _SQLA_AVAILABLE = True
except Exception:  # pragma: no cover
    delete = desc = func = select = None  # type: ignore
    Session = object  # type: ignore
    OperationalError = Exception  # type: ignore
    _SQLA_AVAILABLE = False

ALLOWED_REVIEW_LABELS = ["true_positive", "false_positive", "false_negative"]


from hbmon import __version__
from hbmon.config import Roi, ensure_dirs, load_settings, media_dir, roi_to_str, save_settings
from hbmon.db import get_db, init_db
from hbmon.models import Embedding, Individual, Observation
from hbmon.schema import HealthOut, RoiOut
from hbmon.clustering import l2_normalize, suggest_split_two_groups

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Derived from this module path; not user-controlled, safe for git cwd.
_GIT_PATH = shutil.which("git")


def _normalize_timezone(tz: str | None) -> str:
    txt = (tz or "").strip()
    return txt or "local"


def _read_git_head(repo_root: Path) -> str | None:
    git_path = repo_root / ".git"
    git_dir = git_path
    if git_path.is_file():
        try:
            data = git_path.read_text().strip()
        except OSError:
            return None
        if data.startswith("gitdir:"):
            rel = data.partition(":")[2].strip()
            git_dir = (git_path.parent / rel).resolve()
        else:
            return None
    head_path = git_dir / "HEAD"
    try:
        head = head_path.read_text().strip()
    except OSError:
        return None
    if not head:
        return None
    if head.startswith("ref:"):
        ref = head.partition(" ")[2].strip()
        if not ref:
            return None
        ref_path = git_dir / ref
        try:
            commit = ref_path.read_text().strip()
        except OSError:
            packed = git_dir / "packed-refs"
            try:
                with packed.open() as pf:
                    for line in pf:
                        txt = line.strip()
                        if not txt or txt.startswith("#") or txt.startswith("^"):
                            continue
                        parts = txt.split(" ", 1)
                        if len(parts) == 2 and parts[1] == ref:
                            commit = parts[0]
                            break
                    else:
                        commit = ""
            except OSError:
                commit = ""
        return commit[:7] if commit else None
    return head[:7]


def _timezone_label(tz: str | None) -> str:
    clean = _normalize_timezone(tz)
    return "Browser local" if clean.lower() == "local" else clean


def _get_git_commit() -> str:
    env_commit = os.getenv("HBMON_GIT_COMMIT")
    if env_commit:
        return env_commit
    if _REPO_ROOT.is_dir() and _GIT_PATH is not None:
        try:
            commit = subprocess.check_output(
                [_GIT_PATH, "rev-parse", "--short", "HEAD"],
                cwd=_REPO_ROOT,
                timeout=1.0,
                shell=False,
                text=True,
            )
            cleaned = commit.strip()
            if cleaned:
                return cleaned
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            # Fallback to file-based parsing when the git CLI is unavailable or fails.
            pass
    head_commit = _read_git_head(_REPO_ROOT)
    if head_commit:
        return head_commit
    return "unknown"


_GIT_COMMIT = _get_git_commit()


# ----------------------------
# Presentation helpers
# ----------------------------

def species_to_css(label: str) -> str:
    s = (label or "").strip().lower().replace("’", "'")
    if "anna" in s:
        return "species-anna"
    if "allen" in s:
        return "species-allens"
    if "rufous" in s:
        return "species-rufous"
    if "costa" in s:
        return "species-costas"
    if "black" in s and "chinned" in s:
        return "species-black-chinned"
    if "calliope" in s:
        return "species-calliope"
    if "broad" in s and "billed" in s:
        return "species-broad-billed"
    return "species-unknown"


def build_hour_heatmap(hours_rows: list[tuple[int, int]]) -> list[dict[str, int]]:
    """
    hours_rows: [(hour_int, count_int), ...]
    Returns 24 dicts: {"hour": h, "count": c, "level": 0..5}
    """
    counts = {int(h): int(c) for (h, c) in hours_rows}
    vals = [counts.get(h, 0) for h in range(24)]
    mx = max(vals) if vals else 0

    def lvl(c: int) -> int:
        if c <= 0 or mx <= 0:
            return 0
        frac = c / mx
        if frac <= 0.20:
            return 1
        if frac <= 0.40:
            return 2
        if frac <= 0.60:
            return 3
        if frac <= 0.80:
            return 4
        return 5

    return [{"hour": h, "count": counts.get(h, 0), "level": lvl(counts.get(h, 0))} for h in range(24)]


def pretty_json(text: str | None) -> str | None:
    """
    Best-effort pretty formatting of a JSON string.

    Returns the original text if parsing fails.
    """
    if not text:
        return None
    try:
        obj = json.loads(text)
    except JSONDecodeError:
        return text
    try:
        return json.dumps(obj, indent=4, sort_keys=True)
    except (TypeError, ValueError):
        return text


def _as_utc_str(dt) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_detection_inputs(raw: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    """
    Validate and coerce detection/ML tuning inputs from the config form.

    Parameters
    ----------
    raw: dict[str, str]
        Form values keyed by the expected field names. Values are parsed as:
        - detect_conf, detect_iou: float in [0.05, 0.95]
        - min_box_area: int in [1, 200000]
        - cooldown_seconds: float in [0.0, 120.0]
        - min_species_prob, match_threshold, ema_alpha: float in [0.0, 1.0]

    Returns
    -------
    tuple[dict[str, Any], list[str]]
        Parsed numeric values (floats/ints) and a list of validation error
        messages such as "Detection confidence must be a number." or
        "Minimum box area must be between 1 and 200000.".
    """
    parsed: dict[str, Any] = {}
    errors: list[str] = []

    def parse_float(key: str, label: str, lo: float, hi: float) -> None:
        text = str(raw.get(key, "")).strip()
        try:
            val = float(text)
        except ValueError:
            errors.append(f"{label} must be a number.")
            return
        if not (lo <= val <= hi):
            errors.append(f"{label} must be between {lo} and {hi}.")
            return
        parsed[key] = val

    def parse_int(key: str, label: str, lo: int, hi: int) -> None:
        text = str(raw.get(key, "")).strip()
        try:
            val_float = float(text)
        except ValueError:
            errors.append(f"{label} must be a whole number.")
            return
        if not val_float.is_integer():
            errors.append(f"{label} must be a whole number.")
            return
        val = int(val_float)
        if not (lo <= val <= hi):
            errors.append(f"{label} must be between {lo} and {hi}.")
            return
        parsed[key] = val

    parse_float("detect_conf", "Detection confidence", 0.05, 0.95)
    parse_float("detect_iou", "IOU threshold", 0.05, 0.95)
    parse_int("min_box_area", "Minimum box area", 1, 200000)
    parse_float("cooldown_seconds", "Cooldown seconds", 0.0, 120.0)
    parse_float("min_species_prob", "Minimum species probability", 0.0, 1.0)
    parse_float("match_threshold", "Match threshold", 0.0, 1.0)
    parse_float("ema_alpha", "EMA alpha", 0.0, 1.0)

    tz_text = str(raw.get("timezone", "")).strip()
    if not tz_text:
        parsed["timezone"] = "local"
    else:
        tz_clean = _normalize_timezone(tz_text)
        if tz_clean.lower() == "local":
            parsed["timezone"] = "local"
        else:
            try:
                ZoneInfo(tz_clean)
                parsed["timezone"] = tz_clean
            except ZoneInfoNotFoundError:
                errors.append("Timezone must be a valid IANA name (e.g., America/Los_Angeles) or 'local'.")

    return parsed, errors


def paginate(total_count: int, page: int, page_size: int, max_page_size: int = 100) -> tuple[int, int, int, int]:
    """
    Clamp page/page_size and return (page, page_size, total_pages, offset).
    total_pages is at least 1 even when there are zero rows.
    """
    safe_total = max(0, int(total_count))
    size = max(1, min(int(page_size), max_page_size))
    total_pages = max(1, math.ceil(safe_total / size))
    current = max(1, min(int(page), total_pages))
    offset = (current - 1) * size
    return current, size, total_pages, offset


# ----------------------------
# App factory
# ----------------------------

def make_app() -> Any:
    """
    Create and return a FastAPI application configured with routes and static
    file mounts.  This function will raise ``RuntimeError`` if either
    FastAPI or SQLAlchemy is unavailable.  The return type is ``Any`` to
    avoid import-time type errors when the dependencies are missing.
    """
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is not installed; cannot create the web application."
        )
    if not _SQLA_AVAILABLE:
        raise RuntimeError(
            "SQLAlchemy is not installed; cannot create the web application."
        )

    ensure_dirs()
    init_db()

    app = FastAPI(title="hbmon")

    here = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(here / "templates"))

    # Static assets (CSS/JS)
    app.mount("/static", StaticFiles(directory=str(here / "static")), name="static")

    # Media (snapshots/clips)
    mdir = media_dir()
    mdir.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=str(mdir)), name="media")

    def _safe_unlink_media(rel_path: str | None) -> None:
        if not rel_path:
            return
        p = media_dir() / rel_path
        try:
            p.unlink(missing_ok=True)
        except Exception:
            # Best-effort cleanup; log for visibility but do not block user action.
            print(f"[web] failed to remove media file {p}")

    def _recompute_individual_stats(db: Session, individual_id: int) -> None:
        """Update visit_count/last_seen_at for an individual. Caller must commit."""
        ind = db.get(Individual, individual_id)
        if ind is None:
            print(f"[web] individual {individual_id} not found while recomputing stats")
            return
        rows = db.execute(
            select(func.count(Observation.id), func.max(Observation.ts))
            .where(Observation.individual_id == individual_id)
        ).one()
        ind.visit_count = int(rows[0] or 0)
        ind.last_seen_at = rows[1]

    def _commit_with_retry(db: Session, retries: int = 3, delay: float = 0.5) -> None:
        for i in range(max(1, retries)):
            try:
                db.commit()
                return
            except OperationalError as e:  # pragma: no cover
                msg = str(e).lower()
                if "database is locked" in msg and i < retries - 1:
                    print(f"[web] commit retry due to lock (attempt {i + 1}/{retries})")
                    time.sleep(delay)
                    continue
                raise

    # ----------------------------
    # UI routes
    # ----------------------------

    def _config_form_values(settings, raw: dict[str, str] | None = None) -> dict[str, str]:
        vals = {
            "detect_conf": f"{float(settings.detect_conf):.2f}",
            "detect_iou": f"{float(settings.detect_iou):.2f}",
            "min_box_area": str(int(settings.min_box_area)),
            "cooldown_seconds": f"{float(settings.cooldown_seconds):.2f}",
            "min_species_prob": f"{float(settings.min_species_prob):.2f}",
            "match_threshold": f"{float(settings.match_threshold):.2f}",
            "ema_alpha": f"{float(settings.ema_alpha):.2f}",
            "timezone": str(getattr(settings, "timezone", "local")),
        }
        if raw:
            for k, v in raw.items():
                if k in vals:
                    vals[k] = str(v)
        return vals

    def _context(request: Request, title: str, settings=None, **extra: Any) -> dict[str, Any]:
        s_local = settings or load_settings()
        tz_value = _normalize_timezone(getattr(s_local, "timezone", "local"))
        base = {
            "request": request,
            "title": title,
            "timezone": tz_value,
            "timezone_label": _timezone_label(tz_value),
            "app_version": __version__,
            "git_commit": _GIT_COMMIT,
        }
        base.update(extra)
        return base

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        page: int = 1,
        page_size: int = 10,
        db: Session = Depends(get_db),
    ) -> HTMLResponse:
        s = load_settings()
        title = "Hummingbird Monitor"

        top_inds = db.execute(
            select(Individual.id, Individual.name, Individual.visit_count, Individual.last_seen_at)
            .order_by(desc(Individual.visit_count))
            .limit(20)
        ).all()

        # Convert last_seen to ISO for template
        top_inds_out: list[tuple[int, str, int, str | None]] = []
        for iid, name, visits, last_seen in top_inds:
            top_inds_out.append((int(iid), str(name), int(visits), _as_utc_str(last_seen)))

        total_recent = db.execute(select(func.count(Observation.id))).scalar_one()
        current_page, clamped_page_size, total_pages, offset = paginate(
            total_recent, page=page, page_size=page_size, max_page_size=200
        )

        recent = (
            db.execute(
                select(Observation)
                .order_by(desc(Observation.ts))
                .offset(offset)
                .limit(clamped_page_size)
            )
            .scalars()
            .all()
        )

        for o in recent:
            # attach computed presentation attrs (not in DB)
            o.species_css = species_to_css(o.species_label)  # type: ignore[attr-defined]

        latest_ts = db.execute(
            select(Observation.ts).order_by(desc(Observation.ts)).limit(1)
        ).scalar_one_or_none()
        last_capture_utc = _as_utc_str(latest_ts) if latest_ts else None

        roi_str = roi_to_str(s.roi) if s.roi else ""
        rtsp = s.rtsp_url or ""

        return templates.TemplateResponse(
            "index.html",
            _context(
                request,
                title,
                settings=s,
                top_inds=top_inds_out,
                recent=recent,
                recent_page=current_page,
                recent_page_size=clamped_page_size,
                recent_total_pages=total_pages,
                recent_total=int(total_recent),
                recent_page_size_options=[10, 20, 50, 100],
                roi_str=roi_str,
                rtsp_url=rtsp,
                last_capture_utc=last_capture_utc,
            ),
        )

    @app.get("/observations", response_class=HTMLResponse)
    def observations(
        request: Request,
        individual_id: int | None = None,
        limit: int = 200,
        db: Session = Depends(get_db),
    ) -> HTMLResponse:
        s = load_settings()

        limit = max(10, min(int(limit), 2000))

        q = select(Observation).order_by(desc(Observation.ts)).limit(limit)
        if individual_id is not None:
            q = q.where(Observation.individual_id == individual_id)

        obs = db.execute(q).scalars().all()
        for o in obs:
            o.species_css = species_to_css(o.species_label)  # type: ignore[attr-defined]

        inds = db.execute(
            select(Individual).order_by(desc(Individual.visit_count)).limit(2000)
        ).scalars().all()

        total = db.execute(select(func.count(Observation.id))).scalar_one()

        return templates.TemplateResponse(
            "observations.html",
            _context(
                request,
                "Observations",
                settings=s,
                observations=obs,
                individuals=inds,
                selected_individual=individual_id,
                selected_limit=limit,
                count_shown=len(obs),
                count_total=int(total),
                rtsp_url=s.rtsp_url,
            ),
        )

    @app.get("/observations/{obs_id}", response_class=HTMLResponse)
    def observation_detail(obs_id: int, request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
        o = db.get(Observation, obs_id)
        if o is None:
            raise HTTPException(status_code=404, detail="Observation not found")

        o.species_css = species_to_css(o.species_label)  # type: ignore[attr-defined]
        extra = o.get_extra() or {}
        o.extra_json_pretty = pretty_json(o.extra_json)  # type: ignore[attr-defined]
        return templates.TemplateResponse(
            "observation_detail.html",
            _context(
                request,
                f"Observation {o.id}",
                o=o,
                extra=extra,
                allowed_review_labels=ALLOWED_REVIEW_LABELS,
            ),
        )

    @app.post("/observations/{obs_id}/label")
    def label_observation(
        obs_id: int,
        label: str = Form(...),
        db: Session = Depends(get_db),
    ) -> RedirectResponse:
        """Update the review label for a single observation and redirect to its detail page.

        This endpoint is intended for the observation detail UI. It accepts a form field
        ``label`` for the review label, normalizes it (strip + lower-case), and only
        persists it if it is contained in :data:`ALLOWED_REVIEW_LABELS`. Labels longer
        than 64 characters are rejected with HTTP 400.

        If a valid review label is provided, the observation's ``extra`` JSON is updated
        to include a ``"review"`` section with the label and a ``"labeled_at_utc"``
        timestamp. If the provided label is empty or not allowed, any existing review
        label and timestamp are removed from ``extra`` (and the ``"review"`` section is
        dropped entirely if it becomes empty).

        On success, the change is committed and the client is redirected (HTTP 303) back
        to ``/observations/{obs_id}``.
        """
        o = db.get(Observation, obs_id)
        if o is None:
            raise HTTPException(status_code=404, detail="Observation not found")

        raw_label = label or ""
        if len(raw_label) > 64:
            raise HTTPException(status_code=400, detail="Label too long")
        clean = raw_label.strip().lower()
        allowed = set(ALLOWED_REVIEW_LABELS)
        review_label = clean if clean in allowed else ""

        if review_label:
            o.merge_extra(
                {
                    "review": {
                        "label": review_label,
                        "labeled_at_utc": _as_utc_str(datetime.now(timezone.utc)),
                    }
                }
            )
        else:
            extra = o.get_extra() or {}
            if isinstance(extra, dict):
                # Work on a copy to avoid mutating the dict returned by get_extra() in place.
                extra_copy = dict(extra)
                raw_review = extra_copy.get("review")
                review = dict(raw_review) if isinstance(raw_review, dict) else {}
                review.pop("label", None)
                review.pop("labeled_at_utc", None)
                if review:
                    extra_copy["review"] = review
                else:
                    # Drop the review section entirely if it's now empty.
                    extra_copy.pop("review", None)
                o.set_extra(extra_copy)
        _commit_with_retry(db)

        return RedirectResponse(url=f"/observations/{obs_id}", status_code=303)

    @app.post("/observations/{obs_id}/delete")
    def delete_observation(obs_id: int, db: Session = Depends(get_db)) -> RedirectResponse:
        o = db.get(Observation, obs_id)
        if o is None:
            raise HTTPException(status_code=404, detail="Observation not found")

        ind_id = o.individual_id

        # Clean up media
        _safe_unlink_media(o.snapshot_path)
        _safe_unlink_media(o.video_path)

        db.execute(delete(Embedding).where(Embedding.observation_id == obs_id))
        db.delete(o)
        _commit_with_retry(db)

        if ind_id is not None:
            _recompute_individual_stats(db, int(ind_id))
            _commit_with_retry(db)

        return RedirectResponse(url="/observations", status_code=303)

    @app.get("/individuals", response_class=HTMLResponse)
    def individuals(
        request: Request,
        sort: str = "visits",
        limit: int = 200,
        db: Session = Depends(get_db),
    ) -> HTMLResponse:
        limit = max(10, min(int(limit), 5000))
        sort = (sort or "visits").lower()

        q = select(Individual)
        if sort == "id":
            q = q.order_by(Individual.id)
        elif sort == "recent":
            q = q.order_by(desc(Individual.last_seen_at.nulls_last()))
        else:
            q = q.order_by(desc(Individual.visit_count))

        inds = db.execute(q.limit(limit)).scalars().all()
        total = db.execute(select(func.count(Individual.id))).scalar_one()

        return templates.TemplateResponse(
            "individuals.html",
            _context(
                request,
                "Individuals",
                individuals=inds,
                sort=sort,
                limit=limit,
                count_shown=len(inds),
                count_total=int(total),
            ),
        )

    @app.get("/individuals/{individual_id}", response_class=HTMLResponse)
    def individual_detail(
        individual_id: int,
        request: Request,
        db: Session = Depends(get_db),
    ) -> HTMLResponse:
        ind = db.get(Individual, individual_id)
        if ind is None:
            raise HTTPException(status_code=404, detail="Individual not found")

        obs = db.execute(
            select(Observation)
            .where(Observation.individual_id == individual_id)
            .order_by(desc(Observation.ts))
            .limit(500)
        ).scalars().all()

        for o in obs:
            o.species_css = species_to_css(o.species_label)  # type: ignore[attr-defined]

        total = int(ind.visit_count)

        last_seen = _as_utc_str(ind.last_seen_at)

        # SQLite hour-of-day counts in UTC:
        # Observation.ts stored as timezone-aware; SQLite stores as text.
        # Use strftime('%H', ts) which yields 00..23.
        rows = db.execute(
            select(func.strftime("%H", Observation.ts).label("hh"), func.count(Observation.id))
            .where(Observation.individual_id == individual_id)
            .group_by("hh")
            .order_by("hh")
        ).all()

        hours_rows: list[tuple[int, int]] = [(int(hh), int(cnt)) for (hh, cnt) in rows if hh is not None]
        heatmap = build_hour_heatmap(hours_rows)

        return templates.TemplateResponse(
            "individual_detail.html",
            _context(
                request,
                f"Individual {ind.id}",
                individual=ind,
                observations=obs,
                heatmap=heatmap,
                total=total,
                last_seen=last_seen,
            ),
        )

    @app.post("/individuals/{individual_id}/rename")
    def rename_individual(
        individual_id: int,
        name: str = Form(...),
        db: Session = Depends(get_db),
    ) -> RedirectResponse:
        ind = db.get(Individual, individual_id)
        if ind is None:
            raise HTTPException(status_code=404, detail="Individual not found")

        new_name = (name or "").strip()
        if not new_name:
            new_name = "(unnamed)"
        ind.name = new_name[:128]
        db.commit()

        return RedirectResponse(url=f"/individuals/{individual_id}", status_code=303)

    @app.post("/individuals/{individual_id}/delete")
    def delete_individual(individual_id: int, db: Session = Depends(get_db)) -> RedirectResponse:
        ind = db.get(Individual, individual_id)
        if ind is None:
            raise HTTPException(status_code=404, detail="Individual not found")

        obs_rows = db.execute(
            select(Observation.id, Observation.snapshot_path, Observation.video_path)
            .where(Observation.individual_id == individual_id)
        ).all()
        obs_ids = [int(r[0]) for r in obs_rows]
        for _, snap, vid in obs_rows:
            _safe_unlink_media(snap)
            _safe_unlink_media(vid)

        if obs_ids:
            db.execute(
                delete(Embedding).where(Embedding.observation_id.in_(obs_ids))
            )
            db.execute(delete(Observation).where(Observation.id.in_(obs_ids)))

        db.delete(ind)
        _commit_with_retry(db, retries=5, delay=0.6)

        return RedirectResponse(url="/individuals", status_code=303)

    @app.post("/individuals/{individual_id}/refresh_embedding")
    def refresh_embedding(individual_id: int, db: Session = Depends(get_db)) -> RedirectResponse:
        ind = db.get(Individual, individual_id)
        if ind is None:
            raise HTTPException(status_code=404, detail="Individual not found")

        embs = db.execute(
            select(Embedding).where(Embedding.individual_id == individual_id).order_by(desc(Embedding.created_at)).limit(500)
        ).scalars().all()

        if not embs:
            return RedirectResponse(url=f"/individuals/{individual_id}", status_code=303)

        vecs = [e.get_vec() for e in embs]
        proto = l2_normalize(sum(vecs) / max(1, len(vecs)))
        ind.set_prototype(proto)
        db.commit()

        return RedirectResponse(url=f"/individuals/{individual_id}", status_code=303)

    @app.get("/individuals/{individual_id}/split_review", response_class=HTMLResponse)
    def split_review(individual_id: int, request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
        ind = db.get(Individual, individual_id)
        if ind is None:
            raise HTTPException(status_code=404, detail="Individual not found")

        obs = db.execute(
            select(Observation)
            .where(Observation.individual_id == individual_id)
            .order_by(desc(Observation.ts))
            .limit(120)
        ).scalars().all()

        for o in obs:
            o.species_css = species_to_css(o.species_label)  # type: ignore[attr-defined]
            o.suggested_side = "A"  # type: ignore[attr-defined]

        emb_rows = db.execute(
            select(Embedding).join(Observation, Embedding.observation_id == Observation.id)
            .where(Observation.individual_id == individual_id)
            .order_by(desc(Observation.ts))
            .limit(120)
        ).scalars().all()

        # Map obs_id -> embedding vec
        emb_map: dict[int, Any] = {}
        for e in emb_rows:
            emb_map[int(e.observation_id)] = e.get_vec()

        # Build aligned list
        aligned_vecs = []
        aligned_obs_ids = []
        for o in obs:
            v = emb_map.get(int(o.id))
            if v is not None:
                aligned_vecs.append(v)
                aligned_obs_ids.append(int(o.id))

        suggestion = suggest_split_two_groups(aligned_vecs, min_samples=12)

        if suggestion.ok and suggestion.labels and suggestion.centroid_a is not None and suggestion.centroid_b is not None:
            side_map = {oid: lab for oid, lab in zip(aligned_obs_ids, suggestion.labels)}
            for o in obs:
                if int(o.id) in side_map:
                    o.suggested_side = side_map[int(o.id)]  # type: ignore[attr-defined]
        else:
            # Keep A default
            pass

        return templates.TemplateResponse(
            "split_review.html",
            _context(
                request,
                f"Split review {ind.id}",
                individual=ind,
                observations=obs,
                suggestion_reason=suggestion.reason,
            ),
        )

    @app.post("/individuals/{individual_id}/split_apply")
    async def split_apply(individual_id: int, request: Request, db: Session = Depends(get_db)) -> RedirectResponse:
        ind_a = db.get(Individual, individual_id)
        if ind_a is None:
            raise HTTPException(status_code=404, detail="Individual not found")

        form = await request.form()
        # Parse assignments: assign_<obsid> => "A" or "B"
        assign: dict[int, str] = {}
        for k, v in form.items():
            if not k.startswith("assign_"):
                continue
            try:
                oid = int(k.split("_", 1)[1])
            except Exception:
                continue
            side = str(v).strip().upper()
            assign[oid] = "B" if side == "B" else "A"

        # Determine which observations move to B
        obs_to_b = [oid for (oid, side) in assign.items() if side == "B"]
        if not obs_to_b:
            # no-op
            return RedirectResponse(url=f"/individuals/{individual_id}", status_code=303)

        # Create individual B
        ind_b = Individual(name=f"(split from {ind_a.id})", visit_count=0, last_seen_at=None)
        db.add(ind_b)
        db.flush()  # get ind_b.id

        # Reassign observations to B
        moved_obs = db.execute(select(Observation).where(Observation.id.in_(obs_to_b))).scalars().all()
        for o in moved_obs:
            o.individual_id = ind_b.id

        # Reassign embeddings rows to B too
        moved_embs = db.execute(select(Embedding).where(Embedding.observation_id.in_(obs_to_b))).scalars().all()
        for e in moved_embs:
            e.individual_id = ind_b.id

        db.commit()

        # Recompute visit counts + last seen
        def recompute_stats(ind: Individual) -> None:
            rows = db.execute(
                select(func.count(Observation.id), func.max(Observation.ts))
                .where(Observation.individual_id == ind.id)
            ).one()
            ind.visit_count = int(rows[0] or 0)
            ind.last_seen_at = rows[1]

        recompute_stats(ind_a)
        recompute_stats(ind_b)

        # Recompute prototypes from embeddings (if available)
        def recompute_proto(ind: Individual) -> None:
            embs = db.execute(select(Embedding).where(Embedding.individual_id == ind.id).limit(2000)).scalars().all()
            if not embs:
                return
            vecs = [e.get_vec() for e in embs]
            proto = l2_normalize(sum(vecs) / max(1, len(vecs)))
            ind.set_prototype(proto)

        recompute_proto(ind_a)
        recompute_proto(ind_b)

        db.commit()

        return RedirectResponse(url=f"/individuals/{ind_b.id}", status_code=303)

    @app.get("/config", response_class=HTMLResponse)
    def config_page(request: Request) -> HTMLResponse:
        s = load_settings()
        saved = request.query_params.get("saved") == "1"
        return templates.TemplateResponse(
            "config.html",
            _context(
                request,
                "Config",
                settings=s,
                form_values=_config_form_values(s),
                errors=[],
                saved=saved,
            ),
        )

    @app.post("/config", response_class=HTMLResponse)
    async def config_save(request: Request) -> HTMLResponse:
        s = load_settings()
        form = await request.form()
        field_names = (
            "detect_conf",
            "detect_iou",
            "min_box_area",
            "cooldown_seconds",
            "min_species_prob",
            "match_threshold",
            "ema_alpha",
        )
        raw = {name: str(form.get(name, "") or "").strip() for name in field_names}
        raw["timezone"] = str(form.get("timezone", "") or "").strip()
        parsed, errors = _validate_detection_inputs(raw)

        if errors:
            return templates.TemplateResponse(
                "config.html",
                _context(
                    request,
                    "Config",
                    settings=s,
                    form_values=_config_form_values(s, raw),
                    errors=errors,
                    saved=False,
                ),
                status_code=400,
            )

        s.detect_conf = parsed["detect_conf"]
        s.detect_iou = parsed["detect_iou"]
        s.min_box_area = parsed["min_box_area"]
        s.cooldown_seconds = parsed["cooldown_seconds"]
        s.min_species_prob = parsed["min_species_prob"]
        s.match_threshold = parsed["match_threshold"]
        s.ema_alpha = parsed["ema_alpha"]
        s.timezone = parsed["timezone"]
        save_settings(s)

        return RedirectResponse(url="/config?saved=1", status_code=303)

    @app.get("/calibrate", response_class=HTMLResponse)
    def calibrate(request: Request) -> HTMLResponse:
        s = load_settings()
        roi_str = roi_to_str(s.roi) if s.roi else ""
        return templates.TemplateResponse(
            "calibrate.html",
            _context(
                request,
                "Calibrate ROI",
                settings=s,
                roi=s.roi,
                roi_str=roi_str,
                ts=int(time.time()),
            ),
        )

    # ----------------------------
    # API routes
    # ----------------------------

    @app.get("/api/health", response_model=HealthOut)
    def health(db: Session = Depends(get_db)) -> HealthOut:
        s = load_settings()
        db_ok = True
        last_obs = None
        try:
            last_obs = db.execute(select(Observation).order_by(desc(Observation.ts)).limit(1)).scalars().first()
        except Exception:
            db_ok = False

        return HealthOut(
            ok=True,
            version="0.1.0",
            db_ok=db_ok,
            last_observation_utc=(last_obs.ts_utc if last_obs else None),
            rtsp_url=(s.rtsp_url or None),
        )

    @app.get("/api/roi", response_model=RoiOut)
    def get_roi() -> RoiOut:
        s = load_settings()
        if s.roi is None:
            return RoiOut(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
        r = s.roi.clamp()
        return RoiOut(x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2)

    @app.post("/api/roi")
    def set_roi(
        x1: float = Form(...),
        y1: float = Form(...),
        x2: float = Form(...),
        y2: float = Form(...),
    ) -> RedirectResponse:
        s = load_settings()
        r = Roi(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)).clamp()
        s.roi = r
        save_settings(s)
        return RedirectResponse(url="/calibrate", status_code=303)

    @app.get("/api/frame.jpg")
    def frame_jpg(db: Session = Depends(get_db)) -> Any:
        """
        Return latest snapshot image, or a placeholder if none exist yet.
        """
        last = db.execute(select(Observation).order_by(desc(Observation.ts)).limit(1)).scalars().first()
        if last is not None:
            p = media_dir() / last.snapshot_path
            if p.exists():
                return FileResponse(str(p), media_type="image/jpeg")

        # Placeholder
        try:
            from PIL import Image, ImageDraw
        except Exception:
            raise HTTPException(status_code=404, detail="No frames yet")

        img = Image.new("RGB", (960, 540), (18, 18, 28))
        d = ImageDraw.Draw(img)
        d.text((24, 24), "No snapshots yet.\nStart the worker and wait for a visit.", fill=(220, 220, 235))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")

    # ----------------------------
    # Export routes
    # ----------------------------

    def _stream_csv(rows: Iterator[list[Any]], header: list[str]) -> StreamingResponse:
        def gen() -> Iterator[bytes]:
            sio = io.StringIO()
            w = csv.writer(sio)
            w.writerow(header)
            yield sio.getvalue().encode("utf-8")
            sio.seek(0)
            sio.truncate(0)

            for r in rows:
                w.writerow(r)
                yield sio.getvalue().encode("utf-8")
                sio.seek(0)
                sio.truncate(0)

        return StreamingResponse(gen(), media_type="text/csv")

    @app.get("/export/observations.csv")
    def export_observations_csv(db: Session = Depends(get_db)) -> StreamingResponse:
        q = db.execute(select(Observation).order_by(desc(Observation.ts))).scalars()

        def rows() -> Iterator[list[Any]]:
            for o in q:
                yield [
                    o.id,
                    o.ts_utc,
                    o.camera_name or "",
                    o.species_label,
                    f"{o.species_prob:.6f}",
                    o.individual_id if o.individual_id is not None else "",
                    f"{o.match_score:.6f}",
                    o.snapshot_path,
                    o.video_path,
                    o.bbox_str or "",
                ]

        return _stream_csv(
            rows(),
            header=[
                "observation_id",
                "ts_utc",
                "camera_name",
                "species_label",
                "species_prob",
                "individual_id",
                "match_score",
                "snapshot_path",
                "video_path",
                "bbox_xyxy",
            ],
        )

    @app.get("/export/individuals.csv")
    def export_individuals_csv(db: Session = Depends(get_db)) -> StreamingResponse:
        q = db.execute(select(Individual).order_by(desc(Individual.visit_count))).scalars()

        def rows() -> Iterator[list[Any]]:
            for i in q:
                yield [
                    i.id,
                    i.name,
                    i.visit_count,
                    _as_utc_str(i.created_at) or "",
                    _as_utc_str(i.last_seen_at) or "",
                    i.last_species_label or "",
                ]

        return _stream_csv(
            rows(),
            header=["individual_id", "name", "visit_count", "created_utc", "last_seen_utc", "last_species_label"],
        )

    @app.get("/export/media_bundle.tar.gz")
    def export_media_bundle() -> FileResponse:
        """
        Create a tar.gz containing /media/snapshots and /media/clips.
        (Created on-demand under /data/exports.)
        """
        from hbmon.config import data_dir, snapshots_dir, clips_dir

        ensure_dirs()
        out_dir = data_dir() / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        out_path = out_dir / f"hbmon-media-{stamp}.tar.gz"

        snap = snapshots_dir()
        clips = clips_dir()

        with tarfile.open(out_path, "w:gz") as tf:
            if snap.exists():
                tf.add(snap, arcname="snapshots")
            if clips.exists():
                tf.add(clips, arcname="clips")

        return FileResponse(str(out_path), filename=out_path.name, media_type="application/gzip")

    return app


# Default ASGI app for uvicorn
app = make_app()
