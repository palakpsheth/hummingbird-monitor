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
import io
import tarfile
import time
from datetime import timezone
from pathlib import Path
from typing import Any, Iterator

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
    Depends = FastAPI = Form = HTTPException = Request = object  # type: ignore
    FileResponse = HTMLResponse = RedirectResponse = StreamingResponse = object  # type: ignore
    StaticFiles = object  # type: ignore
    Jinja2Templates = object  # type: ignore
    _FASTAPI_AVAILABLE = False

try:
    from sqlalchemy import desc, func, select  # type: ignore
    from sqlalchemy.orm import Session  # type: ignore
    _SQLA_AVAILABLE = True
except Exception:  # pragma: no cover
    desc = func = select = None  # type: ignore
    Session = object  # type: ignore
    _SQLA_AVAILABLE = False


from hbmon.config import Roi, ensure_dirs, load_settings, media_dir, roi_to_str, save_settings
from hbmon.db import get_db, init_db
from hbmon.models import Embedding, Individual, Observation
from hbmon.schema import HealthOut, RoiOut
from hbmon.clustering import l2_normalize, suggest_split_two_groups


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


def _as_utc_str(dt) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_detection_inputs(raw: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    """
    Validate and coerce detection sensitivity inputs from the config form.
    Returns (parsed_values, errors).
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

    return parsed, errors


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

    # ----------------------------
    # UI routes
    # ----------------------------

    def _config_form_values(s, raw: dict[str, str] | None = None) -> dict[str, str]:
        vals = {
            "detect_conf": f"{float(s.detect_conf):.2f}",
            "detect_iou": f"{float(s.detect_iou):.2f}",
            "min_box_area": str(int(s.min_box_area)),
            "cooldown_seconds": f"{float(s.cooldown_seconds):.2f}",
            "min_species_prob": f"{float(s.min_species_prob):.2f}",
            "match_threshold": f"{float(s.match_threshold):.2f}",
            "ema_alpha": f"{float(s.ema_alpha):.2f}",
        }
        if raw:
            for k, v in raw.items():
                if k in vals:
                    vals[k] = str(v)
        return vals

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
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

        recent = db.execute(
            select(Observation).order_by(desc(Observation.ts)).limit(24)
        ).scalars().all()

        for o in recent:
            # attach computed presentation attrs (not in DB)
            o.species_css = species_to_css(o.species_label)  # type: ignore[attr-defined]

        last_capture_utc = recent[0].ts_utc if recent else None

        roi_str = roi_to_str(s.roi) if s.roi else ""
        rtsp = s.rtsp_url or ""

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "title": title,
                "top_inds": top_inds_out,
                "recent": recent,
                "roi_str": roi_str,
                "rtsp_url": rtsp,
                "last_capture_utc": last_capture_utc,
            },
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
            {
                "request": request,
                "title": "Observations",
                "observations": obs,
                "individuals": inds,
                "selected_individual": individual_id,
                "selected_limit": limit,
                "count_shown": len(obs),
                "count_total": int(total),
                "rtsp_url": s.rtsp_url,
            },
        )

    @app.get("/observations/{obs_id}", response_class=HTMLResponse)
    def observation_detail(obs_id: int, request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
        o = db.get(Observation, obs_id)
        if o is None:
            raise HTTPException(status_code=404, detail="Observation not found")

        o.species_css = species_to_css(o.species_label)  # type: ignore[attr-defined]
        return templates.TemplateResponse(
            "observation_detail.html",
            {"request": request, "title": f"Observation {o.id}", "o": o},
        )

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
            {
                "request": request,
                "title": "Individuals",
                "individuals": inds,
                "sort": sort,
                "limit": limit,
                "count_shown": len(inds),
                "count_total": int(total),
            },
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
            {
                "request": request,
                "title": f"Individual {ind.id}",
                "individual": ind,
                "observations": obs,
                "heatmap": heatmap,
                "total": total,
                "last_seen": last_seen,
            },
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
            {
                "request": request,
                "title": f"Split review {ind.id}",
                "individual": ind,
                "observations": obs,
                "suggestion_reason": suggestion.reason,
            },
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
            {
                "request": request,
                "title": "Config",
                "form_values": _config_form_values(s),
                "errors": [],
                "saved": saved,
            },
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
        parsed, errors = _validate_detection_inputs(raw)

        if errors:
            return templates.TemplateResponse(
                "config.html",
                {
                    "request": request,
                    "title": "Config",
                    "form_values": _config_form_values(s, raw),
                    "errors": errors,
                    "saved": False,
                },
                status_code=400,
            )

        s.detect_conf = parsed["detect_conf"]
        s.detect_iou = parsed["detect_iou"]
        s.min_box_area = parsed["min_box_area"]
        s.cooldown_seconds = parsed["cooldown_seconds"]
        s.min_species_prob = parsed["min_species_prob"]
        s.match_threshold = parsed["match_threshold"]
        s.ema_alpha = parsed["ema_alpha"]
        save_settings(s)

        return RedirectResponse(url="/config?saved=1", status_code=303)

    @app.get("/calibrate", response_class=HTMLResponse)
    def calibrate(request: Request) -> HTMLResponse:
        s = load_settings()
        roi_str = roi_to_str(s.roi) if s.roi else ""
        return templates.TemplateResponse(
            "calibrate.html",
            {
                "request": request,
                "title": "Calibrate ROI",
                "roi": s.roi,
                "roi_str": roi_str,
                "ts": int(time.time()),
            },
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
