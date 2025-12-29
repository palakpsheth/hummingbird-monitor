from __future__ import annotations

import io
import tarfile
from pathlib import Path

from fastapi.testclient import TestClient

from hbmon.config import ensure_dirs, media_dir
from hbmon.db import init_db, reset_db_state, session_scope
from hbmon.models import Candidate, utcnow
from hbmon.web import make_app


def _setup_app(tmp_path: Path, monkeypatch) -> TestClient:
    data_dir = tmp_path / "data"
    media = tmp_path / "media"
    db_path = tmp_path / "db.sqlite"
    monkeypatch.setenv("HBMON_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(media))
    monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("HBMON_DB_ASYNC_URL", f"sqlite+aiosqlite:///{db_path}")
    reset_db_state()
    ensure_dirs()
    init_db()
    app = make_app()
    return TestClient(app)


def _write_media_file(rel_path: str, payload: bytes = b"data") -> None:
    path = media_dir() / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _create_candidate(
    *,
    snapshot_path: str,
    extra: dict[str, object],
    annotated_snapshot_path: str | None = None,
    mask_path: str | None = None,
    mask_overlay_path: str | None = None,
    clip_path: str | None = None,
) -> int:
    with session_scope() as db:
        candidate = Candidate(
            ts=utcnow(),
            snapshot_path=snapshot_path,
            annotated_snapshot_path=annotated_snapshot_path,
            mask_path=mask_path,
            mask_overlay_path=mask_overlay_path,
            clip_path=clip_path,
        )
        candidate.set_extra(extra)
        db.add(candidate)
        db.commit()
        return int(candidate.id)


def test_candidates_list_filters(tmp_path, monkeypatch) -> None:
    client = _setup_app(tmp_path, monkeypatch)

    first_id = _create_candidate(
        snapshot_path="snap-1.jpg",
        extra={
            "reason": "motion",
            "review": {"label": "false_negative"},
            "media": {"snapshot_annotated_path": "snap-1-ann.jpg"},
        },
        annotated_snapshot_path="snap-1-ann.jpg",
    )
    second_id = _create_candidate(
        snapshot_path="snap-2.jpg",
        extra={"reason": "noise", "review": {"label": "true_negative"}},
    )

    response = client.get("/candidates?label=false_negative&reason=motion")
    assert response.status_code == 200
    assert f"/candidates/{first_id}" in response.text
    assert f"/candidates/{second_id}" not in response.text


def test_candidates_list_fallback_filtering(tmp_path, monkeypatch) -> None:
    client = _setup_app(tmp_path, monkeypatch)

    monkeypatch.setattr("hbmon.web._get_db_dialect_name", lambda db: "mysql")

    first_id = _create_candidate(
        snapshot_path="snap-1.jpg",
        extra={"reason": "motion", "review": {"label": "false_negative"}},
    )
    second_id = _create_candidate(
        snapshot_path="snap-2.jpg",
        extra={"reason": "other", "review": {"label": "true_negative"}},
    )

    response = client.get("/candidates?label=false_negative&reason=motion")
    assert response.status_code == 200
    assert f"/candidates/{first_id}" in response.text
    assert f"/candidates/{second_id}" not in response.text


def test_candidate_label_and_clear(tmp_path, monkeypatch) -> None:
    client = _setup_app(tmp_path, monkeypatch)

    candidate_id = _create_candidate(
        snapshot_path="snap-1.jpg",
        extra={"reason": "motion"},
    )

    response = client.post(
        f"/candidates/{candidate_id}/label",
        data={"label": "false_negative"},
        follow_redirects=False,
    )
    assert response.status_code == 303

    with session_scope() as db:
        candidate = db.get(Candidate, candidate_id)
        assert candidate is not None
        extra = candidate.get_extra() or {}
        review = extra.get("review") if isinstance(extra, dict) else {}
        assert isinstance(review, dict)
        assert review.get("label") == "false_negative"

    response = client.post(
        f"/candidates/{candidate_id}/label",
        data={"label": "  "},
        follow_redirects=False,
    )
    assert response.status_code == 303
    with session_scope() as db:
        candidate = db.get(Candidate, candidate_id)
        assert candidate is not None
        extra = candidate.get_extra() or {}
        assert "review" not in extra


def test_candidate_export_bundle_includes_media(tmp_path, monkeypatch) -> None:
    client = _setup_app(tmp_path, monkeypatch)

    _write_media_file("snap-1.jpg")
    _write_media_file("snap-1-ann.jpg")
    _write_media_file("mask.png")
    _write_media_file("mask-overlay.png")
    _write_media_file("clip.mp4")

    candidate_id = _create_candidate(
        snapshot_path="snap-1.jpg",
        annotated_snapshot_path="snap-1-ann.jpg",
        mask_path="mask.png",
        mask_overlay_path="mask-overlay.png",
        clip_path="clip.mp4",
        extra={
            "reason": "motion",
            "media": {
                "snapshot_annotated_path": "snap-1-ann.jpg",
                "mask_path": "mask.png",
                "mask_overlay_path": "mask-overlay.png",
                "clip_path": "clip.mp4",
            },
        },
    )

    response = client.post(
        f"/candidates/{candidate_id}/export_integration_test",
        data={"case_name": "My Candidate"},
    )
    assert response.status_code == 200

    tar_bytes = io.BytesIO(response.content)
    with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tf:
        names = tf.getnames()
    assert any(name.endswith("/metadata.json") for name in names)
    assert any(name.endswith("/snapshot.jpg") for name in names)
    assert any(name.endswith("/snapshot_annotated.jpg") for name in names)
    assert any(name.endswith("/mask.png") for name in names)
    assert any(name.endswith("/mask_overlay.png") for name in names)
    assert any(name.endswith("/clip.mp4") for name in names)


def test_candidate_detail_page_renders_media(tmp_path, monkeypatch) -> None:
    client = _setup_app(tmp_path, monkeypatch)

    candidate_id = _create_candidate(
        snapshot_path="snap-1.jpg",
        annotated_snapshot_path="snap-1-ann.jpg",
        mask_path="mask.png",
        mask_overlay_path="mask-overlay.png",
        extra={"media": {"snapshot_annotated_path": "snap-1-ann.jpg"}},
    )

    response = client.get(f"/candidates/{candidate_id}")
    assert response.status_code == 200
    assert "mask.png" in response.text
    assert "mask-overlay.png" in response.text


def test_bulk_delete_candidates_empty(tmp_path, monkeypatch) -> None:
    client = _setup_app(tmp_path, monkeypatch)

    response = client.post("/candidates/bulk_delete", data={}, follow_redirects=False)
    assert response.status_code == 303


def test_bulk_delete_candidates_removes_media(tmp_path, monkeypatch) -> None:
    client = _setup_app(tmp_path, monkeypatch)

    _write_media_file("snap-1.jpg")
    _write_media_file("snap-1-ann.jpg")
    _write_media_file("mask.png")
    _write_media_file("mask-overlay.png")
    _write_media_file("clip.mp4")
    _write_media_file("snap-raw.jpg")

    candidate_id = _create_candidate(
        snapshot_path="snap-1.jpg",
        annotated_snapshot_path="snap-1-ann.jpg",
        mask_path="mask.png",
        mask_overlay_path="mask-overlay.png",
        clip_path="clip.mp4",
        extra={
            "media": {
                "snapshot_raw_path": "snap-raw.jpg",
                "snapshot_annotated_path": "snap-1-ann.jpg",
                "mask_path": "mask.png",
                "mask_overlay_path": "mask-overlay.png",
                "clip_path": "clip.mp4",
            }
        },
    )

    response = client.post(
        "/candidates/bulk_delete",
        data={"candidate_ids": [candidate_id]},
        follow_redirects=False,
    )
    assert response.status_code == 303

    with session_scope() as db:
        assert db.get(Candidate, candidate_id) is None

    for rel_path in [
        "snap-1.jpg",
        "snap-1-ann.jpg",
        "mask.png",
        "mask-overlay.png",
        "clip.mp4",
        "snap-raw.jpg",
    ]:
        assert not (media_dir() / rel_path).exists()
