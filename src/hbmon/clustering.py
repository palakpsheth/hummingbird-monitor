# src/hbmon/clustering.py
"""
Individual re-identification ("re-ID") via CLIP image embeddings.

Goal
----
Given an embedding for a new hummingbird observation, assign it to an existing
"individual" (cluster) if it is sufficiently close; otherwise create a new
individual.

This module intentionally uses simple, inspectable logic that works well on CPU:
- Each individual has a prototype embedding (L2-normalized float32 vector)
- Matching uses cosine distance (1 - cosine_similarity)
- Update is an exponential moving average (EMA) on the prototype
- Optional "pending buffer" per individual to reduce drift (store a small queue)

We also provide an optional "split suggestion" mechanism that can help the user
split a messy individual into two via a lightweight k-means on recent samples.

Important: embeddings MUST be L2-normalized before calling into this module.

Environment variables (optional):
- HBMON_MATCH_THRESHOLD: cosine distance threshold; default 0.25
- HBMON_EMA_ALPHA: EMA update weight for new embeddings; default 0.10
- HBMON_MIN_VISITS_FOR_LOCK: minimum visits before becoming "stable"; default 5
- HBMON_MAX_PENDING: max pending embeddings per individual for split review; default 64
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    KMeans = None  # type: ignore[assignment]


# ----------------------------
# Config helpers
# ----------------------------

def _get_env_float(name: str, default: float) -> float:
    import os
    v = os.getenv(name)
    if v is None or not v.strip():
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _get_env_int(name: str, default: int) -> int:
    import os
    v = os.getenv(name)
    if v is None or not v.strip():
        return default
    try:
        return int(v)
    except ValueError:
        return default


MATCH_THRESHOLD = _get_env_float("HBMON_MATCH_THRESHOLD", 0.25)  # lower = stricter
EMA_ALPHA = _get_env_float("HBMON_EMA_ALPHA", 0.10)
MIN_VISITS_FOR_LOCK = _get_env_int("HBMON_MIN_VISITS_FOR_LOCK", 5)
MAX_PENDING = _get_env_int("HBMON_MAX_PENDING", 64)


# ----------------------------
# Math utilities
# ----------------------------

def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    n = float(np.linalg.norm(vec) + eps)
    return (vec / n).astype(np.float32, copy=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    a, b: 1D normalized vectors
    """
    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)


def batch_cosine_distance(x: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """
    x: [D] normalized
    prototypes: [N, D] normalized
    returns distances: [N]
    """
    # cosine similarity is dot product for normalized vectors
    sims = prototypes @ x  # [N]
    return (1.0 - sims).astype(np.float32, copy=False)


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class MatchResult:
    matched: bool
    individual_id: int
    distance: float
    similarity: float


@dataclass
class IndividualState:
    """
    In-memory representation of an individual (cluster).

    prototype: [D] normalized
    visit_count: number of assigned observations
    pending: FIFO buffer of recent embeddings (normalized) for split review
    """
    individual_id: int
    prototype: np.ndarray
    visit_count: int = 0
    pending: list[np.ndarray] | None = None

    def ensure_pending(self) -> None:
        if self.pending is None:
            self.pending = []

    def add_pending(self, emb: np.ndarray) -> None:
        self.ensure_pending()
        assert self.pending is not None
        self.pending.append(emb)
        if len(self.pending) > MAX_PENDING:
            # drop oldest
            self.pending.pop(0)


# ----------------------------
# Core matching logic
# ----------------------------

def choose_individual(
    emb: np.ndarray,
    individuals: Sequence[IndividualState],
    *,
    threshold: float = MATCH_THRESHOLD,
) -> MatchResult | None:
    """
    Choose best matching existing individual.

    Returns None if there are no individuals.
    """
    if len(individuals) == 0:
        return None

    emb = l2_normalize(emb)

    protos = np.stack([i.prototype for i in individuals], axis=0).astype(np.float32, copy=False)
    dists = batch_cosine_distance(emb, protos)
    idx = int(np.argmin(dists))
    best = float(dists[idx])
    sim = 1.0 - best

    return MatchResult(
        matched=(best <= threshold),
        individual_id=individuals[idx].individual_id,
        distance=best,
        similarity=sim,
    )


def update_prototype_ema(prototype: np.ndarray, emb: np.ndarray, *, alpha: float = EMA_ALPHA) -> np.ndarray:
    """
    Exponential moving average update on normalized vectors.

    prototype <- normalize((1-alpha)*prototype + alpha*emb)
    """
    prototype = prototype.astype(np.float32, copy=False)
    emb = emb.astype(np.float32, copy=False)
    out = (1.0 - alpha) * prototype + alpha * emb
    return l2_normalize(out)


def assign_or_create(
    emb: np.ndarray,
    individuals: list[IndividualState],
    *,
    threshold: float = MATCH_THRESHOLD,
    alpha: float = EMA_ALPHA,
    next_id: int | None = None,
    keep_pending: bool = True,
) -> tuple[MatchResult, IndividualState, bool]:
    """
    Assign embedding to an existing individual or create a new one.

    Args:
      emb: normalized embedding vector [D] (we normalize again for safety)
      individuals: list of existing IndividualState
      threshold: cosine distance threshold for matching
      alpha: EMA update weight
      next_id: optional id to use if creating; if None uses max+1 or 1
      keep_pending: if True, store embedding in pending buffer

    Returns:
      (match_result, individual_state, created_new)
    """
    emb = l2_normalize(emb)

    m = choose_individual(emb, individuals, threshold=threshold)
    if m is None or not m.matched:
        if next_id is None:
            next_id = (max([i.individual_id for i in individuals]) + 1) if individuals else 1
        state = IndividualState(individual_id=next_id, prototype=emb.copy(), visit_count=1)
        if keep_pending:
            state.add_pending(emb)
        individuals.append(state)
        return (
            MatchResult(matched=False, individual_id=state.individual_id, distance=1.0, similarity=0.0),
            state,
            True,
        )

    # matched existing
    state = next(i for i in individuals if i.individual_id == m.individual_id)
    state.visit_count += 1

    # update prototype more conservatively once "stable"
    # (reduces drift when we have a lot of visits)
    effective_alpha = alpha
    if state.visit_count >= MIN_VISITS_FOR_LOCK:
        effective_alpha = min(alpha, 0.05)

    state.prototype = update_prototype_ema(state.prototype, emb, alpha=effective_alpha)

    if keep_pending:
        state.add_pending(emb)

    return m, state, False


# ----------------------------
# Split suggestion (optional)
# ----------------------------

@dataclass
class SplitSuggestion:
    """
    Suggest splitting a set of embeddings into two groups.
    """
    ok: bool
    reason: str
    labels: list[str] | None = None  # "A" or "B" for each embedding
    centroid_a: np.ndarray | None = None
    centroid_b: np.ndarray | None = None


def suggest_split_two_groups(
    embeddings: Sequence[np.ndarray],
    *,
    min_samples: int = 12,
    random_state: int = 0,
) -> SplitSuggestion:
    """
    Attempt to split embeddings into two clusters via k-means (k=2).

    This is used for the UI "split review" tool. The UI can show a default
    assignment (A/B) that the user can override.

    Returns:
      SplitSuggestion(ok=False, reason=...) if not enough samples or sklearn missing.
    """
    if KMeans is None:
        return SplitSuggestion(ok=False, reason="scikit-learn not installed")

    if len(embeddings) < min_samples:
        return SplitSuggestion(ok=False, reason=f"need >= {min_samples} samples to suggest split")

    X = np.stack([l2_normalize(e) for e in embeddings], axis=0).astype(np.float32, copy=False)

    km = KMeans(n_clusters=2, n_init="auto", random_state=random_state)
    y = km.fit_predict(X)

    # Convert 0/1 to A/B
    labels = ["A" if int(v) == 0 else "B" for v in y.tolist()]

    ca = l2_normalize(km.cluster_centers_[0].astype(np.float32, copy=False))
    cb = l2_normalize(km.cluster_centers_[1].astype(np.float32, copy=False))

    # A sanity check: if centroids are too close, split isn't meaningful.
    dist = cosine_distance(ca, cb)
    if dist < 0.10:
        return SplitSuggestion(ok=False, reason=f"clusters not separable (centroid distance {dist:.3f})")

    return SplitSuggestion(
        ok=True,
        reason=f"ok (centroid distance {dist:.3f})",
        labels=labels,
        centroid_a=ca,
        centroid_b=cb,
    )


def assign_to_centroid(
    emb: np.ndarray,
    centroid_a: np.ndarray,
    centroid_b: np.ndarray,
) -> str:
    """
    Assign embedding to the closer of centroid_a/centroid_b.
    Returns "A" or "B".
    """
    emb = l2_normalize(emb)
    da = cosine_distance(emb, centroid_a)
    db = cosine_distance(emb, centroid_b)
    return "A" if da <= db else "B"
