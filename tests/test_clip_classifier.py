# tests/test_clip_classifier.py
from __future__ import annotations

import numpy as np


def test_species_classifier_returns_topk(sample_rgb_image):
    """
    Spec: CLIP classifier returns (label, score) pairs with a stable interface,
    and always includes a fallback like "unknown".
    """
    from hbmon.classify.clip_species import ClipSpeciesClassifier  # type: ignore

    class DummyEmbedder:
        def encode_image(self, img):
            # 512-d embedding
            return np.ones((512,), dtype=np.float32)

        def encode_text(self, texts):
            # [n, 512]
            return np.stack([np.ones((512,), dtype=np.float32) for _ in texts], axis=0)

    clf = ClipSpeciesClassifier(
        embedder=DummyEmbedder(),
        labels=["anna_hummingbird", "allen_hummingbird", "unknown"],
    )

    out = clf.predict(sample_rgb_image, topk=2)
    assert isinstance(out, list)
    assert len(out) == 2
    assert all("label" in x and "score" in x for x in out)
    assert out[0]["score"] >= out[1]["score"]


def test_classifier_handles_empty_labels(sample_rgb_image):
    from hbmon.classify.clip_species import ClipSpeciesClassifier  # type: ignore

    class DummyEmbedder:
        def encode_image(self, img):
            return np.ones((128,), dtype=np.float32)

        def encode_text(self, texts):
            return np.zeros((len(texts), 128), dtype=np.float32)

    clf = ClipSpeciesClassifier(embedder=DummyEmbedder(), labels=["unknown"])
    out = clf.predict(sample_rgb_image, topk=1)
    assert out[0]["label"] == "unknown"
