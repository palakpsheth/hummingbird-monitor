from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hbmon.clip_model as clip


def _install_openclip_stub(monkeypatch):
    if clip.torch is None:
        pytest.skip("torch not available")
    torch = clip.torch

    class DummyModel:
        def to(self, device: str):
            return self

        def eval(self):
            return self

        def encode_text(self, tokens):
            return torch.ones((tokens.shape[0], 4), dtype=torch.float32)

        def encode_image(self, inp):
            return torch.tensor([[1.0, 0.5, 0.25, 0.0]], dtype=torch.float32)

    def create_model_and_transforms(model_name: str, pretrained: str):
        return DummyModel(), None, lambda pil: torch.ones((3, 2, 2), dtype=torch.float32)

    def get_tokenizer(model_name: str):
        return lambda prompts: torch.ones((len(prompts), 4), dtype=torch.float32)

    stub = SimpleNamespace(
        create_model_and_transforms=create_model_and_transforms,
        get_tokenizer=get_tokenizer,
    )
    monkeypatch.setattr(clip, "open_clip", stub)
    clip._load_openclip.cache_clear()
    return torch


def test_load_openclip_quickgelu_adjustment(monkeypatch):
    if clip.torch is None:
        pytest.skip("torch not available")
    captured = {}

    class DummyModel:
        def to(self, device: str):
            return self

        def eval(self):
            return self

    def create_model_and_transforms(model_name: str, pretrained: str):
        captured["model_name"] = model_name
        return DummyModel(), None, lambda pil: clip.torch.ones((3, 2, 2), dtype=clip.torch.float32)

    def get_tokenizer(model_name: str):
        return lambda prompts: clip.torch.ones((len(prompts), 4), dtype=clip.torch.float32)

    stub = SimpleNamespace(
        create_model_and_transforms=create_model_and_transforms,
        get_tokenizer=get_tokenizer,
    )
    monkeypatch.setattr(clip, "open_clip", stub)
    clip._load_openclip.cache_clear()

    class NoLower:
        def __str__(self) -> str:
            return "vit-b-32"

    clip._load_openclip(NoLower(), "openai", "cpu")
    assert captured["model_name"].endswith("-quickgelu")

    captured.clear()
    clip._load_openclip("vit-b-32-quickgelu", "openai", "cpu")
    assert captured["model_name"] == "vit-b-32-quickgelu"


def test_clipmodel_predictions_and_embeddings(monkeypatch):
    if clip.Image is None or clip.torch is None:
        pytest.skip("clip model dependencies missing")
    _install_openclip_stub(monkeypatch)

    cm = clip.ClipModel(
        labels=["Alpha", "Beta"],
        prompts={"Alpha": ["alpha"], "Beta": ["beta"]},
        prompt_prefix="",
    )

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    embedding = cm.encode_embedding(image)
    assert embedding.shape == (4,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-6)

    pred = cm.predict_species(image, topk=1)
    assert isinstance(pred, clip.SpeciesPrediction)

    preds = cm.predict_species(image, topk=2)
    assert isinstance(preds, list)
    assert len(preds) == 2

    with pytest.raises(ValueError):
        cm.predict_species(image, topk=0)

    label, prob = cm.predict_species_label_prob(image)
    assert label == pred.label
    assert prob == pred.probability

    cm.set_label_space(["Beta"], prompts={"Beta": ["beta"]})
    assert cm.get_labels() == ["Beta"]
    assert cm.get_prompt_map() == {"Beta": ["beta"]}

    sims = cm.debug_similarity(image)
    assert list(sims.keys()) == ["Beta"]
