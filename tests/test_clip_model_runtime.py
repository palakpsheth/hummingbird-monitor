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


def _install_openvino_stubs(monkeypatch, *, text_output: np.ndarray, image_output: np.ndarray) -> None:
    if clip.torch is None or clip.Image is None:
        pytest.skip("clip model dependencies missing")
    torch = clip.torch

    class FakeCompiledModel:
        def __init__(self, output: np.ndarray) -> None:
            self._output = np.asarray(output, dtype=np.float32)

        def __call__(self, inputs: np.ndarray):
            output = self._output
            if output.ndim == 2:
                batch = inputs.shape[0]
                if output.shape[0] != batch:
                    output = np.repeat(output, batch, axis=0)
            return [output]

    import hbmon.openvino_utils as openvino_utils

    monkeypatch.setattr(openvino_utils, "is_openvino_available", lambda: True)
    monkeypatch.setattr(openvino_utils, "select_clip_device", lambda backend: "CPU")
    monkeypatch.setattr(
        openvino_utils,
        "load_openvino_clip",
        lambda model_name, pretrained, ov_device: (
            FakeCompiledModel(image_output),
            FakeCompiledModel(text_output),
        ),
    )

    def fake_load_openclip(model_name: str, pretrained: str, device: str):
        def preprocess(pil):
            return torch.ones((3, 2, 2), dtype=torch.float32)

        def tokenizer(prompts):
            return torch.ones((len(prompts), 4), dtype=torch.float32)

        return SimpleNamespace(), preprocess, tokenizer

    monkeypatch.setattr(clip, "_load_openclip", fake_load_openclip)
    monkeypatch.setattr(clip, "open_clip", SimpleNamespace())


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


def test_init_openvino_backend_falls_back_to_pytorch_cpu(monkeypatch):
    if clip.torch is None or clip.Image is None:
        pytest.skip("clip model dependencies missing")

    import hbmon.openvino_utils as openvino_utils

    monkeypatch.setattr(openvino_utils, "is_openvino_available", lambda: False)
    monkeypatch.setattr(clip, "open_clip", SimpleNamespace())

    def fake_load_openclip(model_name: str, pretrained: str, device: str):
        class DummyModel:
            def encode_text(self, tokens):
                return clip.torch.ones((tokens.shape[0], 4), dtype=clip.torch.float32)

            def encode_image(self, inp):
                return clip.torch.ones((inp.shape[0], 4), dtype=clip.torch.float32)

        def preprocess(pil):
            return clip.torch.ones((3, 2, 2), dtype=clip.torch.float32)

        def tokenizer(prompts):
            return clip.torch.ones((len(prompts), 4), dtype=clip.torch.float32)

        return DummyModel(), preprocess, tokenizer

    monkeypatch.setattr(clip, "_load_openclip", fake_load_openclip)

    cm = clip.ClipModel(
        backend="openvino-cpu",
        labels=["Alpha"],
        prompts={"Alpha": ["alpha"]},
        prompt_prefix="",
    )

    assert cm.backend == "cpu"
    assert cm.use_openvino is False


def test_openvino_features_are_unit_normalized(monkeypatch):
    if clip.torch is None or clip.Image is None:
        pytest.skip("clip model dependencies missing")

    _install_openvino_stubs(
        monkeypatch,
        text_output=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        image_output=np.array([[2.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    )

    cm = clip.ClipModel(
        backend="openvino-cpu",
        labels=["Alpha"],
        prompts={"Alpha": ["alpha", "alpha two"]},
        prompt_prefix="",
    )

    text_features = cm._build_text_features_openvino(["Alpha"], {"Alpha": ["alpha"]})
    assert np.isclose(np.linalg.norm(text_features[0].numpy()), 1.0, atol=1e-6)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    embedding = cm._encode_embedding_openvino(image)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-6)


def test_predict_species_openvino_topk(monkeypatch):
    if clip.torch is None or clip.Image is None:
        pytest.skip("clip model dependencies missing")

    _install_openvino_stubs(
        monkeypatch,
        text_output=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        image_output=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )

    cm = clip.ClipModel(
        backend="openvino-cpu",
        labels=["Alpha", "Beta", "Gamma"],
        prompts={"Alpha": ["alpha"], "Beta": ["beta"], "Gamma": ["gamma"]},
        prompt_prefix="",
    )

    monkeypatch.setattr(
        cm,
        "_predict_logits_openvino",
        lambda image, bbox_xyxy: np.array([0.5, 3.0, 1.5], dtype=np.float64),
    )

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    pred = cm.predict_species(image, topk=1)
    assert pred.label == "Beta"

    preds = cm.predict_species(image, topk=2)
    assert [p.label for p in preds] == ["Beta", "Gamma"]
    assert preds[0].probability >= preds[1].probability
