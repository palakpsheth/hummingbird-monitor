# src/hbmon/clip_model.py
"""
CLIP-powered species classification + image embeddings.

This module uses OpenCLIP (open-clip-torch) to:
  1) classify a cropped hummingbird image into a small set of likely species
  2) produce a stable, normalized embedding vector for re-identification / clustering

Design notes:
- Supports multiple backends: PyTorch (CPU/CUDA) and OpenVINO (CPU/GPU)
- Uses cosine similarity between image and per-class text embeddings.
- Returns a probability distribution via softmax over class similarities.
- Embeddings are L2-normalized float32 numpy arrays.
- OpenVINO models are converted and cached on first initialization

Environment variables (optional):
- HBMON_DEVICE: "cpu" (default), "cuda", "openvino-cpu", or "openvino-gpu"
- HBMON_CLIP_MODEL: e.g. "ViT-B-32" (default)
- HBMON_CLIP_PRETRAINED: e.g. "openai" (default)
- HBMON_CLIP_PROMPT_PREFIX: e.g. "a photo of " (default)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import numpy as np

"""
This module wraps the OpenCLIP model for species classification and embedding
extraction. In order to keep the hbmon package importable on systems where
heavy optional dependencies (torch, open-clip-torch, Pillow) are not
available, we avoid importing those libraries at module import time. Instead we
attempt the imports and, if they fail, defer raising an exception until a
ClipModel is actually constructed. This means that simply importing
``hbmon.clip_model`` will not immediately raise an ImportError if the
dependencies are missing; however attempting to instantiate ``ClipModel`` will.

During testing on constrained environments where torch or open_clip are not
installed, you can still import this module and test unrelated helper
functions. Only the parts of the API that rely on the unavailable libraries
will raise an error at runtime.
"""

# Attempt to import heavy optional dependencies.  If unavailable, we set
# corresponding symbols to ``None`` and defer raising until runtime when
# constructing a ClipModel.
try:  # pragma: no cover
    import torch  # type: ignore[attr-defined]
    import open_clip  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    open_clip = None  # type: ignore[assignment]

try:  # pragma: no cover
    from PIL import Image  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]


# ----------------------------
# Public data structures
# ----------------------------

@dataclass(frozen=True)
class SpeciesPrediction:
    label: str
    probability: float
    # Full probability distribution across all known labels
    probs: dict[str, float]
    # Raw cosine similarities (before softmax), useful for debugging
    logits: dict[str, float]


# ----------------------------
# Defaults (Southern California friendly)
# ----------------------------

DEFAULT_SPECIES: list[str] = [
    "Anna's Hummingbird",
    "Allen's Hummingbird",
    "Rufous Hummingbird",
    "Costa's Hummingbird",
    "Black-chinned Hummingbird",
    "Calliope Hummingbird",
    "Broad-billed Hummingbird",
    "Hummingbird (unknown species)",
]

# Multiple prompts per class tends to improve CLIP performance a bit.
DEFAULT_PROMPTS: dict[str, list[str]] = {
    "Anna's Hummingbird": [
        "an Anna's hummingbird at a bird feeder",
        "a close-up photo of an Anna's hummingbird",
        "an Anna's hummingbird hovering",
    ],
    "Allen's Hummingbird": [
        "an Allen's hummingbird at a bird feeder",
        "a close-up photo of an Allen's hummingbird",
        "an Allen's hummingbird hovering",
    ],
    "Rufous Hummingbird": [
        "a rufous hummingbird at a bird feeder",
        "a close-up photo of a rufous hummingbird",
        "a rufous hummingbird hovering",
    ],
    "Costa's Hummingbird": [
        "a Costa's hummingbird at a bird feeder",
        "a close-up photo of a Costa's hummingbird",
        "a Costa's hummingbird hovering",
    ],
    "Black-chinned Hummingbird": [
        "a black-chinned hummingbird at a bird feeder",
        "a close-up photo of a black-chinned hummingbird",
        "a black-chinned hummingbird hovering",
    ],
    "Calliope Hummingbird": [
        "a Calliope hummingbird at a bird feeder",
        "a close-up photo of a Calliope hummingbird",
        "a Calliope hummingbird hovering",
    ],
    "Broad-billed Hummingbird": [
        "a broad-billed hummingbird at a bird feeder",
        "a close-up photo of a broad-billed hummingbird",
        "a broad-billed hummingbird hovering",
    ],
    "Hummingbird (unknown species)": [
        "a hummingbird at a bird feeder",
        "a close-up photo of a hummingbird",
        "a hummingbird hovering",
    ],
}


# ----------------------------
# Helpers
# ----------------------------

def _get_env(name: str, default: str) -> str:
    import os
    v = os.getenv(name)
    return v if v is not None and v.strip() else default


def _softmax(x: np.ndarray) -> np.ndarray:
    # numerically stable softmax
    x = x.astype(np.float64, copy=False)
    x = x - np.max(x)
    exp = np.exp(x)
    return (exp / np.sum(exp)).astype(np.float64)


def _l2_normalize_torch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _ensure_rgb_pil(image_bgr: np.ndarray) -> Image.Image:
    """
    Convert OpenCV-style BGR uint8 image to PIL RGB.
    """
    if image_bgr is None:
        raise ValueError("image_bgr is None")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR image; got shape {image_bgr.shape}")

    if image_bgr.dtype != np.uint8:
        # We can support float images too, but uint8 is strongly preferred.
        arr = np.clip(image_bgr, 0, 255).astype(np.uint8)
    else:
        arr = image_bgr

    # BGR -> RGB
    rgb = arr[:, :, ::-1]
    # Construct a PIL image from the RGB array.  Do not pass the deprecated
    # 'mode' argument; Pillow will infer RGB from the array shape.  See
    # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray
    img = Image.fromarray(rgb)
    # Ensure the mode is RGB (convert if necessary)
    return img.convert("RGB")


def crop_bgr(image_bgr: np.ndarray, bbox_xyxy: tuple[int, int, int, int] | None) -> np.ndarray:
    """
    Crop image to bbox (x1,y1,x2,y2). Coordinates are clamped to image bounds.
    Returns original if bbox is None.
    """
    if bbox_xyxy is None:
        return image_bgr
    x1, y1, x2, y2 = bbox_xyxy
    h, w = image_bgr.shape[:2]
    x1c = max(0, min(w - 1, int(x1)))
    y1c = max(0, min(h - 1, int(y1)))
    x2c = max(0, min(w, int(x2)))
    y2c = max(0, min(h, int(y2)))
    if x2c <= x1c or y2c <= y1c:
        return image_bgr
    return image_bgr[y1c:y2c, x1c:x2c]


# ----------------------------
# Model loading (cached)
# ----------------------------

@lru_cache(maxsize=4)
def _load_openclip(model_name: str, pretrained: str, device: str) -> tuple[Any, Any, Any]:
    """
    Returns (model, preprocess, tokenizer)
    """
    if open_clip is None or torch is None:  # pragma: no cover
        raise RuntimeError(
            "open-clip-torch and torch must be installed to load OpenCLIP models"
        )
    # When using pretrained weights, ensure the model config matches the activation used
    # during training.  Many OpenAI pretrained checkpoints were trained with the
    # QuickGELU activation even though the base model config uses GELU.  If we detect
    # that the caller has not explicitly opted into a quickgelu model variant, append
    # ``-quickgelu`` to the model name to avoid mismatched activations (see
    # https://github.com/mlfoundations/open_clip for details).  Only adjust the
    # model name when the ``pretrained`` argument is non-empty and the supplied
    # name does not already contain ``quickgelu``.  This prevents the user from
    # inadvertently hitting a runtime warning issued by open_clip.
    adj_name = model_name
    try:
        lower = model_name.lower()
    except Exception:
        lower = ""
    if pretrained and isinstance(pretrained, str):
        if "quickgelu" not in lower:
            # Heuristically adjust to the quickgelu variant when using a known
            # pretrained model.  We avoid appending twice if the suffix is already
            # present.  This behaviour parallels guidance in the OpenCLIP documentation.
            adj_name = f"{model_name}-quickgelu"
    # Suppress user warnings about activation mismatches during model creation.
    import warnings
    with warnings.catch_warnings():  # type: ignore[attr-defined]
        warnings.filterwarnings(
            "ignore",
            message=".*QuickGELU.*",
            category=UserWarning,
            module="open_clip.*",
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=adj_name,
            pretrained=pretrained,
        )
    tokenizer = open_clip.get_tokenizer(model_name)

    model = model.to(device)
    model.eval()

    return model, preprocess, tokenizer


# ----------------------------
# Main class
# ----------------------------

class ClipModel:
    """
    Small wrapper around OpenCLIP for:
      - classifying species via text prompts
      - producing normalized embeddings for re-ID

    Typical usage:
      cm = ClipModel()
      pred = cm.predict_species(crop_bgr_img)
      emb = cm.encode_embedding(crop_bgr_img)
    """

    def __init__(
        self,
        *,
        device: str | None = None,  # Kept for backward compatibility, maps to backend
        backend: str | None = None,
        model_name: str | None = None,
        pretrained: str | None = None,
        prompt_prefix: str | None = None,
        labels: list[str] | None = None,
        prompts: Mapping[str, list[str]] | None = None,
    ) -> None:
        # If optional dependencies are missing, raise a clear error now.  We do
        # this here rather than at module import time so that other parts of
        # hbmon can still be imported without the ML stack available.
        if torch is None or open_clip is None or Image is None:  # pragma: no cover
            raise RuntimeError(
                "open-clip-torch, torch and Pillow are required to instantiate ClipModel."
            )

        # Backend selection: prefer explicit backend, fall back to device (for compatibility)
        self.backend = backend or device or _get_env("HBMON_DEVICE", "cpu")
        self.model_name = model_name or _get_env("HBMON_CLIP_MODEL", "ViT-B-32")
        self.pretrained = pretrained or _get_env("HBMON_CLIP_PRETRAINED", "openai")
        self.prompt_prefix = prompt_prefix or _get_env("HBMON_CLIP_PROMPT_PREFIX", "a photo of ")

        self.labels = labels or list(DEFAULT_SPECIES)
        self.prompts = dict(prompts) if prompts is not None else dict(DEFAULT_PROMPTS)

        # Determine if we're using OpenVINO
        self.use_openvino = "openvino" in self.backend.lower()
        
        # Initialize models based on backend
        if self.use_openvino:
            self._init_openvino_backend()
        else:
            self._init_pytorch_backend()

        # Prepare class text embeddings (cached per instance)
        self._text_features = self._build_text_features(self.labels, self.prompts)
    
    def _init_pytorch_backend(self) -> None:
        """Initialize PyTorch backend (CPU or CUDA)."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Map backend to PyTorch device
        if self.backend == "cuda":
            device = "cuda"
        else:
            device = "cpu"
        
        self.device = device
        logger.info(f"Loading CLIP model: {self.model_name} ({self.pretrained}) - PyTorch {device.upper()} backend")
        
        # Load model components (cached)
        try:
            self._model, self._preprocess, self._tokenizer = _load_openclip(
                self.model_name, self.pretrained, self.device
            )
            self._ov_image_model = None
            self._ov_text_model = None
        except Exception as exc:  # pragma: no cover - exercised indirectly in tests
            raise RuntimeError(f"Failed to load CLIP model: {exc}") from exc
    
    def _init_openvino_backend(self) -> None:
        """Initialize OpenVINO backend (CPU or GPU) with model conversion if needed."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Import OpenVINO utilities
        try:
            from hbmon.openvino_utils import (
                is_openvino_available,
                load_openvino_clip,
                select_clip_device,
            )
        except ImportError:  # pragma: no cover
            logger.warning("OpenVINO not available, falling back to PyTorch CPU")
            self.backend = "cpu"
            self.use_openvino = False
            self._init_pytorch_backend()
            return
        
        if not is_openvino_available():  # pragma: no cover
            logger.warning("OpenVINO not installed, falling back to PyTorch CPU")
            self.backend = "cpu"
            self.use_openvino = False
            self._init_pytorch_backend()
            return
        
        # Select OpenVINO device
        ov_device = select_clip_device(self.backend)
        logger.info(f"Loading CLIP model: {self.model_name} ({self.pretrained}) - OpenVINO {ov_device} backend")
        
        # Try to load cached OpenVINO model
        try:
            cached_models = load_openvino_clip(self.model_name, self.pretrained, ov_device)
            if cached_models is not None:
                self._ov_image_model, self._ov_text_model = cached_models
                logger.info("Loaded cached OpenVINO CLIP model")
            else:
                # No cached model, need to convert
                logger.info("No cached OpenVINO model found, converting from PyTorch...")
                self._convert_and_cache_openvino(ov_device)
        except Exception as exc:  # pragma: no cover
            logger.error(f"Failed to load OpenVINO CLIP model: {exc}")
            logger.warning("Falling back to PyTorch CPU")
            self.backend = "cpu"
            self.use_openvino = False
            self._init_pytorch_backend()
            return
        
        # Still need PyTorch components for preprocessing and tokenization
        self.device = "cpu"  # Use CPU for PyTorch preprocessing
        try:
            self._model, self._preprocess, self._tokenizer = _load_openclip(
                self.model_name, self.pretrained, self.device
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to load CLIP preprocessing components: {exc}") from exc
    
    def _convert_and_cache_openvino(self, ov_device: str) -> None:
        """Convert PyTorch CLIP model to OpenVINO and cache it."""
        import logging
        logger = logging.getLogger(__name__)
        
        from hbmon.openvino_utils import convert_clip_to_openvino
        
        # First load PyTorch model for conversion
        logger.info("Loading PyTorch model for conversion...")
        pytorch_model, preprocess, tokenizer = _load_openclip(
            self.model_name, self.pretrained, "cpu"
        )
        
        # Create example inputs for tracing
        logger.info("Creating example inputs for model tracing...")
        # Image input: batch_size=1, channels=3, height=224, width=224 (typical CLIP input)
        example_image = torch.randn(1, 3, 224, 224)
        # Text input: batch_size=1, sequence_length=77 (CLIP's max text length)
        example_text = torch.randint(0, 49408, (1, 77))  # 49408 is vocab size
        
        # Convert to OpenVINO
        logger.info("Converting CLIP model to OpenVINO IR (this may take 30-60 seconds)...")
        image_model_ov, text_model_ov = convert_clip_to_openvino(
            pytorch_model,
            self.model_name,
            self.pretrained,
            example_image,
            example_text,
        )
        
        # Compile for target device
        logger.info(f"Compiling OpenVINO models for {ov_device}...")
        from openvino import Core  # type: ignore
        core = Core()
        self._ov_image_model = core.compile_model(image_model_ov, ov_device)
        self._ov_text_model = core.compile_model(text_model_ov, ov_device)
        
        logger.info("OpenVINO CLIP model conversion and caching complete")

    def _build_text_features(self, labels: list[str], prompts: Mapping[str, list[str]]) -> torch.Tensor:
        """
        Build a [num_labels, dim] tensor of normalized text features.
        For each label: average features over its prompt list.
        """
        if self.use_openvino:
            return self._build_text_features_openvino(labels, prompts)
        else:
            return self._build_text_features_pytorch(labels, prompts)
    
    def _build_text_features_pytorch(self, labels: list[str], prompts: Mapping[str, list[str]]) -> torch.Tensor:
        """Build text features using PyTorch backend."""
        with torch.no_grad():
            feats: list[torch.Tensor] = []

            for lab in labels:
                plist = prompts.get(lab, [lab])
                # Apply optional prefix to prompts (helps CLIP)
                full_prompts = [self.prompt_prefix + p for p in plist]

                tokens = self._tokenizer(full_prompts)
                tokens = tokens.to(self.device)

                text = self._model.encode_text(tokens)
                text = _l2_normalize_torch(text)

                # average prompt variants
                text_mean = text.mean(dim=0, keepdim=True)
                text_mean = _l2_normalize_torch(text_mean)

                feats.append(text_mean)

            out = torch.cat(feats, dim=0)  # [N, D]
            out = out.float()
            out = _l2_normalize_torch(out)
            return out
    
    def _build_text_features_openvino(self, labels: list[str], prompts: Mapping[str, list[str]]) -> torch.Tensor:
        """Build text features using OpenVINO backend."""
        import numpy as np
        
        feats: list[np.ndarray] = []

        for lab in labels:
            plist = prompts.get(lab, [lab])
            # Apply optional prefix to prompts (helps CLIP)
            full_prompts = [self.prompt_prefix + p for p in plist]

            tokens = self._tokenizer(full_prompts)
            
            # Run OpenVINO inference
            text_outputs = self._ov_text_model(tokens.numpy())[0]  # Get first output
            text = torch.from_numpy(text_outputs).float()
            text = _l2_normalize_torch(text)

            # average prompt variants
            text_mean = text.mean(dim=0, keepdim=True)
            text_mean = _l2_normalize_torch(text_mean)

            feats.append(text_mean.numpy())

        # Convert back to torch tensor for consistency
        out = torch.from_numpy(np.concatenate(feats, axis=0)).float()
        out = _l2_normalize_torch(out)
        return out

    def encode_embedding(self, image_bgr: np.ndarray, *, bbox_xyxy: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """
        Return an L2-normalized float32 embedding vector for the (cropped) image.
        """
        if self.use_openvino:
            return self._encode_embedding_openvino(image_bgr, bbox_xyxy=bbox_xyxy)
        else:
            return self._encode_embedding_pytorch(image_bgr, bbox_xyxy=bbox_xyxy)
    
    def _encode_embedding_pytorch(self, image_bgr: np.ndarray, *, bbox_xyxy: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """Encode embedding using PyTorch backend."""
        img = crop_bgr(image_bgr, bbox_xyxy)
        pil = _ensure_rgb_pil(img)
        with torch.no_grad():
            inp = self._preprocess(pil).unsqueeze(0).to(self.device)
            img_feat = self._model.encode_image(inp)
            img_feat = _l2_normalize_torch(img_feat).float()
            vec = img_feat[0].detach().cpu().numpy().astype(np.float32, copy=False)
        # Ensure unit norm in numpy space too
        n = float(np.linalg.norm(vec) + 1e-12)
        return (vec / n).astype(np.float32, copy=False)
    
    def _encode_embedding_openvino(self, image_bgr: np.ndarray, *, bbox_xyxy: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """Encode embedding using OpenVINO backend."""
        img = crop_bgr(image_bgr, bbox_xyxy)
        pil = _ensure_rgb_pil(img)
        # Preprocess using PyTorch (lightweight operation)
        inp = self._preprocess(pil).unsqueeze(0)
        # Run OpenVINO inference
        img_feat = self._ov_image_model(inp.numpy())[0]  # Get first output
        vec = img_feat[0].astype(np.float32, copy=False)
        # Ensure unit norm
        n = float(np.linalg.norm(vec) + 1e-12)
        return (vec / n).astype(np.float32, copy=False)

    def predict_species(
        self,
        image_bgr: np.ndarray,
        *,
        bbox_xyxy: tuple[int, int, int, int] | None = None,
        topk: int = 1,
    ) -> SpeciesPrediction | list[SpeciesPrediction]:
        """
        Predict species label(s) from the (cropped) image.

        Returns:
          - topk=1 -> SpeciesPrediction (best label)
          - topk>1 -> list[SpeciesPrediction] sorted best->worst
        """
        if topk < 1:
            raise ValueError("topk must be >= 1")

        # Get logits from appropriate backend
        if self.use_openvino:
            logits = self._predict_logits_openvino(image_bgr, bbox_xyxy)
        else:
            logits = self._predict_logits_pytorch(image_bgr, bbox_xyxy)

        probs = _softmax(logits)
        order = np.argsort(-probs)

        def make_pred(i: int) -> SpeciesPrediction:
            label = self.labels[i]
            prob = float(probs[i])
            probs_dict = {self.labels[j]: float(probs[j]) for j in range(len(self.labels))}
            logits_dict = {self.labels[j]: float(logits[j]) for j in range(len(self.labels))}
            return SpeciesPrediction(label=label, probability=prob, probs=probs_dict, logits=logits_dict)

        if topk == 1:
            return make_pred(int(order[0]))

        topk = min(topk, len(self.labels))
        preds = [make_pred(int(order[k])) for k in range(topk)]
        return preds
    
    def _predict_logits_pytorch(self, image_bgr: np.ndarray, bbox_xyxy: tuple[int, int, int, int] | None) -> np.ndarray:
        """Predict logits using PyTorch backend."""
        img = crop_bgr(image_bgr, bbox_xyxy)
        pil = _ensure_rgb_pil(img)

        with torch.no_grad():
            inp = self._preprocess(pil).unsqueeze(0).to(self.device)
            image_feat = self._model.encode_image(inp)
            image_feat = _l2_normalize_torch(image_feat).float()  # [1, D]

            # cosine similarities to per-class text features
            # text: [N, D], image: [1, D] => logits: [1, N]
            logits_t = (image_feat @ self._text_features.T).squeeze(0)  # [N]
            logits = logits_t.detach().cpu().numpy().astype(np.float64, copy=False)
        return logits
    
    def _predict_logits_openvino(self, image_bgr: np.ndarray, bbox_xyxy: tuple[int, int, int, int] | None) -> np.ndarray:
        """Predict logits using OpenVINO backend."""
        img = crop_bgr(image_bgr, bbox_xyxy)
        pil = _ensure_rgb_pil(img)
        
        # Preprocess using PyTorch (lightweight operation)
        inp = self._preprocess(pil).unsqueeze(0)
        
        # Run OpenVINO inference
        img_feat = self._ov_image_model(inp.numpy())[0]  # Get first output
        img_feat_torch = torch.from_numpy(img_feat).float()
        img_feat_torch = _l2_normalize_torch(img_feat_torch)
        
        # cosine similarities to per-class text features
        logits_t = (img_feat_torch @ self._text_features.T).squeeze(0)
        logits = logits_t.numpy().astype(np.float64, copy=False)
        return logits

    def predict_species_label_prob(
        self,
        image_bgr: np.ndarray,
        *,
        bbox_xyxy: tuple[int, int, int, int] | None = None,
    ) -> tuple[str, float]:
        """
        Convenience method: returns (best_label, best_probability)
        """
        pred = self.predict_species(image_bgr, bbox_xyxy=bbox_xyxy, topk=1)
        assert isinstance(pred, SpeciesPrediction)
        return pred.label, pred.probability

    def set_label_space(self, labels: list[str], prompts: Mapping[str, list[str]] | None = None) -> None:
        """
        Replace the label set and rebuild text embeddings.
        Useful if you want to restrict species to those in your area.
        """
        self.labels = list(labels)
        if prompts is not None:
            self.prompts = dict(prompts)
        self._text_features = self._build_text_features(self.labels, self.prompts)

    def get_labels(self) -> list[str]:
        return list(self.labels)

    def get_prompt_map(self) -> dict[str, list[str]]:
        return dict(self.prompts)

    def debug_similarity(self, image_bgr: np.ndarray, *, bbox_xyxy: tuple[int, int, int, int] | None = None) -> dict[str, float]:
        """
        Return raw cosine similarities for each label (no softmax).
        Useful for debugging which labels are close.
        """
        # Reuse the logits prediction methods
        if self.use_openvino:
            logits = self._predict_logits_openvino(image_bgr, bbox_xyxy)
        else:
            logits = self._predict_logits_pytorch(image_bgr, bbox_xyxy)
        return {self.labels[i]: float(logits[i]) for i in range(len(self.labels))}
