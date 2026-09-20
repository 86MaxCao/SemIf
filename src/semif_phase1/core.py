"""Shared input validation, prompts, model loading, and numeric helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def _validate_images(images) -> None:
    """Validate the multimodal branch of state: local images only.

    Remote URLs are rejected so scoring never triggers implicit network
    requests; callers that want remote images download and pin them first.
    """
    if not isinstance(images, list) or not images:
        raise ValueError("state.images must be a nonempty list when present")
    for image in images:
        if not isinstance(image, dict):
            raise ValueError("Each image must be an object with a path or bytes field")
        has_path = bool(isinstance(image.get("path"), str) and image["path"])
        has_bytes = bool(isinstance(image.get("bytes"), (bytes, bytearray)) and image.get("format"))
        if not (has_path ^ has_bytes):
            raise ValueError("Each image needs exactly one of: local path, or bytes with format")
        if has_path and re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", image["path"]):
            raise ValueError(f"Image paths must be local files, not URLs: {image['path']!r}")
        if has_path and not Path(image["path"]).is_file():
            raise ValueError(f"Image file not found: {image['path']!r}")


def row_images(row: dict) -> list[dict]:
    """Return the multimodal image entries of a validated row, or []."""
    state = row["state"]
    if isinstance(state, dict) and "images" in state:
        _validate_images(state["images"])
        return state["images"]
    return []


def image_digest(image: dict) -> str:
    """SHA-256 over the image payload, independent of path or transport."""
    if isinstance(image.get("bytes"), (bytes, bytearray)):
        return hashlib.sha256(image["bytes"]).hexdigest()
    path = Path(image["path"])
    digest_ = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest_.update(chunk)
    return digest_.hexdigest()


def validate_row(row: dict) -> None:
    required = {"id", "state", "question", "options"}
    if not required <= row.keys():
        raise ValueError(f"Row is missing fields: {sorted(required - row.keys())}")
    if not all(isinstance(row[key], str) and row[key] for key in ("id", "question")):
        raise ValueError("id and question must be nonempty strings")
    state = row["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("state must be a nonempty string, object, or array")
    if isinstance(state, dict) and "images" in state:
        # Multimodal rows carry structured evidence; validate the images and
        # skip the JSON-serialization text path entirely.
        _validate_images(state["images"])
        if not isinstance(state.get("text"), (str, type(None))):
            raise ValueError("state.text, when present, must be a string")
    else:
        try:
            json.dumps(state, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("state must be finite JSON-compatible data") from error
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS):
        raise ValueError("options must contain 2-16 entries")
    ids = []
    for option in options:
        if not isinstance(option, dict) or not isinstance(option.get("id"), str) or not isinstance(option.get("description"), str):
            raise ValueError("Each option needs string id and description fields")
        ids.append(option["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Option IDs must be unique")


def direct_messages(row: dict) -> list[dict]:
    validate_row(row)
    payload = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [
            {"letter": LETTERS[index], "description": option["description"]}
            for index, option in enumerate(row["options"])
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def softmax(values: list[float]) -> list[float]:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("Need at least two finite scores")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_causal_model(source: str, revision: str):
    """Load one pinned causal model on the sole visible CUDA device."""
    import torch
    import transformers

    local = Path(source).exists()
    if not local and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Remote models require a pinned 40-character commit revision")
    if local and not revision:
        raise ValueError("Local models require an explicit manifest/revision string")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one CUDA GPU, for example with CUDA_VISIBLE_DEVICES")
    common = {"revision": None if local else revision, "local_files_only": local, "trust_remote_code": False}
    config = transformers.AutoConfig.from_pretrained(source, **common)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    cls = transformers.AutoModelForCausalLM
    if config.model_type in {"qwen3_5", "qwen3_5_text"}:
        cls = getattr(transformers, "Qwen3_5ForCausalLM", None)
        if cls is None:
            raise RuntimeError("Installed transformers lacks the native Qwen3.5 model")
        config = config.get_text_config()
    model, loading = cls.from_pretrained(
        source,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        output_loading_info=True,
        **common,
    )
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Checkpoint did not load completely: {loading}")
    model.eval()
    metadata = {
        "source": source,
        "revision": revision,
        "dtype": "bfloat16",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    return model, tokenizer, metadata
