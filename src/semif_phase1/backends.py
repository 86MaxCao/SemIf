"""Pluggable decision backends.

A backend owns model loading and row scoring. The CLI dispatches to a
backend based on --backend; the protocol below is the seam that lets the
nano-vLLM engines plug in without touching the core scoring contract
(prompt construction, letter-slot validation, and probability
normalization stay in this package).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Protocol

from .core import (
    digest,
    load_causal_model,
    multimodal_messages,
    row_images,
    image_digest,
    image_payload,
    softmax,
)
from .direct import PROMPT_VERSION, _slot_ids, encode_prompt, score as direct_score


class DecisionBackend(Protocol):
    """Scores decision rows and exposes audit metadata for results."""

    model_info: dict

    def score_rows(self, rows: list[dict], max_tokens: int) -> list[dict]:
        ...


class TorchDirectBackend:
    """Existing Transformers path: one row, one full forward per call."""

    def __init__(self, source: str, revision: str):
        self.model, self.tokenizer, self.model_info = load_causal_model(source, revision)

    def score_rows(self, rows: list[dict], max_tokens: int) -> list[dict]:
        return [
            direct_score(self.model, self.tokenizer, row, self.model_info, max_tokens)
            for row in rows
        ]


def _require_pinned_revision(source: str, revision: str) -> None:
    """Mirror core.load_causal_model's revision rules for the nano-vLLM path."""
    local = Path(source).exists()
    if not local and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Remote models require a pinned 40-character commit revision")
    if local and not revision:
        raise ValueError("Local models require an explicit manifest/revision string")


class NanoVLLMBackend:
    """nano-vLLM prefill engine: one batched forward, candidate logits only.

    Requires the nano-vLLM ``prefill_last_logits`` API (deterministic
    last-position logits, no sampling). Rows keep their input order.
    """

    def __init__(self, source: str, revision: str):
        _require_pinned_revision(source, revision)
        try:
            from nanovllm import LLM
        except ImportError as error:
            raise RuntimeError(
                "The nanovllm backend requires the nano-vllm-prefillonly package"
            ) from error
        self.engine = LLM(
            source,
            revision=revision,
            prefill_only_mode=True,
            max_tokens_hint=1,
        )
        self.tokenizer = self.engine.tokenizer
        self.model_info = {
            "source": source,
            "revision": revision,
            "backend": "nanovllm",
            "nanovllm_model_revision": getattr(self.engine.config, "model_revision", None),
        }

    def score_rows(self, rows: list[dict], max_tokens: int) -> list[dict]:
        started = time.perf_counter()
        encoded = [encode_prompt(self.tokenizer, row, max_tokens) for row in rows]
        forward_start = time.perf_counter()
        result = self.engine.prefill_last_logits(
            [ids for ids, _, _ in encoded],
            candidate_token_ids=[slots for _, slots, _ in encoded],
        )
        forward_seconds = time.perf_counter() - forward_start
        logits = result["logits"].tolist()
        outputs = []
        for row, (ids, slots, prompt_hash), row_logits in zip(rows, encoded, logits):
            outputs.append({
                "id": row["id"],
                "option_ids": [option["id"] for option in row["options"]],
                "probabilities": softmax(row_logits[: len(slots)]),
                "option_logits": row_logits[: len(slots)],
                "input_tokens": len(ids),
                "forward_seconds": forward_seconds,
                "total_seconds": time.perf_counter() - started,
                "prompt_sha256": prompt_hash,
                "prompt_version": PROMPT_VERSION,
                "model": self.model_info,
                "backend": "nanovllm",
                "readout": "prefill last-position candidate logits from the nano-vLLM engine",
                "probability_status": "conditional option score; uncalibrated as decision confidence",
            })
        return outputs


def _load_pil_image(image: dict):
    """Decode one validated image entry (bytes, local path, or http(s) URL) with PIL."""
    from io import BytesIO

    from PIL import Image

    return Image.open(BytesIO(image_payload(image))).convert("RGB")


class NanoVLLMMultimodalBackend:
    """nano-vLLM multimodal prefill: images plus text, candidate logits only.

    Requires the nano-vLLM ``prefill_last_logits_multimodal`` API. Rows may
    mix text and image evidence; both read out the same answer-slot
    contract (single uppercase-letter tokens).
    """

    def __init__(self, source: str, revision: str):
        _require_pinned_revision(source, revision)
        try:
            from nanovllm import LLM
        except ImportError as error:
            raise RuntimeError(
                "The nanovllm backend requires the nano-vllm-prefillonly package"
            ) from error
        self.engine = LLM(
            source,
            revision=revision,
            prefill_only_mode=True,
            max_tokens_hint=1,
        )
        self.tokenizer = self.engine.tokenizer
        self.model_info = {
            "source": source,
            "revision": revision,
            "backend": "nanovllm",
            "nanovllm_model_revision": getattr(self.engine.config, "model_revision", None),
        }

    def score_rows(self, rows: list[dict], max_tokens: int) -> list[dict]:
        started = time.perf_counter()
        requests = []
        slots_per_row = []
        hashes_per_row = []
        prompt_hashes = []
        for row in rows:
            images = row_images(row)
            hashes_per_row.append([image_digest(image) for image in images])
            if images:
                messages = multimodal_messages(row)
                slots = _slot_ids(self.tokenizer, len(row["options"]))
                slots_per_row.append(slots)
                prompt_hashes.append(digest(json.dumps(messages, ensure_ascii=False, default=repr)))
                requests.append({
                    "messages": messages,
                    "images": [_load_pil_image(image) for image in images],
                })
            else:
                # Text rows reuse encode_prompt so ids, slot validation, and
                # prompt hash match the text backend byte for byte.
                ids, slots, prompt_hash = encode_prompt(self.tokenizer, row, max_tokens)
                slots_per_row.append(slots)
                prompt_hashes.append(prompt_hash)
                requests.append({"input_ids": ids})
        forward_start = time.perf_counter()
        result = self.engine.prefill_last_logits_multimodal(
            requests,
            candidate_token_ids=slots_per_row,
        )
        forward_seconds = time.perf_counter() - forward_start
        logits = result["logits"].tolist()
        outputs = []
        for row, slots, hashes, prompt_hash, row_logits in zip(
            rows, slots_per_row, hashes_per_row, prompt_hashes, logits
        ):
            outputs.append({
                "id": row["id"],
                "option_ids": [option["id"] for option in row["options"]],
                "probabilities": softmax(row_logits[: len(slots)]),
                "option_logits": row_logits[: len(slots)],
                "input_tokens": result["sequence_lengths"][len(outputs)],
                "forward_seconds": forward_seconds,
                "total_seconds": time.perf_counter() - started,
                "prompt_sha256": prompt_hash,
                "prompt_version": PROMPT_VERSION,
                "model": self.model_info,
                "backend": "nanovllm",
                "modality": "vision-language" if hashes else "text",
                "image_hashes": hashes,
                "readout": "prefill last-position candidate logits from the nano-vLLM engine",
                "probability_status": "conditional option score; uncalibrated as decision confidence",
            })
        return outputs
