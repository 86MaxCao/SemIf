"""Pluggable decision backends.

A backend owns model loading and row scoring. The CLI dispatches to a
backend based on --backend; the protocol below is the seam that lets the
nano-vLLM engines plug in without touching the core scoring contract
(prompt construction, letter-slot validation, and probability
normalization stay in this package).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Protocol

from .core import load_causal_model, softmax
from .direct import PROMPT_VERSION, encode_prompt, score as direct_score


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
