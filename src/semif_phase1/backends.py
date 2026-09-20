"""Pluggable decision backends.

A backend owns model loading and row scoring. The CLI dispatches to a
backend based on --backend; the protocol below is the seam that lets the
nano-vLLM engines plug in without touching the core scoring contract
(prompt construction, letter-slot validation, and probability
normalization stay in this package).
"""

from __future__ import annotations

from typing import Protocol

from .core import load_causal_model
from .direct import score as direct_score


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
