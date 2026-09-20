"""Compare the torch and nanovllm backends on a committed fixture.

Runs both backends on the same rows with the same weights, then reports
row-level alignment (argmax agreement, logit deltas, prompt hashes) and
warm wall-clock throughput. Timing scope matches shape777.py: warm model,
including prompt construction, tokenization, transfers, forwards, and
CPU readout; the comparison pass runs after a warmup scoring pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

from semif_phase1.backends import NanoVLLMBackend, NanoVLLMRerankerBackend, TorchDirectBackend


def timed_pass(backend, rows, max_tokens):
    torch.cuda.synchronize()
    started = time.perf_counter()
    results = backend.score_rows(rows, max_tokens)
    torch.cuda.synchronize()
    return results, time.perf_counter() - started


def run_backend(kind, model, revision, rows, max_tokens, runs):
    if kind == "direct":
        torch_backend = TorchDirectBackend(model, revision)
        warm = torch_backend.score_rows(rows[:2], max_tokens)
        timings, results = [], None
        for _ in range(runs):
            results, elapsed = timed_pass(torch_backend, rows, max_tokens)
            timings.append(elapsed)
        del torch_backend
        torch.cuda.empty_cache()
        nano_backend = NanoVLLMBackend(model, revision)
        nano_backend.score_rows(rows[:2], max_tokens)
        nano_timings, nano_results = [], None
        for _ in range(runs):
            nano_results, elapsed = timed_pass(nano_backend, rows, max_tokens)
            nano_timings.append(elapsed)
        return {
            "torch": {"results": results, "timings": timings},
            "nanovllm": {"results": nano_results, "timings": nano_timings},
        }
    else:
        from semif_phase1.reranker import score as reranker_score
        from semif_phase1.core import load_causal_model

        model_handle, tokenizer, metadata = load_causal_model(model, revision)

        class TorchReranker:
            def __init__(self):
                self.model_info = metadata

            def score_rows(self, rows, max_tokens):
                return [
                    reranker_score(model_handle, tokenizer, row, metadata, max_tokens)
                    for row in rows
                ]

        torch_backend = TorchReranker()
        torch_backend.score_rows(rows[:2], max_tokens)
        timings, results = [], None
        for _ in range(runs):
            results, elapsed = timed_pass(torch_backend, rows, max_tokens)
            timings.append(elapsed)
        del model_handle
        torch.cuda.empty_cache()
        nano_backend = NanoVLLMRerankerBackend(model, revision)
        nano_backend.score_rows(rows[:2], max_tokens)
        nano_timings, nano_results = [], None
        for _ in range(runs):
            nano_results, elapsed = timed_pass(nano_backend, rows, max_tokens)
            nano_timings.append(elapsed)
        return {
            "torch": {"results": results, "timings": timings},
            "nanovllm": {"results": nano_results, "timings": nano_timings},
        }


def compare(torch_results, nano_results, mode, near_tie_margin=0.1):
    assert len(torch_results) == len(nano_results)
    argmax_match = 0
    decisive_total = decisive_match = 0
    near_tie_disagreements = 0
    max_abs_logit_diff = 0.0
    hash_match = hash_total = 0
    max_prob_diff = 0.0
    for torch_row, nano_row in zip(torch_results, nano_results):
        assert torch_row["id"] == nano_row["id"]
        torch_probs = torch_row["probabilities"]
        nano_probs = nano_row["probabilities"]
        assert len(torch_probs) == len(nano_probs)
        torch_argmax = max(range(len(torch_probs)), key=lambda i: torch_probs[i])
        nano_argmax = max(range(len(nano_probs)), key=lambda i: nano_probs[i])
        agrees = torch_argmax == nano_argmax
        if agrees:
            argmax_match += 1
        # A near tie is a row where the torch path itself is undecided:
        # its top-two probability gap is below the margin. Disagreements on
        # such rows carry little signal; report them separately.
        ordered = sorted(torch_probs, reverse=True)
        decisive = len(ordered) < 2 or (ordered[0] - ordered[1]) >= near_tie_margin
        if decisive:
            decisive_total += 1
            if agrees:
                decisive_match += 1
        elif not agrees:
            near_tie_disagreements += 1
        max_abs_logit_diff = max(
            max_abs_logit_diff,
            max(abs(a - b) for a, b in zip(torch_row["option_logits"], nano_row["option_logits"])),
        )
        max_prob_diff = max(
            max_prob_diff, max(abs(a - b) for a, b in zip(torch_probs, nano_probs))
        )
        if mode == "direct":
            if torch_row["prompt_sha256"] == nano_row["prompt_sha256"]:
                hash_match += 1
            hash_total += 1
        else:
            for a, b in zip(torch_row["option_prompt_sha256"], nano_row["option_prompt_sha256"]):
                if a == b:
                    hash_match += 1
                hash_total += 1
    return {
        "rows": len(torch_results),
        "argmax_agreement": argmax_match / len(torch_results),
        "decisive_rows": decisive_total,
        "decisive_argmax_agreement": (decisive_match / decisive_total) if decisive_total else None,
        "near_tie_disagreements": near_tie_disagreements,
        "near_tie_margin": near_tie_margin,
        "max_abs_logit_diff": max_abs_logit_diff,
        "max_abs_prob_diff": max_prob_diff,
        "prompt_hash_match": f"{hash_match}/{hash_total}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "reranker"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")

    raw = run_backend(args.mode, args.model, args.revision, rows, args.max_tokens, args.runs)

    report = {
        "version": "backend-compare-v1",
        "mode": args.mode,
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "model": {
            "source": args.model,
            "revision": args.revision,
        },
        "hardware": torch.cuda.get_device_name(0),
        "timing_scope": (
            "Warm model; median of N timed passes, each covering prompt construction, "
            "tokenization, transfers, forwards, and CPU readout over all rows."
        ),
        "runs": args.runs,
        "alignment": compare(raw["torch"]["results"], raw["nanovllm"]["results"], args.mode),
        "timing": {},
    }
    for side in ("torch", "nanovllm"):
        timings = raw[side]["timings"]
        median = statistics.median(timings)
        report["timing"][side] = {
            "median_seconds": median,
            "all_seconds": timings,
            "decisions_per_second": len(rows) / median,
            "forward_seconds": raw[side]["results"][0].get("forward_seconds"),
        }
    report["timing"]["speedup"] = (
        report["timing"]["torch"]["median_seconds"] / report["timing"]["nanovllm"]["median_seconds"]
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "mode": args.mode,
        "rows": report["alignment"]["rows"],
        "argmax_agreement": report["alignment"]["argmax_agreement"],
        "decisive_argmax_agreement": report["alignment"]["decisive_argmax_agreement"],
        "near_tie_disagreements": report["alignment"]["near_tie_disagreements"],
        "max_abs_logit_diff": round(report["alignment"]["max_abs_logit_diff"], 4),
        "prompt_hash_match": report["alignment"]["prompt_hash_match"],
        "torch_median_s": round(report["timing"]["torch"]["median_seconds"], 3),
        "nanovllm_median_s": round(report["timing"]["nanovllm"]["median_seconds"], 3),
        "speedup": round(report["timing"]["speedup"], 2),
    }, indent=2))


if __name__ == "__main__":
    main()
