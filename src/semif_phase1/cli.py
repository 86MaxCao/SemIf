"""Create-only JSONL command line scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import row_images, validate_row
from .backends import TorchDirectBackend
from .reranker import score as reranker_score
from .serial import SerialPrefixScorer
from .shared import score_shared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "serial", "shared", "reranker"), required=True)
    parser.add_argument("--backend", choices=("torch", "mlx", "nanovllm"), default="torch")
    parser.add_argument("--mlx-bits", type=int, choices=(4, 8), help="Quantize MLX weights in memory; default preserves source precision")
    parser.add_argument("--mlx-cache-limit-mib", type=int,
                        help="MLX inactive allocation cache in MiB (default: 256; 0 disables caching)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists() or args.max_tokens < 1:
        parser.error("Output must be new and max-tokens must be positive")
    if args.mlx_bits and args.backend != "mlx":
        parser.error("--mlx-bits requires --backend mlx")
    if args.mlx_cache_limit_mib is not None:
        if args.backend != "mlx":
            parser.error("--mlx-cache-limit-mib requires --backend mlx")
        if args.mlx_cache_limit_mib < 0:
            parser.error("--mlx-cache-limit-mib must be nonnegative")
    if args.backend == "mlx" and args.mode == "reranker":
        parser.error("MLX supports direct, serial, and shared modes; reranker requires torch")
    if args.backend == "nanovllm" and args.mode != "direct":
        parser.error("The nanovllm backend supports direct mode only")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    for row in rows:
        validate_row(row)
    if args.backend == "mlx":
        from . import mlx_backend

        cache_limit_mib = (mlx_backend.DEFAULT_CACHE_LIMIT_MIB if args.mlx_cache_limit_mib is None
                           else args.mlx_cache_limit_mib)
        model, tokenizer, metadata = mlx_backend.load_model(
            args.model, args.revision, args.mlx_bits, cache_limit_mib=cache_limit_mib)
        direct, serial, shared = mlx_backend.score, mlx_backend.SerialPrefixScorer, mlx_backend.score_shared
        backend = None
    elif args.backend == "nanovllm":
        from .backends import NanoVLLMBackend, NanoVLLMMultimodalBackend

        if any(row_images(row) for row in rows):
            backend = NanoVLLMMultimodalBackend(args.model, args.revision)
        else:
            backend = NanoVLLMBackend(args.model, args.revision)
        model = tokenizer = None
        metadata = backend.model_info
    else:
        backend = TorchDirectBackend(args.model, args.revision)
        model, tokenizer, metadata = backend.model, backend.tokenizer, backend.model_info
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        if args.mode == "shared":
            results, timing = shared(model, tokenizer, rows, metadata, args.max_tokens)
            for result in results:
                destination.write(json.dumps({**result, "shared_timing": timing}, allow_nan=False) + "\n")
        elif args.mode == "serial":
            scorer = serial(model, tokenizer, metadata, args.max_tokens)
            for row in rows:
                destination.write(json.dumps(scorer.score(row), allow_nan=False) + "\n")
                destination.flush()
        elif backend is not None and args.mode == "direct":
            # DecisionBackend protocol path: the backend owns model loading
            # and batch scoring; rows keep their input order.
            for result in backend.score_rows(rows, args.max_tokens):
                destination.write(json.dumps(result, allow_nan=False) + "\n")
                destination.flush()
        else:
            scorer = direct if args.mode == "direct" else reranker_score
            for row in rows:
                destination.write(json.dumps(scorer(model, tokenizer, row, metadata, args.max_tokens), allow_nan=False) + "\n")
                destination.flush()


if __name__ == "__main__":
    main()
