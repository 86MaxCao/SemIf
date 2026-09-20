# Repository instructions

- Run commands from the repository root in an isolated environment installed with `pip install -e '.[test]'`.
- Validate changes with `pytest -q`, `(cd results/raw && sha256sum -c SHA256SUMS)`, and `python benchmarks/verify_published.py`.
- Benchmark outputs are create-only. Use a new output path and expose exactly one CUDA GPU per scorer process.
- Do not change headline claims or `results/phase1-summary.json` without committing the supporting row-level evidence, regenerating the relevant raw report, updating `results/raw/SHA256SUMS`, and updating the method/results text.
- Preserve exact model and source revisions. Use `benchmarks/fetch_sources.py` only for its listed redistributable inputs; do not commit model weights, caches, or third-party raw records.
- `webgpu-demo/` is static and has no build step. Preserve `_headers`, runtime version pins, browser-only inference, and the explicit probability limitations.
- The `nanovllm` backend (`src/semif_phase1/backends.py`) must reuse `direct.encode_prompt`/`multimodal_messages` so prompts, letter-slot validation, and prompt hashes match the torch backend byte for byte. The engine's `prefill_last_logits` APIs are deterministic readouts (no sampling); candidate order equals option order and result order equals input order.
- Image evidence is local-only: `state.images` entries are local paths or inline bytes, URLs are rejected, and every result records `image_hashes`. Do not add implicit network fetches.
- GPU parity tests (`tests/test_nanovllm_backend.py`) skip without the nano-vllm-prefillonly package, a CUDA device, or `--nanovllm-model`. Do not mark nanovllm results as headline benchmark evidence until the parity tests pass on pinned revisions.
