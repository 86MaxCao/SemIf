"""NanoVLLM backend tests.

The heavy parity test requires the nano-vllm-prefillonly package, a CUDA
GPU, and a local Qwen3 checkpoint; it skips automatically otherwise. The
validation tests run everywhere.
"""

import pytest

from semif_phase1.backends import NanoVLLMBackend, _require_pinned_revision


def test_remote_model_requires_pinned_revision():
    with pytest.raises(ValueError, match="40-character commit revision"):
        _require_pinned_revision("Qwen/Qwen3-0.6B", "main")
    _require_pinned_revision("Qwen/Qwen3-0.6B", "a" * 40)


def test_local_model_requires_manifest_revision(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    with pytest.raises(ValueError, match="manifest/revision string"):
        _require_pinned_revision(str(source), "")
    _require_pinned_revision(str(source), "local-manifest-1")


@pytest.mark.parametrize("mode", ["serial", "shared", "reranker"])
def test_nanovllm_backend_rejects_non_direct_modes(tmp_path, monkeypatch, capsys, mode):
    import json
    import sys

    from semif_phase1.cli import main

    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({
        "id": "test", "state": "Evidence", "question": "Supported?",
        "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}],
    }) + "\n")
    monkeypatch.setattr(sys, "argv", [
        "semif-score", "--backend", "nanovllm", "--mode", mode,
        "--model", "unused", "--revision", "a" * 40,
        "--input", str(source), "--output", str(tmp_path / "out.jsonl"),
    ])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "direct mode only" in capsys.readouterr().err
    assert not (tmp_path / "out.jsonl").exists()


def test_nanovllm_vs_torch_parity():
    nanovllm = pytest.importorskip("nanovllm")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("parity test requires a CUDA GPU")

    import json

    from semif_phase1.backends import TorchDirectBackend

    model_source = pytest.config.getoption("--nanovllm-model") if hasattr(pytest, "config") else None
    if model_source is None:
        pytest.skip("set --nanovllm-model to a local Qwen3 checkpoint to run parity")

    rows = [
        {
            "id": "row-1",
            "state": "The sensor logged a steady 22.4C for the last hour.",
            "question": "Is the temperature stable?",
            "options": [
                {"id": "yes", "description": "Yes"},
                {"id": "no", "description": "No"},
                {"id": "unknown", "description": "Insufficient evidence"},
            ],
        },
        {
            "id": "row-2",
            "state": "Records show two conflicting readings minutes apart.",
            "question": "Which reading is newer?",
            "options": [
                {"id": "first", "description": "The first reading"},
                {"id": "second", "description": "The second reading"},
            ],
        },
    ]

    torch_backend = TorchDirectBackend(model_source, "parity-manifest")
    nano_backend = NanoVLLMBackend(model_source, "parity-manifest")
    expected = torch_backend.score_rows(rows, 4096)
    actual = nano_backend.score_rows(rows, 4096)

    assert len(expected) == len(actual)
    for exp, act in zip(expected, actual):
        assert exp["id"] == act["id"]
        assert exp["prompt_sha256"] == act["prompt_sha256"]
        assert exp["option_ids"] == act["option_ids"]
        for reference, candidate in zip(exp["probabilities"], act["probabilities"]):
            assert candidate == pytest.approx(reference, abs=1e-2, rel=1e-2)
