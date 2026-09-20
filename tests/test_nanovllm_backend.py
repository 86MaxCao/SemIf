"""NanoVLLM backend tests.

The heavy parity test requires the nano-vllm-prefillonly package, a CUDA
GPU, and a local Qwen3 checkpoint; it skips automatically otherwise. The
validation tests run everywhere.
"""

import json

import pytest
from transformers import AutoTokenizer

from semif_phase1.backends import NanoVLLMBackend, _require_pinned_revision
from semif_phase1.core import image_digest, multimodal_messages


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


@pytest.mark.parametrize("mode", ["serial", "shared"])
def test_nanovllm_backend_rejects_kv_cache_modes(tmp_path, monkeypatch, capsys, mode):
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
    assert "direct and reranker modes only" in capsys.readouterr().err
    assert not (tmp_path / "out.jsonl").exists()


def test_cli_routes_reranker_image_rows_to_multimodal(tmp_path, monkeypatch):
    import sys

    from semif_phase1 import cli
    from semif_phase1 import backends as backends_module

    image = tmp_path / "scene.jpg"
    image.write_bytes(b"\xff\xd8first")
    row = {
        "id": "scene-1",
        "state": {"text": "Judge the road layout.", "images": [{"path": str(image)}]},
        "question": "Which road is wider?",
        "options": [
            {"id": "left", "description": "Left road"},
            {"id": "right", "description": "Right road"},
        ],
    }
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(row) + "\n")
    output = tmp_path / "out.jsonl"

    selected = {}

    def fake_init(self, source, revision, multimodal=False):
        self.model_info = {"backend": "nanovllm"}
        selected["multimodal"] = multimodal

    def fake_score_rows(self, rows, max_tokens):
        return [{"id": rows[0]["id"]}]

    monkeypatch.setattr(backends_module.NanoVLLMRerankerBackend, "__init__", fake_init)
    monkeypatch.setattr(backends_module.NanoVLLMRerankerBackend, "score_rows", fake_score_rows)
    monkeypatch.setattr(sys, "argv", [
        "semif-score", "--backend", "nanovllm", "--mode", "reranker",
        "--model", "unused", "--revision", "a" * 40,
        "--input", str(source), "--output", str(output),
    ])
    cli.main()
    assert selected["multimodal"] is True
    assert json.loads(output.read_text()) == {"id": "scene-1"}


def test_cli_routes_reranker_text_rows_to_text_backend(tmp_path, monkeypatch):
    import sys

    from semif_phase1 import cli
    from semif_phase1 import backends as backends_module

    row = {
        "id": "text-1",
        "state": "The sensor logged a steady 22.4C.",
        "question": "Is the temperature stable?",
        "options": [
            {"id": "yes", "description": "Yes"},
            {"id": "no", "description": "No"},
        ],
    }
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(row) + "\n")
    output = tmp_path / "out.jsonl"

    selected = {}

    def fake_init(self, source, revision, multimodal=False):
        self.model_info = {"backend": "nanovllm"}
        selected["multimodal"] = multimodal

    def fake_score_rows(self, rows, max_tokens):
        return [{"id": rows[0]["id"]}]

    monkeypatch.setattr(backends_module.NanoVLLMRerankerBackend, "__init__", fake_init)
    monkeypatch.setattr(backends_module.NanoVLLMRerankerBackend, "score_rows", fake_score_rows)
    monkeypatch.setattr(sys, "argv", [
        "semif-score", "--backend", "nanovllm", "--mode", "reranker",
        "--model", "unused", "--revision", "a" * 40,
        "--input", str(source), "--output", str(output),
    ])
    cli.main()
    assert selected["multimodal"] is False
    assert json.loads(output.read_text()) == {"id": "text-1"}


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


def _mm_row(first_bytes=b"\xff\xd8first", second_bytes=b"\xff\xd8second"):
    return {
        "id": "scene-1",
        "state": {
            "text": "Judge the road layout.",
            "images": [
                {"bytes": first_bytes, "format": "jpeg"},
                {"bytes": second_bytes, "format": "jpeg"},
            ],
        },
        "question": "Which road is wider?",
        "options": [
            {"id": "left", "description": "Left road"},
            {"id": "right", "description": "Right road"},
            {"id": "unknown", "description": "Insufficient evidence"},
        ],
    }


def test_multimodal_messages_structure():
    messages = multimodal_messages(_mm_row())
    assert messages[0]["role"] == "system"
    content = messages[1]["content"]
    assert [part["type"] for part in content] == ["image", "image", "text"]
    payload = json.loads(content[-1]["text"])
    assert payload["criterion"] == "Which road is wider?"
    assert [option["letter"] for option in payload["options"]] == ["A", "B", "C"]
    # Image bytes never enter the text evidence; digests stand in.
    for image in payload["evidence"]["images"]:
        assert "bytes" not in image
        assert len(image["sha256"]) == 64


def test_multimodal_messages_digest_order_sensitive():
    row = _mm_row()
    swapped = _mm_row(b"\xff\xd8second", b"\xff\xd8first")
    assert multimodal_messages(row) != multimodal_messages(swapped)


def test_multimodal_messages_requires_images():
    text_row = {
        "id": "t", "state": "plain", "question": "Q?",
        "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
    }
    with pytest.raises(ValueError, match="state.images"):
        multimodal_messages(text_row)


def test_cli_routes_multimodal_rows_to_multimodal_backend(tmp_path, monkeypatch):
    import sys

    from semif_phase1 import cli
    from semif_phase1 import backends as backends_module

    image = tmp_path / "scene.jpg"
    image.write_bytes(b"\xff\xd8first")
    row = {
        "id": "scene-1",
        "state": {"text": "Judge the road layout.", "images": [{"path": str(image)}]},
        "question": "Which road is wider?",
        "options": [
            {"id": "left", "description": "Left road"},
            {"id": "right", "description": "Right road"},
        ],
    }
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(row) + "\n")
    output = tmp_path / "out.jsonl"

    selected = {}

    def fake_init(self, source, revision):
        self.model_info = {"backend": "nanovllm"}
        selected["cls"] = type(self).__name__

    monkeypatch.setattr(backends_module.NanoVLLMBackend, "__init__", fake_init)
    monkeypatch.setattr(backends_module.NanoVLLMMultimodalBackend, "__init__", fake_init)

    def fake_score_rows(self, rows, max_tokens):
        return [{"id": rows[0]["id"]}]

    monkeypatch.setattr(backends_module.NanoVLLMMultimodalBackend, "score_rows", fake_score_rows)
    monkeypatch.setattr(sys, "argv", [
        "semif-score", "--backend", "nanovllm", "--mode", "direct",
        "--model", "unused", "--revision", "a" * 40,
        "--input", str(source), "--output", str(output),
    ])
    cli.main()
    assert selected["cls"] == "NanoVLLMMultimodalBackend"
    assert json.loads(output.read_text()) == {"id": "scene-1"}


def test_nanovllm_reranker_backend_score_rows():
    """Scoring logic with a stubbed engine: pair expansion, sigmoid inversion,
    and the torch-reranker output schema."""
    import math

    pytest.importorskip("transformers")

    from semif_phase1.backends import NanoVLLMRerankerBackend
    from semif_phase1.reranker import _encode

    class StubEngine:
        def __init__(self):
            self.calls = []

        def rerank_ids(self, input_ids):
            self.calls.append(input_ids)
            import torch
            # Deliberately distinct sigmoid outputs in pair order.
            return torch.tensor([0.9, 0.2, 0.8])

    backend = NanoVLLMRerankerBackend.__new__(NanoVLLMRerankerBackend)
    backend.engine = StubEngine()
    backend.model_info = {"backend": "nanovllm"}
    backend.multimodal = False
    backend.tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/nas-tbt/tbt/checkpoint/hf_cache/Qwen3-Reranker-0.6B"
    )

    rows = [{
        "id": "row-1",
        "state": "The sensor logged a steady 22.4C for the last hour.",
        "question": "Is the temperature stable?",
        "options": [
            {"id": "yes", "description": "Yes"},
            {"id": "no", "description": "No"},
            {"id": "unknown", "description": "Insufficient evidence"},
        ],
    }]

    results = backend.score_rows(rows, 4096)
    assert len(results) == 1
    result = results[0]

    # One rerank_ids call carrying every (row, option) pair.
    expected_ids = [
        _encode(backend.tokenizer, rows[0], option, 4096)[0]
        for option in rows[0]["options"]
    ]
    assert backend.engine.calls == [expected_ids]

    # Sigmoid inverted back to log-odds; softmax matches torch reranker output.
    assert result["option_ids"] == ["yes", "no", "unknown"]
    for odds, probability, score in zip(
        result["option_logits"], result["probabilities"], (0.9, 0.2, 0.8)
    ):
        assert odds == pytest.approx(math.log(score / (1 - score)))
    assert sum(result["probabilities"]) == pytest.approx(1.0)
    assert result["independent_binary_relevance"] == pytest.approx([0.9, 0.2, 0.8])
    # Prompt hashes are identical to the torch path's encoding.
    assert result["option_prompt_sha256"] == [
        _encode(backend.tokenizer, rows[0], option, 4096)[1]
        for option in rows[0]["options"]
    ]
    assert result["prompt_version"] == "qwen3-reranker-native-options-v1"
    assert result["backend"] == "nanovllm"
    assert result["input_tokens"] == sum(len(ids) for ids in expected_ids)
    assert result["max_option_input_tokens"] == max(len(ids) for ids in expected_ids)
    assert result["pair_batches"][0]["pair_batch_size"] == 3


def test_nanovllm_reranker_multimodal_score_rows():
    """Multimodal path: rerank_batch receives (query, doc) pairs plus
    per-pair images (list for image rows, None for text rows)."""
    import math
    from io import BytesIO

    pytest.importorskip("transformers")
    PIL = pytest.importorskip("PIL.Image")

    from semif_phase1.backends import NanoVLLMRerankerBackend
    from semif_phase1.core import digest

    class StubEngine:
        def __init__(self):
            self.calls = []

        def rerank_batch(self, pairs, images=None, use_tqdm=False):
            self.calls.append((pairs, images))
            import torch
            return torch.tensor([0.9, 0.2, 0.8, 0.6])

    def png_bytes(color):
        buffer = BytesIO()
        PIL.new("RGB", (8, 8), color=color).save(buffer, format="PNG")
        return buffer.getvalue()

    backend = NanoVLLMRerankerBackend.__new__(NanoVLLMRerankerBackend)
    backend.engine = StubEngine()
    backend.model_info = {"backend": "nanovllm"}
    backend.multimodal = True
    backend.tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/nas-tbt/tbt/checkpoint/hf_cache/Qwen3-Reranker-0.6B"
    )

    rows = [
        {
            "id": "scene-1",
            "state": {
                "text": "Judge the road layout.",
                "images": [
                    {"bytes": png_bytes((255, 0, 0)), "format": "png"},
                    {"bytes": png_bytes((0, 0, 255)), "format": "png"},
                ],
            },
            "question": "Which road is wider?",
            "options": [
                {"id": "left", "description": "Left road"},
                {"id": "right", "description": "Right road"},
            ],
        },
        {
            "id": "text-1",
            "state": "The sensor logged a steady 22.4C for the last hour.",
            "question": "Is the temperature stable?",
            "options": [
                {"id": "yes", "description": "Yes"},
                {"id": "no", "description": "No"},
            ],
        },
    ]

    results = backend.score_rows(rows, 4096)
    assert len(results) == 2

    pairs, images = backend.engine.calls[0]
    assert len(pairs) == 4
    assert len(images) == 4
    # Image row: both options carry the row's two PIL images.
    for pair_images in images[:2]:
        assert isinstance(pair_images, list) and len(pair_images) == 2
        assert all(image.size == (8, 8) for image in pair_images)
    # Text row: pairs stay text-only.
    assert images[2] is None and images[3] is None

    # Pair text mirrors the torch reranker's sections.
    assert pairs[0] == (
        "Question: Which road is wider?\nCandidate answer: Left road",
        "<Instruct>: Given evidence and one possible answer to a question, determine whether "
        "the evidence supports that answer under the question's criterion. Use only the "
        "supplied evidence.\nJudge the road layout.",
    )
    assert pairs[2] == (
        "Question: Is the temperature stable?\nCandidate answer: Yes",
        "<Instruct>: Given evidence and one possible answer to a question, determine whether "
        "the evidence supports that answer under the question's criterion. Use only the "
        "supplied evidence.\nThe sensor logged a steady 22.4C for the last hour.",
    )

    # Audit hashes cover the pair text actually sent.
    scene = results[0]
    assert scene["option_prompt_sha256"] == [
        digest(f"{pairs[0][0]}\n{pairs[0][1]}"),
        digest(f"{pairs[1][0]}\n{pairs[1][1]}"),
    ]
    assert scene["independent_binary_relevance"] == pytest.approx([0.9, 0.2])
    for odds, score in zip(scene["option_logits"], (0.9, 0.2)):
        assert odds == pytest.approx(math.log(score / (1 - score)))
    assert sum(scene["probabilities"]) == pytest.approx(1.0)

    text = results[1]
    assert text["independent_binary_relevance"] == pytest.approx([0.8, 0.6])
    assert sum(text["probabilities"]) == pytest.approx(1.0)
    assert text["pair_batches"][0]["pair_batch_size"] == 4
