"""Evidence storage must preserve original payload bytes and reject corruption."""
import gzip
import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "mlx_evidence", Path(__file__).resolve().parents[1] / "benchmarks/mlx_evidence.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


@pytest.mark.parametrize("compressed", [False, True])
def test_original_checksums_accept_plain_or_gzip(tmp_path, compressed):
    payload = b'{"probabilities": [0.125, 0.875]}\n'
    path = tmp_path / "predictions.json"
    if compressed:
        path = path.with_suffix(".json.gz")
    path.write_bytes(gzip.compress(payload, mtime=0) if compressed else payload)
    manifest = tmp_path / "UNCOMPRESSED_SHA256SUMS"
    manifest.write_text(hashlib.sha256(payload).hexdigest() + "  predictions.json\n")
    assert evidence.verify_checksums(manifest) == 1
    assert evidence.read_bytes(path) == payload
    assert evidence.read_json(tmp_path / "predictions.json")["probabilities"] == [0.125, 0.875]
    path.write_bytes(gzip.compress(b"changed", mtime=0) if compressed else b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        evidence.verify_checksums(manifest)


def test_rejects_ambiguous_or_corrupt_compressed_evidence(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b"{}\n")
    compressed = path.with_suffix(".jsonl.gz")
    compressed.write_bytes(gzip.compress(b"{}\n", mtime=0))
    with pytest.raises(ValueError, match="Both plain and compressed"):
        evidence.read_bytes(path)
    path.unlink()
    compressed.write_bytes(b"not gzip")
    with pytest.raises(gzip.BadGzipFile):
        evidence.read_bytes(path)
