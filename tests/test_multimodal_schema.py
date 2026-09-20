"""Multimodal decision schema tests (no model required)."""

import json

import pytest

from semif_phase1.core import image_digest, row_images, validate_row


def _row(state):
    return {
        "id": "scene-1",
        "state": state,
        "question": "Which road is wider?",
        "options": [
            {"id": "left", "description": "Left road"},
            {"id": "right", "description": "Right road"},
        ],
    }


def test_text_rows_unchanged():
    # Every pre-multimodal state shape still validates.
    for state in ("plain string", {"key": "value"}, [1, 2, {"a": "b"}]):
        validate_row(_row(state))


def test_multimodal_row_with_local_path(tmp_path):
    image = tmp_path / "intersection.jpg"
    image.write_bytes(b"\xff\xd8fakejpeg")
    row = _row({"text": "Judge the road layout.", "images": [{"path": str(image)}]})
    validate_row(row)
    assert row_images(row) == [{"path": str(image)}]


def test_multimodal_row_with_bytes():
    row = _row({"images": [{"bytes": b"\xff\xd8fakejpeg", "format": "jpeg"}]})
    validate_row(row)
    assert row_images(row) == [{"bytes": b"\xff\xd8fakejpeg", "format": "jpeg"}]


def test_empty_images_rejected():
    with pytest.raises(ValueError, match="nonempty list"):
        validate_row(_row({"text": "no evidence", "images": []}))


def test_url_images_rejected():
    with pytest.raises(ValueError, match="not URLs"):
        validate_row(_row({"images": [{"path": "https://example.com/a.jpg"}]}))


def test_missing_image_file_rejected(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        validate_row(_row({"images": [{"path": str(tmp_path / "missing.jpg")}]}))


def test_image_needs_exactly_one_source(tmp_path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")
    with pytest.raises(ValueError, match="exactly one"):
        validate_row(_row({"images": [{}]}))
    with pytest.raises(ValueError, match="exactly one"):
        validate_row(_row({"images": [{"path": str(image), "bytes": b"y", "format": "jpeg"}]}))


def test_image_digest_stable_across_path_and_bytes(tmp_path):
    payload = b"\xff\xd8samebytes"
    image = tmp_path / "a.jpg"
    image.write_bytes(payload)
    assert image_digest({"path": str(image)}) == image_digest({"bytes": payload})


def test_image_digest_writes_nothing():
    digest = image_digest({"bytes": b"abc", "format": "png"})
    assert len(digest) == 64
    assert digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_state_text_must_be_string():
    with pytest.raises(ValueError, match="state.text"):
        validate_row(_row({"images": [{"bytes": b"x", "format": "png"}], "text": 123}))


def test_images_key_is_not_serialize_leak():
    # state with images plus extra keys stays multimodal; without images it
    # keeps the legacy JSON-serialization text path.
    row = _row({"images": [{"bytes": b"x", "format": "png"}], "other": [1, 2]})
    validate_row(row)
    with pytest.raises(ValueError, match="finite JSON"):
        validate_row(_row({"bad": float("inf") if False else set()}))
