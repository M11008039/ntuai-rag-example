from pathlib import Path

import pytest

from telegram_rag_chatbot.loaders import (
    UnsupportedFileTypeError,
    is_supported_file,
    load_document,
)


def test_is_supported_file_accepts_known_extensions() -> None:
    assert is_supported_file(Path("a.pdf"))
    assert is_supported_file(Path("a.DOCX"))
    assert is_supported_file(Path("a.markdown"))


def test_load_text_document(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("# 標題\n\n內容", encoding="utf-8")

    documents = load_document(path)

    assert len(documents) == 1
    assert "內容" in documents[0].page_content
    assert documents[0].metadata["source"] == "note.md"


def test_load_empty_text_document_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   ", encoding="utf-8")

    assert load_document(path) == []


def test_load_document_rejects_unsupported_file_type(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"not really an image")

    with pytest.raises(UnsupportedFileTypeError):
        load_document(path)
