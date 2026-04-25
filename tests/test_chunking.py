from langchain_core.documents import Document
import pytest

from telegram_rag_chatbot.chunking import split_documents


def test_split_documents_returns_empty_for_no_documents() -> None:
    assert split_documents([], chunk_size=100, chunk_overlap=10) == []


def test_split_documents_rejects_overlap_larger_than_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_documents([Document(page_content="hello")], chunk_size=10, chunk_overlap=10)


def test_split_documents_preserves_metadata_and_adds_chunk_number() -> None:
    docs = [
        Document(
            page_content="第一段內容。\n\n第二段內容。\n\n第三段內容。",
            metadata={"source": "example.md"},
        )
    ]

    chunks = split_documents(docs, chunk_size=12, chunk_overlap=2)

    assert chunks
    assert all(chunk.metadata["source"] == "example.md" for chunk in chunks)
    assert [chunk.metadata["chunk"] for chunk in chunks] == list(range(1, len(chunks) + 1))
