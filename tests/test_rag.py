from pathlib import Path

from langchain_core.documents import Document

from telegram_rag_chatbot.rag import (
    RagService,
    _format_excerpts,
    _unique_sources,
    list_ingestable_files,
)


def test_unique_sources_preserves_order_and_removes_duplicates() -> None:
    documents = [
        Document(page_content="a", metadata={"source": "a.md", "chunk": 1}),
        Document(page_content="b", metadata={"source": "a.md", "chunk": 1}),
        Document(page_content="c", metadata={"source": "b.md", "chunk": 2}),
    ]

    assert _unique_sources(documents) == ["a.md / chunk 1", "b.md / chunk 2"]


def test_format_excerpts_includes_source_and_text() -> None:
    documents = [
        Document(
            page_content="這是被檢索到的原文內容",
            metadata={"source": "manual.pdf", "page": 1, "chunk": 2},
        )
    ]

    assert _format_excerpts(documents) == [
        "[1] manual.pdf / p.1 / chunk 2\n這是被檢索到的原文內容"
    ]


def test_ingest_directory_filters_unsupported_files(tmp_path: Path) -> None:
    supported = tmp_path / "a.md"
    unsupported = tmp_path / "a.png"
    supported.write_text("hello", encoding="utf-8")
    unsupported.write_text("not indexed", encoding="utf-8")

    assert list_ingestable_files(tmp_path) == [supported]


def test_clear_resets_collection_without_removing_persist_directory(tmp_path: Path) -> None:
    class FakeVectorStore:
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset_collection(self) -> None:
            self.reset_calls += 1

    service = RagService.__new__(RagService)
    service._lock = __import__("threading").RLock()
    service._state_lock = __import__("threading").Lock()
    service._indexing = False
    service.vectorstore = FakeVectorStore()

    service.clear()

    assert service.vectorstore.reset_calls == 1


def test_is_indexing_returns_current_state() -> None:
    service = RagService.__new__(RagService)
    service._lock = __import__("threading").RLock()
    service._state_lock = __import__("threading").Lock()
    service._indexing = True

    assert service.is_indexing() is True
