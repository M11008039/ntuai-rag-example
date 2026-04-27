from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from telegram_rag_chatbot.rag import (
    INSUFFICIENT_INFORMATION_TEXT,
    Answer,
    RagService,
    _format_excerpts,
    _is_insufficient_answer,
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


def test_retrieve_relevant_documents_uses_score_threshold() -> None:
    document = Document(page_content="RAG content", metadata={"source": "guide.md"})

    class FakeVectorStore:
        def similarity_search_with_relevance_scores(
            self,
            question: str,
            k: int,
            score_threshold: float,
        ) -> list[tuple[Document, float]]:
            assert question == "RAG 是什麼？"
            assert k == 3
            assert score_threshold == 0.7
            return [(document, 0.8)]

    service = RagService.__new__(RagService)
    service.settings = SimpleNamespace(retrieval_k=3, relevance_score_threshold=0.7)
    service.vectorstore = FakeVectorStore()

    assert service._retrieve_relevant_documents("RAG 是什麼？") == [document]


def test_answer_without_relevant_documents_returns_no_sources() -> None:
    class FakeVectorStore:
        def similarity_search_with_relevance_scores(
            self,
            question: str,
            k: int,
            score_threshold: float,
        ) -> list[tuple[Document, float]]:
            return []

    service = RagService.__new__(RagService)
    service._lock = __import__("threading").RLock()
    service.settings = SimpleNamespace(retrieval_k=4, relevance_score_threshold=0.55)
    service.vectorstore = FakeVectorStore()

    answer = service.answer("你是誰？")

    assert answer == Answer(
        text=f"{INSUFFICIENT_INFORMATION_TEXT}請先上傳相關文件，或使用 /reindex 重建知識庫。",
        sources=[],
        excerpts=[],
    )


def test_answer_that_says_insufficient_information_returns_no_sources() -> None:
    document = Document(page_content="unrelated", metadata={"source": "a.md"})

    class FakeVectorStore:
        def similarity_search_with_relevance_scores(
            self,
            question: str,
            k: int,
            score_threshold: float,
        ) -> list[tuple[Document, float]]:
            return [(document, 0.9)]

    class FakeLlm:
        def invoke(self, prompt: str) -> object:
            return SimpleNamespace(content=INSUFFICIENT_INFORMATION_TEXT)

    service = RagService.__new__(RagService)
    service._lock = __import__("threading").RLock()
    service.settings = SimpleNamespace(retrieval_k=4, relevance_score_threshold=0.55)
    service.vectorstore = FakeVectorStore()
    service.llm = FakeLlm()

    assert service.answer("未知問題") == Answer(
        text=INSUFFICIENT_INFORMATION_TEXT,
        sources=[],
        excerpts=[],
    )


def test_is_insufficient_answer_matches_standard_phrase() -> None:
    assert _is_insufficient_answer("我在目前知識庫中找不到足夠資訊。") is True
    assert _is_insufficient_answer("這是有來源支持的回答。") is False


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
