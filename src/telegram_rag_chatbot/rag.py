from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from telegram_rag_chatbot.chunking import split_documents
from telegram_rag_chatbot.config import Settings
from telegram_rag_chatbot.loaders import is_supported_file, load_document
from telegram_rag_chatbot.prompts import build_prompt, format_source


@dataclass(frozen=True)
class IngestResult:
    files: int
    raw_documents: int
    chunks: int


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[str]
    excerpts: list[str]


INSUFFICIENT_INFORMATION_TEXT = "我在目前知識庫中找不到足夠資訊。"


class RagService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = RLock()
        self._state_lock = Lock()
        self._indexing = False
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.gemini_api_key,
        )
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.2,
        )
        self.vectorstore = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(settings.chroma_dir),
        )

    def ingest_file(self, path: Path) -> IngestResult:
        with self._lock:
            documents = load_document(path)
            chunks = split_documents(
                documents,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            self._add_chunks(chunks)
            return IngestResult(files=1, raw_documents=len(documents), chunks=len(chunks))

    def ingest_directory(self, directory: Path | None = None) -> IngestResult:
        target = directory or self.settings.upload_dir
        with self._lock:
            return self._ingest_paths(list_ingestable_files(target))

    def rebuild_directory(self, directory: Path | None = None) -> IngestResult:
        target = directory or self.settings.upload_dir
        self._set_indexing(True)
        with self._lock:
            try:
                self._reset_collection()
                return self._ingest_paths(list_ingestable_files(target))
            finally:
                self._set_indexing(False)

    def answer(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            return Answer(text="請輸入想詢問的問題。", sources=[], excerpts=[])

        with self._lock:
            documents = self._retrieve_relevant_documents(question)
        if not documents:
            return Answer(
                text=f"{INSUFFICIENT_INFORMATION_TEXT}請先上傳相關文件，或使用 /reindex 重建知識庫。",
                sources=[],
                excerpts=[],
            )

        response = self.llm.invoke(build_prompt(question, documents))
        text = getattr(response, "content", str(response))
        if _is_insufficient_answer(str(text)):
            return Answer(text=INSUFFICIENT_INFORMATION_TEXT, sources=[], excerpts=[])

        sources = _unique_sources(documents)
        excerpts = _format_excerpts(documents)
        return Answer(text=str(text).strip(), sources=sources, excerpts=excerpts)

    def clear(self) -> None:
        with self._lock:
            self._reset_collection()

    def count_documents(self) -> int:
        with self._lock:
            # LangChain's Chroma wrapper does not expose a public count helper.
            collection = self.vectorstore._collection
            return int(collection.count())

    def is_indexing(self) -> bool:
        with self._state_lock:
            return self._indexing

    def _set_indexing(self, value: bool) -> None:
        with self._state_lock:
            self._indexing = value

    def _reset_collection(self) -> None:
        self.vectorstore.reset_collection()

    def _ingest_paths(self, paths: list[Path]) -> IngestResult:
        files = 0
        raw_documents = 0
        chunks = 0
        for path in paths:
            documents = load_document(path)
            split = split_documents(
                documents,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            self._add_chunks(split)
            files += 1
            raw_documents += len(documents)
            chunks += len(split)
        return IngestResult(files=files, raw_documents=raw_documents, chunks=chunks)

    def _add_chunks(self, chunks: list[Document]) -> None:
        if not chunks:
            return
        batch_size = self.settings.index_batch_size
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            ids = [str(uuid4()) for _ in batch]
            self.vectorstore.add_documents(batch, ids=ids)

    def _retrieve_relevant_documents(self, question: str) -> list[Document]:
        scored_documents = self.vectorstore.similarity_search_with_relevance_scores(
            question,
            k=self.settings.retrieval_k,
            score_threshold=self.settings.relevance_score_threshold,
        )
        return [document for document, _score in scored_documents]


def _unique_sources(documents: list[Document]) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for document in documents:
        source = format_source(document)
        if source not in seen:
            sources.append(source)
            seen.add(source)
    return sources


def _format_excerpts(documents: list[Document], max_chars: int = 700) -> list[str]:
    excerpts: list[str] = []
    for index, document in enumerate(documents, start=1):
        text = " ".join(document.page_content.split())
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        excerpts.append(f"[{index}] {format_source(document)}\n{text}")
    return excerpts


def _is_insufficient_answer(text: str) -> bool:
    return INSUFFICIENT_INFORMATION_TEXT in text


def list_ingestable_files(directory: Path) -> list[Path]:
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and is_supported_file(path)
    ]
