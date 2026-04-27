from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    telegram_bot_token: str
    data_dir: Path
    chroma_dir: Path
    upload_dir: Path
    collection_name: str
    gemini_chat_model: str
    gemini_embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    relevance_score_threshold: float
    index_batch_size: int
    auto_reindex_on_startup: bool


def _get_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return parsed


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not 0 <= parsed <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return parsed


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_settings() -> Settings:
    load_dotenv()

    data_dir = Path(os.getenv("DATA_DIR", "data"))
    chroma_dir = Path(os.getenv("CHROMA_DIR", str(data_dir / "chroma")))
    upload_dir = Path(os.getenv("UPLOAD_DIR", str(data_dir / "uploads")))
    chunk_size = _get_int("CHUNK_SIZE", 1600)
    chunk_overlap = _get_int("CHUNK_OVERLAP", 200)

    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    return Settings(
        gemini_api_key=_get_required("GEMINI_API_KEY"),
        telegram_bot_token=_get_required("TELEGRAM_BOT_TOKEN"),
        data_dir=data_dir,
        chroma_dir=chroma_dir,
        upload_dir=upload_dir,
        collection_name=os.getenv("COLLECTION_NAME", "telegram_rag"),
        gemini_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
        gemini_embedding_model=os.getenv(
            "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"
        ),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_k=_get_int("RETRIEVAL_K", 4),
        relevance_score_threshold=_get_float("RELEVANCE_SCORE_THRESHOLD", 0.55),
        index_batch_size=_get_int("INDEX_BATCH_SIZE", 64),
        auto_reindex_on_startup=_get_bool("AUTO_REINDEX_ON_STARTUP", True),
    )
