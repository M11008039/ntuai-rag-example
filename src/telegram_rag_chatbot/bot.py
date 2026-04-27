from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from telegram_rag_chatbot.config import load_settings
from telegram_rag_chatbot.loaders import UnsupportedFileTypeError, is_supported_file
from telegram_rag_chatbot.rag import RagService

logger = logging.getLogger(__name__)


HELP_TEXT = """可用指令：
/start - 查看簡介
/help - 查看指令
/status - 查看知識庫狀態
/reindex - 重新索引 uploads 內的文件
/clear - 清空向量知識庫

你也可以直接傳 PDF、DOCX、TXT、MD 給我，我會把文件加入知識庫。
文件加入後，直接輸入問題即可查詢。"""


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "你好，我是 Telegram RAG chatbot。先傳文件給我，接著直接問問題即可。\n\n"
        + HELP_TEXT
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT)


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rag = _rag(context)
    count = await asyncio.to_thread(rag.count_documents)
    indexing = await asyncio.to_thread(rag.is_indexing)
    suffix = "\n目前正在建立索引，完成前查詢結果可能不完整。" if indexing else ""
    await update.message.reply_text(f"目前 Chroma 知識庫共有 {count} 個 chunks。{suffix}")


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rag = _rag(context)
    await update.message.reply_text("正在清空知識庫...")
    await asyncio.to_thread(rag.clear)
    await update.message.reply_text("已清空知識庫。上傳文件後即可重新建立。")


async def reindex(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rag = _rag(context)
    await update.message.reply_text(
        "正在重新索引 uploads 內的文件，請稍等。大型 PDF 可能需要 1 到 2 分鐘。"
    )
    result = await asyncio.to_thread(rag.rebuild_directory)
    await update.message.reply_text(
        f"完成索引：{result.files} 個檔案，{result.raw_documents} 份原始文件，{result.chunks} 個 chunks。"
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rag = _rag(context)
    document = update.message.document
    if document is None:
        return

    filename = _safe_filename(document.file_name or f"telegram-file-{document.file_unique_id}")
    destination = rag.settings.upload_dir / filename
    if not is_supported_file(destination):
        await update.message.reply_text("目前只支援 PDF、DOCX、TXT、MD 文件。")
        return

    await update.message.reply_text(f"收到 {filename}，正在下載並建立索引...")
    telegram_file = await document.get_file()
    await telegram_file.download_to_drive(custom_path=destination)

    try:
        result = await asyncio.to_thread(rag.ingest_file, destination)
    except UnsupportedFileTypeError as exc:
        await update.message.reply_text(str(exc))
        return

    if result.chunks == 0:
        await update.message.reply_text("文件沒有讀到可索引的文字內容。")
        return

    await update.message.reply_text(
        f"完成索引：{filename}\n原始文件：{result.raw_documents}\nchunks：{result.chunks}\n現在可以直接問我問題。"
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rag = _rag(context)
    question = update.message.text or ""
    if await asyncio.to_thread(rag.is_indexing):
        await update.message.reply_text(
            "我正在建立索引，請稍等一下再問。大型文件需要一點時間處理。"
        )
        return

    await update.message.chat.send_action(action="typing")
    answer = await asyncio.to_thread(rag.answer, question)
    for message in _telegram_messages(
        _format_answer(answer.text, answer.sources, answer.excerpts)
    ):
        await update.message.reply_text(message)


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Telegram handler failed", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text("處理時發生錯誤，請稍後再試或查看服務 logs。")


def build_application() -> Application:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    settings = load_settings()
    rag = RagService(settings)
    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(_post_init)
        .build()
    )
    application.bot_data["rag"] = rag

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("clear", clear))
    application.add_handler(CommandHandler("reindex", reindex))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_error_handler(handle_error)
    return application


def main() -> None:
    application = build_application()
    application.run_polling(allowed_updates=Update.ALL_TYPES)


def _rag(context: ContextTypes.DEFAULT_TYPE) -> RagService:
    return context.application.bot_data["rag"]


def _safe_filename(filename: str) -> str:
    path = Path(filename)
    stem = "".join(char for char in path.stem if char.isalnum() or char in {"-", "_", "."})
    suffix = "".join(char for char in path.suffix if char.isalnum() or char == ".")
    if not stem:
        stem = "uploaded"
    return f"{stem}{suffix.lower()}"


def _telegram_safe(text: str) -> str:
    limit = 3900
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n...（回答過長，已截斷）"


def _telegram_messages(text: str, limit: int = 3900) -> list[str]:
    if len(text) <= limit:
        return [text]

    messages: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            messages.append(remaining)
            break
        split_at = remaining.rfind("\n\n", 0, limit)
        if split_at < limit // 2:
            split_at = limit
        messages.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    return messages


def _format_answer(text: str, sources: list[str], excerpts: list[str] | None = None) -> str:
    excerpts = excerpts or []
    if not sources and not excerpts:
        return text

    sections = [text]
    if sources:
        source_lines = "\n".join(f"- {source}" for source in sources)
        sections.append(f"來源：\n{source_lines}")
    if excerpts:
        sections.append("參考片段：\n" + "\n\n".join(excerpts))
    return "\n\n".join(sections)


async def _post_init(application: Application) -> None:
    rag: RagService = application.bot_data["rag"]
    if not rag.settings.auto_reindex_on_startup:
        return

    logger.info("AUTO_REINDEX_ON_STARTUP enabled; rebuilding vector index.")
    result = await asyncio.to_thread(rag.rebuild_directory)
    logger.info(
        "Startup reindex completed: files=%s raw_documents=%s chunks=%s",
        result.files,
        result.raw_documents,
        result.chunks,
    )


if __name__ == "__main__":
    main()
