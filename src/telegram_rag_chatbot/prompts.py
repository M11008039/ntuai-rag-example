from __future__ import annotations

from langchain_core.documents import Document

SYSTEM_PROMPT = """你是一個 Telegram RAG 助理。
請只根據「參考資料」回答使用者問題。
如果參考資料不足，請明確說「我在目前知識庫中找不到足夠資訊」。
只有在答案確實由參考資料支持時，才列出引用來源。
如果參考資料不足，不要猜測答案，也不要列出引用來源。
回答請使用繁體中文，保持精簡、清楚。"""


def format_context(documents: list[Document]) -> str:
    if not documents:
        return "目前沒有可用的參考資料。"
    blocks = []
    for index, document in enumerate(documents, start=1):
        blocks.append(f"[{index}] {document.page_content}")
    return "\n\n".join(blocks)


def format_source(document: Document) -> str:
    source = str(document.metadata.get("source", "unknown"))
    page = document.metadata.get("page")
    chunk = document.metadata.get("chunk")
    parts = [source]
    if page:
        parts.append(f"p.{page}")
    if chunk:
        parts.append(f"chunk {chunk}")
    return " / ".join(parts)


def build_prompt(question: str, documents: list[Document]) -> str:
    sources = "\n".join(
        f"[{index}] {format_source(document)}"
        for index, document in enumerate(documents, start=1)
    )
    return f"""{SYSTEM_PROMPT}

參考資料：
{format_context(documents)}

引用來源對照：
{sources or "無"}

使用者問題：
{question}

請輸出：
1. 直接回答
2. 若有足夠參考資料，再列出引用來源"""
