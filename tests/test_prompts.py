from langchain_core.documents import Document

from telegram_rag_chatbot.prompts import build_prompt, format_source


def test_format_source_includes_page_and_chunk_when_available() -> None:
    document = Document(
        page_content="content",
        metadata={"source": "manual.pdf", "page": 3, "chunk": 2},
    )

    assert format_source(document) == "manual.pdf / p.3 / chunk 2"


def test_build_prompt_contains_question_context_and_source() -> None:
    document = Document(
        page_content="Gemini 可以用於 RAG。",
        metadata={"source": "guide.md", "chunk": 1},
    )

    prompt = build_prompt("RAG 是什麼？", [document])

    assert "RAG 是什麼？" in prompt
    assert "Gemini 可以用於 RAG。" in prompt
    assert "guide.md / chunk 1" in prompt
