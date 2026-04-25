from telegram_rag_chatbot.bot import (
    _format_answer,
    _safe_filename,
    _telegram_messages,
    _telegram_safe,
)


def test_safe_filename_removes_path_and_unsafe_characters() -> None:
    assert _safe_filename("../../我的 文件!.PDF") == "我的文件.pdf"


def test_safe_filename_falls_back_when_stem_is_empty() -> None:
    assert _safe_filename("!!!.txt") == "uploaded.txt"


def test_telegram_safe_keeps_short_text_unchanged() -> None:
    assert _telegram_safe("hello") == "hello"


def test_telegram_safe_truncates_long_text() -> None:
    text = "a" * 4000

    result = _telegram_safe(text)

    assert len(result) <= 4096
    assert "已截斷" in result


def test_format_answer_appends_sources() -> None:
    result = _format_answer("答案", ["manual.pdf / p.1"])

    assert "答案" in result
    assert "來源" in result
    assert "manual.pdf / p.1" in result


def test_format_answer_appends_excerpts() -> None:
    result = _format_answer("答案", ["manual.pdf / p.1"], ["[1] manual.pdf / p.1\n原文"])

    assert "參考片段" in result
    assert "原文" in result


def test_telegram_messages_splits_long_text() -> None:
    messages = _telegram_messages("a" * 8000, limit=3900)

    assert len(messages) == 3
    assert all(len(message) <= 3900 for message in messages)
