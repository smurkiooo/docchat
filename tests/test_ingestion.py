from itertools import pairwise
from pathlib import Path

import pytest

from app.ingestion.chunker import chunk_pages
from app.ingestion.cleaner import clean_pages
from app.ingestion.pdf_parser import InvalidPdfError, ScannedPdfError, parse_pdf
from app.models import Page
from scripts.pdfgen import make_pdf

LOREM = (
    "Первое предложение про настройку. Второе предложение описывает датчик температуры. "
    "Третье предложение рассказывает о подключении к сети Wi-Fi! Четвёртое — про гарантию? "
)


def test_parse_pdf_keeps_page_numbers(sample_pdf: Path) -> None:
    pages = parse_pdf(sample_pdf)
    assert [p.number for p in pages] == [1, 2, 3]
    assert "230 вольт" in pages[1].text
    assert "36 месяцев" in pages[2].text


def test_parse_rejects_non_pdf(tmp_path: Path) -> None:
    path = tmp_path / "fake.pdf"
    path.write_text("not a pdf")
    with pytest.raises(InvalidPdfError):
        parse_pdf(path)


def test_parse_rejects_scans(tmp_path: Path) -> None:
    path = make_pdf([["x"], ["y"]], tmp_path / "scan.pdf", footer=None)
    with pytest.raises(ScannedPdfError):
        parse_pdf(path)


def test_cleaner_removes_running_headers_and_footers(sample_pdf: Path) -> None:
    pages = clean_pages(parse_pdf(sample_pdf))
    for page in pages:
        assert "ООО «Тест»" not in page.text
        assert "Страница" not in page.text
    assert "Гарантийный срок" in pages[2].text


def test_cleaner_joins_hyphenated_words_and_lines() -> None:
    pages = clean_pages([Page(1, "Темпера-\nтура в комнате\nподдерживается.\n\nНовый абзац.")])
    assert pages[0].text == "Температура в комнате поддерживается.\n\nНовый абзац."


def test_chunks_respect_size_and_pages() -> None:
    pages = [Page(n, LOREM * 6) for n in (1, 2, 3)]
    chunks = chunk_pages(pages, "doc", chunk_size=300, chunk_overlap=60, min_chunk_chars=100)

    assert all(len(c.text) <= 300 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert {c.page_start for c in chunks} == {1, 2, 3}
    assert all(c.page_start == c.page_end for c in chunks)


def test_chunks_overlap_within_page() -> None:
    chunks = chunk_pages([Page(1, LOREM * 6)], "doc", chunk_size=300, chunk_overlap=80)
    for prev, nxt in pairwise(chunks):
        assert nxt.text.split(" ", 3)[0] in prev.text[-100:]


def test_short_page_tail_merges_into_next_page() -> None:
    pages = [Page(1, "Короткая страница."), Page(2, "Продолжение на второй странице.")]
    chunks = chunk_pages(pages, "doc", chunk_size=500, chunk_overlap=50, min_chunk_chars=200)
    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_very_long_word_is_hard_split() -> None:
    chunks = chunk_pages([Page(1, "x" * 1000)], "doc", chunk_size=300, chunk_overlap=0)
    assert all(len(c.text) <= 300 for c in chunks)
    assert "".join(c.text for c in chunks) == "x" * 1000


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "text")], "doc", chunk_size=100, chunk_overlap=100)
