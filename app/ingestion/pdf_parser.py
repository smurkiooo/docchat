from pathlib import Path

import pymupdf

from app.models import Page

MIN_AVG_CHARS_PER_PAGE = 50


class PdfError(Exception):
    pass


class InvalidPdfError(PdfError):
    pass


class ScannedPdfError(PdfError):
    pass


def parse_pdf(path: Path) -> list[Page]:
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise InvalidPdfError("Не удалось открыть файл: он повреждён или не является PDF.") from exc

    with doc:
        if doc.needs_pass:
            raise InvalidPdfError("PDF защищён паролем.")
        if doc.page_count == 0:
            raise InvalidPdfError("PDF не содержит страниц.")
        pages = [Page(number=i + 1, text=_page_text(page)) for i, page in enumerate(doc)]

    total_chars = sum(len(p.text.strip()) for p in pages)
    if total_chars / len(pages) < MIN_AVG_CHARS_PER_PAGE:
        raise ScannedPdfError(
            "В PDF почти нет текстового слоя — похоже на скан. OCR пока не поддерживается."
        )
    return pages


def _page_text(page: pymupdf.Page) -> str:

    blocks = page.get_text("blocks", sort=True)
    return "\n\n".join(b[4].strip() for b in blocks if b[6] == 0 and b[4].strip())
