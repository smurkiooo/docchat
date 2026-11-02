from pathlib import Path

import pymupdf

PAGE = pymupdf.paper_rect("a4")
MARGIN = 56
FONT = "F0"


def make_pdf(
    pages: list[list[str]],
    path: Path,
    header: str | None = None,
    footer: str | None = "Страница {page}",
) -> Path:
    font = pymupdf.Font("cjk")
    doc = pymupdf.open()
    for number, paragraphs in enumerate(pages, start=1):
        page = doc.new_page(width=PAGE.width, height=PAGE.height)
        page.insert_font(fontname=FONT, fontbuffer=font.buffer)
        if header:
            _write(page, header, MARGIN, 24, size=8, bottom=44)
        if footer:
            _write(
                page,
                footer.format(page=number),
                MARGIN,
                PAGE.height - 40,
                size=8,
                bottom=PAGE.height - 16,
            )
        y = MARGIN
        for paragraph in paragraphs:
            heading = paragraph.startswith("# ")
            text = paragraph.removeprefix("# ")
            y = _write(page, text, MARGIN, y, size=14 if heading else 10.5) + (6 if heading else 8)
    doc.subset_fonts()
    doc.save(path, garbage=4, deflate=True)
    doc.close()
    return path


def _write(
    page: pymupdf.Page, text: str, x: float, y: float, size: float, bottom: float | None = None
) -> float:
    rect = pymupdf.Rect(x, y, PAGE.width - MARGIN, bottom or PAGE.height - 50)
    spare = page.insert_textbox(rect, text, fontname=FONT, fontsize=size, lineheight=1.35)
    if spare < 0:
        raise ValueError(f"Text does not fit on page {page.number + 1}: {text[:40]!r}")
    return rect.y1 - spare
