import re
from collections import Counter

from app.models import Page

_HYPHEN_BREAK = re.compile(r"(\w)[-­]\n(\w)")
_SOFT_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
_SPACES = re.compile(r"[ \t ]+")
_MANY_BREAKS = re.compile(r"\n{3,}")
_DIGITS = re.compile(r"\d+")


BOILERPLATE_MIN_SHARE = 0.5
BOILERPLATE_MIN_PAGES = 3
BOILERPLATE_MAX_LEN = 120


def clean_pages(pages: list[Page]) -> list[Page]:
    boilerplate = _find_boilerplate(pages)
    return [Page(p.number, _clean_text(p.text, boilerplate)) for p in pages]


def _clean_text(text: str, boilerplate: set[str]) -> str:
    blocks = [b for b in text.split("\n\n") if b.strip()]
    blocks = [
        b
        for i, b in enumerate(blocks)
        if i not in (0, len(blocks) - 1) or _signature(b) not in boilerplate
    ]
    text = "\n\n".join(blocks)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SOFT_NEWLINE.sub(" ", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MANY_BREAKS.sub("\n\n", text).strip()


def _signature(block: str) -> str:
    return _DIGITS.sub("#", " ".join(block.split()).lower())


def _find_boilerplate(pages: list[Page]) -> set[str]:
    if len(pages) < BOILERPLATE_MIN_PAGES:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        blocks = [b for b in page.text.split("\n\n") if b.strip()]
        edge = {_signature(b) for b in blocks[:1] + blocks[-1:]}
        counts.update(s for s in edge if len(s) <= BOILERPLATE_MAX_LEN)
    threshold = len(pages) * BOILERPLATE_MIN_SHARE
    return {sig for sig, n in counts.items() if n > threshold}
