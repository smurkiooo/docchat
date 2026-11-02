import re
from dataclasses import dataclass

from app.models import Chunk, Page

_SENTENCE_END = re.compile(r"(?<=[.!?…;:])\s+")


@dataclass(slots=True)
class _Segment:
    text: str
    page: int
    starts_paragraph: bool


def chunk_pages(
    pages: list[Page],
    doc_id: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    min_chunk_chars: int = 200,
) -> list[Chunk]:
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    drafts: list[list[_Segment]] = []
    current: list[_Segment] = []
    for segment in _segments(pages, chunk_size):
        new_page = bool(current) and segment.page != current[-1].page
        if current and new_page and _length(current) >= min_chunk_chars:
            drafts.append(current)
            current = []
        elif current and _length([*current, segment]) > chunk_size:
            drafts.append(current)
            current = _overlap(current, segment, chunk_size, chunk_overlap)
        current.append(segment)
    if current:
        drafts.append(current)

    return [
        Chunk(
            id=f"{doc_id}:{i}",
            doc_id=doc_id,
            index=i,
            text=_join(draft),
            page_start=draft[0].page,
            page_end=draft[-1].page,
        )
        for i, draft in enumerate(drafts)
    ]


def _segments(pages: list[Page], size: int) -> list[_Segment]:
    result: list[_Segment] = []
    for page in pages:
        for paragraph in page.text.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            for i, piece in enumerate(_split(paragraph, size)):
                result.append(_Segment(piece, page.number, starts_paragraph=i == 0))
    return result


def _split(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    sentences = [s for s in _SENTENCE_END.split(text) if s]
    if len(sentences) > 1:
        return [piece for sentence in sentences for piece in _split(sentence, size)]
    words = text.split()
    if len(words) > 1:
        return _pack_words(words, max(size // 4, 1))
    return [text[i : i + size] for i in range(0, len(text), size)]


def _pack_words(words: list[str], target: int) -> list[str]:
    pieces: list[str] = []
    for word in words:
        if len(word) > target:
            pieces.extend(word[i : i + target] for i in range(0, len(word), target))
        elif pieces and len(pieces[-1]) + 1 + len(word) <= target:
            pieces[-1] = f"{pieces[-1]} {word}"
        else:
            pieces.append(word)
    return pieces


def _overlap(
    previous: list[_Segment], upcoming: _Segment, size: int, overlap: int
) -> list[_Segment]:
    if overlap == 0 or (upcoming.starts_paragraph and upcoming.page != previous[-1].page):
        return []
    tail = _join([s for s in previous if s.page == previous[-1].page])[-overlap:]
    if " " in tail:
        tail = tail.split(" ", 1)[1]
    tail = tail.strip()
    if not tail or len(tail) + (2 if upcoming.starts_paragraph else 1) + len(upcoming.text) > size:
        return []
    return [_Segment(tail, previous[-1].page, starts_paragraph=True)]


def _join(segments: list[_Segment]) -> str:
    out = segments[0].text
    for seg in segments[1:]:
        out += ("\n\n" if seg.starts_paragraph else " ") + seg.text
    return out


def _length(segments: list[_Segment]) -> int:
    return len(_join(segments))
