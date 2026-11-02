import math
import re
import threading
from collections.abc import Callable
from functools import lru_cache

import snowballstemmer
from rank_bm25 import BM25Okapi

from app.models import Chunk

_TOKEN = re.compile(r"\w+", re.UNICODE)
_ALNUM_PART = re.compile(r"[^\W\d_]+|\d+")
_CYRILLIC = re.compile(r"[а-яё]")

STOPWORDS = frozenset(
    """
    а без более бы был была были было быть в вам вас весь во вот все всего всех вы где да
    даже для до его ее её если есть еще ещё же за здесь и из или им их к как ко когда кто
    ли либо мне может мы на над надо наш не него нее неё нет ни них но ну о об однако он
    она они оно от очень по под при про с со так также такой там те тем то того тоже той
    только том ты у уже хотя чего чей чем что чтобы чье чья эта эти это я какой какая какие
    каков сколько который которая которые
    a an and are as at be by for from has have how in is it its of on or that the this to
    was were what when where which who why will with does do did can about
    """.split()
)

_ru = snowballstemmer.stemmer("russian")
_en = snowballstemmer.stemmer("english")
_stem_lock = threading.Lock()


@lru_cache(maxsize=50_000)
def _stem(token: str) -> str:

    with _stem_lock:
        return (_ru if _CYRILLIC.search(token) else _en).stemWord(token)


def _expand(token: str) -> list[str]:
    parts = _ALNUM_PART.findall(token)
    return [token, *parts] if len(parts) > 1 else [token]


def tokenize(text: str) -> list[str]:
    tokens = (
        part
        for token in _TOKEN.findall(text.lower())
        for part in _expand(token)
        if part not in STOPWORDS
    )
    return [_stem(t) for t in tokens if len(t) > 1 or t.isdigit()]


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._ids = [c.id for c in chunks]
        corpus = [tokenize(c.text) or ["∅"] for c in chunks]
        self._vocab = [set(tokens) for tokens in corpus]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        if self._bm25:
            for term in self._bm25.idf:
                frequency = sum(term in vocab for vocab in self._vocab)
                self._bm25.idf[term] = math.log1p(
                    (len(corpus) - frequency + 0.5) / (frequency + 0.5)
                )

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens or self._bm25 is None or k <= 0:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self._ids[i], float(scores[i])) for i in ranked[:k] if scores[i] > 0]

    def coverage(self, query: str) -> float:
        terms = set(tokenize(query))
        if not terms:
            return 0.0
        return max((len(terms & vocab) / len(terms) for vocab in self._vocab), default=0.0)


class BM25Store:
    def __init__(self, load_chunks: Callable[[str], list[Chunk]]) -> None:
        self._load_chunks = load_chunks
        self._indexes: dict[str, BM25Index] = {}
        self._lock = threading.Lock()

    def get(self, doc_id: str) -> BM25Index:
        with self._lock:
            if doc_id not in self._indexes:
                self._indexes[doc_id] = BM25Index(self._load_chunks(doc_id))
            return self._indexes[doc_id]

    def build(self, doc_id: str, chunks: list[Chunk]) -> None:
        with self._lock:
            self._indexes[doc_id] = BM25Index(chunks)

    def delete(self, doc_id: str) -> None:
        with self._lock:
            self._indexes.pop(doc_id, None)
