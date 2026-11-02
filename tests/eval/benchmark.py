import json
import re
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models import Chunk

DATA = Path(__file__).with_name("data")
K_VALUES = (1, 3, 6)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Evidence(StrictModel):
    page: int = Field(gt=0)
    quote: str = Field(min_length=12)


class Document(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]+$")
    split: Literal["dev", "test"]
    title: str
    pages: list[list[str]] = Field(min_length=1)


class Case(StrictModel):
    id: str
    document_id: str
    split: Literal["dev", "test"]
    question: str = Field(min_length=1)
    answer: str | None

    evidence: list[list[Evidence]]
    tags: list[str] = Field(min_length=1)
    negative_reason: str | None

    @model_validator(mode="after")
    def labels(self):
        if self.answer is None:
            if self.evidence or not self.negative_reason or "unanswerable" not in self.tags:
                raise ValueError("Unanswerable cases need a reason and no positive evidence")
        elif not self.answer or not self.evidence or any(not group for group in self.evidence):
            raise ValueError(
                "Answerable cases need a reference answer and nonempty evidence groups"
            )
        elif self.negative_reason or "unanswerable" in self.tags:
            raise ValueError("Conflicting answerability labels")
        return self

    @property
    def answerable(self) -> bool:
        return self.answer is not None


class Turn(StrictModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    evidence: list[list[Evidence]] = Field(min_length=1)


class Dialogue(StrictModel):
    id: str
    document_id: str
    split: Literal["dev", "test"]
    turns: list[Turn] = Field(min_length=2)


class Corpus(StrictModel):
    version: str
    provenance: str
    documents: list[Document]


def validate_evidence(document: Document, evidence: list[list[Evidence]]) -> None:
    for group in evidence:
        if not group:
            raise ValueError("Empty evidence group")
        for span in group:
            if span.page > len(document.pages):
                raise ValueError(f"Unknown evidence page: {document.id}:{span.page}")
            if normalize(span.quote) not in normalize("\n".join(document.pages[span.page - 1])):
                raise ValueError(f"Evidence not in source: {document.id}:{span.quote}")


def load_dataset(data_dir: Path = DATA) -> tuple[Corpus, list[Case], list[Dialogue]]:
    corpus = Corpus.model_validate_json((data_dir / "documents.json").read_text())
    cases = [
        Case.model_validate_json(line)
        for line in (data_dir / "cases.jsonl").read_text().splitlines()
        if line.strip()
    ]
    dialogues = [
        Dialogue.model_validate(d) for d in json.loads((data_dir / "dialogues.json").read_text())
    ]
    documents = {d.id: d for d in corpus.documents}
    if len(documents) != len(corpus.documents):
        raise ValueError("Duplicate document IDs")
    ids, questions = set(), set()
    for case in cases:
        if case.id in ids or normalize(case.question) in questions:
            raise ValueError(f"Duplicate case or question: {case.id}")
        ids.add(case.id)
        questions.add(normalize(case.question))
        document = documents[case.document_id]
        if case.split != document.split:
            raise ValueError(f"Document leakage across splits: {case.id}")
        validate_evidence(document, case.evidence)
    for dialogue in dialogues:
        if dialogue.id in ids:
            raise ValueError(f"Duplicate dialogue ID: {dialogue.id}")
        ids.add(dialogue.id)
        document = documents[dialogue.document_id]
        if document.split != dialogue.split:
            raise ValueError("Dialogue split differs from document split")
        for turn in dialogue.turns:
            validate_evidence(document, turn.evidence)
    for split in ("dev", "test"):
        subset = [c for c in cases if c.split == split]
        if not any(c.answerable for c in subset) or not any(not c.answerable for c in subset):
            raise ValueError(f"Both answerability classes are required in {split}")
    return corpus, cases, dialogues


def contains_evidence(chunk: Chunk, evidence: Evidence) -> bool:
    return chunk.page_start <= evidence.page <= chunk.page_end and normalize(
        evidence.quote
    ) in normalize(chunk.text)


def evidence_hits(chunks: list[Chunk], groups: list[list[Evidence]]) -> list[bool]:
    return [any(contains_evidence(c, e) for c in chunks for e in group) for group in groups]


def retrieval_metrics(chunks: list[Chunk], groups: list[list[Evidence]]) -> dict:
    if not groups:
        return {}
    metrics = {}
    for k in K_VALUES:
        selected = chunks[:k]
        hits = evidence_hits(selected, groups)
        metrics[f"evidence_recall@{k}"] = sum(hits) / len(hits)
        metrics[f"complete@{k}"] = float(all(hits))
        metrics[f"hit@{k}"] = float(any(hits))
        relevant = sum(any(evidence_hits([c], groups)) for c in selected)
        metrics[f"precision@{k}"] = relevant / len(selected) if selected else 0.0
    metrics["mrr@6"] = next(
        (1 / i for i, c in enumerate(chunks[:6], 1) if any(evidence_hits([c], groups))), 0.0
    )
    return metrics


def citation_diagnostics(answer: str, chunks: list[Chunk]) -> dict:
    cited = []
    for match in re.finditer(
        r"\[(?:стр\.|p\.|pp\.)\s*(\d+)(?:\s*[-–]\s*(\d+))?\]", answer, re.IGNORECASE
    ):
        start, end = int(match[1]), int(match[2] or match[1])
        if start <= end and end - start <= 100:
            cited.extend(range(start, end + 1))
        else:
            cited.append(-1)
    allowed = {p for c in chunks for p in range(c.page_start, c.page_end + 1)}
    return dict(
        cited_pages=cited,
        has_citation=bool(cited),
        cited_pages_in_context=(all(p in allowed for p in cited) if cited else None),
    )
