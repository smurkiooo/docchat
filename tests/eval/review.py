import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

DIMENSIONS = ("correctness", "completeness", "faithfulness", "refusal_appropriate")


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    run_id: str
    case_id: str
    prediction_sha256: str
    reviewer: str
    reviewer_type: Literal["human", "assistant"] | None
    answer_mode: Literal["answer", "partial", "refusal"] | None
    correctness: bool | None
    completeness: bool | None
    faithfulness: bool | None
    refusal_appropriate: bool | None
    citation_support: bool | None
    rationale: str


def prediction_digest(prediction: dict) -> str:
    return hashlib.sha256(
        json.dumps(prediction, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def review_template(run_id: str, prediction: dict) -> dict:
    return dict(
        run_id=run_id,
        case_id=prediction["case_id"],
        prediction_sha256=prediction_digest(prediction),
        reviewer="",
        reviewer_type=None,
        answer_mode=None,
        **dict.fromkeys(DIMENSIONS),
        citation_support=None,
        rationale="",
    )


def score_reviews(report: dict, judgments: list[dict]) -> dict:
    predictions = {p["case_id"]: p for p in report["predictions"]}
    if not predictions:
        raise ValueError("Run has no predictions to review")
    if len(predictions) != len(report["predictions"]):
        raise ValueError("Duplicate prediction IDs")
    reviews = [Judgment.model_validate(j) for j in judgments]
    if len({j.case_id for j in reviews}) != len(reviews):
        raise ValueError("Duplicate review IDs; adjudicate before scoring")
    if {j.case_id for j in reviews} != set(predictions):
        raise ValueError("Review must cover every prediction exactly once")
    for review in reviews:
        if review.prediction_sha256 != prediction_digest(predictions[review.case_id]):
            raise ValueError("Prediction content changed after review template creation")
        if review.run_id != report["run_id"]:
            raise ValueError("Review belongs to a different run")
        if review.reviewer_type is None:
            raise ValueError("Reviewer type must be declared")
        if not review.reviewer.strip() or not review.rationale.strip():
            raise ValueError("Reviewer and evidence-based rationale are required")
        if any(getattr(review, key) is None for key in (*DIMENSIONS, "answer_mode")):
            raise ValueError("All semantic judgments must be completed before scoring")
        if review.answer_mode != "refusal" and review.citation_support is None:
            raise ValueError("Answers/partial answers require a citation-support judgment")
        if predictions[review.case_id]["error"]:
            raise ValueError("A failed generation cannot receive a semantic quality score")
    if not report["complete"]:
        raise ValueError("Incomplete run: fix generation errors and repeat before scoring")
    strata = {}
    for scenario in sorted({p["scenario"] for p in predictions.values()}):
        for answerable in (True, False):
            subset = [
                r
                for r in reviews
                if predictions[r.case_id]["scenario"] == scenario
                and predictions[r.case_id]["answerable"] == answerable
            ]
            if not subset:
                continue
            key = f"{scenario}/{'answerable' if answerable else 'unanswerable'}"
            strata[key] = {}
            for dimension in (*DIMENSIONS, "citation_support"):
                values = [
                    getattr(r, dimension) for r in subset if getattr(r, dimension) is not None
                ]
                strata[key][dimension] = dict(passed=sum(values), reviewed=len(values))
    return dict(
        run_id=report["run_id"],
        reviewed=len(reviews),
        strata=strata,
        reviewers=sorted({r.reviewer for r in reviews}),
        reviewer_types=sorted({r.reviewer_type for r in reviews}),
        note="Rubric judgments; assessor provenance is recorded; no statistical claim",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate rubric assessments for a completed evaluation run."
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.run.read_text())
    judgments = [json.loads(line) for line in args.reviews.read_text().splitlines() if line.strip()]
    summary = score_reviews(report, judgments)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
