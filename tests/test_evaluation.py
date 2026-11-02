import copy
import json
from argparse import Namespace

import pytest
from pydantic import ValidationError

from app.models import Chunk
from tests.eval.benchmark import (
    Case,
    Evidence,
    citation_diagnostics,
    evidence_hits,
    load_dataset,
    retrieval_metrics,
)
from tests.eval.review import review_template, score_reviews
from tests.eval.run_eval import CONFIG, calibrate, run, settings_for_run, write_artifacts


def chunk(text, page=1, index=0):
    return Chunk(f"d:{index}", "d", index, text, page, page)


def test_dataset_has_document_disjoint_splits_and_verified_evidence():
    corpus, cases, dialogues = load_dataset()
    dev = {c.document_id for c in cases if c.split == "dev"}
    test = {c.document_id for c in cases if c.split == "test"}
    assert not dev & test
    assert len(corpus.documents) == 4
    assert len(cases) == 48
    assert len(dialogues) == 4
    assert all(
        len([c for c in cases if c.split == s and c.answerable]) == 16 for s in ("dev", "test")
    )


def test_page_match_without_quote_is_not_relevant():
    groups = [[Evidence(page=1, quote="Гарантия составляет 36 месяцев.")]]
    assert retrieval_metrics([chunk("На этой странице есть другой текст.")], groups)["hit@6"] == 0
    assert not any(evidence_hits([chunk(groups[0][0].quote, page=2)], groups))


def test_compound_questions_require_all_evidence_groups_with_alternatives():
    groups = [
        [
            Evidence(page=1, quote="Напряжение питания 12 вольт."),
            Evidence(page=2, quote="Прибор питается от 12 вольт."),
        ],
        [Evidence(page=3, quote="Потребление не более 8 ватт.")],
    ]
    ranking = [chunk(groups[0][1].quote, page=2), chunk(groups[1][0].quote, page=3, index=1)]
    metrics = retrieval_metrics(ranking, groups)
    assert metrics["hit@1"] == 1
    assert metrics["evidence_recall@1"] == 0.5
    assert metrics["complete@1"] == 0
    assert metrics["complete@3"] == 1
    assert metrics["precision@3"] == 1


def test_mrr_uses_same_truncation_and_does_not_reward_rank_seven():
    groups = [[Evidence(page=1, quote="Гарантия составляет 36 месяцев.")]]
    ranking = [chunk("нерелевантный текст", index=i) for i in range(6)]
    ranking.append(chunk(groups[0][0].quote, index=6))
    assert retrieval_metrics(ranking, groups)["mrr@6"] == 0
    assert retrieval_metrics([], groups)["complete@6"] == 0


def test_no_partial_quote_credit_and_whitespace_normalization():
    groups = [[Evidence(page=1, quote="Гарантия составляет 36 месяцев.")]]
    assert evidence_hits([chunk("Гарантия\nсоставляет  36 месяцев.")], groups) == [True]
    assert evidence_hits([chunk("Гарантия составляет"), chunk("36 месяцев.", index=1)], groups) == [
        False
    ]


def test_conflicting_answerability_labels_fail():
    _, cases, _ = load_dataset()
    data = cases[0].model_dump()
    data["answer"] = None
    with pytest.raises(ValidationError):
        Case.model_validate(data)


def test_citation_checks_are_structural_and_absence_is_not_success():
    assert citation_diagnostics("нет цитаты", [])["cited_pages_in_context"] is None
    assert citation_diagnostics("Факт [стр. 2]", [chunk("text")])["cited_pages_in_context"] is False
    assert (
        citation_diagnostics("Вымысел [стр. 1]", [chunk("text")])["cited_pages_in_context"] is True
    )


def test_eval_config_ignores_local_retrieval_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "999")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    settings = settings_for_run(CONFIG, tmp_path, with_llm=False)
    assert settings.chunk_size == 800
    assert settings.openai_api_key is None
    assert len(settings.embedding_revision) == 40


def test_missing_config_fields_are_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="exactly"):
        settings_for_run(path, tmp_path, False)


def test_calibration_rejects_test_split_and_missing_classes():
    with pytest.raises(ValueError, match="only allowed on dev"):
        calibrate([], "test")
    with pytest.raises(ValueError, match="both answerability"):
        calibrate([], "dev")


def test_calibration_balances_classes_and_is_deterministic():
    rows = [
        dict(
            answerable=True, gate=dict(has_context=True, max_similarity=0.8, lexical_coverage=0.0)
        ),
        dict(
            answerable=False, gate=dict(has_context=True, max_similarity=0.65, lexical_coverage=0.0)
        ),
    ]
    result = calibrate(rows, "dev")
    assert result["selected"]["balanced_error"] == 0
    assert result == calibrate(rows, "dev")


def reviewed_run():
    prediction = dict(case_id="c1", answerable=True, scenario="single_turn", error=[])
    report = dict(run_id="run-1", predictions=[prediction], complete=True)
    review = review_template(report["run_id"], prediction)
    review.update(
        reviewer="reviewer-a",
        reviewer_type="human",
        answer_mode="answer",
        correctness=True,
        completeness=True,
        faithfulness=False,
        refusal_appropriate=True,
        citation_support=False,
        rationale="Unsupported claim; see source paragraph 2.",
    )
    return report, review


def test_review_template_never_becomes_an_automatic_quality_score():
    report, _ = reviewed_run()
    with pytest.raises(ValueError):
        score_reviews(report, [review_template(report["run_id"], report["predictions"][0])])


@pytest.mark.parametrize(
    "mutation", ["run", "duplicate", "missing", "generation_error", "incomplete"]
)
def test_review_provenance_and_coverage(mutation):
    report, review = reviewed_run()
    reviews = [review]
    if mutation == "run":
        review["run_id"] = "other"
    elif mutation == "duplicate":
        reviews.append(copy.deepcopy(review))
    elif mutation == "missing":
        reviews.clear()
    elif mutation == "generation_error":
        report["predictions"][0]["error"] = ["timeout"]
    else:
        report["complete"] = False
    with pytest.raises(ValueError):
        score_reviews(report, reviews)


def test_review_reports_denominators_and_human_failures():
    report, review = reviewed_run()
    result = score_reviews(report, [review])
    assert result["strata"]["single_turn/answerable"]["faithfulness"] == dict(passed=0, reviewed=1)


@pytest.mark.parametrize("backend", ["bm25", "smoke"])
async def test_evaluation_runs_offline_and_persists_auditable_artifacts(tmp_path, backend):
    args = Namespace(backend=backend, split="dev", config=CONFIG, with_llm=False, calibrate=False)
    report = await run(args)
    assert report["complete"]
    assert len(report["rows"]) == 24
    assert all(r["oracle_evidence_recall"] == 1 for r in report["rows"] if r["answerable"])
    assert report["predictions"] == []
    output = tmp_path / "run.md"
    write_artifacts(report, output)
    stored = json.loads(output.with_suffix(".json").read_text())
    assert stored["config"]["chunk_size"] == 800
    assert stored["summary"]["bm25"]["n"] == 16
    assert "openai_api_key" not in output.with_suffix(".json").read_text()
    if backend == "smoke":
        assert "SMOKE ONLY" in output.read_text()


def test_review_detects_edited_prediction():
    report, review = reviewed_run()
    report["predictions"][0]["answer"] = "edited after template"
    with pytest.raises(ValueError, match="content changed"):
        score_reviews(report, [review])


def test_settings_ignore_invalid_retrieval_env_even_with_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "not-an-integer")
    settings = settings_for_run(CONFIG, tmp_path, with_llm=True)
    assert settings.chunk_size == 800


async def test_generation_trace_contains_actual_context_and_terminal_status(client, sample_pdf):
    from tests.eval.run_eval import RecordingRetriever, generate
    from tests.test_api import _upload

    doc_id = _upload(client, sample_pdf)["id"]
    container = client.app.state.container
    trace = RecordingRetriever(container.rag._retriever)
    container.rag._retriever = trace
    case = Case(
        id="trace",
        document_id=doc_id,
        split="dev",
        question="Какой гарантийный срок?",
        answer="36 месяцев.",
        evidence=[
            [Evidence(page=3, quote="Гарантийный срок составляет 36 месяцев с даты покупки.")]
        ],
        tags=["exact"],
        negative_reason=None,
    )
    prediction = await generate(container, trace, case, "single_turn")
    assert prediction["error"] == []
    assert prediction["search_query"] == case.question
    assert prediction["context_evidence_recall"] == 1
    assert prediction["sources"]
    assert "correctness" not in prediction["diagnostics"]


def test_secrets_cannot_be_added_to_evaluation_config(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["openai_api_key"] = "private"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="exactly"):
        settings_for_run(path, tmp_path, False)


def test_existing_artifacts_are_never_overwritten(tmp_path):
    path = tmp_path / "run.md"
    path.write_text("reviewed report")
    with pytest.raises(FileExistsError):
        write_artifacts(dict(predictions=[], calibration=None), path)
    assert path.read_text() == "reviewed report"
