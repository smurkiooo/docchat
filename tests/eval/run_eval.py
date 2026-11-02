import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings
from app.container import build_container
from app.ingestion.chunker import chunk_pages
from app.ingestion.cleaner import clean_pages
from app.ingestion.pdf_parser import parse_pdf
from app.llm import prompts
from app.llm.base import LLMError
from app.models import RetrievedChunk
from app.retrieval.bm25_store import BM25Index
from app.retrieval.hybrid import context_relevance
from app.services.rag_service import DoneEvent, ErrorEvent, SourcesEvent, TokenEvent
from scripts.pdfgen import make_pdf
from tests.eval.benchmark import (
    DATA,
    Case,
    citation_diagnostics,
    evidence_hits,
    load_dataset,
    normalize,
    retrieval_metrics,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("config.json")

CONFIG_FIELDS = frozenset(
    {
        "chunk_size",
        "chunk_overlap",
        "min_chunk_chars",
        "embedding_model",
        "embedding_batch_size",
        "top_k_vector",
        "top_k_bm25",
        "top_k_final",
        "rrf_k",
        "use_reranker",
        "reranker_model",
        "min_relevance_score",
        "min_lexical_coverage",
        "history_turns",
        "llm_temperature",
        "llm_max_tokens",
        "llm_timeout_s",
        "embedding_revision",
        "reranker_revision",
        "model_device",
    }
)


class DisabledLLM:
    name, model = "disabled", "disabled"

    async def complete(self, *args):
        raise LLMError("Generation is disabled in retrieval-only evaluation")

    async def stream(self, *args):
        raise LLMError("Generation is disabled in retrieval-only evaluation")
        yield

    async def is_available(self):
        return False


class RecordingRetriever:
    def __init__(self, delegate):
        self.delegate = delegate
        self.query = None
        self.result = None

    def retrieve(self, doc_id, query):
        self.query = query
        self.result = self.delegate.retrieve(doc_id, query)
        return self.result


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_digest() -> str:
    files = sorted(
        [
            *ROOT.glob("app/**/*.py"),
            *ROOT.glob("tests/eval/**/*.py"),
            ROOT / "scripts/pdfgen.py",
            ROOT / "tests/conftest.py",
        ]
    )
    return digest(
        b"".join(str(p.relative_to(ROOT)).encode() + b"\0" + p.read_bytes() for p in files)
    )


def settings_for_run(config: Path, directory: Path, with_llm: bool) -> Settings:
    values = json.loads(config.read_text())
    if set(values) != CONFIG_FIELDS:
        raise ValueError("Config must contain exactly the fields in tests/eval/config.json")

    defaults = {name: field.default for name, field in Settings.model_fields.items()}
    if with_llm:
        connection_fields = (
            "llm_provider",
            "openai_base_url",
            "openai_api_key",
            "openai_model",
            "ollama_base_url",
            "ollama_model",
            "anthropic_api_key",
            "anthropic_model",
            "anthropic_effort",
        )
        local = Settings(
            **{name: value for name, value in defaults.items() if name not in connection_fields}
        )
        for name in connection_fields:
            defaults[name] = getattr(local, name)
    defaults.update(values)
    defaults["data_dir"] = directory
    settings = Settings(_env_file=None, **defaults)
    if min(settings.top_k_vector, settings.top_k_bm25, settings.top_k_final) < 6:
        raise ValueError("This protocol reports @6: all candidate/context budgets must be >= 6")
    return settings


def summarize(rows: list[dict]) -> dict:
    modes = sorted({mode for row in rows for mode in row.get("metrics", {})})
    summary = {}
    for mode in modes:
        eligible = [r for r in rows if r["answerable"] and mode in r.get("metrics", {})]
        if not eligible:
            continue
        keys = eligible[0]["metrics"][mode]
        per_document = {}
        for doc_id in sorted({r["document_id"] for r in eligible}):
            subset = [r for r in eligible if r["document_id"] == doc_id]
            per_document[doc_id] = {
                "n": len(subset),
                **{key: statistics.mean(r["metrics"][mode][key] for r in subset) for key in keys},
            }
        by_tag = {}
        for tag in sorted({tag for r in eligible for tag in r["tags"]}):
            subset = [r for r in eligible if tag in r["tags"]]
            by_tag[tag] = {
                "n": len(subset),
                "complete@6": statistics.mean(r["metrics"][mode]["complete@6"] for r in subset),
            }
        summary[mode] = {
            "n": len(eligible),
            "question_mean": {
                key: statistics.mean(r["metrics"][mode][key] for r in eligible) for key in keys
            },
            "document_macro": {
                key: statistics.mean(d[key] for d in per_document.values()) for key in keys
            },
            "per_document": per_document,
            "by_tag": by_tag,
        }
    return summary


def gate_summary(rows: list[dict]) -> dict:
    eligible = [r for r in rows if "gate" in r]
    positive = [r for r in eligible if r["answerable"]]
    negative = [r for r in eligible if not r["answerable"]]
    return {
        "answerable_n": len(positive),
        "false_rejections": sum(not r["gate"]["accepted"] for r in positive),
        "unanswerable_n": len(negative),
        "false_acceptances": sum(r["gate"]["accepted"] for r in negative),
        "accepted_without_complete_evidence": sum(
            r["gate"]["accepted"] and not r["gate"]["complete_evidence"] for r in positive
        ),
        "negative_strata": {
            tag: {
                "n": sum(tag in r["tags"] for r in negative),
                "accepted": sum(tag in r["tags"] and r["gate"]["accepted"] for r in negative),
            }
            for tag in sorted({t for r in negative for t in r["tags"] if t != "unanswerable"})
        },
    }


def calibrate(rows: list[dict], split: str) -> dict:
    if split != "dev":
        raise ValueError("Calibration is only allowed on dev")
    if any("gate" not in r for r in rows):
        raise ValueError("Calibration needs a complete real-model gate run")
    positive = [r for r in rows if r["answerable"]]
    negative = [r for r in rows if not r["answerable"]]
    if not positive or not negative:
        raise ValueError("Calibration requires both answerability classes")
    trials = []
    for cosine in (0.55, 0.6, 0.65, 0.7, 0.72, 0.75, 0.8, 0.85, 0.9):
        for coverage in (0.25, 0.5, 0.75, 1.0):

            def accepts(row, cosine=cosine, coverage=coverage):
                gate = row["gate"]
                return gate["has_context"] and (
                    gate["max_similarity"] >= cosine or gate["lexical_coverage"] >= coverage
                )

            fn = sum(not accepts(r) for r in positive)
            fp = sum(accepts(r) for r in negative)
            trials.append(
                dict(
                    cosine=cosine,
                    coverage=coverage,
                    balanced_error=(fn / len(positive) + fp / len(negative)) / 2,
                    false_rejections=fn,
                    false_acceptances=fp,
                )
            )

    best = min(
        trials,
        key=lambda t: (t["balanced_error"], t["false_rejections"], t["cosine"], t["coverage"]),
    )
    return {
        "objective": "minimum balanced error; ties: FN, cosine, coverage ascending",
        "selected": best,
        "trials": trials,
    }


async def generate(container, trace, case, scenario):
    trace.result, trace.query = None, None
    started = time.perf_counter()
    first_token_s = None
    events = []
    async for event in container.rag.answer(case.document_id, case.question):
        if isinstance(event, TokenEvent) and first_token_s is None:
            first_token_s = time.perf_counter() - started
        events.append(event)
    duration_s = time.perf_counter() - started
    errors = [e.detail for e in events if isinstance(e, ErrorEvent)]
    answer = "".join(e.text for e in events if isinstance(e, TokenEvent)).strip()
    done = any(isinstance(e, DoneEvent) for e in events)
    result = trace.result

    emitted_sources = [
        s.model_dump() for e in events if isinstance(e, SourcesEvent) for s in e.sources
    ]
    chunks = [rc.chunk for rc in result.chunks] if result and emitted_sources else []
    return dict(
        case_id=case.id,
        scenario=scenario,
        duration_s=duration_s,
        first_token_s=first_token_s,
        used_llm=bool(emitted_sources),
        gate_rejected=(result is not None and not emitted_sources and not errors),
        document_id=case.document_id,
        question=case.question,
        expected_answer=case.answer,
        expected_evidence=[[e.model_dump() for e in group] for group in case.evidence],
        answerable=case.answerable,
        answer=answer,
        search_query=trace.query,
        sources=emitted_sources,
        retrieved_context=[asdict(rc.chunk) for rc in result.chunks] if result else [],
        context_evidence_recall=(
            sum(evidence_hits(chunks, case.evidence)) / len(case.evidence)
            if case.evidence
            else None
        ),
        error=errors or ([] if done else ["Missing terminal done event"]),
        diagnostics={
            **citation_diagnostics(answer, chunks),
            "exact_refusal_template": normalize(answer)
            in {normalize(prompts.NO_ANSWER_RU), normalize(prompts.NO_ANSWER_EN)},
        },
    )


async def run(args) -> dict:
    if args.calibrate and (args.split != "dev" or args.backend != "real"):
        raise ValueError("--calibrate requires --split dev --backend real")
    if args.with_llm and args.backend != "real":
        raise ValueError("--with-llm requires --backend real")
    code_before = code_digest()
    data_before = {p.name: digest(p.read_bytes()) for p in sorted(DATA.iterdir())}
    config_before = digest(args.config.read_bytes())
    corpus, all_cases, all_dialogues = load_dataset()
    cases = [c for c in all_cases if c.split == args.split]
    documents = [d for d in corpus.documents if d.split == args.split]
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        settings = settings_for_run(args.config, directory / "data", args.with_llm)
        if args.backend != "real" and settings.use_reranker:
            raise ValueError("Reranker requires --backend real")
        container = None
        if args.backend != "bm25":
            embedder = None
            if args.backend == "smoke":
                from tests.conftest import HashEmbedder

                embedder = HashEmbedder()
            container = build_container(
                settings, embedder=embedder, llm=None if args.with_llm else DisabledLLM()
            )
            if args.backend == "real":
                container.embedder.warmup()
        chunks_by_doc, indexes, extracted_hashes = {}, {}, {}
        for doc in documents:
            pdf = make_pdf(doc.pages, directory / f"{doc.id}.pdf", header=doc.title)
            pages = parse_pdf(pdf)
            extracted_hashes[doc.id] = digest(
                json.dumps([asdict(p) for p in pages], ensure_ascii=False).encode()
            )
            if container:
                container.db.create_document(doc.id, pdf.name, doc.id)
                container.ingestion.ingest(doc.id, pdf)
                status = container.db.get_document(doc.id)
                if status.status != "ready":
                    raise RuntimeError(f"Indexing failed: {doc.id}: {status.error}")
                chunks = container.db.list_chunks(doc.id)
            else:
                chunks = chunk_pages(
                    clean_pages(pages),
                    doc.id,
                    settings.chunk_size,
                    settings.chunk_overlap,
                    settings.min_chunk_chars,
                )
            chunks_by_doc[doc.id] = chunks
            indexes[doc.id] = BM25Index(chunks)

        def rank(case):
            if container:
                return container.rag._retriever.rank(case.document_id, case.question)
            by_id = {c.id: c for c in chunks_by_doc[case.document_id]}
            return {
                "bm25": [
                    RetrievedChunk(by_id[cid], score, bm25_score=score)
                    for cid, score in indexes[case.document_id].search(
                        case.question, settings.top_k_bm25
                    )
                ]
            }

        for doc in documents:
            rank(next(c for c in cases if c.document_id == doc.id))
        rows = []
        deployed_mode = "rerank" if settings.use_reranker else "hybrid"
        for case in cases:
            t0 = time.perf_counter()
            rankings = rank(case)
            elapsed = (time.perf_counter() - t0) * 1000
            oracle = evidence_hits(chunks_by_doc[case.document_id], case.evidence)
            row = dict(
                case_id=case.id,
                document_id=case.document_id,
                question=case.question,
                answerable=case.answerable,
                tags=case.tags,
                evidence_groups=len(case.evidence),
                oracle_evidence_recall=sum(oracle) / len(oracle) if oracle else None,
                ranking_pass_ms=elapsed,
                rankings={
                    mode: [
                        dict(chunk_id=r.chunk.id, score=r.score, similarity=r.similarity)
                        for r in ranking
                    ]
                    for mode, ranking in rankings.items()
                },
                metrics={
                    mode: retrieval_metrics([r.chunk for r in ranking], case.evidence)
                    for mode, ranking in rankings.items()
                },
            )
            if container:
                selected = rankings[deployed_mode][: settings.top_k_final]
                result = context_relevance(selected, case.question)
                row["gate"] = dict(
                    has_context=bool(selected),
                    max_similarity=result.max_similarity,
                    lexical_coverage=result.lexical_coverage,
                    accepted=bool(selected)
                    and result.is_relevant(
                        settings.min_relevance_score, settings.min_lexical_coverage
                    ),
                    complete_evidence=all(evidence_hits([r.chunk for r in selected], case.evidence))
                    if case.answerable
                    else None,
                )
            rows.append(row)

        predictions = []
        if args.with_llm:
            trace = RecordingRetriever(container.rag._retriever)
            container.rag._retriever = trace
            for case in cases:
                container.db.clear_messages(case.document_id)
                prediction = await generate(container, trace, case, "single_turn")
                predictions.append(prediction)
                print(
                    f"{case.id}: {'error' if prediction['error'] else 'done'} "
                    f"in {prediction['duration_s']:.1f}s",
                    flush=True,
                )
            for dialogue in (d for d in all_dialogues if d.split == args.split):
                container.db.clear_messages(dialogue.document_id)
                for turn_number, turn in enumerate(dialogue.turns, 1):
                    case = Case(
                        id=f"{dialogue.id}:{turn_number}",
                        document_id=dialogue.document_id,
                        split=dialogue.split,
                        question=turn.question,
                        answer=turn.answer,
                        evidence=turn.evidence,
                        tags=["dialogue"],
                        negative_reason=None,
                    )
                    prediction = await generate(container, trace, case, "dialogue")
                    predictions.append(prediction)
                    print(
                        f"{case.id}: {'error' if prediction['error'] else 'done'} "
                        f"in {prediction['duration_s']:.1f}s",
                        flush=True,
                    )
                    if prediction["error"]:
                        break

        packages = {}
        for name in (
            "chromadb",
            "sentence-transformers",
            "transformers",
            "torch",
            "PyMuPDF",
            "rank-bm25",
            "snowballstemmer",
        ):
            packages[name] = importlib.metadata.version(name)
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        config = {name: getattr(settings, name) for name in sorted(CONFIG_FIELDS)}
        if code_digest() != code_before or digest(args.config.read_bytes()) != config_before:
            raise RuntimeError("Code/config changed during the run; repeat with frozen inputs")
        if any(digest((DATA / name).read_bytes()) != value for name, value in data_before.items()):
            raise RuntimeError("Dataset changed during the run; repeat with frozen inputs")
        identity = dict(
            config=config,
            backend=args.backend,
            split=args.split,
            data_hashes={p.name: digest(p.read_bytes()) for p in sorted(DATA.iterdir())},
            llm_identity={"provider": container.llm.name, "model": container.llm.model}
            if args.with_llm
            else None,
            packages=packages,
            prompts={"system": prompts.SYSTEM_PROMPT, "condense": prompts.CONDENSE_SYSTEM_PROMPT}
            if args.with_llm
            else None,
            code_sha256=code_digest(),
            lock_sha256=digest((ROOT / "uv.lock").read_bytes()),
        )
        return dict(
            schema_version=1,
            run_id=uuid.uuid4().hex,
            experiment_id=digest(json.dumps(identity, sort_keys=True).encode())[:16],
            **identity,
            corpus_version=corpus.version,
            provenance=corpus.provenance,
            timestamp_utc=datetime.now(UTC).isoformat(),
            python=platform.python_version(),
            platform=platform.platform(),
            git_commit=git.stdout.strip() if git.returncode == 0 else None,
            model_revision=(
                settings.embedding_revision
                if args.backend == "real"
                else "not applicable: no pretrained model"
            ),
            llm={"provider": container.llm.name, "model": container.llm.model}
            if args.with_llm
            else None,
            complete=(
                not any(p["error"] for p in predictions)
                and (
                    not args.with_llm
                    or len(predictions)
                    == len(cases)
                    + sum(len(d.turns) for d in all_dialogues if d.split == args.split)
                )
            ),
            expected_predictions=(
                len(cases) + sum(len(d.turns) for d in all_dialogues if d.split == args.split)
            )
            if args.with_llm
            else 0,
            elapsed_s=time.perf_counter() - started,
            extracted_text_hashes=extracted_hashes,
            chunks={key: [asdict(c) for c in value] for key, value in chunks_by_doc.items()},
            cases=[c.model_dump() for c in cases],
            rows=rows,
            summary=summarize(rows),
            gate=gate_summary(rows),
            predictions=predictions,
            calibration=calibrate(rows, args.split) if args.calibrate else None,
        )


def markdown(report: dict) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"Run: `{report['run_id']}` · split: **{report['split']}** · "
        f"backend: **{report['backend']}**",
        "",
        "Synthetic, jointly authored fixtures; not an independent generalization benchmark.",
        "No statistical significance or production-quality claim is made.",
        "",
    ]
    if report["backend"] == "smoke":
        lines += [
            "**SMOKE ONLY: hash embeddings are not E5; these are not model quality results.**",
            "",
        ]
    if not report["complete"]:
        lines += ["**INCOMPLETE: generation errors occurred; inspect the JSON artifact.**", ""]
    lines += [
        "All methods use identical queries, chunks, evidence labels and output cutoffs.",
        "Hybrid can use the union of both candidate pools; its compute budget is larger.",
        "",
        "| Mode | n | Hit@1 | Evidence recall@6 | Complete@6 | MRR@6 | Document macro Complete@6 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, stats in report["summary"].items():
        m = stats["question_mean"]
        lines.append(
            f"| {mode} | {stats['n']} | {m['hit@1']:.3f} | "
            f"{m['evidence_recall@6']:.3f} | {m['complete@6']:.3f} | "
            f"{m['mrr@6']:.3f} | {stats['document_macro']['complete@6']:.3f} |"
        )
    positive = [r for r in report["rows"] if r["answerable"]]
    lines += [
        "",
        f"Answerable cases: {len(positive)}; unanswerable: {len(report['rows']) - len(positive)}.",
        f"Oracle complete evidence after parsing/chunking: "
        f"{sum(r['oracle_evidence_recall'] == 1 for r in positive)}/{len(positive)}.",
    ]
    if report["backend"] == "real":
        g = report["gate"]
        lines += [
            "",
            "## Gate (deployed context only)",
            "",
            f"Thresholds used: cosine >= {report['config']['min_relevance_score']} or "
            f"lexical coverage >= {report['config']['min_lexical_coverage']}.",
            f"False rejections: {g['false_rejections']}/{g['answerable_n']} answerable.",
            f"False acceptances: {g['false_acceptances']}/{g['unanswerable_n']} unanswerable.",
            f"Accepted with incomplete evidence: {g['accepted_without_complete_evidence']}.",
        ]
    if report["calibration"]:
        selected = report["calibration"]["selected"]
        lines += [
            "",
            "## Dev-only calibration (not applied to the gate counts above)",
            "",
            f"Selected cosine {selected['cosine']}, coverage {selected['coverage']}; "
            f"balanced error {selected['balanced_error']:.4f}; "
            f"false rejections {selected['false_rejections']}, "
            f"false acceptances {selected['false_acceptances']}.",
            "Full grid and selection rule are in JSON. Always-pass/always-reject "
            "baselines each have balanced error 0.5 when both classes are present.",
        ]
    latencies = [r["ranking_pass_ms"] for r in report["rows"]]
    lines += [
        "",
        f"Median shared ranking pass: {statistics.median(latencies):.1f} ms. "
        "One warm-up per document; includes all enabled rankings; not per-mode latency.",
        "",
        "Per-case rankings, all @1/3/6 metrics, document/tag breakdowns, config, hashes, "
        "chunks and raw predictions are in the adjacent JSON artifact.",
        "",
        "## Generation",
        "",
    ]
    if report["predictions"]:
        lines += [
            f"Generated {len(report['predictions'])}/{report['expected_predictions']} records "
            "(single-turn and dialogues).",
            "Citation presence and refusal templates are diagnostics, not correctness scores.",
            "Semantic quality requires completing and reviewing the adjacent review JSONL.",
        ]
    else:
        lines += ["Not run. No answer-quality or citation-correctness score is claimed."]
    lines += [
        "",
        f"Model revision: {report['model_revision']}.",
        "See docs/EVALUATION.md for the protocol and limitations.",
    ]
    return "\n".join(lines) + "\n"


def write_artifacts(report: dict, output: Path):
    targets = [output, output.with_suffix(".json")]
    if report["predictions"]:
        targets.append(output.with_suffix(".review.jsonl"))
    if report["calibration"]:
        targets.append(output.with_suffix(".calibrated.json"))
    if any(path.exists() for path in targets):
        raise FileExistsError(
            "Choose a new output path; existing runs/reviews are never overwritten"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown(report))
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["predictions"]:
        from tests.eval.review import review_template

        output.with_suffix(".review.jsonl").write_text(
            "".join(
                json.dumps(review_template(report["run_id"], p), ensure_ascii=False) + "\n"
                for p in report["predictions"]
            )
        )
    if report["calibration"]:
        selected = report["calibration"]["selected"]
        config = {
            **report["config"],
            "min_relevance_score": selected["cosine"],
            "min_lexical_coverage": selected["coverage"],
        }
        output.with_suffix(".calibrated.json").write_text(json.dumps(config, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval, gate and grounded generation."
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--backend", choices=("real", "bm25", "smoke"), default="real")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--output", type=Path, help="New .md path; defaults to a unique run name")
    args = parser.parse_args()
    if args.output and args.output.suffix != ".md":
        parser.error("--output must end in .md")
    if args.output and any(
        args.output.with_suffix(suffix).exists()
        for suffix in (".md", ".json", ".review.jsonl", ".calibrated.json")
    ):
        parser.error("Output already exists; choose a new path to preserve prior runs")
    report = asyncio.run(run(args))
    args.output = args.output or (
        ROOT / "artifacts/eval" / f"{args.backend}-{args.split}-{report['run_id'][:8]}.md"
    )
    write_artifacts(report, args.output)
    print(markdown(report))
    print(f"Artifacts: {args.output}")
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
