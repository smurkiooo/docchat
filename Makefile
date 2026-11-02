.PHONY: install run test lint format eval eval-test eval-bm25 eval-smoke eval-llm sample docker

install:
	uv sync --frozen

run:
	uv run uvicorn app.main:app --reload --reload-dir app --port 8000

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

eval:
	uv run python -m tests.eval.run_eval --split dev

eval-test:
	uv run python -m tests.eval.run_eval --split test

eval-bm25:
	uv run python -m tests.eval.run_eval --backend bm25 --split dev
	uv run python -m tests.eval.run_eval --backend bm25 --split test

eval-smoke:
	uv run python -m tests.eval.run_eval --backend smoke --split dev

eval-llm:
	uv run python -m tests.eval.run_eval --split dev --with-llm

sample:
	uv run python -m scripts.make_sample_pdf

docker:
	docker compose up --build
