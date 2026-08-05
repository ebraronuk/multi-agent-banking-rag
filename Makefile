.PHONY: install dev lint typecheck test cov seed run mcp docker-build docker-up docker-down

install:
	pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload --app-dir src --port 8000

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	mypy src

test:
	pytest

cov:
	pytest --cov=src --cov-report=html && open htmlcov/index.html

seed:
	python scripts/seed_vectorstore.py

eval:
	PYTHONPATH=src python -m evaluation.eval_harness

run:
	python -m app.main --app-dir src

mcp:
	python -m mcp_server.server --app-dir src

docker-build:
	docker build -f docker/Dockerfile -t multi-agent-banking-rag:local .

docker-up:
	docker compose -f docker/docker-compose.yml up --build

docker-down:
	docker compose -f docker/docker-compose.yml down -v
