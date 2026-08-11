# Короткие команды поверх того, что уже умеют CLI и compose.
# Makefile ничего не решает сам — он только избавляет от печатания.

.DEFAULT_GOAL := help
VENV := .venv
PY   := $(VENV)/bin/python

.PHONY: help
help: ## показать список команд
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- окружение ---------------------------------------------------------------

.PHONY: venv
venv: ## создать venv и поставить проект в режиме разработки
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q --upgrade pip
	$(VENV)/bin/pip install -q -e '.[dev]'

# --- проверки ----------------------------------------------------------------

.PHONY: test
test: ## прогнать тесты (тесты pgvector пропустятся без базы)
	$(PY) -m pytest -q

.PHONY: lint
lint: ## ruff
	$(VENV)/bin/ruff check src/ scripts/ tests/

.PHONY: format
format: ## отформатировать код
	$(VENV)/bin/ruff format src/ scripts/ tests/
	$(VENV)/bin/ruff check src/ scripts/ tests/ --fix

.PHONY: check
check: lint test ## линтер плюс тесты

# --- стек --------------------------------------------------------------------

.PHONY: up
up: ## поднять всё (первый запуск строит индекс, это занимает минуты)
	docker compose up -d
	@echo "Готовность: curl -s localhost:8000/health"

.PHONY: down
down: ## остановить контейнеры
	docker compose down

.PHONY: clean-volumes
clean-volumes: ## остановить и стереть базу, кэш весов и кэш эмбеддингов
	docker compose down -v

.PHONY: db
db: ## поднять только базу (нужна для тестов pgvector)
	docker compose up -d db

.PHONY: logs
logs: ## логи api
	docker compose logs -f api

# --- корпус и индекс ---------------------------------------------------------

.PHONY: corpus
corpus: ## скачать корпус по команде из data/raw/SOURCE.md
	mkdir -p data/raw
	curl -sS -L --max-time 300 --compressed \
		-A "Mozilla/5.0 (compatible; kz-labor-rag/0.1)" \
		-o data/raw/adilet_K1500000414_rus.html \
		https://adilet.zan.kz/rus/docs/K1500000414
	shasum -a 256 -c data/raw/CHECKSUM || \
		echo "ВНИМАНИЕ: корпус отличается от зафиксированного — вероятно, новая редакция"

.PHONY: corpus-stats
corpus-stats: ## статистика разбора корпуса
	$(VENV)/bin/kzrag-corpus stats

.PHONY: index
index: ## построить индекс
	$(VENV)/bin/kzrag-index build

.PHONY: reindex
reindex: ## пересоздать индекс с нуля
	$(VENV)/bin/kzrag-index build --rebuild

.PHONY: index-status
index-status: ## чем построен текущий индекс
	$(VENV)/bin/kzrag-index status

# --- датасет и eval ----------------------------------------------------------

.PHONY: dataset
dataset: ## пересобрать синтетическую часть датасета и файл ревью
	$(PY) scripts/build_synthetic_dataset.py

.PHONY: dataset-check
dataset-check: ## сверить разметку эталона с текстом кодекса
	$(VENV)/bin/kzrag-corpus check-dataset

.PHONY: validate
validate: ## проверить гейт готовности датасета
	$(VENV)/bin/kzrag-eval validate

.PHONY: failures
failures: ## показать вопросы, где эталон не попал в топ-k
	$(VENV)/bin/kzrag-search --failures

.PHONY: eval
eval: ## прогнать eval и записать результат
	$(VENV)/bin/kzrag-eval run
