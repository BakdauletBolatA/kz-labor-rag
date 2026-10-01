# Короткие команды поверх того, что уже умеют CLI и compose.
# Makefile ничего не решает сам — он только избавляет от печатания.

.DEFAULT_GOAL := help
VENV := .venv
PY   := $(VENV)/bin/python

# Интерпретатор, на котором собирается venv. Проект требует Python >= 3.12
# (requires-python в pyproject.toml), а `python3` в системе бывает старее.
# Тогда venv создавался молча, а падал уже pip — и не про версию Python, а
# сообщением резолвера «requires a different Python», из которого причина не
# читается: выглядит как проблема с зависимостями, а не с интерпретатором.
# Поэтому версия проверяется до создания venv.
#
# Свой интерпретатор: make venv PYTHON=/usr/bin/python3.12
PYTHON ?=
PYTHON_CANDIDATES := $(if $(PYTHON),$(PYTHON),python3 python3.12 python3.13 python3.14)

.PHONY: help
help: ## показать список команд
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- окружение ---------------------------------------------------------------

.PHONY: venv
venv: ## создать venv и поставить проект в режиме разработки
	@found=""; \
	for p in $(PYTHON_CANDIDATES); do \
		command -v "$$p" >/dev/null 2>&1 || continue; \
		"$$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' \
			2>/dev/null || continue; \
		found="$$p"; break; \
	done; \
	if [ -z "$$found" ]; then \
		echo "Нужен Python >= 3.12: этого требует requires-python в pyproject.toml." >&2; \
		echo "Проверены: $(PYTHON_CANDIDATES)" >&2; \
		echo "Укажите свой: make venv PYTHON=/путь/к/python3.12" >&2; \
		exit 1; \
	fi; \
	echo "venv на $$("$$found" -V)"; \
	"$$found" -m venv $(VENV)
	$(VENV)/bin/pip install -q --upgrade pip
	$(VENV)/bin/pip install -q -e '.[dev]'

# --- проверки ----------------------------------------------------------------

.PHONY: test
test: ## прогнать тесты (тесты pgvector пропустятся без базы)
	$(PY) -m pytest -q

.PHONY: lint
lint: ## ruff
	$(VENV)/bin/ruff check .

.PHONY: format
format: ## отформатировать код
	$(VENV)/bin/ruff format .
	$(VENV)/bin/ruff check . --fix

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

.PHONY: dataset-check
dataset-check: ## сверить разметку эталона с текстом кодекса
	$(VENV)/bin/kzrag-corpus check-dataset

.PHONY: validate
validate: ## проверить гейт готовности датасета
	$(VENV)/bin/kzrag-eval validate

.PHONY: failures
failures: ## показать вопросы, где эталон не попал в топ-k
	$(VENV)/bin/kzrag-search --failures

.PHONY: compare
compare: ## таблица «до/после»: make compare A=... B=...
	$(VENV)/bin/kzrag-eval compare $(A) $(B)

.PHONY: review
review: ## ревью тестового набора
	$(PY) evals/review.py

.PHONY: eval
eval: ## все замеры и обновление таблиц в README
	$(PY) eval.py

.PHONY: eval-run
eval-run: ## один прогон метрик поиска по текущему конфигу
	$(VENV)/bin/kzrag-eval run
