# Python 3.12 по требованию проекта (локально может стоять другой).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# torch ставится отдельно и только в CPU-сборке. Обычный индекс тянет за
# собой пакеты nvidia и triton — около 3.5 ГБ CUDA, которые в этом образе
# не используются никогда: индексация идёт на CPU, GPU в Docker на Mac нет.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Остальные зависимости — отдельным слоем: правка кода не должна тянуть
# переустановку torch, а это самая долгая часть сборки.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

COPY config/ ./config/
COPY scripts/ ./scripts/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "kz_labor_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
