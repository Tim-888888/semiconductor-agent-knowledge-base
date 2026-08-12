FROM python:3.12.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY requirements.lock ./
RUN pip install --require-hashes -r requirements.lock
COPY src ./src
COPY data ./data

RUN addgroup --system --gid 10001 semikb \
    && adduser --system --uid 10001 --ingroup semikb --home /app semikb \
    && chown -R semikb:semikb /app

USER 10001:10001

EXPOSE 8000
CMD ["uvicorn", "semikb.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
