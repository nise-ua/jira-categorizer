# Parent / local container image (also used to build the enclave .eif)
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY sample_data ./sample_data

RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

# Train a bootstrap model from sample data (replace with sealed prod data mount)
RUN jira-cat-pipeline --config /app/config/default.yaml

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["jira-cat-serve", "--host", "0.0.0.0", "--port", "8080"]
