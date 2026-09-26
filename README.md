# Jira Enclave Categorizer

Python-only MLOps pipeline that categorizes Jira tickets from **summary** + **description**, learning historical **labels** and **impacted area**. Runs in a **secure enclave** pattern (AWS Nitro Enclaves oriented). **No LLM.**

## What it learns

| Input (new ticket) | Targets (from history) |
|---|---|
| `summary` | `labels` (multi-label) |
| `description` | `impacted_area` (multiclass; multi-value supported in training data) |

Features are classical **TF-IDF word + character n-grams**. Models are **One-Vs-Rest logistic regression** (labels) and **multiclass logistic regression** (area). Swap to `linear_svc` via config.

## Architecture

```
Jira export (CSV/JSON) --> ingest/preprocess (inside enclave)
                              |
                              v
                     TF-IDF feature union
                              |
              +---------------+---------------+
              v                               v
     Labels model (multi-label)     Impacted-area model
              |                               |
              +---------------+---------------+
                              v
                    Model registry (+ optional seal)
                              |
                              v
              Inference server (HTTP / vsock proxy)
```

### Secure enclave MLOps loop

1. **Ingest** historical tickets with known labels / impacted area (offline export preferred; no SaaS LLM calls).
2. **Train + evaluate** inside the enclave (or local_dev mode).
3. **Gate** on F1 thresholds; **promote** to `production` registry.
4. **Seal** model bundle (local Fernet for dev; production = Nitro attestation + KMS-unwrapped DEK).
5. **Serve** predictions over enclave loopback; parent instance proxies vsock.
6. **Drift check** on mean text length vs last training frame (extend with PSI on top terms).

Production attestation hook: `jira_categorizer/enclave/attestation.py` — set `enclave.mode: nitro` and supply `NITRO_ATTESTATION_DOC`. KMS key alias is configured under `enclave.kms_key_alias`.

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Train + register + smoke predict
jira-cat-pipeline

# Single prediction
jira-cat-predict --summary "Checkout 504 on card charge" --description "EU customers fail at payment"

# HTTP inference
jira-cat-serve --host 127.0.0.1 --port 8080
curl -s localhost:8080/predict -H 'content-type: application/json' \
  -d '{"summary":"SSO blank page after Azure callback","description":"Mobile web login"}'
```

### Bring your Jira export

CSV columns (aliases accepted):

- required: `summary`, `description`
- targets: `labels`, `impacted_area` (also accepts `components`)
- optional: `issue_key`

Multi-value fields: `payments|p1` or `auth;sso`.

```bash
jira-cat-pipeline --data /path/to/jira_export.csv
```

## Config

See [`config/default.yaml`](config/default.yaml) for feature sizes, model hyperparameters, promotion gates, and enclave settings.

## Containers / Nitro

```bash
# Local container (parent-style)
docker build -f enclave/Dockerfile -t jira-cat:local .

# Enclave image -> build .eif on Nitro-capable parent
docker build -f enclave/Dockerfile.enclave -t jira-cat:enclave .
# nitro-cli build-enclave --docker-uri jira-cat:enclave --output-file jira-cat.eif
# nitro-cli run-enclave --eif-path jira-cat.eif --cpu-count 2 --memory 4096 --enclave-cid 16
```

Host proxy sketch: `vsock-proxy` / `socat` from parent TCP to enclave CID port `5000`/`8080`. Only release sealed model DEK after PCR verification against the expected EIF measurements.

## Ops UI

Simple browser UI for historical surge charts and human triage:

```bash
# ensure a model exists
jira-cat-pipeline --config config/kafka_public.yaml

# serve API + UI
jira-cat-serve --config config/kafka_public.yaml --host 127.0.0.1 --port 8080
# open http://127.0.0.1:8080/
```

- **Historical view**: line charts by driver (`labels`, `impacted_area`, `issuetype`, `status`) and window (`1d` / `1w` / `1m` / `3m`), plus known-vs-novel volume and surge / new-label callouts.
- **Triage inbox**: table of recent tickets with ML-suggested label + impacted area dropdowns; expand description; **Save** writes feedback locally and pushes to Jira when credentials are set (`JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`).
- **Learning loop**: disagreements land in `artifacts/feedback/corrections.jsonl`; **Retrain with feedback** merges corrections into training and promotes a new model.
- **Offline-first data**: import a Jira **Excel/CSV** export in the UI, or connect **Jira API** when available. Local cache keeps learning/triage working without network.
- **Day watch**: forecast (dashed) vs actual hourly lines for **today + next day**, with a short management summary (on-track / surge / novel category).

## Monitoring: category surges + new label suggestions

Classical monitors (no LLM), run after train or via `jira-cat-monitor`:

1. **Category surge** — compare recent vs baseline daily rates for each `label` and `impacted_area` using rate-ratio + z-score. Alerts when a category spikes.
2. **New label suggestion** — tickets with low label *and* area confidence are treated as novel, clustered (TF-IDF + MiniBatchKMeans), and a slug is proposed from top terms + exemplar issues.

```bash
jira-cat-monitor --config config/kafka_public.yaml
# reports: artifacts/reports/.../category_surges.json
#          artifacts/reports/.../new_label_suggestions.json
```

Inference also attaches a `novelty` block per ticket. HTTP: `GET /monitor/surges`, `GET /monitor/new-labels`.

## Public Jira experiment (Apache Kafka)

Fetched **100** anonymously readable issues from [Apache Jira / KAFKA](https://issues.apache.org/jira) with non-empty `labels` and `components`:

```bash
python scripts/fetch_apache_jira.py --limit 100 --out sample_data/public_jira/kafka_100.csv
jira-cat-pipeline --config config/kafka_public.yaml
```

Field mapping: `components` → `impacted_area`, `labels` → multi-label target.

Holdout results on this slice (75 train / 25 test, classical TF-IDF + logistic regression):

| Target | Metric | Score |
|---|---|---|
| Impacted area (6 classes) | Accuracy | ~0.62 |
| Impacted area | F1 weighted | ~0.54 |
| Labels (13 classes) | F1 micro | ~0.50 |
| Labels | F1 macro | ~0.41 |

Area prediction from summary/description is usable even at n=100. Labels are noisier (process tags like `kip` / `newbie` / `gradle`) and benefit from taxonomy cleanup + more volume. Full write-up: `artifacts/reports/kafka_public/kafka_public_experiment.md`.

## Tests

```bash
pytest -q
```

## Design constraints

- **No LLM / no external generative APIs**
- **Python only** (scikit-learn stack)
- Training data and models stay on sealed paths under `artifacts/`
- Suitable for air-gapped or enclave-bound Jira metadata categorization
