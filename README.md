# PromptShield — LLM Prompt-Injection Gateway

> A security gateway that evaluates prompts before they reach an LLM and blocks likely prompt-injection or jailbreak attempts.

## What this project does

PromptShield sits between a client and a target language model. It applies a fast pattern layer, a semantic-similarity layer, and an optional local LLM judge. Prompts are classified as allowed, suspicious, or blocked before any forwarding occurs.

## Core functions

- Match known injection and jailbreak patterns with a regex library.
- Compare prompts against injection and safe corpora with sentence-transformer embeddings.
- Optionally use a local Ollama model as a third decision layer.
- Forward allowed prompts to a configured target LLM endpoint.
- Provide a FastAPI interface, dashboard, SQLite history, and Prometheus metrics.

## Architecture

```text
Client prompt → pattern matcher → semantic classifier → optional local LLM judge
                                                        │
                                      block ◀───────────┼───────────▶ allow → target LLM
```

## Technology and local run

Python, FastAPI, sentence-transformers, scikit-learn, optional Ollama, SQLite/Peewee, and Prometheus.

```bash
cd promptshield
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python scripts/train_classifier.py
python run.py
```

## Limitations

Prompt security is probabilistic. Tune thresholds, review logs, and combine this gateway with model-level safeguards and least-privilege tool access.
