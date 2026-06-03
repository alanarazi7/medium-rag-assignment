# Medium Article RAG Assistant

A Retrieval-Augmented Generation system over ~7,600 Medium articles, deployed as a public API on Vercel.

## Live URLs

- **App / Demo UI**: `https://medium-rag-assignment.vercel.app/`
- **POST** `https://medium-rag-assignment.vercel.app/api/prompt`
- **GET** `https://medium-rag-assignment.vercel.app/api/stats`

## Deployment Decision

The assignment requires two public REST endpoints (`POST /api/prompt`, `GET /api/stats`) hosted on **Vercel**. We initially considered Streamlit, but Streamlit runs a stateful WebSocket server and cannot expose standard REST endpoints — so it does not satisfy the API contract the grader tests.

We instead use **Vercel's Python serverless runtime**: each file in `api/` becomes a standalone function invoked per HTTP request. This is purely Python (no Node.js needed), scales automatically, and exposes exactly the endpoints required. A lightweight HTML demo page at `/` serves as the interactive front-end, calling the same `/api/prompt` endpoint from the browser.

## RAG Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `chunk_size` | 512 tokens | Focused, single-topic chunks; better embedding signal than 1024 |
| `overlap_ratio` | 0.2 | 102-token overlap bridges chunk boundaries without bloating index |
| `top_k` | 8 | Covers all 4 query types with headroom; with C=3, guarantees at least 3 distinct articles |
| `max_chunks_per_article` | 3 | Ensures retrieval diversity while still allowing deep context on a single article for summary queries |

For multi-result queries (type 2: "list 3 articles"), over-fetching 15 candidates (top_k × 3) and capping at 3 chunks per article guarantees at least 2 distinct articles in every response.

## API

### `POST /api/prompt`

```json
{ "question": "Your question here" }
```

Returns:
```json
{
  "response": "...",
  "context": [{ "article_id": "...", "title": "...", "chunk": "...", "score": 0.85 }],
  "Augmented_prompt": { "System": "...", "User": "..." }
}
```

### `GET /api/stats`

```json
{ "chunk_size": 512, "overlap_ratio": 0.2, "top_k": 5 }
```

## Local Setup

```bash
cp .env.example .env   # fill in credentials

# Ingest (100-article test first)
pip install -r scripts/requirements_ingest.txt
python scripts/ingest.py --csv /path/to/medium-english-50mb.csv --limit 100

# Full corpus
python scripts/ingest.py --csv /path/to/medium-english-50mb.csv

# Run 5-question local test
python scripts/test_local.py
```
