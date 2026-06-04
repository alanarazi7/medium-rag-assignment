# Medium Article RAG Assistant

A Retrieval-Augmented Generation (RAG) system over ~7,600 Medium articles. Given a natural language question, it retrieves the most relevant article passages from a Pinecone vector index and answers strictly from that context using a chat model.

## Live URLs

- **Demo UI**: `https://medium-rag-assignment-livid.vercel.app/`
- **POST** `https://medium-rag-assignment-livid.vercel.app/api/prompt`
- **GET** `https://medium-rag-assignment-livid.vercel.app/api/stats`

---

## How It Works

```
Question
   │
   ▼
Embed with text-embedding-3-small (1536-dim)
   │
   ▼
Query Pinecone — fetch top 28 candidates
   │
   ▼
Filter: max 3 chunks per article → keep top 7
   │
   ▼
Build augmented prompt (system + context + question)
   │
   ▼
Call gpt-5-mini → return response + context + prompts
```

The app is deployed as **Python serverless functions on Vercel** — one file per endpoint, no persistent server. Each request is stateless: embed → retrieve → generate → return.

---

## Dataset

7,682 English-language Medium articles. Fields: `title`, `text`, `url`, `authors`, `timestamp`, `tags`.

---

## Hyperparameter Decisions

### Chunk size — 300 tokens

The course recommends **512–1024 tokens** for general long-form text. We measured the actual paragraph structure of the corpus before deciding:

| Statistic | Tokens |
|-----------|--------|
| Mean paragraph length | 44 |
| Median paragraph length | 33 |
| 90th percentile | 91 |
| 99th percentile | 180 |

66.8% of paragraphs are under 50 tokens. A 512-token chunk spans roughly 10–15 paragraphs and mixes multiple distinct ideas into one embedding vector — diluting cosine similarity for any single query. We chose **300 tokens** to capture 3–5 natural paragraphs (one coherent section) while keeping the embedding tight enough to retrieve precisely.

### Chunking strategy — paragraph-aware

Rather than a blind sliding window, we accumulate whole paragraphs until the next one would exceed the 300-token limit. The overlap carries trailing paragraphs (up to 20% of chunk size, ~60 tokens) into the next chunk. This guarantees no sentence is ever split mid-paragraph.

### Overlap — 0.20 (20%)

Medium articles sit between two course categories:

| Category | Chunk size | Overlap |
|---|---|---|
| General, long articles | 512–1024 | 5–15% |
| Conversational | 200–400 | — |

The writing style is conversational (short punchy paragraphs) but total article length is long. We landed at 20% — slightly above the long-article recommendation — to compensate for the conversational style where a key idea sometimes bridges two paragraphs.

### top_k — 7

The course recommends **3–5 chunks** for general text. We chose **7** because our per-article cap (C=3) means a single highly-relevant article can occupy up to 3 slots, leaving only 4 for other articles. With k=5 and C=3, worst case is 3+2 = only 2 distinct articles — not enough for multi-result queries that ask for 3. With k=7 and C=3, worst case is 3+3+1 = **3 distinct articles guaranteed**.

### max_chunks_per_article (C) — 3

A cap is needed to prevent one article from dominating all k slots. Setting C too low (e.g. 1) risks missing information spread across sections of a genuinely relevant article. C=3 allows meaningful depth — three 300-token chunks cover ~900 tokens, roughly half a Medium article — while still leaving room for other sources.

### Over-fetching

Pinecone is queried for `top_k × max(3, C+1) = 7 × 4 = 28` candidates. Results are walked in score order; any article that has already contributed C=3 chunks is skipped. Over-fetching is necessary because if Pinecone's top 7 results were dominated by one article, applying the cap would leave fewer than 7 final chunks — the buffer ensures all 7 slots are always filled. The final context contains exactly 7 chunks from at least 3 distinct articles.

---

## API

### `POST /api/prompt`

**Input:**
```json
{ "question": "Your natural language question here" }
```

**Output:**
```json
{
  "response": "Final answer from the model.",
  "context": [
    {
      "article_id": "42",
      "title": "How The Media Can Prevent Copycat Suicides",
      "chunk": "The Werther effect was coined in the late 1700s...",
      "score": 0.4271
    }
  ],
  "Augmented_prompt": {
    "System": "You are a Medium-article assistant...",
    "User": "Context:\n[1] Article: ...\n\nQuestion: ..."
  }
}
```

### `GET /api/stats`

```json
{ "chunk_size": 300, "overlap_ratio": 0.2, "top_k": 7 }
```

---

## Deployment

**Vercel Python serverless** — each `api/*.py` file becomes an HTTP function. Streamlit was considered but ruled out: it runs a stateful WebSocket server and cannot expose the REST endpoints the assignment requires. Vercel's Python runtime handles this natively with no Node.js dependency.

Environment variables required (set in Vercel dashboard):
```
OPENAI_API_KEY
OPENAI_BASE_URL
PINECONE_API_KEY
PINECONE_INDEX_NAME
```

---

## Local Setup

```bash
cp .env.example .env        # fill in credentials

pip install -r scripts/requirements_ingest.txt

# Test with 100 articles first
python scripts/ingest.py --csv /path/to/medium-english-50mb.csv --limit 100

# Full corpus (~$0.29, ~45k chunks)
python scripts/ingest.py --csv /path/to/medium-english-50mb.csv

# Run 5-question end-to-end test
python scripts/test_local.py
```
