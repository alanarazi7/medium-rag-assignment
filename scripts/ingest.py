"""One-time ingestion script: chunk articles, embed, upsert to Pinecone.

Usage:
    # Copy .env.example to .env and fill in your credentials, then:

    # Test with 100 articles first:
    python scripts/ingest.py --csv /path/to/medium-english-50mb.csv --limit 100

    # Full corpus:
    python scripts/ingest.py --csv /path/to/medium-english-50mb.csv
"""

import argparse
import os
import time

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import tiktoken
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

CHUNK_SIZE = 300
OVERLAP_RATIO = 0.2
EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
DIMENSIONS = 1536
EMBED_BATCH = 100
UPSERT_BATCH = 100
MIN_CHUNK_TOKENS = 50


def chunk_text(text: str, enc) -> list[str]:
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 0]
    overlap_budget = int(CHUNK_SIZE * OVERLAP_RATIO)

    chunks = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(enc.encode(para))
        if current_tokens + para_tokens > CHUNK_SIZE and current:
            chunks.append('\n\n'.join(current))
            # Carry trailing paragraphs into next chunk up to overlap budget
            overlap: list[str] = []
            overlap_tokens = 0
            for p in reversed(current):
                t = len(enc.encode(p))
                if overlap_tokens + t > overlap_budget:
                    break
                overlap.insert(0, p)
                overlap_tokens += t
            current = overlap + [para]
            current_tokens = overlap_tokens + para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current_tokens >= MIN_CHUNK_TOKENS:
        chunks.append('\n\n'.join(current))
    return chunks


def ensure_index(pc: Pinecone, index_name: str):
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"Creating Pinecone index '{index_name}' (dim={DIMENSIONS}, cosine)...")
        pc.create_index(
            name=index_name,
            dimension=DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait until ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(2)
        print("Index ready.")
    else:
        print(f"Index '{index_name}' already exists.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to medium-english-50mb.csv")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N articles (for testing)")
    args = parser.parse_args()

    openai_client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX_NAME"]
    ensure_index(pc, index_name)
    index = pc.Index(index_name)

    enc = tiktoken.get_encoding("cl100k_base")

    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)
    print(f"Articles to process: {len(df)}")

    # Build all (vector_id, chunk_text, metadata) records
    records = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        text = str(row.get("text", "") or "")
        title = str(row.get("title", "") or "")
        authors = str(row.get("authors", "") or "")
        url = str(row.get("url", "") or "")
        tags = str(row.get("tags", "") or "")
        chunks = chunk_text(text, enc)
        for chunk_idx, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{idx}_{chunk_idx}",
                    "chunk": chunk,
                    "metadata": {
                        "article_id": str(idx),
                        "title": title,
                        "authors": authors,
                        "url": url,
                        "tags": tags,
                        "chunk": chunk,
                    },
                }
            )

    print(f"Total chunks: {len(records)}")

    # Embed in batches and upsert
    for batch_start in tqdm(range(0, len(records), EMBED_BATCH), desc="Embedding+Upserting"):
        batch = records[batch_start : batch_start + EMBED_BATCH]
        texts = [r["chunk"] for r in batch]

        emb_response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        embeddings = [e.embedding for e in emb_response.data]

        pinecone_vectors = [
            {
                "id": batch[i]["id"],
                "values": embeddings[i],
                "metadata": batch[i]["metadata"],
            }
            for i in range(len(batch))
        ]

        # Upsert in sub-batches to stay within Pinecone request size limits
        for sub_start in range(0, len(pinecone_vectors), UPSERT_BATCH):
            index.upsert(vectors=pinecone_vectors[sub_start : sub_start + UPSERT_BATCH])

    stats = index.describe_index_stats()
    print(f"Done. Pinecone vector count: {stats.total_vector_count}")


if __name__ == "__main__":
    main()
