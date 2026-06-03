from http.server import BaseHTTPRequestHandler
import json
import os

from openai import OpenAI
from pinecone import Pinecone

CHUNK_SIZE = 512
OVERLAP_RATIO = 0.2
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a Medium-article assistant that answers questions strictly and only "
    "based on the Medium articles dataset context provided to you (metadata and "
    "article passages). You must not use any external knowledge, the open internet, "
    "or information that is not explicitly contained in the retrieved context. "
    "If the answer cannot be determined from the provided context, respond: "
    '"I don\'t know based on the provided Medium articles data." '
    "Always explain your answer using the given context, quoting or paraphrasing "
    "the relevant article passage or metadata when helpful."
)

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL = "4UHRUIN-gpt-5-mini"


def _build_openai_client():
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


def _build_pinecone_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(os.environ["PINECONE_INDEX_NAME"])


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            question = body.get("question", "").strip()
            if not question:
                self._respond(400, {"error": "question field is required"})
                return

            client = _build_openai_client()

            emb = client.embeddings.create(model=EMBEDDING_MODEL, input=question)
            query_vector = emb.data[0].embedding

            index = _build_pinecone_index()
            results = index.query(vector=query_vector, top_k=TOP_K, include_metadata=True)

            context = []
            for match in results.matches:
                meta = match.metadata or {}
                context.append(
                    {
                        "article_id": str(meta.get("article_id", "")),
                        "title": str(meta.get("title", "")),
                        "chunk": str(meta.get("chunk", "")),
                        "score": float(match.score),
                    }
                )

            context_text = "\n\n".join(
                f'[{i + 1}] Article: "{c["title"]}"\n{c["chunk"]}'
                for i, c in enumerate(context)
            )
            user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"

            completion = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            self._respond(
                200,
                {
                    "response": completion.choices[0].message.content,
                    "context": context,
                    "Augmented_prompt": {
                        "System": SYSTEM_PROMPT,
                        "User": user_prompt,
                    },
                },
            )

        except Exception as exc:
            self._respond(500, {"error": str(exc)})

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _respond(self, status: int, data: dict):
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass
