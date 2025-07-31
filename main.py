import os
import weaviate
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import get_embedding, cosine_similarity

app = FastAPI(title="Gemini + Weaviate Best Similarity")

# Configura el cliente de Weaviate
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_KEY = os.getenv("WEAVIATE_API_KEY") or None

client = (
    weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_KEY)
    ) if WEAVIATE_KEY
    else weaviate.Client(url=WEAVIATE_URL)
)

# Modelos Pydantic
class QueryRequest(BaseModel):
    text: str
    top_k: int = 5

class BestCandidate(BaseModel):
    id: str
    text: str
    weaviate_distance: float
    gemini_similarity: float

@app.post("/best_similarity", response_model=BestCandidate)
def best_similarity(req: QueryRequest):
    try:
        # 1) Embedding de la consulta
        q_emb = get_embedding(req.text)

        # 2) Recupera top-k de Weaviate (incluye distancia interna)
        near = (
            client.query
                  .get("Document", ["text"])             # Ajusta "Document" a tu clase si es otro nombre
                  .with_near_vector({"vector": q_emb})
                  .with_limit(req.top_k)
                  .with_additional(["id", "distance"])
                  .do()
        )
        hits = near["data"]["Get"]["Document"]

        # 3) Para cada hit, calcula similitud con Gemini
        results = []
        for hit in hits:
            doc_id   = hit["_additional"]["id"]
            doc_text = hit["text"]
            w_dist   = hit["_additional"]["distance"]
            emb_doc  = get_embedding(doc_text)
            g_sim    = cosine_similarity(q_emb, emb_doc)
            results.append({
                "id": doc_id,
                "text": doc_text,
                "weaviate_distance": w_dist,
                "gemini_similarity": g_sim
            })

        # 4) Ordena por gemini_similarity y devuelve el mejor
        best = max(results, key=lambda x: x["gemini_similarity"])
        return BestCandidate(
            id=best["id"],
            text=best["text"],
            weaviate_distance=round(best["weaviate_distance"], 4),
            gemini_similarity=round(best["gemini_similarity"], 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
