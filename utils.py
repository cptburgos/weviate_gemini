import os
import requests
from dotenv import load_dotenv
from scipy.spatial.distance import cosine

load_dotenv()

#Gemini Embeddings
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"

def get_embedding(text: str):
    payload = {
        "model": "models/embedding-001",
        "content": { "parts": [{ "text": text }] }
    }
    resp = requests.post(
        f"{GEMINI_ENDPOINT}?key={GEMINI_KEY}",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]

def cosine_similarity(v1, v2) -> float:
    return 1 - cosine(v1, v2)
