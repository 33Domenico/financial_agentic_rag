import os
from functools import lru_cache
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch
from qdrant_client import models, QdrantClient
from langchain_openai import ChatOpenAI




EMBED_MODEL = "google/embeddinggemma-2"
LLM_MODEL = "nvidia/nemotron-3.5-super-vl:free"            # Nemotron su OpenRouter
JUDGE_MODEL = "thinkingmachines/inkling:free"          # Inkling su OpenRouter
COLLECTION = "financebench_pages_embgemma2_title"
QDRANT_URL = "http://localhost:6333"
TOP_K = 5

load_dotenv()


@lru_cache
def get_encoder():
    """Load embeddinggemma2 on the GPU."""
    device= "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embeddinggemma2 on {device} ")
    encoder=SentenceTransformer(EMBED_MODEL,model_kwargs={"torch_dtype": torch.bfloat16},device=device)
    return encoder

@lru_cache
def get_qdrant():
    """Client connected to Qdrant in Docker."""
    client = QdrantClient(url=QDRANT_URL)
    return client

@lru_cache
def get_llm(model_name: str = LLM_MODEL ):
    """ChatOpenAI puntato a openrouter, temperature=0.0 """
    api_key= os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY or OPENAI_API_KEY in your .env file")
    
    llm=ChatOpenAI(model=model_name,base_url="https://openrouter.ai/api/v1", api_key=api_key ,temperature=0.0)
    return llm

if __name__ == "__main__":
    encoder = get_encoder()
    qdrant = get_qdrant()
    llm = get_llm()
    print("All components loaded successfully.")