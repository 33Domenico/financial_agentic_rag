from finrag.config import get_encoder, get_qdrant, get_llm,TOP_K,COLLECTION

def search_pages(query: str, k: int= TOP_K)->list[dict]:
    """Returns the k pages most similar to the question asked."""
    encoder=get_encoder()
    clinet=get_qdrant()
    query_vec=encoder.encode(query,prompt_name="QuestionAnswering",normalize_embeddings=True)
    
    result=clinet.query_points(collection_name=COLLECTION,
                               query=query_vec.tolist(),
                               limit=k,
                               with_payload=True).points
    
    pages=[{**p.payload,"score": p.score} for p in result]
    return pages
    
    
    
    
    