from finrag.config import get_encoder, get_qdrant, get_llm,TOP_K,COLLECTION
from finrag.schema import QueryFilters
from langchain_core.messages import SystemMessage, HumanMessage
from finrag.prompts import FILTER_SYSTEM_PROMPT
from qdrant_client import models
from functools import lru_cache

@lru_cache
def get_filter_llm():
    """LLM with structured output used to extract search filters from a question."""
    return get_llm().with_structured_output(QueryFilters)

def search_pages(query: str,company: str | None = None, fiscal_year: int | None= None ,doc_type: str | None = None, k: int= TOP_K)->list[dict]:
    """Search SEC filings for the pages most relevant to a question.

    Performs a semantic search over the pages of 10-K, 10-Q, 8-K, earnings
    releases and annual reports. Optional filters restrict the search to the
    pages of a specific company, fiscal year or document type. Use the filters
    only when the question clearly specifies them: a wrong filter excludes the
    document that contains the answer.

    Args:
        query: The question in natural language.
        company: Company code as stored in the index (e.g. "3M", "COCACOLA",
            "JOHNSON_JOHNSON"). If None, all companies are searched.
        fiscal_year: Fiscal year of the filing (e.g. 2018). If None, all years
            are searched.
        doc_type: Type of filing: "10K", "10Q", "8K", "EARNINGS" or
            "ANNUALREPORT". If None, all document types are searched.
        k: Number of pages to return.

    Returns:
        A list of up to k pages, ordered by relevance. Each page is a dict with
        "doc_name", "page_num", "text", "n_chars", "company", "fiscal_year",
        "period", "doc_type" and "score" (cosine similarity to the question).
    """
    encoder=get_encoder()
    client=get_qdrant()
    query_vec=encoder.encode(query,prompt_name="QuestionAnswering",normalize_embeddings=True)
    filter_list=[]
    if company is not None:
        filter_list.append(models.FieldCondition(key="company",match=models.MatchValue(value=company)))
    
    if fiscal_year is not None:
        filter_list.append(models.FieldCondition(key="fiscal_year",match=models.MatchValue(value=fiscal_year)))
    
    if doc_type is not None:
        filter_list.append(models.FieldCondition(key="doc_type",match=models.MatchValue(value=doc_type)))
        
    
    result=client.query_points(collection_name=COLLECTION,
                               query=query_vec.tolist(),
                               limit=k,
                               with_payload=True,
                               query_filter=models.Filter(must=filter_list) if len(filter_list)>0 else None
                               ).points
    
    pages=[{**p.payload,"score": p.score} for p in result]
    return pages
    
def  extract_filter_from_query(query: str)->QueryFilters:
    """Extract search filters (company, fiscal year, document type) from a question.

    Uses an LLM with structured output. A field is filled only when the
    question states it clearly; otherwise it is left as None, so that the
    search is not restricted on that field.

    Args:
        query: The question in natural language.

    Returns:
        A QueryFilters object with the fields "company", "fiscal_year" and
        "doc_type", each possibly None.
    """
    return get_filter_llm().invoke([
        SystemMessage(content=FILTER_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ])
    
    
    
    