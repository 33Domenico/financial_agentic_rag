from fastmcp import FastMCP
from finrag.agents import build_graph
from typing import Annotated,Literal
from pydantic import Field,ValidationError
from langchain_core.messages import HumanMessage
import logging
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
import openai
from fastmcp.exceptions import ToolError
from finrag.tools import calculate,COMPANIES,METRICS,search_pages,rerank_pages
from finrag.xbrl import get_metric
from finrag.schema import DOC_TYPES
import torch
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


mcp = FastMCP("finrag",mask_error_details=True)
app=build_graph()



@mcp.tool()
def ask_finrag(
    question: Annotated[str, Field(min_length=10,description=(
        "A self-contained question about a single company's SEC filings, "
        "including the company name and fiscal year, e.g. "
        "'What was 3M's capital expenditure in FY2018?'. "
        "Rewrite follow-up questions so they do not depend on previous turns."
    ))]
) -> dict:
    """Answer a financial question about a public company using its SEC filings
    (10-K, 10-Q, 8-K, earnings releases) from the FinanceBench corpus.

    Numeric questions on annual financial statement items (revenue, net income,
    capex, cash flow, balance sheet items, and ratios built from them) are computed
    from SEC XBRL data; all other questions are answered from the filing pages.
    Ask one company per call: for comparisons, call the tool once per company.

    Returns a dict with:
    - answer: the final answer, with reasoning and sources
    - route: "numeric" (XBRL agent) or "text" (retrieval over filings)
    - used_rag: whether filing pages were retrieved (also true when a numeric
      question fell back to retrieval)
    - sources: list of {doc, page} for retrieved pages; empty for XBRL answers,
      whose sources are listed inside the answer
    """
    try:
        result=app.invoke({"messages":[HumanMessage(content=question)]})
    except (ResponseHandlingException, UnexpectedResponse):
        logger.exception("Qdrant error")
        raise ToolError("The filings index is currently unavailable. Try again later.")
    except openai.RateLimitError:
        logger.exception("OpenRouter rate limit")
        raise ToolError("The language model is rate-limited. Wait a minute and retry.")
    except (openai.APITimeoutError, openai.APIConnectionError):
        logger.exception("OpenRouter connection error")
        raise ToolError("The language model did not respond. Retry the request.")
    except (openai.LengthFinishReasonError, ValidationError):
        logger.exception("Invalid structured output")
        raise ToolError("The question could not be processed. Retry or rephrase it more explicitly.")
    except openai.APIStatusError as e:
        logger.exception("OpenRouter error")
        raise ToolError(f"The language model returned an error (HTTP {e.status_code}). Retry later.")
    documents = result.get("documents") or []
    return {
        "answer": result["messages"][-1].content,
        "route": result["route"],
        "used_rag": bool(documents),
        "sources": [{"doc": d["doc_name"], "page": d["page_num"]} for d in documents]
    }


@mcp.tool
def calculator(
    expression: Annotated[str, Field(
        min_length=1,
        description=(
            "An arithmetic expression using only numbers, + - * / ** and parentheses, "
            "e.g. '(1469502000 / 2213556000)' or '(4258 + 56 + 4947) / 10936'. "
            "No variables, functions or units: write raw numbers."
        ),
    )]
) -> float:
    """Evaluate an arithmetic expression exactly and return the result.

    Use it for any calculation on financial figures (ratios, growth rates, averages,
    margins) instead of computing mentally. When the inputs come from
    get_financial_metric, copy the values exactly as returned (raw USD, not millions).
    """
    try:
        return calculate(expression)
    except ZeroDivisionError:
        raise ToolError("Division by zero: check that the denominator value is not zero.")
    except (ValueError, SyntaxError, TypeError):
        raise ToolError(
            "Invalid expression: use only numbers, + - * / ** and parentheses, "
            "e.g. '(1469502000 / 2213556000)'."
        )

@mcp.tool
def get_financial_metric(
    company: Annotated[Literal[COMPANIES], Field(
        description="Company code as used in the corpus, e.g. '3M', 'MICROSOFT', 'JOHNSON_JOHNSON'.",
    )],
    metric: Annotated[Literal[METRICS], Field(
        description="Standardized annual line item from the income statement, balance sheet or cash flow statement.",
    )],
    fiscal_year: Annotated[int, Field(
        ge=2009,
        description="Fiscal year of the figure (not the filing year), e.g. 2018 for the year ended in 2018.",
    )],
) -> dict:
    """Read one reported annual value of a company from its 10-K filings (SEC XBRL data).

    Covers annual GAAP line items only: no quarterly figures, segments, non-GAAP or
    adjusted metrics. Call it once per metric and per fiscal year (e.g. an average of
    total assets over FY2019-FY2020 needs two calls), then combine the values with
    the calculator tool.
    Values are in raw USD (not millions). Balance sheet items are measured at the end
    of the fiscal year. The result includes the source: XBRL concept, period end and
    filing date.

    If the value is not available in XBRL, the result has "value": null and the figure
    may still be found in the filings: use search_filings with a question about it.
    """
    try:
        result = get_metric(company, metric, fiscal_year)
    except Exception:
        logger.exception("XBRL lookup failed")
        raise ToolError("Financial data is currently unavailable. Try search_filings instead.")
    
    if result is None:
        return {
            "company": company,
            "metric": metric,
            "year": fiscal_year,
            "value": None,
            "note": (
                f"{metric} is not available in XBRL for {company} FY{fiscal_year}. "
                "It may be reported under a different line item in the filings: "
                "use search_filings to look it up."
            ),
        }
    return result

@mcp.tool
def search_filings(
    query: Annotated[str, Field(
        min_length=10,
        description=(
            "What to look for in the filings, phrased as a question or a description of "
            "the content, e.g. 'depreciation and amortization in the cash flow statement'."
        ),
    )],
    company: Annotated[Literal[COMPANIES] | None, Field(
        description="Restrict the search to one company, e.g. 'MICROSOFT'. Strongly recommended.",
    )] = None,
    fiscal_year: Annotated[int | None, Field(
        ge=2009,
        description="Restrict the search to filings of this fiscal year, e.g. 2023.",
    )] = None,
    doc_type: Annotated[Literal[DOC_TYPES] | None, Field(
        description="Restrict the search to one document type, e.g. '10K' for annual reports.",
    )] = None,
    k: Annotated[int, Field(
        ge=1, le=8,
        description="Number of pages to return. Start with 3 and ask for more only if needed.",
    )] = 3,
) -> list[dict]:
    """Search the SEC filings of the FinanceBench corpus (10-K, 10-Q, 8-K, earnings
    releases) and return the most relevant pages, ranked by a cross-encoder reranker.

    Use it to read figures or text that get_financial_metric does not cover: quarterly
    data, segments, non-GAAP metrics, footnotes, management commentary, or values
    missing from XBRL. Set company and fiscal_year whenever they are known: without
    filters the search runs over all 53,000 pages and results are much less precise.

    Returns a list of pages, best first, each with the document name, page number
    (zero-indexed), relevance score and full page text. Cite pages as [document, page].
    """
    
    try:
        docs = search_pages(query=query, company=company, fiscal_year=fiscal_year, doc_type=doc_type)
        ranked_docs = rerank_pages(query=query, doc_list=docs, k=k)
    except (ResponseHandlingException, UnexpectedResponse):
        logger.exception("Qdrant error")
        raise ToolError("The filings index is currently unavailable. Try again later.")
    except torch.cuda.OutOfMemoryError:
        logger.exception("Reranker out of memory")
        raise ToolError("Not enough resources to process the request. Retry with a smaller k.")
    
    return [{"doc": r["doc_name"], "page": r["page_num"] , "score": float(r["rerank_score"]), "text": r["text"]} for r in ranked_docs]
    

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)