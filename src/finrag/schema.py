from typing import Literal
from pydantic import BaseModel, Field
from finrag.config import get_companies

COMPANIES = tuple(get_companies())
DOC_TYPES = ("10K", "10Q", "8K", "EARNINGS", "ANNUALREPORT")


class QueryFilters(BaseModel):
    """Filters extracted from a user question, used to restrict the document search."""

    company: Literal[COMPANIES] | None = Field(
        default=None,
        description=(
            "The company the question is about, as one of the allowed codes. "
            "Map the company name in the question to its code (e.g. 'Coca-Cola' -> 'COCACOLA', "
            "'Johnson & Johnson' -> 'JOHNSON_JOHNSON'). "
            "Leave empty if the question does not clearly name a single company."
        ),
    )
    fiscal_year: int | None = Field(
        default=None,
        description=(
            "The fiscal year of the document to search, e.g. 2018 for 'FY2018'. "
            "Years can also appear inside dates (e.g. 'the 8-K dated July 1, 2022' -> 2022). "
            "If the question covers several years (e.g. 'FY2017-FY2019'), use the most recent one, "
            "since the latest filing usually reports the previous years too. "
            "If the question asks about forecasts, expectations, targets or guidance for a year "
            "(e.g. 'What revenue growth does the company expect for FY2023?'), use the previous year (2022), "
            "since forecasts for a year are published in the filings of the year before. "
            "Leave empty if no year is mentioned."
        ),
    )
    doc_type: Literal[DOC_TYPES] | None = Field(
        default=None,
        description=(
            "The type of filing, only if the question explicitly refers to it: "
            "'10K' for annual reports on Form 10-K, '10Q' for quarterly reports, "
            "'8K' for current reports, 'EARNINGS' for earnings releases or calls, "
            "'ANNUALREPORT' for the annual report to shareholders. "
            "Leave empty if the question does not specify the type of document."
        ),
    )
    
class Route(BaseModel):
    reasoning: str = Field(description="One short sentence explaining the choice.")
    route: Literal["numeric", "text"] = Field(
        description="'numeric' if the answer can be computed from annual financial statement line items, "
                    "'text' otherwise."
    )   