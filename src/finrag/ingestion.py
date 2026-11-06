import re

DOC_NAME_PATTERN = re.compile(
    r"^(?P<company>.+?)"
    r"(?:_(?P<year>\d{4})(?P<quarter>Q[1-4])?)?"          # anno ORA opzionale
    r"_(?P<doc_type>10K|10Q|8K|EARNINGS|ANNUALREPORT)"     # + ANNUALREPORT
    r"(?:[_-]dated[_-](?P<date_year>\d{4})-\d{2}-\d{2})?"  # data: separatore _ o -
    r"(?:[_-].*)?$",                                       # eventuale coda residua
    re.IGNORECASE,                                         # maiuscole/minuscole indifferenti
)


def parse_doc_name(doc_name: str) -> dict | None:
    """Extracts company, fiscal_year, period and doc_type from a document name."""
    m = DOC_NAME_PATTERN.match(doc_name)
    if m is None:
        return None
    year = m.group("year") or m.group("date_year")   # anno dal nome o, in mancanza, dalla data
    if year is None:
        return None
    return {
        "company": m.group("company").upper(),
        "fiscal_year": int(year),
        "period": (m.group("quarter") or "FY").upper(),
        "doc_type": m.group("doc_type").upper(),
    }
