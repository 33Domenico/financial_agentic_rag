import json
import os
import time
from pathlib import Path
from functools import lru_cache

import requests
from dotenv import load_dotenv

from finrag.config import get_companies
import pandas as pd

load_dotenv()
HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT")}
ROOT = Path(__file__).resolve().parents[2]  # financial_agentic_rag/
RAW_DIR = ROOT / "data" / "raw" / "xbrl"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_company_facts():
    index_companies = set(get_companies())
    mapping = json.loads(
        (PROCESSED_DIR / "company_cik.json").read_text(encoding="utf-8")
    )

    for code, info in mapping.items():
        if code not in index_companies:
            continue
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{info['cik']:010d}.json"
        response = requests.get(url, headers=HEADERS, timeout=30)

        if response.status_code != 200:
            print(f"{code:22s}  ERRORE {response.status_code}")
        else:
            data = response.json()
            (RAW_DIR / f"{code}.json").write_text(response.text, encoding="utf-8")
            print(f"{code:22s}  {data['entityName']}")

        time.sleep(0.2)

def flatten_comapnyfacts(data: dict, company: str)-> pd.DataFrame:
    """Trasforma il JSON companyfacts di un'azienda in una tabella: una riga per numero depositato."""
    rows = []
    for concept, info in data["facts"]["us-gaap"].items():
        for unit, observations in info["units"].items():
            for obs in observations:
                rows.append({"company": company, "concept": concept, "unit": unit, **obs})
    return pd.DataFrame(rows)

def build_facts_parquet() -> Path:
    """Legge tutti i JSON in RAW_DIR e salva un unico parquet in data/processed."""
    frames = []
    for path in sorted(RAW_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        frames.append(flatten_comapnyfacts(data, company=path.stem))
    df = pd.concat(frames, ignore_index=True)
    out = PROCESSED_DIR / "xbrl_facts.parquet"
    df.to_parquet(out, index=False)
    return out


def clean_annual_facts(df: pd.DataFrame) -> pd.DataFrame:
    """Riduce i fatti XBRL grezzi a un solo valore annuale per azienda, concetto, unità e anno."""
    df = df.copy()

    # a. Solo dati annuali depositati nei 10-K
    df = df[(df["fp"] == "FY") & (df["form"] == "10-K")]

    # b. Solo periodi di circa un anno; i concetti istantanei (senza start) si tengono tutti
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    duration = (df["end"] - df["start"]).dt.days
    is_instant = df["start"].isna()
    is_annual = duration.between(350, 380)
    df = df[is_instant | is_annual]

    # c. L'anno a cui si riferisce il dato, dalla data di fine (non da fy).
    # Gli esercizi a 52/53 settimane (es. J&J) possono chiudersi nei primi giorni
    # di gennaio: si sottraggono 15 giorni perché contino per l'anno appena finito.
    df["year"] = (df["end"] - pd.Timedelta(days=15)).dt.year

    # d. Un solo valore per azienda, concetto, unità e anno: il deposito originale (il più vecchio)
    df = (
        df.sort_values("filed")
        .drop_duplicates(subset=["company", "concept", "unit", "year"], keep="first")
        .sort_values(["company", "concept", "year"])
        .reset_index(drop=True)
    )

    return df

@lru_cache
def get_metric_vocabulary()->dict:
    """Return the list of metrics"""
    with open(PROCESSED_DIR/"metrics.json",encoding="utf-8") as f:
        metrics=json.load(f)
    return metrics


@lru_cache
def get_annual_facts()->pd.DataFrame:
    """Carica i metadati xrbl"""
    annual_facts= pd.read_parquet(PROCESSED_DIR/"xbrl_facts_annual.parquet")
    return annual_facts

def get_metric(company: str, metric: str, year: int) -> dict | None:
    """Return the annual value of a financial metric for a company, from SEC XBRL data.

    Tries the XBRL concepts listed for the metric in metrics.json, in order,
    and returns the first one reported by the company for that fiscal year.

    Args:
        company: company code as stored in the index (e.g. "3M", "ADOBE").
        metric: metric name from the vocabulary (e.g. "capex", "current_liabilities").
        year: fiscal year of the value (e.g. 2018).

    Returns:
        A dict with the value and its source (concept, period end, filing date, form),
        or None if no concept for that metric is available.
    """
    vocabulary = get_metric_vocabulary()
    if metric not in vocabulary:
        raise ValueError(f"Unknown metric '{metric}'. Allowed: {sorted(vocabulary)}")

    facts = get_annual_facts()
    company_year = facts[(facts["company"] == company) & (facts["year"] == year)]

    for concept in vocabulary[metric]:
        match = company_year[company_year["concept"] == concept]
        if not match.empty:
            row = match.iloc[0]
            return {
                "company": company,
                "metric": metric,
                "year": year,
                "value": float(row["val"]),
                "unit": row["unit"],
                "concept": concept,
                "period_end": row["end"].date().isoformat(),
                "filed": row["filed"],
                "form": row["form"],
            }

    return None
    
    

if __name__=="__main__":
    download_company_facts()
    

