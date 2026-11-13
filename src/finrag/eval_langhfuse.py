import re

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import Evaluation
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

from finrag.agents import build_graph
from finrag.config import get_llm, JUDGE_MODEL
from finrag.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE

app = build_graph()


class Verdict(BaseModel):
    reasoning: str = Field(description="Brief explanation of the grading")
    is_correct: bool = Field(description="True if the candidate answer is correct")


_judge_llm = get_llm(JUDGE_MODEL)
# Il giudice a volte non chiude il JSON e continua a generare spazi/newline
# all'infinito: un tetto sui token evita risposte da 35k righe.
_judge_llm.max_tokens = 2000
judge = _judge_llm.with_structured_output(Verdict, include_raw=True)


def _parse_verdict(out: dict) -> Verdict:
    """Usa l'output strutturato; se il JSON e' troncato, recupera i campi a mano."""
    if out.get("parsed") is not None:
        return out["parsed"]
    text = out["raw"].content if out.get("raw") is not None else ""
    m = re.search(r'"is_correct"\s*:\s*(true|false)', text)
    if not m:
        raise ValueError(f"Judge output non interpretabile: {text[:200]!r}")
    r = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)', text)
    return Verdict(
        is_correct=m.group(1) == "true",
        reasoning=(r.group(1) if r else "")[:500] or "(JSON troncato)",
    )


# --- Task (era "target") ---------------------------------------------------
def task(*, item, **kwargs) -> dict:
    """Esegue il grafo su un elemento del dataset."""
    # Con i dataset di Langfuse "item" è un oggetto (item.input);
    # con dati locali è un dizionario (item["input"]).
    item_input = item["input"] if isinstance(item, dict) else item.input

    handler = CallbackHandler() 
    result = app.invoke(
        {"messages": [HumanMessage(content=item_input["question"])]},
        config={"callbacks": [handler]},
    )
    return {
        "answer": result["messages"][-1].content,
        "retrieved_pages": [[d["doc_name"], d["page_num"]] for d in result["documents"]],
    }


# --- Evaluator --------------------------------------------------------------

def hit_rate(*, output, expected_output, **kwargs) -> Evaluation:
    retrieved = output["retrieved_pages"]
    evidence = expected_output["evidence_pages"]
    hit = any(page in retrieved for page in evidence)
    return Evaluation(name="hit_rate", value=float(hit))


def evidence_recall(*, output, expected_output, **kwargs) -> Evaluation:
    retrieved = output["retrieved_pages"]
    evidence = expected_output["evidence_pages"]
    if not evidence:
        return Evaluation(name="evidence_recall", value=0.0)
    found = sum(1 for page in evidence if page in retrieved)
    return Evaluation(name="evidence_recall", value=found / len(evidence))


def correctness(*, input, output, expected_output, **kwargs) -> Evaluation:
    user_prompt = JUDGE_USER_TEMPLATE.format(
        question=input["question"],
        reference=expected_output["answer"],
        candidate=output["answer"],
    )
    verdict = _parse_verdict(judge.invoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]))
    return Evaluation(name="correctness", value=float(verdict.is_correct), comment=verdict.reasoning)


def average_scores(*, item_results, **kwargs) -> list[Evaluation]:
    """Media di ogni metrica sull'intero esperimento."""
    results = []
    #per ogni domanda, e per goni voto di quella domanda se corrisponde alla metrica aggiungilo alla lista 
    for name in ["hit_rate", "evidence_recall", "correctness"]:
        values = [e.value for r in item_results for e in r.evaluations if e.name == name]
        if values:
            results.append(Evaluation(name=f"avg_{name}", value=sum(values) / len(values)))
    return results