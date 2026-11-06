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


judge = get_llm(JUDGE_MODEL).with_structured_output(Verdict)


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
    verdict = judge.invoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
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