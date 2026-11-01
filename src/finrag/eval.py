from langsmith import Client, wrappers
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from openai import OpenAI
from finrag.agents import build_graph
from langchain_core.messages import  HumanMessage,SystemMessage
from pydantic import BaseModel, Field
from finrag.config import get_llm, JUDGE_MODEL
from finrag.prompts import JUDGE_SYSTEM_PROMPT,JUDGE_USER_TEMPLATE



class Verdict(BaseModel):
    reasoning: str= Field(description="Brief explanation of the grading")
    is_correct: bool= Field(description="True if the candidate answer is correct")
 
#grafo costruito una volta sola quando importo il file
app=build_graph()    
judge = get_llm(JUDGE_MODEL).with_structured_output(Verdict)


def target(inputs: dict)->dict:
    """Function called to generate a response from the graph for each sample in the LangSmith dataset."""
    
    response=app.invoke({"messages": [HumanMessage(content=inputs["question"])]})
    return {"answer": response["messages"][-1].content, "retrieved_pages":[[d.get("doc_name"),d.get("page_num")]for d in response["documents"]] }


def hit_rate(outputs: dict, reference_outputs: dict)-> bool:
    """Checks whether at least one ground-truth page is among those retrieved."""
    # LangSmith receives exactly these inputs; do not change them.
    retrived_page=outputs["retrieved_pages"]
    gt_page=reference_outputs["evidence_pages"]
    check=False
    for page in gt_page:
        if page in retrived_page:
            check= True
            break
    return check
  
def evidence_recall(outputs: dict, reference_outputs: dict)-> float:
    """Checks that all ground-truth pages were retrieved as references."""
    retrived_page=outputs["retrieved_pages"]
    gt_page=reference_outputs["evidence_pages"]
    if len(gt_page)==0:
        return 0.0
    counter=0
    for page in gt_page:
         if page in retrived_page:
            counter+=1
    return counter/len(gt_page)
    
def correctness(inputs: dict ,outputs: dict, reference_outputs: dict)-> dict:
    user_prompt=JUDGE_USER_TEMPLATE.format(question=inputs["question"],reference=reference_outputs["answer"],candidate=outputs["answer"])
        
    verdict=judge.invoke([SystemMessage(content=JUDGE_SYSTEM_PROMPT),HumanMessage(content=user_prompt)])
    
    return {"key": "correctness","score": verdict.is_correct, "comment": verdict.reasoning }
    
    


if __name__=="__main__":
    #out = target({"question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement."})
    #print(out["answer"])
    #print(out["retrieved_pages"])
    outputs = {
    "answer": "...",
    "retrieved_pages": [["3M_2018_10K", 48], ["3M_2018_10K", 45], ["3M_2018_10K", 59]],
    }
    reference_outputs = {
        "answer": "$1577.00",
        "evidence_pages": [["3M_2018_10K", 59]],
    }

    # Caso 1: la pagina di evidenza è tra quelle recuperate
    print(hit_rate(outputs=outputs, reference_outputs=reference_outputs))   # atteso: True

    # Caso 2: tolgo la pagina 59 dalle recuperate
    outputs_miss = {
        "answer": "...",
        "retrieved_pages": [["3M_2018_10K", 48], ["3M_2018_10K", 45]],
    }
    print(hit_rate(outputs=outputs_miss, reference_outputs=reference_outputs)) 
    reference_two = {"answer": "...", "evidence_pages": [["ADOBE_2015_10K", 50], ["ADOBE_2015_10K", 48]]}
    outputs_one = {"answer": "...", "retrieved_pages": [["ADOBE_2015_10K", 50], ["ADOBE_2015_10K", 12]]}

    print(evidence_recall(outputs=outputs_one, reference_outputs=reference_two))   # atteso: 0.5