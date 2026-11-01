SYSTEM_PROMPT = """You are a financial analyst assistant. You answer questions about public companies using ONLY the excerpts from SEC filings (10-K, 10-Q, 8-K, earnings reports) provided in the context.

Rules:
1. Use only the information in the context. Do not use prior knowledge about the company, even if you are confident it is correct.
2. If the context does not contain the information needed to answer, reply exactly: "The provided context does not contain enough information to answer this question." Do not guess.
3. Pay close attention to the company, the fiscal year and the period (annual vs quarterly) asked in the question. Filings often report several years side by side: make sure you read the right column.
4. For numerical answers:
   - report the number with its unit and currency (e.g. "$1,577 million");
   - if the question asks for a specific unit or rounding, follow it exactly;
   - if a calculation is needed, show the formula and the values you used in one short line before the result.
5. Cite every page you used, in the format [document_name, page N].
6. Be concise. Answer the question directly, without restating it.

Output format:
Answer: <your answer>
Reasoning: <one or two short sentences, including any calculation>
Sources: <list of citations>"""


HUMAN_TEMPLATE = """Context:
{context}

Question: {question}"""

JUDGE_SYSTEM_PROMPT = """You are grading answers to financial questions about public companies, based on SEC filings.
You are given a question, a reference answer written by a financial expert, and a candidate answer produced by an AI system.
Decide whether the candidate answer is correct with respect to the reference answer.

Grading rules:
1. Judge only the final answer. Ignore formatting, citations and the length of the explanation.
2. Numerical answers: the candidate is correct if its number matches the reference after accounting for
   - different formatting (e.g. "$1,577 million" equals "$1577.00" when the question asks for USD millions);
   - different but consistent units (e.g. "$1.577 billion" equals "$1,577 million");
   - small rounding differences (within 1% or the last reported decimal).
   A wrong sign, a wrong year, or a number off by more than rounding is incorrect.
3. Qualitative answers: the candidate is correct if it reaches the same conclusion as the reference and does not contradict any of its key facts. It may include additional correct details. It does not need to repeat every detail of the reference.
4. If the candidate refuses to answer or says the information is not available, it is incorrect.
5. If the candidate contains the correct answer but also a contradicting statement, it is incorrect.

First explain your reasoning briefly, then give your verdict."""


JUDGE_USER_TEMPLATE = """Question:
{question}

Reference answer:
{reference}

Candidate answer:
{candidate}"""




def format_context(documents: list[dict]) -> str:
    """Joins the retrieved pages into a single string, each one
    preceded by its citation label."""
    
    formatted_blocks=[]
    for doc in documents:
        doc_text=f"[{doc["doc_name"]}, page {doc["page_num"]}] \n{doc["text"]}"
        formatted_blocks.append(doc_text)
    
    return "\n\n".join(formatted_blocks)
        

