from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from finrag.tools import search_pages,extract_filter_from_query
from finrag.prompts import format_context,SYSTEM_PROMPT, HUMAN_TEMPLATE
from finrag.config import get_llm



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]
    documents: list[dict]
    filters: dict
    
def extract_filters(state: AgentState)->dict:
    """Create the rigth filter based on the question so thath the extraction is easier"""
    filters= extract_filter_from_query(state["messages"][-1].content).model_dump()
    return{"filters": filters}
  
def retrieve(state: AgentState)->dict:
    """Search the documents for pages relevant to the question and return the list of retrieved pages."""
    docs=search_pages(state["messages"][-1].content,**state["filters"])
    return {"documents": docs}

def generate(state:AgentState)->AgentState:
    """Builds the prompt, calls the LLM, and updates the message by appending the response."""
    question=state["messages"][-1].content
    context=format_context(state["documents"])
    user_text =HUMAN_TEMPLATE.format(context=context,question=question)
    
    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT)]+
        list(state["messages"][:-1])+
        [HumanMessage(content=user_text),]
        )
    return {"messages": [response]}
    
    
    
def build_graph():
    graph=StateGraph(AgentState)
    graph.add_node("extract_filters",extract_filters)
    graph.add_node("generate",generate)
    graph.add_node("retriver",retrieve)
    
    graph.set_entry_point("extract_filters")
    graph.add_edge("extract_filters","retriver")
    graph.add_edge("retriver","generate")
    graph.add_edge("generate",END)
    return graph.compile()