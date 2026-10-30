from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from finrag.tools import search_pages
from finrag.prompts import format_context,SYSTEM_PROMPT, HUMAN_TEMPLATE
from finrag.config import get_llm



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]
    documents: list[dict]
    
    
def retrieve(state: AgentState)->AgentState:
    """Search the documents for pages relevant to the question and return the list of retrieved pages."""
    docs=search_pages(state["messages"][-1].content)
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
    graph.add_node("generate",generate)
    graph.add_node("retriver",retrieve)
    
    graph.set_entry_point("retriver")
    graph.add_edge("retriver","generate")
    graph.add_edge("generate",END)
    return graph.compile()