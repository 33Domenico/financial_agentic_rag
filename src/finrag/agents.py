from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from finrag.tools import search_pages,extract_filter_from_query,rerank_pages, FINANCIAL_TOOLS
from finrag.prompts import format_context,SYSTEM_PROMPT, HUMAN_TEMPLATE,ROUTER_SYSTEM_PROMPT,FINANCIAL_AGENT_PROMPT
from finrag.config import get_llm
from finrag.schema import Route
from langgraph.prebuilt import ToolNode


re_act=get_llm().bind_tools(FINANCIAL_TOOLS)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]
    documents: list[dict]
    filters: dict
    route: str
    question: str
    
def extract_filters(state: AgentState)->dict:
    """Create the rigth filter based on the question so thath the extraction is easier"""
    filters= extract_filter_from_query(state["question"]).model_dump()
    return{"filters": filters}


def router(state: AgentState)->dict:
    """Route the question to the numeric agent with xbrl data or text agent with rag pdf data"""
    query=state["messages"][-1].content
    response=get_llm(reasoning_effort="none").with_structured_output(Route).invoke([SystemMessage(content=ROUTER_SYSTEM_PROMPT),HumanMessage(content=query)])
    return {"route":response.route, "question":query}

def route_question(state: AgentState) -> str:
    return  state["route"]    
    
    
def financial_agent(state:AgentState)->dict:
    """React tool agent that use xbrl data to answer the question"""
    response=re_act.invoke([SystemMessage(content=FINANCIAL_AGENT_PROMPT)]+list(state["messages"]))
    return {"messages": [response]}
    
def after_agent(state:AgentState)->str:
    last=state["messages"][-1]
    if last.tool_calls:
        return "continue"
    elif "MISSING_DATA" in last.content:
        return "rag"
    else:
        return "end"

def retrieve(state: AgentState)->dict:
    """Search the documents for pages relevant to the question and return the list of retrieved pages."""
    docs=search_pages(state["question"],**state["filters"])
    ranked_docs=rerank_pages(state["question"],docs)
    return {"documents": ranked_docs}

def generate(state:AgentState)->dict:
    """Builds the prompt, calls the LLM, and updates the message by appending the response."""
    question=state["question"]
    context=format_context(state["documents"])
    user_text =HUMAN_TEMPLATE.format(context=context,question=question)
    
    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT)]+
        [HumanMessage(content=user_text),]
        )
    return {"messages": [response]}
    

    
def build_graph():
    graph=StateGraph(AgentState)
    graph.add_node("router",router)
    graph.add_node("extract_filters",extract_filters)
    graph.add_node("generate",generate)
    graph.add_node("retriver",retrieve)
    graph.add_node("financial_agent",financial_agent)
    tools_node=ToolNode(tools=FINANCIAL_TOOLS)
    graph.add_node("tools",tools_node)
    
    
    graph.set_entry_point("router")
    graph.add_conditional_edges("router",route_question,{"numeric": "financial_agent","text":"extract_filters" })
    
    #Route 1
    graph.add_edge("extract_filters","retriver")
    graph.add_edge("retriver","generate")
    graph.add_edge("generate",END)
    #Route 2
    graph.add_conditional_edges("financial_agent",after_agent,{"continue": "tools","rag": "extract_filters", "end": END })
    graph.add_edge("tools","financial_agent")
    
    
    
    
    return graph.compile()