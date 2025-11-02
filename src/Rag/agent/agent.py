from typing import TypedDict, Annotated, List, Literal, Optional, cast
import traceback
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import (
    PromptTemplate,
)
from langchain_core.messages.base import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from src.logger.logg import logs
from .utils import *

logger = logs("main.log")

tools_reception = [database_retriever_tool]
tool_node = ToolNode(tools_reception)
clinical_node_tools = [web_search_tool, vector_retriever_tool]
clinical_tool_node = ToolNode(clinical_node_tools)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_inputs: Annotated[List[HumanMessage], add_messages]

def reception_node(state: AgentState):     #[Humanmessage - > AIMessage -> ToolMessage -> AIMessage -> HumanMessage -> next agent]
    """
    ---------------------------------------------      NODE 1      ---------------------------------------------
    Reception agent which receives input from user and is responsible to fetch discharge data using tool

    """
    try:
        query = next(
                (m for m in state["user_inputs"] if isinstance(m, HumanMessage)),
                None
        )

        if query is None:
            raise ValueError("No message in state")
        
        has_tool_message = any(isinstance(m, ToolMessage) and m.name=='database_retriever_tool' for m in state["messages"])
        if has_tool_message:
            print(f"data_retrieval_tool message was found {state['messages']}")
            data_tool_message = next(
                (m for m in state["messages"] if isinstance(m, ToolMessage) and m.name == 'database_retriever_tool'),
                None
            )
            if data_tool_message is None:
                logger.warning("The agent didnt call the data_retriever_tool for fetching reports!!")
            else:
                system_prompt = PromptTemplate(
                input_variables=["query", "discharge_report_content"],
                template=reception_prompt_template,
                )
                final_system_template = system_prompt.invoke(
                {"query": query.content, "discharge_report_content": data_tool_message.content}
                )
                
                question_patient = instance_decision_llm.invoke(final_system_template)
                            
                print(f"tool message llm questions : {question_patient}")
                print(f"full state messages till now : {state['messages']}")

                return {"messages" : [question_patient]}
            
        # extract query text
        query_text = query.content if isinstance(query.content, str) else str(query.content)
        system_prompt = PromptTemplate(
            input_variables=["query", "discharge_report_content"],
            template=reception_prompt_template,
        )
        final_system_template = system_prompt.invoke(
            {"query": query_text, "discharge_report_content": ""}
        )
        
        context_retriever_answer = instance_decision_llm.invoke(final_system_template)
        logger.info("decision maker llm was successfully invoked")
        
        if isinstance(context_retriever_answer, AIMessage):
            print("AIMessage content:", context_retriever_answer)
        else:
            print("Received non-AIMessage response;", repr(context_retriever_answer))

        print(f"final system template : {final_system_template}")
        print(f"test_new_template_reception : {state['messages']}")
        return {"messages" : [context_retriever_answer]}
    
    except Exception as e:
        logger.exception("decision maker node can't be executed: %s", e)
        print(traceback.format_exc())

def clinical_node(state: AgentState):
    """
    ---------------------------------------------      NODE 2      ---------------------------------------------
    Clinical agent which answers patient query related to rag data . 
    has 2 tools which can be used : web search and rag data tool which retrives relevant chunk of text from vectorstore
    """
    try:
        print(f"all User_inputs till clinical_node {state['user_inputs']}")
        
        query = next(
            (m for m in reversed(state["user_inputs"]) if isinstance(m, HumanMessage)),
            None  # default if no HumanMessage found
        )
        
        logger.info("Entering clinical node")
        
        context_enhancer_prompt = PromptTemplate(
            input_variables=["queries", "retrieved_rag_data", "web_search_output"], template=clinical_llm_template
        )
        final_context_prompt = context_enhancer_prompt.invoke(
            {"queries": query.content, "retrieved_rag_data": "", "web_search_output": ""}
        )

        out = clinical_llm.invoke(final_context_prompt)
        logger.info("clinical llm output is : %s", out)
        return {"messages": [out]}
    except Exception as e:
        logger.exception("Couldn't execute context enhancer node %s", e)

def should_continue(state: AgentState) -> Literal["data", "next_agent"]:
    """Function to decide what to do next"""
    messages = state["messages"]
    last_message = messages[-1]
    print(f"the last message in should_continue is  : {last_message}")

    # Check if there's a ToolMessage named "data_retriever_tool"
    has_data_retriever_tool = any(
        isinstance(m, ToolMessage) and getattr(m, "name", None) == "data_retriever_tool"
        for m in messages
    )

    # Only check .tool_calls if the last message actually supports it
    if hasattr(last_message, "tool_calls") and last_message.tool_calls and not has_data_retriever_tool:
        print("ENTERING LOOP")
        return "data"
    else:
        return "next_agent"
    
def should_continue_clinical(state: AgentState) -> Optional[Literal["clinical_tools", "exit"]]:
    """Function to decide what to do next."""
    messages = state["messages"]
    last_message = messages[-1]
    print(f"should_continue states clinical : {state['messages']}")

    # If the last message is a ToolMessage, it means the tool has already executed
    if isinstance(last_message, ToolMessage):
        print("✅ Tool finished — returning to agent.")
        return "exit"

    # If the LLM just made a tool call, perform the tool action
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("ENTERING web_search / clinical_tools")
        return "clinical_tools"

    # Otherwise, end or move to next stage
    return "exit"

memory = MemorySaver()

def graph():
    """
    Logic which decides how will each node be connected and how information and control will flow
    """
    graph = StateGraph(AgentState)

    graph.add_node("reception", reception_node)
    graph.add_node("data_retrieve_tool", tool_node)
    graph.add_edge(START, "reception")
    graph.add_edge("data_retrieve_tool", "reception")
    graph.add_node("clinical_agent", clinical_node)

    graph.add_conditional_edges(
        "reception",
        should_continue,
    {
        "data": "data_retrieve_tool",
        "next_agent": "clinical_agent"

    })
    
    graph.add_node("clinical_tools", clinical_tool_node)
    graph.add_conditional_edges(
        "clinical_agent",
        should_continue_clinical,
        {
            "clinical_tools" : "clinical_tools",
            "exit" : END
        }
    )
    print("returning output from clinical tools node!!")
    graph.add_edge("clinical_tools","clinical_agent")
    chat = graph.compile(checkpointer=memory, interrupt_before=["clinical_agent"])
    return chat