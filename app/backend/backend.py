from src.Rag.agent.agent import graph, AgentState
from langchain_core.messages import HumanMessage

thread = {"configurable" : {"thread_id" : "1"}}

def run_reception_graph(user_input: str):
    try:
        # Initialize state with patient message
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_input)],
            "user_inputs": [HumanMessage(content=user_input)]
        }
        
        print("🧩 Running Reception Graph...")
        workflow = graph()
        result = workflow.invoke(initial_state, config=thread)
        print("✅ Graph Execution Completed.\n")
        print("---- Result State ----")
        print(result)
        
        # Return the graph output directly
        return result
    
    except Exception as e:
        print(f"❌ Error running graph: {e}")
        raise