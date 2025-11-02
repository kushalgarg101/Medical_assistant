from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend import run_reception_graph

app = FastAPI()

@app.post("/chat")
def chat(question: str):
    if not question:
        raise HTTPException(status_code=400, detail='No question was provided')

    # Run your LangGraph flow
    full_response = run_reception_graph(user_input=question)

    # Extract messages
    ai_messages = []
    try:
        for msg in full_response.get("messages", []):
            # Only AI messages and non-empty content
            if msg.__class__.__name__ == "AIMessage" and msg.content.strip():
                ai_messages.append(msg.content.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing AI messages: {e}")

    # Return JSON for frontend
    return JSONResponse(content={"responses": ai_messages})