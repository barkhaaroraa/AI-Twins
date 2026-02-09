from fastapi import FastAPI
from pydantic import BaseModel
from app.llm.ollama_client import generate_response

app = FastAPI(title="AI Twin System")

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    prompt = f"""
You are an AI assistant.

User Message:
{request.message}
"""
    response = generate_response(prompt)
    return {"response": response}
