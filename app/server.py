from fastapi import FastAPI
from app.api.schema import QuestionRequest, AnswerResponse
from app.pipeline.rag import answer

app = FastAPI(title="Financial RAG")

@app.post("/ask", response_model= AnswerResponse)

def ask_question (req: QuestionRequest):
    return answer(req.question)

