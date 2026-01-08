from langchain_openai import ChatOpenAI
from app.config.settings import Model, Temprature
from app.agent.prompts import DECISION_PROMPT

def decide_retrival_or_refusal(question : str, metadata: str):

    llm = ChatOpenAI(
        model= Model,
        temperature= Temprature
    )

    response = llm.invoke(
        DECISION_PROMPT.format(
            document_metadata = metadata,
            question = question
        )
    )

    decision = response.content.strip().upper()

    # print(decision)

    if decision not in ("YES", "NO"):
        return "NO"

    return decision

