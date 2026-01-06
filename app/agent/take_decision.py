from langchain_openai import ChatOpenAI
from app.config.settings import Model, Temprature
from app.agent.prompt import DECISION_PROMPT

def decide_retrival_or_refusal(question : str):

    llm = ChatOpenAI(
        model= Model,
        temperature= Temprature
    )

    response = llm.invoke(
        DECISION_PROMPT.format(
            question = question
        )
    )

    decision = response.content.strip().upper()

    # print(decision)

    if decision not in ("RETRIEVE", "REFUSE"):
        return "REFUSE"

    return decision

