from langchain_openai import ChatOpenAI
from app.agent.prompt import VERIFY_PROMPT
from app.config.settings import Model, Temprature

def verify_output(answer: str, context : str):

    llm = ChatOpenAI(
        model= Model,
        temperature= Temprature
    )

    response = llm.invoke(
        VERIFY_PROMPT.format(
            answer = answer,
            context = context
        )
    )

    verification = response.content.strip().upper()

    if verification not in ("PASS", "FAIL"):
        return "FAIL"
    
    return verification