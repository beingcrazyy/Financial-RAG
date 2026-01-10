from langchain_openai import ChatOpenAI
from app.agent.prompts import ANSWER_VERIFICATION_PROMPT
from app.config.settings import Model, Temprature

def verify_output(answer: str, context : str):

    llm = ChatOpenAI(
        model= Model,
        temperature= Temprature
    )

    response = llm.invoke(
        ANSWER_VERIFICATION_PROMPT.format(
            answer = answer,
            context = context
        )
    )

    verification = response.content.strip().upper()

    if verification not in ("PASS", "FAIL", "PARTIAL"):
        return "FAIL"
    
    return verification