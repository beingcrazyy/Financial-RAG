from langchain_openai import ChatOpenAI
from app.agent.prompts import VERIFY_QUESTION_PROMPT
from app.config.settings import Model, Temprature

def verify_question(question: str):

    llm = ChatOpenAI(
        model = Model,
        temperature= Temprature
    )


    response = llm.invoke(
        VERIFY_QUESTION_PROMPT.format(
            question = question
        )
    )

    verdict = response.content.strip().upper()

    return verdict




