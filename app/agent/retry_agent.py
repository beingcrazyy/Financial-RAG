from langchain_openai import ChatOpenAI
from app.agent.prompts import RETRY_FIX_PROMPT
from app.config.settings import Model, Temprature

def retry(answer : str, context : str):

    llm = ChatOpenAI(
        model= Model,
        temperature= Temprature
    )

    response = llm.invoke(
        RETRY_FIX_PROMPT.format(
            context = context,
            answer = answer
        )
    )

    answer = response.content.strip()

    return answer
