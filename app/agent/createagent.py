from langchain_openai import OpenAI
from app.config.settings import Model, Temprature
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

#----------------------------------------------------------
# THIS IS NOT USED IN THIS AGENT 
#----------------------------------------------------------


def create_agent(prompt : str, context):

    prompt = PromptTemplate(
        template = prompt,
        input_variables=["Question", "Context"]
    )

    llm = OpenAI(
        model= Model,
        temperature= Temprature
    )

    agent = {
        ["input", "context"] : RunnablePassthrough() | prompt | llm 
    }

    return agent