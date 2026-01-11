from langchain_openai import ChatOpenAI
from app.agent.prompts import QUESTION_UNDERSTANDING_PROMPT
from app.config.settings import Model, Temprature
from app.helper.clean_llm_json import clean_json
import os
import json
import pprint
from pydantic import BaseModel
from typing import Dict, List

class QuestionSpec(BaseModel):
    intent : str
    entities : List[str]
    time_scope : str|None
    retrieval_queries : List[str]
    
def understand_question(question : str):

    llm = ChatOpenAI(
        model = Model,
        temperature = Temprature,
    )

    response = llm.invoke(
        QUESTION_UNDERSTANDING_PROMPT.format(
            question = question
        )
    )
    # cleaned_response = clean_json(response.content)

    parsed = json.loads(response.content.strip())

    spec = QuestionSpec.model_validate_json(response.content)

    return spec

if __name__ == "__main__":
    q = "compare the revenue of microsoft and google"
    result = understand_question(q)

    print(result)



