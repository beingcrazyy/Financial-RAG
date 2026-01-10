from langchain_openai import ChatOpenAI
from app.agent.prompts import QUESTION_UNDERSTANDING_PROMPT
from app.config.settings import Model, Temprature
import os
import json
import pprint
from pydantic import BaseModel
from typing import Dict, List

class QuestionSpec(BaseModel):
    intent : str
    entities : Dict
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

    # parsed = json.loads(response.content.strip())

    spec = QuestionSpec.model_validate_json(response.content)

    return spec




