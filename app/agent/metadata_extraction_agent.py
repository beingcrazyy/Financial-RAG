from langchain_openai import ChatOpenAI
from app.agent.prompts import DOCUMENT_METADATA_EXTRACTION_PROMPT
from app.config.settings import Model, Temprature
from app.helper.clean_llm_json import clean_json
from typing import List, Optional
import os
from pydantic import BaseModel
import re


class DocumentMetadata(BaseModel):
    document_type: str
    domain: str
    entities: List[str]
    time_scope: Optional[str]

def extract_metadata(text : str) :

    llm = ChatOpenAI(
        model = Model,
        temperature = Temprature,
    )

    doc_context = text[:3000]

    response = llm.invoke(
        DOCUMENT_METADATA_EXTRACTION_PROMPT.format(
            document_excerpt = doc_context
        )
    )

    cleaned_response = clean_json(response.content)

    # return response
    return DocumentMetadata.model_validate_json(cleaned_response)


