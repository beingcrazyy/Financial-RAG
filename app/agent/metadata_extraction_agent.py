from langchain_openai import ChatOpenAI
from app.agent.prompts import DOCUMENT_METADATA_EXTRACTION_PROMPT
from app.config.settings import Model, Temprature
from typing import List, Optional
from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    document_type: str
    domain: str
    entities: List[str]
    time_scope: Optional[str]

def extract_metadata(text : str) -> DocumentMetadata:

    llm = ChatOpenAI(
        model= Model,
        temperature= Temprature
    )

    doc_context = text[:3000]

    response = llm.invoke(
        DOCUMENT_METADATA_EXTRACTION_PROMPT.format(
            document_excerpt = doc_context
        )
    )

    return DocumentMetadata.model_validate_json(response.content)
