from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.agent.metadata_extraction_agent import extract_metadata
from pathlib import Path
import pprint
import os

#----------------------------------------------------------
# GETTING THE CHUNKS OF THE DOCUMENT 
#----------------------------------------------------------

def get_text(docs):
    text = []
    for d in docs[:8]:
        if d.page_content.strip():
            text.append(d.page_content)
        if sum(len(i) for i in text) >= 3000:
            break
    return "\n".join(text)
         

def load_and_chunk_document(doc_path : str):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )

    file_path = Path(doc_path)

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    representative_text = get_text(docs)
    metadata = extract_metadata(representative_text)
    
    for d in docs:
            d.metadata.update(metadata.model_dump())

    chunks = text_splitter.split_documents(docs)

    return chunks
