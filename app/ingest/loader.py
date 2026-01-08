from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import pprint
import os

#----------------------------------------------------------
# GETTING THE CHUNKS OF THE DOCUMENT 
#----------------------------------------------------------

def load_and_chunk_documents(files : list):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    all_docs = []

    for file in files:
        file_path = Path(file)
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        for d in docs:
            d.metadata["source_file"] = file

        all_docs.extend(docs)

    chunks = text_splitter.split_documents(all_docs)

    # pprint.pp(docs[0].metadata)
    # print(docs[3].page_content)

    # print(len(chunks))
    # print(chunks[9].page_content)
    # pprint.pp(chunks[0].metadata)

    return chunks
