from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.ingest.loader import load_and_chunk_documents
from dotenv import load_dotenv
load_dotenv()

#----------------------------------------------------------
# CREATING THE VECTORS STORE (FAISS) FROM THE CHUNKS USING OPEN AI EMBEDDINGS  
#----------------------------------------------------------


def build_vectorstore(files : list):
    chunks = load_and_chunk_documents(files)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

#----------------------------------------------------------
# WE CAN USE THE SIMILARITY SEARCH FOR ANY VECTOR STORE  
#----------------------------------------------------------


if __name__ == "__main__":
    vs = build_vectorstore("doc/MicrosoftAnnualReport.pdf")

    query = "What was Microsoft's total revenue?"
    results = vs.similarity_search(query, k=3)

    for r in results:
        print("----")
        print(r.page_content[:300])
        print(r.metadata)