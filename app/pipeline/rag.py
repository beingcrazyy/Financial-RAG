from app.retriver.vectorstore import build_vectorstore
from langchain_openai import ChatOpenAI
from app.config.settings import Model, Temprature
from app.agent.prompts import ANSWER_GENERATION_PROMPT
from app.agent.question_understanding_agent import understand_question
from app.agent.question_verification_agent import verify_question
from app.agent.answer_verification_agent import verify_output
from app.agent.retry_agent import retry

MAX_RETRY = 2

def build_context(docs):
    return "\n\n".join(d.page_content for d in docs)

def extract_sources(docs):
    sources = []
    for d in docs:
        sources.append({
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "entity": d.metadata.get("entities"),
            "document_type": d.metadata.get("document_type")
        })
    return sources

def dedupe_sources(sources):
    unique = {}
    for s in sources:
        key = (s["source"], s["page"])
        unique[key] = s
    return list(unique.values())

def answer(question : str):

    docs = ["doc/MicrosoftAnnualReport.pdf","doc/GoogleAnnualReport.pdf"]

    vector_stores = []

    for d in docs:
        vs = build_vectorstore(d)
        vector_stores.append(vs)

    question_status = verify_question(question)

    entity_vs_map = {
        "Microsoft" : vector_stores[0],
        "Google" : vector_stores[1]
    }

    if question_status == "OUT_OF_SCOPE" :
        return "Sorry! The question you are asking is out of scope for Financial RAG"

    question_spec = understand_question(question)

    intent = question_spec.intent
    entities = question_spec.entities
    retrieval_queries = question_spec.retrieval_queries

    retrived_docs = []

    if intent == "COMPARISON":
        for entity in entities:
            vs = entity_vs_map.get(entity)
            if not vs:
                continue

            for q in retrieval_queries:
                docs = vs.similarity_search(q, k=3)
                retrived_docs.extend(docs)
    else :
        for vs in vector_stores:
            for q in retrieval_queries:
                docs = vs.similarity_search(q, k=2)
                retrived_docs.extend(docs)

    unique_docs = {}
    for d in retrived_docs:
        key = d.metadata.get("source"), d.metadata.get("page")
        unique_docs[key] = d

    retrived_docs = list(unique_docs.values())
    print("the type of retived docs is :", type(retrived_docs))

    # print(retrived_docs[1])

    context = build_context(retrived_docs)

    llm = ChatOpenAI(
        model = Model,
        temperature = Temprature
    )


    response = llm.invoke(
        ANSWER_GENERATION_PROMPT.format(
            question = question,
            retrieved_context = context
        )
    )

    answer =  {
    "answer": response.content,
    "sources": dedupe_sources(extract_sources(retrived_docs))
    }

    return answer






if __name__ == "__main__":
    q = "Explain the investments of google and microsoft Giants and what we can derive from it that world is going in which direction"
    result = answer(q)

    print("ANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for s in result["sources"]:
        print(s)






