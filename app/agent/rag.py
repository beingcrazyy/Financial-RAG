from app.retriver.vectorstore import build_vectorstore
from langchain_openai import ChatOpenAI
from app.config.settings import Model, Temprature
from app.agent.prompt import RAG_PROMPT

def answer(question : str):
    doc_path = "doc/MicrosoftAnnualReport.pdf"
    vs = build_vectorstore(doc_path)

    docs = vs.similarity_search(question, k= 5)

    context = "\n\n".join(
        d.page_content for d in docs
    )

    llm = ChatOpenAI(
        model= Model,
        temperature= Temprature
    )

    response = llm.invoke(
        RAG_PROMPT.format(
            context = context,
            question = question
        )
    )

    return {
        "answer" : response.content,
        "source" : [d.metadata for d in docs]
    }

if __name__ == "__main__":
    q = "What was Microsofts's total revenue?"
    result = answer(q)

    print("ANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for s in result["source"]:
        print(s)






