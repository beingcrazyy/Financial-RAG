from app.retriver.vectorstore import build_vectorstore
from langchain_openai import ChatOpenAI
from app.config.settings import Model, Temprature
from app.agent.prompt import RAG_PROMPT
from app.agent.take_decision import decide_retrival_or_refusal
from app.agent.verify import verify_output
from app.agent.retry import retry

MAX_RETRY = 2

def answer(question : str):

    decision = decide_retrival_or_refusal(question)
    print (f"The decision is {decision}")

    if decision == "REFUSE" :
        return {
            "answer" : "Sorry! This is not found in the document",
            "source" : []
        }

    doc1_path = "doc/MicrosoftAnnualReport.pdf"
    doc2_path = "doc/GoogleAnnualReport.pdf"
    vs = build_vectorstore(
        [doc1_path, doc2_path]
        )

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

    answer = response.content

    verification = verify_output(answer, context)

    print(verification)

    if verification == "FAIL":
        for i in range (MAX_RETRY):
            print(f"Trying {i} time" )
            answer = retry(answer, context)
            verification = verify_output(answer, context)
            if verification == "PASS" :
                break
    
    if verification == "FAIL":
        return {
            "answer" : "Sorry! This is not found in the document",
            "source" : []
        }


    return {
        "answer" : answer,
        "source" : [d.metadata for d in docs]
    }

if __name__ == "__main__":
    q = "COMPARISON OF 5 YEAR CUMULATIVE TOTAL RETURN"
    result = answer(q)

    print("ANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for s in result["source"]:
        print(s)






