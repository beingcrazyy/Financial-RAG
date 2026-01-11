from app.retriver.vectorstore import build_vectorstore
from langchain_openai import ChatOpenAI
from app.config.settings import Model, Temprature
from app.agent.prompts import ANSWER_GENERATION_PROMPT, ANSWER_VERIFICATION_PROMPT, RETRY_FIX_PROMPT
from app.agent.question_understanding_agent import understand_question
from app.agent.question_verification_agent import verify_question
from app.agent.answer_verification_agent import verify_output
from app.agent.retry_agent import retry
import logging

MAX_RETRY = 2

# LOGIN CONFIG -----------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

# BUILD CONTEXT FUNCTION -------------------------------------------------------------------------

def build_context(docs):
    logger.debug("Building context from %d documents", len(docs))
    return "\n\n".join(d.page_content for d in docs)

# EXTRACT SOURCES FUNCTION ------------------------------------------------------------------------

def extract_sources(docs):
    logger.debug("Extracting sources from documents")
    sources = []
    for d in docs:
        sources.append({
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "entity": d.metadata.get("entities"),
            "document_type": d.metadata.get("document_type")
        })
    return sources


# REMOVING DUPLICATE SOURCES FUNCTION --------------------------------------------------------------

def dedupe_sources(sources):
    logger.debug("Deduplicating %d sources", len(sources))
    unique = {}
    for s in sources:
        key = (s["source"], s["page"])
        unique[key] = s
    return list(unique.values())


# MAIN FUNCTION - TAKES QUESTION - GIVES ANSWER -------------------------------------------------- 

def answer(question : str):

    logger.info("Received question: %s", question)

    docs = ["doc/MicrosoftAnnualReport.pdf","doc/GoogleAnnualReport.pdf"]

    vector_stores = []
    logger.info("Building vector database")

# BUILDING VECTOR STORE ----------------------------------------------------------------------------

    for d in docs:
        try:
            vs = build_vectorstore(d)
            vector_stores.append(vs)
        except Exception as e:
            logger.error("Vector store build failed | doc=%s | error=%s", d, e, exc_info=True)
            raise

    logger.info("Vector database built successfully")

# VERIFYING AND ANALYSING THE QUESTION --------------------------------------------------------------
    
    try:
        question_status = verify_question(question)
        question_spec = understand_question(question)
        logger.info("Question verified with status: %s", question_status)
    except Exception as e:
        logger.error("Question processing failed | error=%s", e, exc_info=True)
        return {"error": "Invalid question processing"}

    logger.info(
        "Parsed question | intent=%s | entities=%s | queries=%s",
        question_spec.intent,
        question_spec.entities,
        question_spec.retrieval_queries
    )

# MAPPING THE ENTITY WITH VECTOR STORE --------------------------------------------------------------

    entity_vs_map = {
        "Microsoft" : vector_stores[0],
        "Google" : vector_stores[1]
    }

    if question_status == "OUT_OF_SCOPE" :
        logger.warning("Question is out of scope")
        return "Sorry! The question you are asking is out of scope for Financial RAG"
    

    intent = question_spec.intent
    entities = question_spec.entities
    retrieval_queries = question_spec.retrieval_queries

    retrived_docs = []

    logger.info("Starting document retrieval")

# RETRIVAL ACCORDING TO THE INTENT OF THE QUESTION ------------------------------------------------

    if intent == "COMPARISON":
        for entity in entities:
            vs = entity_vs_map.get(entity)
            if not vs:
                logger.warning("No vector store found for entity: %s", entity)
                continue

            for q in retrieval_queries:
                logger.debug("Similarity search | entity=%s | query=%s", entity, q)
                try: 
                    docs = vs.similarity_search(q, k=3)
                except Exception as e:
                    logger.warning(
                        "Similarity search failed | query=%s | error=%s",
                        q, e
                    )
                    continue
                retrived_docs.extend(docs)
    else :
        for vs in vector_stores:
            for q in retrieval_queries:
                logger.debug("Similarity search | query=%s", q)
                try: 
                    docs = vs.similarity_search(q, k=3)
                except Exception as e:
                    logger.warning(
                        "Similarity search failed | query=%s | error=%s",
                        q, e
                    )
                    continue
                retrived_docs.extend(docs)

# REMOVING DUPLICATE CHUNKS FROM THE RETRIVALS ---------------------------------------------------

    unique_docs = {}
    for d in retrived_docs:
        key = d.metadata.get("source"), d.metadata.get("page")
        unique_docs[key] = d

    retrived_docs = list(unique_docs.values())
    logger.info("Deduplicated to %d documents", len(retrived_docs))
    print("the type of retived docs is :", type(retrived_docs))

    # print(retrived_docs[1])

    context = build_context(retrived_docs)

    llm = ChatOpenAI(
        model = Model,
        temperature = Temprature
    )

# GENERATING ANSWER FROM THE RETRIVED CHUNKS ------------------------------------------------------

    logger.info("Generating answer")
    try: 
        response = llm.invoke(
            ANSWER_GENERATION_PROMPT.format(
                question = question,
                retrieved_context = context
            )
        )
    except Exception as e:
        logger.error("Answer generation LLM invocation failed | error=%s", e, exc_info=True)
        raise

    logger.info("Verifying answer")

# VERIFYING ANSWER AS IT SHOULD BE FROM THE CONTEXT PROVIDED -------------------------------------

    try:
        answer_verification = llm.invoke(
            ANSWER_VERIFICATION_PROMPT.format(
                answer = response.content,
                context = context
            )
        )
    except Exception as e:
        logger.error("Answer verification LLM invocation failed | error=%s", e, exc_info=True)
        raise

# RETRYING IF THE ANSWER IS NOT FROM THE CONTEXT PROVIDED ------------------------------------------

    for i in range (MAX_RETRY):
        if answer_verification.content.strip().upper() == "FAIL" :
            logger.warning("Answer verification failed | retry=%d", i + 1)
            try :
                response = llm.invoke(
                    RETRY_FIX_PROMPT.format(
                        answer = response.content,
                        context = context
                    )
                )
            except Exception as e:
                logger.error("retry LLM invocation failed | error=%s", e, exc_info=True)
                raise

            try:
                answer_verification = llm.invoke(
                    ANSWER_VERIFICATION_PROMPT.format(
                        answer = response.content,
                        context = context
                    )
                )
            except Exception as e:
                logger.error("Answer verification LLM invocation failed | error=%s", e, exc_info=True)
                raise
        else:
            logger.info("Answer verification passed")
            break
    
# CREATING FINAL ANSWER ----------------------------------------------------------------------------
            
    answer =  {
        "answer": response.content,
        "sources": dedupe_sources(extract_sources(retrived_docs))
    }

    logger.info("Answer pipeline completed successfully")
    return answer



if __name__ == "__main__":
    q = "What is the Revenue of Amazon?"
    result = answer(q)

    print("ANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for s in result["sources"]:
        print(s)






