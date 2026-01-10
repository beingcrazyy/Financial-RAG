from langchain_core.prompts import PromptTemplate

DOCUMENT_METADATA_EXTRACTION_PROMPT = """
    You are a document analysis assistant.

    From the text below, extract high-level metadata for indexing and retrieval.

    Extract:
    - document_type: one of [annual_report, financial_statement, bank_filing, research_report, internal_doc, other]
    - domain: one of [finance, banking, technology, healthcare, legal, general]
    - entities: companies, organizations, or institutions explicitly mentioned
    - time_scope: year, quarter, or period if clearly stated

    Rules:
    - Do NOT infer or guess.
    - If a field is not explicitly stated, return null.
    - Return JSON only.
    - Be concise and accurate.

    TEXT:
    {document_excerpt}

"""

VERIFY_QUESTION_PROMPT = """
    You are checking whether a user question is within the scope of the provided documents.

    Determine whether the question is:
    - IN_SCOPE: can reasonably be answered using finance / business documents
    - OUT_OF_SCOPE: clearly unrelated (e.g., medical advice, coding help, personal questions)

    Rules:
    - Be conservative. If unsure, choose IN_SCOPE.
    - Do NOT answer the question.
    - Return JSON only.

    Question:
    {question}
"""

QUESTION_UNDERSTANDING_PROMPT = """
    You are a query understanding and retrieval planning assistant.

    Your task is to analyze the user question and produce structured outputs
    that help retrieve relevant information from documents.

    Extract:
    - intent: one of [LOOKUP, COMPARISON, SUMMARY, EXPLORATION]
    - entities: companies, organizations, sectors explicitly mentioned
    - time_scope: year or period if explicitly stated, otherwise null
    - retrieval_queries: 1 to 3 short, focused search queries optimized for document retrieval

    **Conditions** 
    LOOKUP:
    - The user asks for a specific fact, number, or statement

    COMPARISON:
    - The user asks to compare two or more entities

    SUMMARY:
    - The user asks for a condensed overview of a document or topic

    EXPLORATION:
    - The user asks broad, open-ended, or trend-based questions

    Rules:
    - Do NOT answer the question.
    - Do NOT infer entities or facts not explicitly mentioned.
    - Retrieval queries should be factual and concise.
    - Each retrieval query should be suitable for semantic search.
    - Return JSON only.

    User question:
    {question}
"""



ANSWER_GENERATION_PROMPT = """
    You are an analytical assistant helping a user using ONLY the provided document excerpts.

    Your task:
    - Answer the question using the given context.
    - If information is partial, say so clearly.
    - If information is missing, explain the limitation.
    - Do NOT use external knowledge.
    - Do NOT guess or infer beyond the text.

    Guidelines:
    - Be clear and structured.
    - Use neutral, professional language.
    - Cite information implicitly from the context.

    Context:
    {retrieved_context}

    Question:
    {question}
"""


ANSWER_VERIFICATION_PROMPT = """
    You are verifying whether an answer is fully supported by the provided context.

    Check:
    - Are all factual claims grounded in the context?
    - Are numbers consistent with the context?
    - Are there unsupported assertions?

    Return one of:
    - PASS: answer is fully grounded
    - PARTIAL: answer is mostly grounded but incomplete
    - FAIL: answer contains unsupported claims

    Return JSON only.

    Answer:
    {answer}

    Context:
    {context}
"""

RETRY_FIX_PROMPT = """

    You are correcting an answer that contains unsupported claims.

    Rules:
    - Remove any claims not supported by the context.
    - Preserve correct information.
    - If necessary, downgrade certainty.
    - Do NOT add new facts.
    - Use only the provided context.

    Return the corrected answer text only.

    Context:
    {context}

    Original Answer:
    {answer}

"""

