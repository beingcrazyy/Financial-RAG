from langchain_core.prompts import PromptTemplate

RAG_PROMPT ="""

    You are a financial analysis assistant.

    Answer the user's question ONLY using the context provided below.
    If the answer is not present in the context, say:
    "Not found in the provided documents."

    Context:
    {context}

    Question:
    {question}

    Rules:
    - Do NOT use outside knowledge.
    - Be precise and factual.
    - If numbers are mentioned, copy them exactly from context.
"""

DECISION_PROMPT = """
    You are a decision-making agent.

    Your task:
    Decide whether the user's question is about these financial documents {document_metadata}

    Return ONLY one word:
    - YES
    - NO

    Rules:
    - YES if the question asks about financial data, numbers, risks, statements, facts or understanding the finance likely in the report.
    - NO if the question is general knowledge or outside the scope of finance 
    Question:
    {question}
"""

VERIFY_PROMPT = """
    You are a verification agent.

    Your task:
    Check whether the answer is fully supported by the provided context.

    Rules:
    - If ALL claims in the answer are directly supported by the context, respond with: PASS
    - If ANY claim is unsupported, partially supported, or inferred, respond with: FAIL
    - Do NOT add explanations.
    - Respond with only one word: PASS or FAIL

    Context:
    {context}

    Answer:
    {answer}
"""

RETRY_FIX_PROMPT = """
    You previously gave an answer that was NOT fully supported by the context.

    Your task:
    Rewrite the answer so that it is STRICTLY supported by the context.

    Rules:
    - Do NOT add new information.
    - Remove any unsupported claims.
    - If the context does not support an answer, respond with:
    "Not found in the provided documents."

    Context:
    {context}

    Previous Answer:
    {answer}
"""

