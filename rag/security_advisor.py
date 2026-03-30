import os
from retriever import retrieve_docs
from utils.llm_utils import get_llama_model

def _get_llm():
    """Return the unified LLM instance."""
    return get_llama_model()


def ask_security_advisor(question, history=None):
    """
    Retrieve relevant security docs via FAISS and answer a cybersecurity
    question using the Phi-3 mini LLM, taking prior chat history into account.

    Args:
        question (str): The current user question.
        history (list[dict] | None): Previous turns, each dict has keys
            'user' (str) and 'assistant' (str).

    Returns:
        tuple[str, list[dict]]: (answer_text, list_of_retrieved_docs)
            Each doc dict has keys: 'control', 'source', 'content'.
    """
    if history is None:
        history = []

    # Step 1 — Retrieve relevant docs for the current question
    docs = retrieve_docs(question, k=3)

    context = ""
    for d in docs:
        context += f"""
Control: {d['control']}
Source: {d['source']}
Description: {d['content']}
"""

    # Step 2 — Build conversation history text
    conversation = ""
    for h in history:
        conversation += f"User: {h['user']}\nAdvisor: {h['assistant']}\n"

    # Step 3 — Build full prompt
    prompt = f"""You are a cybersecurity advisor helping startups secure their systems.

Conversation History:
{conversation}
Security References:
{context}
User Question:
{question}

Provide clear practical advice and mention relevant security controls.
"""

    # Step 4 — Ask LLM
    response = _get_llm()(
        prompt,
        max_tokens=400,
        temperature=0.3,
        stop=["</s>"]
    )

    answer = response["choices"][0]["text"].strip()
    return answer, docs