from functools import lru_cache

@lru_cache(maxsize=100)
def cached_retrieval(query: str, vectorstore) -> str:
    """Retrieve and cache relevant documents for a query."""
    retriever = vectorstore.as_retriever()
    documents = retriever.invoke(query)
    context = "Relevant context from documents:\n"
    for doc in documents:
        context += f"\n\n{doc.page_content}"
    return context