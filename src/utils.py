from functools import lru_cache
import time
import numpy as np
from langchain_community.vectorstores.faiss import FAISS

@lru_cache(maxsize=500)
def cached_retrieval(query: str, vectorstore: FAISS) -> str:
    """Retrieve and cache relevant documents for a query using manual FAISS search."""
    # Embed the query and perform similarity search
    start_time = time.time()
    query_vector = vectorstore.embedding_function(query)
    query_vector = np.array([query_vector]).astype('float32')
    k = 5
    D, I = vectorstore.index.search(query_vector, k)
    end_time = time.time()
    print(f"RAG Retrieval took {end_time - start_time:.2f} seconds")

    # Retrieve documents
    documents = []
    for i in I[0]:
        if i != -1:
            doc_id = vectorstore.index_to_docstore_id.get(i)
            if doc_id is not None:
                doc = vectorstore.docstore.get(doc_id)
                if doc:
                    documents.append(doc)

    if not documents:
        return "No relevant context found."

    # Combine documents into context
    context = "Relevant context from documents:\n"
    for doc in documents:
        context += f"\n\n{doc.page_content}"
    return context