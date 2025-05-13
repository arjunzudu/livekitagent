from functools import lru_cache
import logging
import numpy as np
from langchain_community.vectorstores.faiss import FAISS

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=500)
def cached_retrieval(query: str, vectorstore: FAISS) -> str:
    """Retrieve and cache relevant documents for a query using manual FAISS search."""
    try:
        # Validate vectorstore type
        logger.debug(f"Vectorstore type in cached_retrieval: {type(vectorstore)}")
        if not isinstance(vectorstore, FAISS):
            logger.error(f"Invalid vectorstore type: {type(vectorstore)}. Expected FAISS vectorstore.")
            return "No relevant context found."

        # Validate the FAISS index
        logger.debug(f"FAISS index type: {type(vectorstore.index)}")
        if not hasattr(vectorstore.index, 'search'):
            logger.error(f"Invalid FAISS index: {type(vectorstore.index)}. Expected FAISS index with search method.")
            return "No relevant context found."

        # Validate embedding function is callable
        logger.debug(f"Embedding function type: {type(vectorstore.embedding_function)}")
        if not callable(vectorstore.embedding_function):
            logger.error(f"Invalid embedding function: {type(vectorstore.embedding_function)}. Expected a callable function.")
            return "No relevant context found."

        # Embed the query using the correct method
        query_vector = vectorstore.embedding_function(query)
        query_vector = np.array([query_vector]).astype('float32')  # FAISS expects a 2D array

        # Perform similarity search
        k = 5
        D, I = vectorstore.index.search(query_vector, k)

        # Retrieve documents
        documents = []
        for i in I[0]:
            if i != -1:  # -1 indicates no match
                doc_id = vectorstore.index_to_docstore_id.get(i)
                if doc_id is not None:
                    doc = vectorstore.docstore.get(doc_id)
                    if doc:
                        documents.append(doc)

        if not documents:
            logger.warning(f"No documents retrieved for query: {query}")
            return "No relevant context found."

        # Combine documents into context
        context = "Relevant context from documents:\n"
        for doc in documents:
            context += f"\n\n{doc.page_content}"
        logger.debug(f"Retrieved context: {context[:100].replace(chr(10), ' | ')}...")
        return context
    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        return "No relevant context found."