import faiss
import json
import hashlib
import numpy as np
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def setup_vector_store(data_dir, faiss_index_path):
    """Set up or load an optimized FAISS vector store."""
    # Check if PDFs have changed
    pdf_hashes = {f.name: hashlib.md5(open(f, "rb").read()).hexdigest() for f in data_dir.glob("*.pdf")}
    index_exists = faiss_index_path.exists()
    rebuild = False
    if index_exists:
        try:
            with open(faiss_index_path / "pdf_hashes.json", "r") as f:
                old_hashes = json.load(f)
            if old_hashes != pdf_hashes:
                rebuild = True
        except FileNotFoundError:
            rebuild = True

    if not index_exists or rebuild:
        loader = DirectoryLoader(str(data_dir), glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        # Filter short chunks
        texts = [doc for doc in texts if len(doc.page_content) > 50]
        embeddings = OpenAIEmbeddings()

        # Generate embeddings
        embedding_vectors = np.array([embeddings.embed_query(doc.page_content) for doc in texts]).astype('float32')
        
        # Normalize embeddings
        faiss.normalize_L2(embedding_vectors)

        # Choose index type based on number of vectors
        num_vectors = len(embedding_vectors)
        if num_vectors < 100:
            index = faiss.IndexFlatL2(embedding_vectors.shape[1])
        else:
            nlist = min(100, max(1, int(np.sqrt(num_vectors))))
            index = faiss.IndexIVFFlat(faiss.IndexFlatL2(embedding_vectors.shape[1]), embedding_vectors.shape[1], nlist)
            index.train(embedding_vectors)
        
        # Add vectors to the index
        index.add(embedding_vectors)

        # Create FAISS vector store
        vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore={i: doc for i, doc in enumerate(texts)},
            index_to_docstore_id={i: i for i in range(len(texts))}
        )
        
        # Set the embedding function to the callable method
        vectorstore.embedding_function = embeddings.embed_query
        
        vectorstore.save_local(str(faiss_index_path))
        with open(faiss_index_path / "pdf_hashes.json", "w") as f:
            json.dump(pdf_hashes, f)
    else:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            str(faiss_index_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        # Ensure embedding_function is callable
        vectorstore.embedding_function = embeddings.embed_query
    return vectorstore