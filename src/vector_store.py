from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def setup_vector_store(data_dir, faiss_index_path):
    """Set up or load the FAISS vector store."""
    if not faiss_index_path.exists():
        print("🔍 FAISS index not found, building from PDFs in 'data/'...")
        loader = DirectoryLoader(str(data_dir), glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(str(faiss_index_path))
        print("✅ FAISS index built and saved.")
    else:
        print("📦 Loading existing FAISS index from disk...")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            str(faiss_index_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ FAISS index loaded.")
    
    return vectorstore