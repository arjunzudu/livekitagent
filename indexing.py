import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from Zilliz_cloud.upload_rag import create_and_upload_index

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
zilliz_uri = os.getenv("ZILLIZ_CLOUD_URI")
zilliz_api_key = os.getenv("ZILLIZ_CLOUD_API_KEY")

if not zilliz_uri or not zilliz_api_key:
    raise ValueError("ZILLIZ_CLOUD_URI and ZILLIZ_CLOUD_API_KEY must be set in .env")

def load_or_create_index(persist_dir: Path, data_dir: Path):
    """Load an existing index from Zilliz Cloud or create a new one from PDFs and upload to Zilliz Cloud."""
    # Always create or ensure the index is uploaded to Zilliz Cloud
    if not persist_dir.exists():
        print("🔍 Index not found, creating and uploading to Zilliz Cloud...")
        create_and_upload_index(persist_dir, data_dir)
    else:
        print("📦 Local index exists, ensuring it’s uploaded to Zilliz Cloud...")

    # Load the index from Zilliz Cloud using MilvusVectorStore
    collection_name = "zudu_knowledge_base"
    vector_store = MilvusVectorStore(
        uri=zilliz_uri,
        token=zilliz_api_key,
        collection_name=collection_name,
        dim=1536,
    )
    
    print("Loading index from Zilliz Cloud...")
    index = VectorStoreIndex.from_vector_store(vector_store)
    logging.info(f"Index loaded with vector store: {type(index.vector_store)}")
    print("✅ Index loaded.")
    return index