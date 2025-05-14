import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import MilvusClient, utility

# Load environment variables
load_dotenv()
zilliz_uri = os.getenv("ZILLIZ_CLOUD_URI")
zilliz_api_key = os.getenv("ZILLIZ_CLOUD_API_KEY")

if not zilliz_uri or not zilliz_api_key:
    raise ValueError("ZILLIZ_CLOUD_URI and ZILLIZ_CLOUD_API_KEY must be set in .env")

# Define collection name and data directory
collection_name = "zudu_knowledge_base"
data_dir = Path("data/")

# Initialize Milvus Client to verify upload
milvus_client = MilvusClient(uri=zilliz_uri, token=zilliz_api_key)
print(f"Connected to Zilliz Cloud: {zilliz_uri}")

# Check if collection exists and drop it to start fresh (optional)
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)
    print(f"Dropped existing collection: {collection_name}")

# Initialize Milvus Vector Store for Zilliz Cloud
vector_store = MilvusVectorStore(
    uri=zilliz_uri,
    token=zilliz_api_key,
    collection_name=collection_name,
    dim=1536,  # Default for text-embedding-ada-002
    overwrite=True  # Overwrite to ensure fresh upload
)

# Load PDFs and create embeddings
print("Loading PDFs from data/ directory...")
documents = SimpleDirectoryReader(data_dir).load_data()
print(f"Loaded {len(documents)} documents")

# Create VectorStoreIndex and upload embeddings to Zilliz Cloud
print("Generating embeddings and uploading to Zilliz Cloud...")
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store
)
print("Upload completed successfully!")

# Verify the upload by checking the number of entities
milvus_client.load_collection(collection_name)
entity_count = milvus_client.get_collection_stats(collection_name)["row_count"]
print(f"Total entities in collection {collection_name}: {entity_count}")