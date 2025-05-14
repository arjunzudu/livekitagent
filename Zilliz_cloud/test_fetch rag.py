import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import MilvusClient

# Load environment variables
load_dotenv()
zilliz_uri = os.getenv("ZILLIZ_CLOUD_URI")
zilliz_api_key = os.getenv("ZILLIZ_CLOUD_API_KEY")

if not zilliz_uri or not zilliz_api_key:
    raise ValueError("ZILLIZ_CLOUD_URI and ZILLIZ_CLOUD_API_KEY must be set in .env")

# Define collection name
collection_name = "zudu_knowledge_base"

# Initialize Milvus Client to check collection
milvus_client = MilvusClient(uri=zilliz_uri, token=zilliz_api_key)
print(f"Connected to Zilliz Cloud: {zilliz_uri}")

# Verify collection exists and get entity count
if not milvus_client.has_collection(collection_name):
    raise ValueError(f"Collection {collection_name} does not exist. Run upload_rag_data.py first.")
milvus_client.load_collection(collection_name)
entity_count = milvus_client.get_collection_stats(collection_name)["row_count"]
print(f"Total entities in collection {collection_name}: {entity_count}")

# Initialize Milvus Vector Store for Zilliz Cloud
vector_store = MilvusVectorStore(
    uri=zilliz_uri,
    token=zilliz_api_key,
    collection_name=collection_name,
    dim=1536  # Must match the dimension used during upload
)

# Load the index from Zilliz Cloud
print("Loading index from Zilliz Cloud...")
index = VectorStoreIndex.from_vector_store(vector_store)
print("Index loaded successfully!")

# Create a retriever for vector search
retriever = index.as_retriever(similarity_top_k=3)

# Test query for RAG
query = "Who is the founder of Zudu AI?"
print(f"\nPerforming vector search for query: '{query}'")

# Fetch relevant documents
results = retriever.retrieve(query)

# Display results
print(f"Found {len(results)} results:")
for i, result in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"Text: {result.get_text()[:200]}...")  # Truncate for readability
    print(f"Score: {result.score}")