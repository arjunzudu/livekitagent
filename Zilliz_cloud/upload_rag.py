import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import MilvusClient, utility
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
zilliz_uri = os.getenv("ZILLIZ_CLOUD_URI")
zilliz_api_key = os.getenv("ZILLIZ_CLOUD_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not zilliz_uri or not zilliz_api_key:
    raise ValueError("ZILLIZ_CLOUD_URI and ZILLIZ_CLOUD_API_KEY must be set in .env")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY must be set in .env")

# Define collection name and data directory
collection_name = "zudu_knowledge_base"
data_dir = Path("data/")

# Initialize Milvus Client to verify upload
milvus_client = MilvusClient(uri=zilliz_uri, token=zilliz_api_key)
print(f"Connected to Zilliz Cloud: {zilliz_uri}")

# Check if collection exists and drop it to start fresh
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

# Test embedding generation
print("Testing embedding generation...")
embed_model = OpenAIEmbedding(api_key=openai_api_key)
try:
    test_embedding = embed_model.get_text_embedding("Test text for embedding")
    print(f"Embedding generation successful. Sample embedding length: {len(test_embedding)}")
    if len(test_embedding) != 1536:
        raise ValueError(f"Unexpected embedding dimension: {len(test_embedding)}. Expected 1536.")
except Exception as e:
    raise ValueError(f"Failed to generate embeddings: {str(e)}")

# Load PDFs and inspect their content
print("Loading PDFs from data/ directory...")
reader = SimpleDirectoryReader(data_dir)
documents = reader.load_data()
print(f"Loaded {len(documents)} documents")
if len(documents) == 0:
    raise ValueError("No documents loaded from data/ directory. Ensure data/ contains valid PDFs.")

# Inspect the extracted text from each document
for i, doc in enumerate(documents):
    text = doc.text
    print(f"\nDocument {i+1} content (first 200 characters):")
    print(text[:200] if text else "No text extracted")
    if not text.strip():
        print(f"Warning: Document {i+1} has no extractable text.")

# Generate embeddings for a sample document to debug
if documents:
    sample_text = documents[0].text
    if sample_text.strip():
        sample_embedding = embed_model.get_text_embedding(sample_text[:500])
        print(f"Sample embedding for first document (length): {len(sample_embedding)}")
    else:
        print("Sample document has no text to embed.")

# Test direct insertion using MilvusVectorStore (optional, can be removed if not needed)
print("Testing direct insertion to Zilliz Cloud...")
test_node = TextNode(text="Prem Kumar is the founder of Zudu AI.", id_="test_doc_1")
test_node.embedding = embed_model.get_text_embedding(test_node.text)
vector_store.add([test_node])
milvus_client.flush(collection_name)
entity_count = milvus_client.get_collection_stats(collection_name)["row_count"]
print(f"Entities after direct insertion: {entity_count}")
if entity_count == 0:
    raise ValueError("Direct insertion failed: No entities were stored in the collection.")

# Create nodes from documents and add to vector store
nodes = []
for doc in documents:
    node = TextNode(text=doc.text, id_=doc.id_)
    node.embedding = embed_model.get_text_embedding(doc.text)
    nodes.append(node)

# Add nodes to vector store
print(f"Adding {len(nodes)} nodes to Zilliz Cloud...")
vector_store.add(nodes)

# Flush the collection to persist data
print("Flushing collection to persist data...")
milvus_client.flush(collection_name)

# Verify collection creation and indexing
print("Verifying collection status...")
collection_info = milvus_client.describe_collection(collection_name)
print(f"Collection info: {collection_info}")

# Verify the upload by checking the number of entities
milvus_client.load_collection(collection_name)
entity_count = milvus_client.get_collection_stats(collection_name)["row_count"]
print(f"Total entities in collection {collection_name}: {entity_count}")
if entity_count == 0:
    raise ValueError("Upload failed: No entities were stored in the collection.")