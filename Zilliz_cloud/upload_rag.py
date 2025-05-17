import os
import logging
import re
from pathlib import Path
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Document
from llama_index.core.schema import TextNode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from pymilvus import MilvusClient

# Configure logging
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_embedding(embed_model, text):
    """Generate embedding with retry logic."""
    return embed_model.get_text_embedding(text)

def create_and_upload_index(persist_dir: Path, data_dir: Path):
    """Create an index, save it locally, and upload to Zilliz Cloud."""
    # Initialize embedding model
    embed_model = OpenAIEmbedding(api_key=openai_api_key)

    # Test embedding generation
    try:
        test_embedding = generate_embedding(embed_model, "Test text for embedding")
        if len(test_embedding) != 1536:
            raise ValueError(f"Unexpected embedding dimension: {len(test_embedding)}. Expected 1536.")
        print("✅ Embedding generation test successful.")
    except Exception as e:
        raise ValueError(f"Failed to generate test embedding: {str(e)}")

    # Load documents
    print("Loading PDFs from data/ directory...")
    reader = SimpleDirectoryReader(data_dir)
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents")
    if len(documents) == 0:
        raise ValueError("No documents loaded from data/ directory. Ensure data/ contains valid PDFs.")

    # Validate and filter documents with extractable text, create new Documents with cleaned text
    cleaned_documents = []
    for i, doc in enumerate(documents):
        text = doc.text.strip()
        if not text:
            print(f"Warning: Document {i+1} (ID: {doc.id_}) has no extractable text. Skipping.")
            continue
        # Clean the text: normalize whitespace and remove non-ASCII characters
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)
        if len(cleaned_text) < 10:
            print(f"Warning: Document {i+1} (ID: {doc.id_}) has too short text after cleaning ({len(cleaned_text)} chars). Skipping.")
            continue
        # Create a new Document with the cleaned text
        new_doc = Document(text=cleaned_text, id_=doc.id_, metadata=doc.metadata)
        try:
            test_doc_embedding = generate_embedding(embed_model, cleaned_text[:500])
            if len(test_doc_embedding) != 1536:
                print(f"Warning: Document {i+1} (ID: {doc.id_}) generated an invalid embedding (length: {len(test_doc_embedding)}). Skipping.")
                continue
            print(f"Document {i+1} (ID: {doc.id_}) text (first 200 chars): {cleaned_text[:200]}...")
            cleaned_documents.append(new_doc)
        except Exception as e:
            print(f"Warning: Failed to generate embedding for Document {i+1} (ID: {doc.id_}): {str(e)}. Skipping.")

    if not cleaned_documents:
        raise ValueError("No documents with extractable text and valid embeddings found. Cannot create index.")

    print(f"Proceeding with {len(cleaned_documents)} valid documents.")

    # Create local storage context
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore(),
        vector_store=SimpleVectorStore(),
        index_store=SimpleIndexStore(),
    )

    # Manually generate embeddings and create nodes
    nodes = []
    for doc in cleaned_documents:
        try:
            embedding = generate_embedding(embed_model, doc.text)
            if embedding is None or len(embedding) != 1536:
                raise ValueError(f"Invalid embedding generated for document {doc.id_}.")
            node = TextNode(text=doc.text, id_=doc.id_, embedding=embedding)
            nodes.append(node)
        except Exception as e:
            print(f"Warning: Failed to generate embedding for document {doc.id_}: {str(e)}. Skipping.")

    if not nodes:
        raise ValueError("No documents with valid embeddings.")

    # Create and persist local index
    print("Creating local index...")
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    storage_context.persist(persist_dir=persist_dir)
    print(f"Index saved locally to {persist_dir}")

    # Upload to Zilliz Cloud
    collection_name = "zudu_knowledge_base"
    milvus_client = MilvusClient(uri=zilliz_uri, token=zilliz_api_key)
    print(f"Connected to Zilliz Cloud: {zilliz_uri}")

    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)
        print(f"Dropped existing collection: {collection_name}")

    vector_store = MilvusVectorStore(
        uri=zilliz_uri,
        token=zilliz_api_key,
        collection_name=collection_name,
        dim=1536,
        overwrite=True
    )

    # Validate embeddings before uploading
    valid_nodes = []
    for node in nodes:
        if not hasattr(node, 'embedding') or node.embedding is None:
            print(f"Warning: Node {node.id_} has no embedding. Text (first 200 chars): {node.text[:200]}... Skipping.")
            continue
        if not isinstance(node.embedding, list) or len(node.embedding) != 1536:
            print(f"Warning: Node {node.id_} has invalid embedding (length: {len(node.embedding) if isinstance(node.embedding, list) else 'N/A'}). Expected 1536. Skipping.")
            continue
        valid_nodes.append(node)

    if not valid_nodes:
        raise ValueError("No nodes with valid embeddings to upload.")

    print(f"Uploading {len(valid_nodes)} nodes to Zilliz Cloud...")
    vector_store.add(valid_nodes)

    milvus_client.flush(collection_name)
    entity_count = milvus_client.get_collection_stats(collection_name)["row_count"]
    print(f"Total entities in collection {collection_name}: {entity_count}")
    if entity_count == 0:
        raise ValueError("Upload failed: No entities stored.")

if __name__ == "__main__":
    persist_dir = Path("retrieval-engine-storage")
    data_dir = Path("data")
    create_and_upload_index(persist_dir, data_dir)