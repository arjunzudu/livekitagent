from pathlib import Path
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from Zilliz_cloud.upload_rag import create_and_upload_index

def load_or_create_index(persist_dir: Path, data_dir: Path):
    """Load an existing index or create a new one from PDFs and upload to Zilliz Cloud if necessary."""
    if not persist_dir.exists():
        print("🔍 Index not found, creating and uploading to Zilliz Cloud...")
        create_and_upload_index(persist_dir, data_dir)
    else:
        print("📦 Loading existing index from disk...")

    # Load the local index
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    print("✅ Index loaded.")
    return index