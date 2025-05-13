from pathlib import Path
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

def load_or_create_index(persist_dir: Path, data_dir: Path):
    """Load an existing index or create a new one from PDFs."""
    if not persist_dir.exists():
        print("🔍 Index not found, building from PDF in 'data/'...")
        documents = SimpleDirectoryReader(data_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
        print("✅ Index built and saved.")
    else:
        print("📦 Loading existing index from disk...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        print("✅ Index loaded.")
    return index