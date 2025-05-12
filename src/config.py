import os
from pathlib import Path
from dotenv import load_dotenv

def load_config():
    """Load environment variables and return configuration."""
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("❌ OPENAI_API_KEY not found.")
    print("✅ OPENAI_API_KEY loaded:", openai_key[:8] + "...")

    # Define directory paths
    THIS_DIR = Path(__file__).parent.parent  # Adjust for src/ directory
    DATA_DIR = THIS_DIR / "data"
    FAISS_INDEX_PATH = THIS_DIR / "faiss_index"

    return {
        "openai_key": openai_key,
        "data_dir": DATA_DIR,
        "faiss_index_path": FAISS_INDEX_PATH,
    }