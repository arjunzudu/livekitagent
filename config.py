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
    return {"openai_key": openai_key}

def get_project_dirs():
    """Return project directory paths."""
    this_dir = Path(__file__).parent
    return {
        "this_dir": this_dir,
        "persist_dir": this_dir / "retrieval-engine-storage",
        "data_dir": this_dir / "data"
    }