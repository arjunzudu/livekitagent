from src.config import load_config
from src.vector_store import setup_vector_store
from src.session import entrypoint
from livekit import agents

if __name__ == "__main__":
    # Load configuration
    config = load_config()

    # Set up vector store
    vectorstore = setup_vector_store(config["data_dir"], config["faiss_index_path"])

    # Run the LiveKit agent
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=lambda ctx: entrypoint(ctx, vectorstore)
    ))