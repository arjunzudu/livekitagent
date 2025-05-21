import os
import time
import argparse
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Retrieve API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def validate_environment():
    """Checks if required environment variables are set."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable is not set.")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

def generate_query_embedding(query: str, api_key: str) -> list:
    """
    Generates a vector embedding for the query using OpenAI's API.
    
    Args:
        query (str): The query text.
        api_key (str): OpenAI API key.
    
    Returns:
        list: Vector embedding of the query.
    """
    try:
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        raise Exception(f"Error generating query embedding: {str(e)}")

def query_pinecone(index_name: str, query_embedding: list, api_key: str, top_k: int = 1) -> dict:
    """
    Queries the Pinecone index with the given embedding.
    
    Args:
        index_name (str): Name of the Pinecone index.
        query_embedding (list): Embedding of the query.
        api_key (str): Pinecone API key.
        top_k (int): Number of top matches to return (set to 1).
    
    Returns:
        dict: Query results and response time in seconds.
    """
    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        
        start_time = time.time()
        query_response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        end_time = time.time()
        
        response_time = end_time - start_time
        return {
            "results": query_response,
            "response_time": response_time
        }
    except Exception as e:
        raise Exception(f"Error querying Pinecone: {str(e)}")

def run_chatbot(index_name: str):
    """
    Runs a chatbot loop to continuously query Pinecone with user input, returning only the top match.
    
    Args:
        index_name (str): Name of the Pinecone index.
    """
    top_k = 1  # Enforce returning only one response
    print("Starting Pinecone query chatbot...")
    print("Type your query and press Enter. To exit, type 'exit' or 'quit'.")
    print(f"Using index: {index_name}, returning only the top match")
    print("-" * 50)
    
    while True:
        # Prompt user for input
        query = input("Enter your query: ").strip()
        
        # Check for exit commands
        if query.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break
        
        # Skip empty queries
        if not query:
            print("Please enter a valid query.")
            continue
        
        print(f"\nQuery: {query}")
        
        # Measure total time (embedding + Pinecone query)
        total_start_time = time.time()
        
        # Generate query embedding
        print("Generating query embedding...")
        try:
            query_embedding = generate_query_embedding(query, OPENAI_API_KEY)
        except Exception as e:
            print(f"Failed to generate query embedding: {str(e)}")
            continue
        
        # Query Pinecone
        print(f"Querying Pinecone index '{index_name}'...")
        try:
            result = query_pinecone(index_name, query_embedding, PINECONE_API_KEY, top_k)
            query_results = result["results"]
            pinecone_response_time = result["response_time"]
            
            total_end_time = time.time()
            total_response_time = total_end_time - total_start_time
            
            print(f"\nTotal response time (embedding + query): {total_response_time:.3f} seconds")
            print(f"Pinecone query response time: {pinecone_response_time:.3f} seconds")
            print("\nTop match:")
            if query_results["matches"]:
                match = query_results["matches"][0]  # Only the top match
                print(f"ID: {match['id']}")
                print(f"Score: {match['score']:.3f}")
                print(f"Text: {match['metadata']['text']}\n")
            else:
                print("No matches found.")
        except Exception as e:
            print(f"Failed to query Pinecone: {str(e)}")
            continue
        
        print("-" * 50)

def main():
    """
    Main function to initialize and run the Pinecone query chatbot.
    """
    parser = argparse.ArgumentParser(description="Run a Pinecone query chatbot returning only the top match.")
    parser.add_argument("--index_name", default="pdf-index", help="Pinecone index name")
    args = parser.parse_args()
    
    # Validate environment variables
    validate_environment()
    
    # Run the chatbot
    run_chatbot(args.index_name)

if __name__ == "__main__":
    main()