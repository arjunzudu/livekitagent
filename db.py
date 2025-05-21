import os
import json
from typing import List, Dict
import PyPDF2
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import argparse

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

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    
    Raises:
        FileNotFoundError: If the PDF file does not exist.
        Exception: For other PDF reading errors.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def split_text_into_chunks(text: str) -> List[str]:
    """
    Splits the text into meaningful chunks for embedding.
    
    Args:
        text (str): The full text extracted from the PDF.
    
    Returns:
        List[str]: A list of text chunks.
    """
    # Split by single newlines to catch sections like "Our Vision", "Our Team", etc.
    initial_chunks = text.split('\n')
    
    # Clean and filter chunks
    chunks = []
    current_chunk = ""
    for line in initial_chunks:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        # If the line is a section header (short, standalone line), start a new chunk
        if len(line) < 50 and not current_chunk:
            if chunks and chunks[-1]:  # Ensure the last chunk isn't empty
                chunks[-1] = chunks[-1].strip()
            chunks.append(line)
        else:
            # Append to the current chunk
            if current_chunk:
                current_chunk += " " + line
            else:
                current_chunk = line
            # If the current chunk is long enough, add it to chunks
            if len(current_chunk.split()) > 50:  # Roughly a paragraph
                chunks.append(current_chunk)
                current_chunk = ""
    
    # Add any remaining text as the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    # Final cleanup: ensure chunks are not too short
    final_chunks = [chunk for chunk in chunks if len(chunk.split()) > 10]
    return final_chunks if final_chunks else [text.strip()]  # Fallback to whole text if no chunks

def generate_embeddings(chunks: List[str], api_key: str) -> List[List[float]]:
    """
    Generates vector embeddings for each text chunk using OpenAI's API.
    
    Args:
        chunks (List[str]): List of text chunks.
        api_key (str): OpenAI API key.
    
    Returns:
        List[List[float]]: List of vector embeddings for each chunk.
    
    Raises:
        Exception: For OpenAI API errors.
    """
    try:
        client = OpenAI(api_key=api_key)
        embeddings = []
        for chunk in chunks:
            response = client.embeddings.create(
                input=chunk,
                model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        return embeddings
    except Exception as e:
        raise Exception(f"Error generating embeddings: {str(e)}")

def save_index_locally(pdf_name: str, chunks: List[str], output_dir: str):
    """
    Saves the index (chunk IDs and texts) locally as a JSON file.
    
    Args:
        pdf_name (str): Base name of the PDF file (without extension).
        chunks (List[str]): List of text chunks.
        output_dir (str): Directory to save the JSON file.
    
    Raises:
        Exception: For file writing errors.
    """
    try:
        index_data = [
            {"id": f"{pdf_name}_{i}", "text": chunk}
            for i, chunk in enumerate(chunks)
        ]
        output_path = os.path.join(output_dir, f"{pdf_name}_index.json")
        with open(output_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        print(f"Index saved locally to {output_path}")
    except Exception as e:
        raise Exception(f"Error saving index locally: {str(e)}")

def upload_to_pinecone(index_name: str, vectors: List[Dict[str, any]], api_key: str):
    """
    Uploads the vector embeddings to a Pinecone index.
    
    Args:
        index_name (str): Name of the Pinecone index.
        vectors (List[Dict[str, any]]): List of vectors to upload, each with 'id', 'values', and 'metadata'.
        api_key (str): Pinecone API key.
    
    Raises:
        Exception: For Pinecone API errors.
    """
    try:
        pc = Pinecone(api_key=api_key)
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,  # Dimension for text-embedding-ada-002
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        index = pc.Index(index_name)
        index.upsert(vectors=vectors)
        print(f"Vectors uploaded to Pinecone index '{index_name}'")
    except Exception as e:
        raise Exception(f"Error uploading to Pinecone: {str(e)}")

def main():
    """
    Main function to extract text from PDF, create index, save locally, and upload to Pinecone.
    """
    parser = argparse.ArgumentParser(description="Extract text from PDF, create index, save locally, and upload to Pinecone.")
    parser.add_argument("--index_name", default="pdf-index", help="Pinecone index name")
    args = parser.parse_args()
    
    # Validate environment variables
    validate_environment()
    
    # Hardcode the PDF path
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of db.py (E:\pinecone_db)
    pdf_path = os.path.join(script_dir, "data", "kb.pdf")  # Path to E:\pinecone_db\data\kb.pdf
    
    # Extract text from PDF
    print("Extracting text from PDF...")
    try:
        text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        print(f"Failed to extract text: {str(e)}")
        return
    
    # Split text into chunks
    print("Splitting text into chunks...")
    chunks = split_text_into_chunks(text)
    print(f"Split text into {len(chunks)} chunks.")
    
    # Generate embeddings
    print("Generating embeddings...")
    try:
        embeddings = generate_embeddings(chunks, OPENAI_API_KEY)
    except Exception as e:
        print(f"Failed to generate embeddings: {str(e)}")
        return
    
    # Prepare vectors for Pinecone
    pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
    vectors = [
        {
            "id": f"{pdf_name}_{i}",
            "values": embedding,
            "metadata": {"text": chunk}
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    
    # Save index locally
    output_dir = os.path.dirname(pdf_path)
    try:
        save_index_locally(pdf_name, chunks, output_dir)
    except Exception as e:
        print(f"Failed to save index locally: {str(e)}")
        return
    
    # Upload to Pinecone
    print("Uploading to Pinecone...")
    try:
        upload_to_pinecone(args.index_name, vectors, PINECONE_API_KEY)
    except Exception as e:
        print(f"Failed to upload to Pinecone: {str(e)}")
        return

if __name__ == "__main__":
    main()