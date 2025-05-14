import os
from dotenv import load_dotenv
from pymilvus import MilvusClient, utility

# Load environment variables from .env file
load_dotenv()

# Retrieve Zilliz Cloud URI and API Key
milvus_uri = os.getenv("ZILLIZ_CLOUD_URI")
token = os.getenv("ZILLIZ_CLOUD_API_KEY")

if not milvus_uri or not token:
    raise ValueError("ZILLIZ_CLOUD_URI and ZILLIZ_CLOUD_API_KEY must be set in .env")

try:
    # Initialize Milvus Client for Zilliz Cloud
    milvus_client = MilvusClient(uri=milvus_uri, token=token)
    print(f"Connected to Zilliz Cloud: {milvus_uri} successfully")

    # List collections to verify connection (no data operations)
    collections = milvus_client.list_collections()
    print(f"Collections in cluster: {collections}")

except Exception as e:
    print(f"Failed to connect to Zilliz Cloud: {str(e)}")