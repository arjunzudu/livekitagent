Voice Agent
Overview
This project is an AI-driven conversational voice assistant built using LiveKit Agents and LLaMA Index. It leverages Retrieval-Augmented Generation (RAG) with a vector database to provide context-aware responses, optimized for efficient voice interactions. The assistant selectively retrieves document context based on trigger words, ensuring fast responses for simple queries.
Key features:

Selective RAG Retrieval: Uses trigger words to determine when to retrieve document context.
Customizable Trigger Words: Easily adjust which queries trigger RAG retrieval.
Scalable Knowledge Base: Integrates with a vector database for document retrieval.

Prerequisites

Python 3.8+
A vector database account (e.g., Zilliz Cloud, Milvus)
OpenAI API key for LLM (GPT-4o)
LiveKit Agents for voice interaction
LLaMA Index for RAG integration

Installation

Clone the Repository:
git clone https://github.com/yourusername/voice-agent.git
cd voice-agent


Set Up a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

Example requirements.txt:
livekit-agents
llama-index
openai


Set Environment Variables:Create a .env file in the project root and add your API keys:
OPENAI_API_KEY=your-openai-api-key
VECTOR_DB_API_KEY=your-vector-db-api-key


Prepare Instructions File:Ensure an instructions.txt file exists in the project root with the system prompt for the assistant.


Usage

Run the Application:
python main.py

This starts the LiveKit Agents worker, initializes the vector database index, and begins listening for voice inputs.

Interact with the Assistant:

Use a voice client compatible with LiveKit to interact with the assistant.
Example queries:
"What is the vision of the company?"
"Who is the founder?"
"Hi, my name is Arjun." (RAG skipped for simple queries)




Monitor Logs:Logs will display RAG decisions (performed or skipped) and retrieved context for debugging.


Project Structure

zudu_agent.py: Core logic for the voice agent, including selective RAG retrieval and LLM integration.
main.py: Entry point for starting the LiveKit worker and initializing the agent.
instructions.txt: System prompt for the assistant.
requirements.txt: List of Python dependencies.

Customization

Trigger Words: Modify the TRIGGER_WORDS list in zudu_agent.py to control which queries trigger RAG retrieval. Example:TRIGGER_WORDS = [
    "who", "what", "where", "when", "why", "how",
    "tell me", "explain", "describe", "give me",
    "information about", "details on", "facts about",
    "use case", "vision", "founder", "contact number",
]


RAG Parameters: Adjust similarity_top_k in the as_retriever() call to balance retrieval speed and accuracy.



License
This project is licensed under the MIT License. See the LICENSE file for details.
