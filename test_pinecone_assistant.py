import os
import asyncio
import logging
from dotenv import load_dotenv
from pinecone_assistant_plugin import PineconeAssistantLLM
from livekit.agents.llm import ChatContext

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

async def test_pinecone_assistant():
    """
    Test the Pinecone Assistant LLM plugin locally with a simulated chat interaction.
    """
    # Initialize the Pinecone Assistant LLM
    try:
        llm = PineconeAssistantLLM(
            api_key=os.getenv("PINECONE_API_KEY"),
            assistant_name="rag"
        )
        logger.debug("PineconeAssistantLLM initialized")
    except Exception as e:
        logger.error(f"Failed to initialize PineconeAssistantLLM: {str(e)}")
        return

    # Create a chat context to store conversation history
    chat_ctx = ChatContext()

    # Simulate a conversation with a system message
    test_messages = [
        ("system", "You are a helpful voice AI assistant."),
        ("user", "Hello, what can you tell me about Pinecone?"),
        ("user", "How does RAG work with Pinecone?"),
        ("user", "Who is the founder of Zudu AI?")
    ]

    print("Testing Pinecone Assistant Plugin\n")

    # Process each test message
    for role, content in test_messages:
        # Add message to chat context
        chat_ctx.add_message(role=role, content=[content])
        
        # Get the stream from the chat method
        logger.debug(f"Sending message: {content}")
        try:
            stream = llm.chat(chat_ctx=chat_ctx)
            logger.debug("Chat stream created")
        except Exception as e:
            logger.error(f"Failed to create chat stream: {str(e)}")
            print(f"You: {content}")
            print(f"Assistant: Error creating stream: {str(e)}\n")
            continue
        
        # Collect the response from the stream
        full_response = ""
        try:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    full_response += chunk.delta.content
                    logger.debug(f"Received chunk: {chunk.delta.content}")
        except Exception as e:
            full_response = f"Stream error: {str(e)}"
            logger.error(f"Stream error: {str(e)}")
        
        # Print the conversation
        print(f"You: {content}")
        print(f"Assistant: {full_response}\n")

    # Interactive chat loop
    print("Type your message (or 'exit' to quit):")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            # Add user input to chat context
            chat_ctx.add_message(role="user", content=[user_input])
            
            # Get the stream from the chat method
            logger.debug(f"Sending user input: {user_input}")
            stream = llm.chat(chat_ctx=chat_ctx)
            
            # Collect the response from the stream
            full_response = ""
            try:
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        full_response += chunk.delta.content
                        logger.debug(f"Received chunk: {chunk.delta.content}")
            except Exception as e:
                full_response = f"Stream error: {str(e)}"
                logger.error(f"Stream error: {str(e)}")
            
            # Print response
            print(f"Assistant: {full_response}\n")
        except (EOFError, KeyboardInterrupt):
            print("Goodbye!")
            break

# Run the test
if __name__ == "__main__":
    asyncio.run(test_pinecone_assistant())