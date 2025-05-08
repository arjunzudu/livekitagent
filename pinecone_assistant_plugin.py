import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message as PineconeMessage
from livekit.agents.llm import LLM, ChatContext, ChatMessage, ChatChunk, LLMStream, ChoiceDelta
from livekit.agents.types import APIConnectOptions
from livekit.agents.utils import shortuuid
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class PineconeAssistantLLM(LLM):
    def __init__(self, api_key: Optional[str] = None, assistant_name: str = "rag"):
        super().__init__()
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key is required.")
        self.pc = Pinecone(api_key=self.api_key)
        self.assistant = self.pc.assistant.Assistant(assistant_name=assistant_name)
        logger.debug(f"Initialized PineconeAssistantLLM with assistant: {assistant_name}")

    def chat(self, chat_ctx: ChatContext, **kwargs) -> 'PineconeAssistantLLMStream':
        logger.debug("Creating PineconeAssistantLLMStream")
        return PineconeAssistantLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=[],
            conn_options=APIConnectOptions()
        )

class PineconeAssistantLLMStream(LLMStream):
    async def _run(self) -> None:
        try:
            # Extract messages from self._chat_ctx.items
            messages = []
            for item in self._chat_ctx.items:
                if isinstance(item, ChatMessage):
                    role = item.role
                    content = "".join(item.content) if isinstance(item.content, list) else item.content
                    if role == "system":
                        # Convert system role to user role for Pinecone compatibility
                        role = "user"
                        content = f"System instruction: {content}"
                        logger.debug(f"Converted system message to user: {content}")
                    if not content:
                        logger.warning("Skipping empty content message")
                        continue
                    messages.append(PineconeMessage(role=role, content=content))
            logger.debug(f"Prepared messages for Pinecone: {messages}")

            if not messages:
                logger.warning("No valid messages to send to Pinecone")
                error_chunk = ChatChunk(
                    id=shortuuid(),
                    delta=ChoiceDelta(role="assistant", content="Error: No valid messages to process"),
                    usage=None
                )
                await self._event_ch.send(error_chunk)
                return

            # Call Pinecone Assistant
            logger.debug("Calling Pinecone Assistant API")
            resp = self._llm.assistant.chat(messages=messages)
            generated_text = resp.message.content
            logger.debug(f"Pinecone response: {generated_text}")

            # Create ChatChunk
            delta = ChoiceDelta(role="assistant", content=generated_text)
            chunk = ChatChunk(id=shortuuid(), delta=delta, usage=None)
            
            # Send the chunk
            logger.debug("Sending ChatChunk")
            await self._event_ch.send(chunk)
        except Exception as e:
            logger.error(f"Pinecone API error: {str(e)}")
            error_chunk = ChatChunk(
                id=shortuuid(),
                delta=ChoiceDelta(role="assistant", content=f"Error: {str(e)}"),
                usage=None
            )
            await self._event_ch.send(error_chunk)