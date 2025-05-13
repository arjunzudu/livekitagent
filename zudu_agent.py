import time
import logging
from llama_index.core.schema import MetadataMode
from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.agents.llm import ChatMessage
import livekit.agents.llm as livekit_llm
from livekit.agents.voice.agent import ModelSettings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Assistant(Agent):
    def __init__(self, session: AgentSession, index):
        with open("instructions.txt", "r") as f:
            instructions = f.read()
        super().__init__(instructions=instructions)
        self.index = index
        self._session = session

    async def llm_node(
        self,
        chat_ctx: livekit_llm.ChatContext,
        tools: list[livekit_llm.FunctionTool],
        model_settings: ModelSettings,
    ):
        # Only run retrieval if the last message is from the user
        if chat_ctx.items and isinstance(chat_ctx.items[-1], ChatMessage) and chat_ctx.items[-1].role == "user":
            user_query = chat_ctx.items[-1].text_content or ""
            if user_query.strip():
                # Start timing RAG
                start_time = time.time()
                
                # Fetch RAG context
                retriever = self.index.as_retriever()
                nodes = await retriever.aretrieve(user_query)

                context = "Relevant context from documents:\n"
                for node in nodes:
                    node_content = node.get_content(metadata_mode=MetadataMode.LLM)
                    context += f"\n\n{node_content}"

                # Inject into system message
                if chat_ctx.items and isinstance(chat_ctx.items[0], ChatMessage) and chat_ctx.items[0].role == "system":
                    chat_ctx.items[0].content.append(context)
                else:
                    chat_ctx.items.insert(0, ChatMessage(role="system", content=[context]))

                # Log RAG time
                rag_time = time.time() - start_time
                logger.info(f"RAG query processed in {rag_time:.2f} seconds for query: {user_query[:50]}...")
                print(f"[RAG] Injected context: {context[:100].replace(chr(10), ' | ')}...")

        # Delegate to default llm_node
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk