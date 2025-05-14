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
        self.interaction_count = 0  # Initialize interaction counter

    async def llm_node(
        self,
        chat_ctx: livekit_llm.ChatContext,
        tools: list[livekit_llm.FunctionTool],
        model_settings: ModelSettings,
    ):
        # Log STT task finish
        stt_finish_time = time.time()
        logger.info(f"STT task finished at {stt_finish_time:.2f}")

        # Increment interaction count
        self.interaction_count += 1

        # Decide which chat context to use based on interaction count
        if self.interaction_count <= 2:
            # Use original chat_ctx with full instructions for first two interactions
            chat_ctx_to_use = chat_ctx
        else:
            # Use short system message and last three messages (last two user messages + assistant response)
           
            
            conversation_history = chat_ctx.items[-3:]  # Last 3 messages: user(n-1), assistant(n-1), user(n)
            chat_ctx_to_use = livekit_llm.ChatContext()
            chat_ctx_to_use.items =  conversation_history

        # Only run retrieval if the last message is from the user
        if chat_ctx.items and isinstance(chat_ctx.items[-1], ChatMessage) and chat_ctx.items[-1].role == "user":
            user_query = chat_ctx.items[-1].text_content or ""
            if user_query.strip():
                # Start timing RAG
                rag_start_time = time.time()
                
                # Fetch RAG context
                retriever = self.index.as_retriever()
                nodes = await retriever.aretrieve(user_query)

                context = "Relevant context from documents:\n"
                for node in nodes:
                    node_content = node.get_content(metadata_mode=MetadataMode.LLM)
                    context += f"\n\n{node_content}"

                # Inject into system message of the chat context being used
                if chat_ctx_to_use.items and isinstance(chat_ctx_to_use.items[0], ChatMessage) and chat_ctx_to_use.items[0].role == "system":
                    chat_ctx_to_use.items[0].content.append(context)
                else:
                    chat_ctx_to_use.items.insert(0, ChatMessage(role="system", content=[context]))

                rag_time = time.time() - rag_start_time
                logger.info(f"RAG query processed in {rag_time:.2f} seconds for query: {user_query[:50]}...")
                print(f"[RAG] Injected context: {context[:100].replace(chr(10), ' | ')}...")

        # Log LLM query sent
        llm_query_sent_time = time.time()
        time_from_stt_to_llm_sent = llm_query_sent_time - stt_finish_time
        logger.info(f"LLM query sent at {llm_query_sent_time:.2f} (Time from STT finish: {time_from_stt_to_llm_sent:.2f} seconds)")

        # Process LLM response and log when first chunk is received and TTS starts
        first_chunk = True
        async for chunk in Agent.default.llm_node(self, chat_ctx_to_use, tools, model_settings):
            if first_chunk:
                llm_response_received_time = time.time()
                llm_processing_time = llm_response_received_time - llm_query_sent_time
                logger.info(f"LLM query received at {llm_response_received_time:.2f} (Processing time: {llm_processing_time:.2f} seconds)")
                logger.info(f"TTS start at {llm_response_received_time:.2f}")
                first_chunk = False
            yield chunk