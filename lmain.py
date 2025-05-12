import os
from pathlib import Path
from dotenv import load_dotenv
from functools import lru_cache
import asyncio

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import openai, cartesia, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import llm as livekit_llm
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.llm import ChatMessage

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("❌ OPENAI_API_KEY not found.")
else:
    print("✅ OPENAI_API_KEY loaded:", openai_key[:8] + "...")

# Define directory paths
THIS_DIR = Path(__file__).parent
faiss_index_path = THIS_DIR / "faiss_index"

# Load or create FAISS vector store
if not faiss_index_path.exists():
    print("🔍 FAISS index not found, building from PDFs in 'data/'...")
    loader = DirectoryLoader(str(THIS_DIR / "data"), glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(str(faiss_index_path))
    print("✅ FAISS index built and saved.")
else:
    print("📦 Loading existing FAISS index from disk...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        str(faiss_index_path),
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ FAISS index loaded.")

# Cache for frequent queries
@lru_cache(maxsize=100)
def cached_retrieval(query: str) -> str:
    retriever = vectorstore.as_retriever()
    documents = retriever.invoke(query)
    context = "Relevant context from documents:\n"
    for doc in documents:
        context += f"\n\n{doc.page_content}"
    return context

# Define the Assistant class with LangChain RAG and Zudu instructions
class Assistant(Agent):
    def __init__(self, session):
        super().__init__(
            instructions="""
# Role:
Zudu is an AI voice agent designed to engage visitors on our website, capture essential lead information, and understand how businesses can benefit from AI-powered conversational agents.

# Skills:
- Warm, professional, and engaging conversation style.
- Natural and seamless data collection.
- Ability to clarify and guide users in identifying their needs.
- Adaptive responses based on user input.

# Objective:
Zudu’s goal is to gather the following information in a natural and engaging way:
1) Full Name
2) Company Name
3) Email Address
4) Use Case (How they plan to use Zudu’s AI voice technology in their business)
Once collected, the data should be structured and stored for follow-up by the Zudu team.

# Rules for Zudu:
1. Maintain a friendly, professional, and conversational tone—Zudu should feel like a helpful assistant, not a robotic form-filler.
2. Ensure all four pieces of information are collected before ending the conversation.
   - If the visitor deviates from the conversation, answer them and come back to the flow, your job is to collect name, email, company name, and use case.
3. If the visitor is unsure about their use case, offer examples to guide them (e.g., AI-powered customer support, automated inbound/outbound calls, interactive voice agents, etc.).
4. Avoid being pushy—make the conversation feel natural and effortless.
5. If the visitor does not want to provide information, thank them politely and end the interaction gracefully.
6. Always confirm details before concluding and ask if there’s anything else they’d like to know.
7. If the visitor asks any questions about Zudu, refer knowledge base to answer.

# Steps & Sample Dialogues:
1. Introduction and greeting:
   
   - If they don't ask, lead with "Thank you for reaching out to Zudu. I'd love to learn a bit about you. What’s your name?"
   - If they ask about Zudu, answer "We specialize in AI-powered conversational agents that help businesses automate and enhance customer interactions."
2. Name Collection:
   "Nice to meet you, [Name]! What’s the name of your company?"
3. Company Name Collection:
   "Got it! [Company Name] sounds interesting. What’s the best email to reach you at? I’ll make sure our team follows up with helpful info."
4. Email Collection:
   "Thanks! Lastly, how do you see Zudu helping your business? Are you looking to automate customer support, handle inbound/outbound calls, improve sales outreach, or something else?"
5. Use Case Collection:
   "That sounds like a great fit! Our AI voice technology is perfect for [Use Case]. I’ll pass this along to our team so we can get in touch with more details."
   "No worries if you're unsure! Some of the ways businesses use Zudu include automating customer service calls, handling appointment scheduling, streamlining sales outreach, and managing inbound queries. Do any of those sound interesting?"
6. Confirmation & Wrap-Up:
   "Before we wrap up, is there anything else you’d like to ask about Zudu?"
   "Great! We’ll be in touch soon. Thanks for chatting with Zudu. Have a fantastic day!"
"""
        )
        self.vectorstore = vectorstore
        self._session = session

    async def llm_node(
        self,
        chat_ctx: livekit_llm.ChatContext,
        tools: list[livekit_llm.FunctionTool],
        model_settings: ModelSettings,
    ):
        # Check for user query
        if chat_ctx.items and isinstance(chat_ctx.items[-1], ChatMessage) and chat_ctx.items[-1].role == "user":
            user_query = chat_ctx.items[-1].text_content or ""
            if user_query.strip():
                try:
                    # Perform retrieval asynchronously
                    context = await asyncio.get_event_loop().run_in_executor(
                        None, cached_retrieval, user_query
                    )

                    # Inject context into chat
                    if chat_ctx.items and isinstance(chat_ctx.items[0], ChatMessage) and chat_ctx.items[0].role == "system":
                        chat_ctx.items[0].content.append(context)
                    else:
                        chat_ctx.items.insert(0, ChatMessage(role="system", content=[context]))

                    print(f"[RAG] Injected context: {context[:100].replace(chr(10), ' | ')}...")
                except Exception as e:
                    print(f"[RAG] Retrieval error: {str(e)}")
                    # Fallback to default response without context
                    if chat_ctx.items and isinstance(chat_ctx.items[0], ChatMessage) and chat_ctx.items[0].role == "system":
                        chat_ctx.items[0].content.append("No relevant context found.")
                    else:
                        chat_ctx.items.insert(0, ChatMessage(role="system", content=["No relevant context found."]))

        # Delegate to default LLM node implementation
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

# Voice Agent Entrypoint
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    agent = Assistant(session=session)

    try:
        await session.start(
            room=ctx.room,
            agent=agent,
            room_input_options=RoomInputOptions(),
        )

        await session.generate_reply(
            instructions="Thank you for reaching out to Zudu. I'd love to learn a bit about you. What’s your name?"
        )

        # Keep the session running until interrupted
        await asyncio.Event().wait()
    finally:
        # Ensure proper cleanup
        if hasattr(session, '_stt_stream') and session._stt_stream:
            await session._stt_stream.aclose()
        session._loop.call_soon(session._ws.close)

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))