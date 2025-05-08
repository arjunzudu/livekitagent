import os
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.agents.llm import ChatMessage
from livekit.agents.voice.agent import ModelSettings
from livekit.plugins import deepgram, openai, cartesia, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import llm as livekit_llm

def get_mem0():
    from mem0 import AsyncMemoryClient
    return AsyncMemoryClient

load_dotenv()

# ensure all keys are present
for var in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "CARTESIA_API_KEY", "MEM0_API_KEY"):
    if not os.getenv(var):
        raise RuntimeError(f"Missing {var} in .env")

env_user = os.getenv("USER_ID")
USER_ID = env_user if env_user else "default_user"

class RAGAgent(Agent):
    def __init__(self, memory):
        super().__init__(instructions="You are a helpful voice agent. You may consult memory when needed.")
        self.memory = memory
        self.user_id = USER_ID

    async def llm_node(self, chat_ctx: livekit_llm.ChatContext, tools, model_settings: ModelSettings):
        last = chat_ctx.items[-1]
        if isinstance(last, ChatMessage) and last.role == "user":
            q = (last.text_content or "").strip()
            if q:
                try:
                    await self.memory.add([
                        {"role": "user", "content": q}
                    ], user_id=self.user_id)
                    print(f"💾 MEM0 ADD (user): {q}")
                except Exception as e:
                    print(f"❌ MEM0 ADD FAILED: {e}")

                try:
                    result = await self.memory.search(q, user_id=self.user_id)
                    print(f"🔍 MEM0 SEARCH RESULT: {result}")
                except Exception as e:
                    print(f"❌ MEM0 SEARCH FAILED: {e}")
                    result = []

                # 🔧 FIXED: handle both list and dict types
                if isinstance(result, dict):
                    hits = result.get("hits", [])
                elif isinstance(result, list):
                    hits = result
                else:
                    hits = []

                print(f"🔍 Found {len(hits)} hits")
                if hits:
                    context = "\n".join(h.get("text", "") for h in hits)
                    sys_msg = ChatMessage(role="system", content=[f"Relevant memory:\n{context}"])
                    chat_ctx.items.insert(0, sys_msg)

        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

async def entrypoint(ctx: agents.JobContext):
    AsyncMemoryClient = get_mem0()
    mem0 = AsyncMemoryClient(api_key=os.getenv("MEM0_API_KEY"))

    try:
        await mem0.add([
            {"role": "user", "content": "this is a test memory entry"}
        ], user_id=USER_ID)
        test = await mem0.search("test memory", user_id=USER_ID)
        print(f"✅ SANITY CHECK, got: {test}")
    except Exception as e:
        print(f"❌ SANITY CHECK FAILED: {e}")

    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(api_key=os.getenv("DEEPGRAM_API_KEY")),
        llm=openai.LLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini"),
        tts=cartesia.TTS(api_key=os.getenv("CARTESIA_API_KEY")),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=RAGAgent(mem0),
        room_input_options=RoomInputOptions()
    )

    await session.generate_reply(instructions="Hi there! How can I help you today?")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
