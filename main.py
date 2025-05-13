import asyncio
from livekit import agents
from livekit.agents import AgentSession, RoomInputOptions
from livekit.plugins import openai, cartesia, deepgram, silero, elevenlabs
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from config import load_config, get_project_dirs
from indexing import load_or_create_index
from zudu_agent import Assistant

async def entrypoint(ctx: agents.JobContext):
    # Load configuration and index
    config = load_config()
    dirs = get_project_dirs()
    index = load_or_create_index(dirs["persist_dir"], dirs["data_dir"])

    # Connect to LiveKit
    await ctx.connect()

    # Initialize session
    session = AgentSession(
            stt=deepgram.STT(model="nova-3", language="multi"),
            llm=openai.LLM(model="gpt-4o"),
            tts=elevenlabs.TTS(
                voice_id="pzxut4zZz4GImZNlqQ3H",
                model="eleven_multilingual_v2"
            ),
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
        )

    # Initialize agent
    agent = Assistant(session=session, index=index)

    # Start session
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(),
    )

    # Initial greeting
    await session.generate_reply(
        instructions="Hi, welcome to Zudu"
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))