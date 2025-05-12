import asyncio
from livekit import agents
from livekit.agents import AgentSession, RoomInputOptions
from livekit.plugins import openai, cartesia, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from .agent import Assistant

async def entrypoint(ctx: agents.JobContext, vectorstore):
    """Start the LiveKit voice agent session."""
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    agent = Assistant(session=session, vectorstore=vectorstore)

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