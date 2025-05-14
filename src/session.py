import asyncio
from livekit import agents
from livekit.agents import AgentSession, RoomInputOptions
from livekit.plugins import openai, elevenlabs, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

async def entrypoint(ctx: agents.JobContext, vectorstore):
    tasks = []
    session = None
    try:
        await ctx.connect()

        # Initialize TTS with fallback handling
        try:
            tts = elevenlabs.TTS(
                voice_id="pzxut4zZz4GImZNlqQ3H",
                model="eleven_multilingual_v2"
            )
        except Exception:
            tts = elevenlabs.TTS(
                voice_id="EXAVITQu4vr4xnSDxMaL",
                model="eleven_multilingual_v2"
            )

        # Simplified AgentSession with turn detection
        session = AgentSession(
            stt=deepgram.STT(
                model="nova-2",
                language="en",
            ),
            llm=openai.LLM(model="gpt-4o"),
            tts=tts,
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
        )

        # Import Assistant
        from .agent import Assistant
        agent = Assistant(session=session, vectorstore=vectorstore, ctx=ctx)

        # Start session
        start_task = asyncio.create_task(
            session.start(
                room=ctx.room,
                agent=agent,
                room_input_options=RoomInputOptions(),
            )
        )
        tasks.append(start_task)
        await start_task

        # Play initial message using generate_reply
        await session.generate_reply(
            instructions="Say: 'Thank you for reaching out to Zudu. I'd love to learn a bit about you. What’s your name?'"
        )

        # Keep session running
        keep_alive_task = asyncio.create_task(asyncio.Event().wait())
        tasks.append(keep_alive_task)
        await asyncio.gather(*tasks, return_exceptions=True)

    finally:
        # Cancel all tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        # Properly await task cancellation
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

        # Cleanup session resources
        if session:
            if hasattr(session, '_stt_stream') and session._stt_stream:
                await session._stt_stream.aclose()
            if hasattr(session, '_ws') and session._ws:
                await session._ws.close()