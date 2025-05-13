import asyncio
import logging
from datetime import datetime
from livekit import agents
from livekit.agents import AgentSession, RoomInputOptions
from livekit.plugins import openai, elevenlabs, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from .agent import Assistant

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Custom timestamp format
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

async def entrypoint(ctx: agents.JobContext, vectorstore):
    """Start the LiveKit voice agent session."""
    tasks = []
    session = None
    try:
        logger.debug(f"Vectorstore type in entrypoint: {type(vectorstore)}")
        await ctx.connect()
        logger.info("Connected to LiveKit room")

        # Initialize TTS with fallback handling
        try:
            tts = elevenlabs.TTS(
                voice_id="pzxut4zZz4GImZNlqQ3H",  # Default voice
                model="eleven_multilingual_v2"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize ElevenLabs TTS: {str(e)}. Falling back to default voice.")
            tts = elevenlabs.TTS(
                voice_id="EXAVITQu4vr4xnSDxMaL",  # Fallback voice (Sarah - en-US)
                model="eleven_multilingual_v2"
            )

        session = AgentSession(
            stt=deepgram.STT(model="nova-3", language="multi"),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=tts,
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
        )

        agent = Assistant(session=session, vectorstore=vectorstore)

        # Start session
        start_task = asyncio.create_task(
            session.start(
                room=ctx.room,
                agent=agent,
                room_input_options=RoomInputOptions(),
            )
        )
        tasks.append(start_task)
        await start_task  # Wait for session to start

        # Log STT completion by listening to transcript events
        async def log_transcript():
            async for event in session.transcript():
                logger.info(f"[{get_timestamp()}] STT Finished: Transcript received - {event.text}")

        # Start transcript logging
        transcript_task = asyncio.create_task(log_transcript())
        tasks.append(transcript_task)

        # Generate initial reply with error handling and TTS logging
        try:
            logger.info(f"[{get_timestamp()}] TTS Started: Generating initial reply")
            session.generate_reply(
                instructions="Thank you for reaching out to Zudu. I'd love to learn a bit about you. What’s your name?"
            )
        except Exception as e:
            logger.error(f"Failed to generate initial reply: {str(e)}")
            logger.info(f"[{get_timestamp()}] TTS Started: Generating fallback reply")
            session.generate_reply(
                instructions="Hello! I'm here to assist you. What’s your name?"
            )

        # Keep session running
        keep_alive_task = asyncio.create_task(asyncio.Event().wait())
        tasks.append(keep_alive_task)

        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)

    except Exception as e:
        logger.error(f"Error in entrypoint: {str(e)}")
        raise
    finally:
        logger.info("Shutting down tasks")
        for task in tasks:
            if not task.done():
                task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
        except asyncio.CancelledError:
            logger.debug("Tasks successfully cancelled")
        except asyncio.TimeoutError:
            logger.warning("Task cancellation timed out")
        except Exception as e:
            logger.error(f"Error during task shutdown: {str(e)}")
        if session:
            if hasattr(session, '_stt_stream') and session._stt_stream:
                await session._stt_stream.aclose()
            if hasattr(session, '_ws') and session._ws:
                session._loop.call_soon(session._ws.close)
        logger.info("Session cleanup completed")