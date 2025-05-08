import os
import logging
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from pinecone_assistant_plugin import PineconeAssistantLLM

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="")
        logger.debug("Assistant initialized")


async def entrypoint(ctx: agents.JobContext):
    logger.debug("Connecting to LiveKit room")
    await ctx.connect()

    try:
        session = AgentSession(
            stt=deepgram.STT(model="nova-3", language="multi"),
            llm=PineconeAssistantLLM(api_key=os.getenv("PINECONE_API_KEY"), assistant_name="rag"),
            tts=cartesia.TTS(),
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
        )
        logger.debug("AgentSession created with PineconeAssistantLLM")
    except Exception as e:
        logger.error(f"Failed to create AgentSession: {str(e)}")
        raise

    try:
        await session.start(
            room=ctx.room,
            agent=Assistant(),
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        logger.debug("AgentSession started")
    except Exception as e:
        logger.error(f"Failed to start AgentSession: {str(e)}")
        raise

    try:
        await session.generate_reply(
            instructions="Greet the user and offer your assistance."
        )
        logger.debug("Generated initial reply")
    except Exception as e:
        logger.error(f"Failed to generate reply: {str(e)}")
        raise


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))