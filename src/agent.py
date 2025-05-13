import asyncio
import time
import logging
from datetime import datetime
from livekit import agents
from livekit.agents import llm as livekit_llm
from livekit.agents.llm import ChatMessage
from livekit.agents.voice.agent import ModelSettings

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Custom timestamp format
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

class Assistant(agents.Agent):
    def __init__(self, session, vectorstore):
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
8. If the visitor mentions 'Zulu,' they likely mean the historical Zulu Kingdom, founded by Shaka kaSenzangakhona (Shaka Zulu) around 1816, not Zudu. Clarify the context before responding (e.g., 'Did you mean the Zulu Kingdom, or are you asking about Zudu’s services?').

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
        logger.debug(f"Vectorstore type in Assistant init: {type(self.vectorstore)}")

    async def llm_node(
        self,
        chat_ctx: livekit_llm.ChatContext,
        tools: list[livekit_llm.FunctionTool],
        model_settings: ModelSettings,
    ):
        from .utils import cached_retrieval

        # Check for user query
        user_query = ""
        if chat_ctx.items and isinstance(chat_ctx.items[-1], ChatMessage) and chat_ctx.items[-1].role == "user":
            user_query = chat_ctx.items[-1].text_content or ""
            if user_query.strip():
                try:
                    # Log vectorstore type before retrieval
                    logger.debug(f"Vectorstore type before cached_retrieval: {type(self.vectorstore)}")
                    # Perform retrieval asynchronously with timing
                    start_time = time.time()
                    context = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: cached_retrieval(user_query, self.vectorstore)
                    )
                    end_time = time.time()
                    print(f"[{get_timestamp()}] RAG Timing: Retrieval took {end_time - start_time:.2f} seconds")
                    print(f"[{get_timestamp()}] RAG Timing: Injected context: {context[:100].replace(chr(10), ' | ')}...")

                    # Inject context into chat
                    if chat_ctx.items and isinstance(chat_ctx.items[0], ChatMessage) and chat_ctx.items[0].role == "system":
                        chat_ctx.items[0].content.append(context)
                    else:
                        chat_ctx.items.insert(0, ChatMessage(role="system", content=[context]))
                except Exception as e:
                    print(f"[{get_timestamp()}] RAG Error: Retrieval error: {str(e)}")
                    # Fallback to default response without context
                    if chat_ctx.items and isinstance(chat_ctx.items[0], ChatMessage) and chat_ctx.items[0].role == "system":
                        chat_ctx.items[0].content.append("No relevant context found.")
                    else:
                        chat_ctx.items.insert(0, ChatMessage(role="system", content=["No relevant context found."]))

        # Log query sent to LLM
        print(f"[{get_timestamp()}] LLM Query Sent: {user_query}")

        # Delegate to default LLM node implementation and log response received
        start_time = time.time()
        async for chunk in agents.Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk
        end_time = time.time()
        print(f"[{get_timestamp()}] LLM Query Received: Response took {end_time - start_time:.2f} seconds")