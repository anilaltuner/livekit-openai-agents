from dotenv import load_dotenv, find_dotenv

from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.plugins import (
    openai,
    silero
)
from livekit_openai_agents import OpenAIAgentAdapter

from examples.tutors.tutor_agents import triage_agent

# Load .env file for API keys (e.g., OPENAI_API_KEY)
dotenv_path = find_dotenv()
if dotenv_path:
    print(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    print("No .env file found. Please ensure API keys are set as environment variables if needed.")


class ExampleAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a friendly and helpful voice AI assistant.")


async def entrypoint(ctx: agents.JobContext):
    print("🚀 Agent job starting...")
    await ctx.connect()
    print(f"🔌 Connected to room: {ctx.room.name}")

    # 3. Initialize the OrchestratorLLMAdapter with your orchestrator instance
    custom_llm_adapter = OpenAIAgentAdapter(orchestrator=triage_agent)

    session = AgentSession(
        stt=openai.STT(model="whisper-1"),
        llm=custom_llm_adapter,
        tts=openai.TTS(model="tts-1"),
        vad=silero.VAD.load(),
    )

    print("🎧 AgentSession configured. Starting session...")
    await session.start(
        room=ctx.room,
        agent=ExampleAssistant(),
        room_input_options=RoomInputOptions(),
    )
    print("✅ AgentSession started.")
    print("👋 Generating initial greeting...")

    await session.generate_reply(
        instructions="Greet the user warmly and ask how you can be of service today."
    )

    print("👂 Agent is now listening. Speak into your microphone.")
    print("ℹ️ To stop the agent, you can close the worker (Ctrl+C).")


if __name__ == "__main__":
    print("Starting LiveKit Agent with Custom Orchestrator LLM Adapter Example")
    # Ensure LIVEKIT_URL and LIVEKIT_API_KEY / LIVEKIT_API_SECRET are set in your environment or .env
    # You might also need OPENAI_API_KEY for STT/TTS.

    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
