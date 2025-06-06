# LiveKit OpenAI Agents Adapter

This library provides an adapter to integrate agents built with the [openai-agents](https://github.com/openai/openai-agents-python) library (specifically, its `agents` module) as LLM providers within the [LiveKit Agents](https://github.com/livekit/agents) framework.

It allows you to plug in your `openai-agents` based orchestrators, leveraging their capabilities like conversation history management, tool usage, and handoffs, while still utilizing the real-time audio and agent lifecycle management features of LiveKit Agents.

## Features

- Integrates agents built with the `openai-agents` library into LiveKit.
- Facilitates the use of `openai-agents` features (like handoffs, structured output if designed in your agent) within LiveKit.
- **Configurable streaming**: Choose between the original non-streaming approach (default) or real-time streaming responses.
- Includes a utility to extract the last user message from the chat context for the agent.

## Installation

```bash
pip install livekit_openai_agents
```
*(Note: Ensure this package name matches your published package on PyPI.)*

Alternatively, if you have this project cloned:
```bash
pip install .
```

## Dependencies

Based on `pyproject.toml`:
- `livekit-agents[openai,elevenlabs,silero,turn-detector] == 1.0.17` (or your current version)
- `livekit-plugins-noise-cancellation~=0.2`
- `openai-agents >= 0.0.14`
- `pyee >= 9.0.0`

Please refer to your `pyproject.toml` for the most up-to-date list of dependencies.

## Usage

### 1. Define Your `openai-agents` Agent

First, create your agent(s) using the `agents` library. For example:

```python
# your_openai_agents.py
from agents import Agent
from pydantic import BaseModel

class MathResponse(BaseModel):
    explanation: str
    answer: float

math_tutor_agent = Agent(
    name="MathTutor",
    description="A specialized agent that helps with math problems and provides explanations.",
    instructions="You are a math tutor. Explain your reasoning step-by-step and provide the final answer.",
    output_type=MathResponse, # Example of structured output
    # You can add handoffs, tools, etc. as per openai-agents documentation
)

# You might have other agents or a more complex setup, e.g., a triage agent
# from agents import Runner
# result = await Runner.run(math_tutor_agent, "What is 2+2?")
# print(result.final_output) # Output: MathResponse(explanation='...', answer=4.0)
```

### 2. Use the Adapter in Your LiveKit Agent

Import `OpenAIAgentAdapter` from this library and your `openai-agents` agent. Then, initialize the adapter and use it in your LiveKit `AgentSession`.

```python
# your_livekit_app.py
import asyncio
from dotenv import load_dotenv, find_dotenv

from livekit import agents
from livekit.agents import Agent as LiveKitAgent, AgentSession, RoomInputOptions # Renamed to avoid clash
from livekit.plugins import openai, silero # or your preferred STT/TTS/VAD

# Import the adapter and your openai-agent
from livekit_openai_agents.adapter import OpenAIAgentAdapter # Assuming 'livekit_openai_agents' is the package name
from your_openai_agents import math_tutor_agent # The agent defined in step 1

# Load .env for API keys if necessary
# load_dotenv(find_dotenv())

class MyLiveKitAssistant(LiveKitAgent): # Renamed to avoid clash with openai-agents' Agent
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant that can call specialized tutors.")

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    # 1. Initialize your openai-agent (it's already defined, just use the instance)
    # math_tutor_agent is already an instance

    # 2. Initialize the adapter with your openai-agent instance
    # streaming=False (default): Uses the original non-streaming approach
    # streaming=True: Enables real-time streaming responses
    openai_agent_llm_adapter = OpenAIAgentAdapter(
        orchestrator=math_tutor_agent,
        streaming=False  # Default behavior - uses original non-streaming approach
    )

    # To enable streaming, you can set streaming=True:
    # openai_agent_llm_adapter = OpenAIAgentAdapter(
    #     orchestrator=math_tutor_agent,
    #     streaming=True  # Enable streaming for real-time responses
    # )

    # You can also change streaming mode dynamically:
    # openai_agent_llm_adapter.set_streaming(True)   # Enable streaming
    # openai_agent_llm_adapter.set_streaming(False)  # Disable streaming (back to original approach)
    
    # Check current streaming status:
    # is_streaming = openai_agent_llm_adapter.is_streaming_enabled()

    # 3. Set up the AgentSession
    session = AgentSession(
        stt=openai.STT(), # Replace with your STT
        llm=openai_agent_llm_adapter, # Use the adapter here
        tts=openai.TTS(), # Replace with your TTS
        vad=silero.VAD.load(),
    )

    await session.start(
        room=ctx.room,
        agent=MyLiveKitAssistant(),
        # ... other options
    )

    # Example: Generate an initial greeting
    await session.generate_reply(
        instructions="Greet the user and ask how you can help."
    )

    print("Agent is ready and listening.")

if __name__ == "__main__":
    # Load .env for API keys (e.g., OPENAI_API_KEY, LIVEKIT_URL, LIVEKIT_API_KEY)
    dotenv_path = find_dotenv()
    if dotenv_path:
        print(f"Loading .env file from: {dotenv_path}")
        load_dotenv(dotenv_path)
    else:
        print("No .env file found. Ensure API keys and LiveKit connection info are set as environment variables.")
    
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
```

### 3. Streaming Configuration

The `OpenAIAgentAdapter` supports both the original non-streaming approach and an optional streaming mode:

#### Non-Streaming Mode (Default)
```python
# Use original non-streaming approach (default behavior)
adapter = OpenAIAgentAdapter(orchestrator=your_agent, streaming=False)
```
- Uses the original implementation approach
- Responses are delivered as complete messages
- Maintains backward compatibility
- Uses `Runner.run()` internally

#### Streaming Mode (Optional)
```python
# Enable streaming responses
adapter = OpenAIAgentAdapter(orchestrator=your_agent, streaming=True)
```
- Responses are delivered in real-time as they are generated
- Users see text appearing progressively
- Better user experience for longer responses
- Uses `Runner.run_streamed()` internally

#### Dynamic Streaming Control
```python
adapter = OpenAIAgentAdapter(orchestrator=your_agent)  # Default: streaming=False

# Change streaming mode at runtime
adapter.set_streaming(True)   # Enable streaming
adapter.set_streaming(False)  # Disable streaming (back to original approach)

# Check current streaming status
if adapter.is_streaming_enabled():
    print("Streaming is enabled")
else:
    print("Using original non-streaming approach")
```

See the `examples/` directory (e.g., `examples/tutors/adapter_example.py`) for a more detailed, runnable example.

## API Reference

### OpenAIAgentAdapter

#### Constructor Parameters
- `orchestrator`: The OpenAI Agents Agent instance to adapt
- `guardrail_handler`: Optional function to handle guardrail trips
- `context`: Optional context to provide to the agent
- `streaming`: Whether to enable streaming responses (default: `False`)

#### Methods
- `set_streaming(streaming: bool)`: Sets the streaming mode for future chat calls
- `is_streaming_enabled() -> bool`: Returns whether streaming is currently enabled
- `chat(...)`: Creates a chat stream (uses streaming setting)
- `generate(...)`: Generates a response string (always non-streaming)

## Development

To set up for development:
```bash
git clone https://github.com/anilaltuner/livekit-openai-agents.git
cd livekit-openai-agents
pip install -e .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 