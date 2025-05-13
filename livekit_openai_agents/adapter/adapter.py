from asyncio import ensure_future, Future
from typing import Any, Dict, List, Optional

from agents import Agent as OpenAIAgent, Runner
from livekit.agents import (
    NotGivenOr,
    APIConnectOptions,
    FunctionTool,
    ChatContext,
    NOT_GIVEN,
    DEFAULT_API_CONNECT_OPTIONS
)
from livekit.agents.llm import LLM, LLMStream, ToolChoice, ChatChunk, ChoiceDelta
from livekit.agents.utils import shortuuid
from pyee.asyncio import AsyncIOEventEmitter

from .utils import extract_last_user_message, generate_context


class OpenAIAgentStream(LLMStream):
    def __init__(self,
                 llm: LLM,
                 chat_ctx: ChatContext,
                 response_future: Future,
                 tools: Optional[List[FunctionTool]] = None,
                 conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._response_future = response_future
        self.response_text: str = ""

    async def __aenter__(self):
        await super().__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await super().__aexit__(exc_type, exc_val, exc_tb)

    async def _run(self):
        response = await self._response_future
        raw_output = response.final_output

        self.response_text = str(raw_output) if raw_output is not None else ""
        
        stripped_content = self.response_text.strip()
        if stripped_content:  # Only send a chunk if there's actual content
            chunk = ChatChunk(
                id=shortuuid(),
                delta=ChoiceDelta(role="assistant", content=stripped_content)
            )
            self._event_ch.send_nowait(chunk)


class OpenAIAgentAdapter(LLM, AsyncIOEventEmitter):
    """
    Adapter to use an OpenAI Agents Agent with LiveKit.

    Args:
        orchestrator: The OpenAI Agents Agent instance to adapt.
        message_history: Optional initial list of messages (for warm start).
        context: Optional context to provide to the agent.
        last_n_messages: Number of most recent messages to keep in the running history.
    """
    def __init__(self, orchestrator: OpenAIAgent,
                 context: Optional[List[Dict[str, Any]]] = None,
                 last_n_messages: int = 20):
        super().__init__()
        self.orchestrator = orchestrator
        self.context: List[Dict[str, Any]] = context if context is not None else []
        self.last_n_messages = last_n_messages
    def chat(
            self,
            *,
            chat_ctx: ChatContext,
            tools: Optional[List[FunctionTool]] = None,
            conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
            parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
            tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
            extra_kwargs: NotGivenOr[Dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        user_message = extract_last_user_message(chat_ctx)
        generated_ctx_str = generate_context(chat_ctx.to_dict(), self.context, user_message)
        coro = Runner.run(self.orchestrator, generated_ctx_str)
        future = ensure_future(coro)

        return OpenAIAgentStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            response_future=future
        )

    async def generate(self, prompt: str, chat_ctx: Optional[ChatContext] = None) -> str:
        """
        Generates a response string from the orchestrator.
        """
        response = await Runner.run(self.orchestrator, prompt)
        raw_output = response.final_output
        return str(raw_output) if raw_output is not None else ""
