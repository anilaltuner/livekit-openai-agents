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

from .utils import extract_last_user_message


class OpenAIAgentStream(LLMStream):
    def __init__(self,
                 llm: LLM,
                 chat_ctx: ChatContext,
                 response_future: Future,
                 tools: List[FunctionTool] | None = None,
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
        # Fetch the orchestrator's result when _run() is called
        response = await self._response_future
        self.response_text = response.final_output

        if self.response_text and self.response_text.strip():
            chunk = ChatChunk(
                id=shortuuid(),
                delta=ChoiceDelta(role="assistant", content=self.response_text.strip())
            )
            self._event_ch.send_nowait(chunk)
        else:
            pass


class OpenAIAgentAdapter(LLM, AsyncIOEventEmitter):
    def __init__(self, orchestrator: OpenAIAgent):
        super().__init__()
        self.orchestrator = orchestrator

    def chat(
            self,
            *,
            chat_ctx: ChatContext,
            tools: List[FunctionTool] | None = None,
            conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
            parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
            tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
            extra_kwargs: NotGivenOr[Dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        user_message = extract_last_user_message(chat_ctx)
        coro = Runner.run(self.orchestrator, user_message)
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
        This method also stores the full response (including metadata)
        in `self.last_response_data`.
        """

        response = await Runner.run(self.orchestrator, prompt)
        return response.final_output
