import json
from asyncio import ensure_future, Future
from typing import Any, Dict, List, Optional, Callable

from agents import Agent as OpenAIAgent, Runner, InputGuardrailTripwireTriggered
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
                 conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
                 guardrail_handler: Optional[Callable[[InputGuardrailTripwireTriggered, str], None]] = None):
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._response_future = response_future
        self.response_text: str = ""
        self.guardrail_handler = guardrail_handler

    async def __aenter__(self):
        await super().__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await super().__aexit__(exc_type, exc_val, exc_tb)

    async def _run(self):
        try:
            response = await self._response_future
            final_output = response.final_output
        except InputGuardrailTripwireTriggered as e:
            if self.guardrail_handler:
                final_output = self.guardrail_handler(e,json.dumps(self.chat_ctx.to_dict()))
            else:
                raise e

        raw_output = final_output

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
        guardrail_handler: Optional function to handle guardrail trips.
        context: Optional context to provide to the agent.
    """

    def __init__(self, orchestrator: OpenAIAgent,
                 guardrail_handler: Optional[Callable[[InputGuardrailTripwireTriggered, str], None]] = None,
                 context: Optional[List[Dict[str, Any]]] = None):
        super().__init__()
        self.orchestrator = orchestrator
        self.guardrail_handler = guardrail_handler
        self.context: List[Dict[str, Any]] = context if context is not None else []
        self.message_history: List[Dict[str, Any]] = []

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
        self.message_history = chat_ctx.to_dict()

        return OpenAIAgentStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            response_future=future,
            guardrail_handler=self.guardrail_handler
        )

    async def generate(self, prompt: str, chat_ctx: Optional[ChatContext] = None) -> str:
        """
        Generates a response string from the orchestrator.
        """
        response = await Runner.run(self.orchestrator, prompt)
        raw_output = response.final_output
        return str(raw_output) if raw_output is not None else ""

    async def get_message_history(self) -> List[Dict[str, Any]]:
        """
        Returns the message history of the orchestrator.
        """
        return self.message_history

    async def set_context(self, context: List[Dict[str, Any]]):
        """
        Sets the context of the orchestrator.
        """
        self.context = context
