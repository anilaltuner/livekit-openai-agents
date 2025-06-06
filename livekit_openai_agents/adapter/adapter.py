import json
from asyncio import ensure_future, Future
from typing import Any, Dict, List, Optional, Callable

from agents import Agent as OpenAIAgent, Runner, InputGuardrailTripwireTriggered, RunResultStreaming
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
from openai.types.responses import ResponseTextDeltaEvent

from .utils import extract_last_user_message, generate_context


class OpenAIAgentStream(LLMStream):
    def __init__(self,
                 llm: LLM,
                 chat_ctx: ChatContext,
                 response_future: Optional[Future] = None,
                 streaming_result: Optional[RunResultStreaming] = None,
                 tools: Optional[List[FunctionTool]] = None,
                 conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
                 guardrail_handler: Optional[Callable[[InputGuardrailTripwireTriggered, str], None]] = None,
                 is_streaming: bool = False):
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._response_future = response_future
        self._streaming_result = streaming_result
        self._is_streaming = is_streaming
        self.response_text: str = ""
        self.guardrail_handler = guardrail_handler
        self._accumulated_content: List[str] = []

    async def __aenter__(self):
        await super().__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await super().__aexit__(exc_type, exc_val, exc_tb)

    async def _run(self):
        try:
            if self._is_streaming and self._streaming_result:
                async for event in self._streaming_result.stream_events():
                    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                        delta_text = event.data.delta
                        if delta_text:
                            self._accumulated_content.append(delta_text)

                            chunk = ChatChunk(
                                id=shortuuid(),
                                delta=ChoiceDelta(role="assistant", content=delta_text)
                            )
                            self._event_ch.send_nowait(chunk)


                self.response_text = "".join(self._accumulated_content)
            else:
                response = await self._response_future
                final_output = response.final_output
                raw_output = final_output

                self.response_text = str(raw_output) if raw_output is not None else ""

                stripped_content = self.response_text.strip()
                if stripped_content:  # Only send a chunk if there's actual content
                    chunk = ChatChunk(
                        id=shortuuid(),
                        delta=ChoiceDelta(role="assistant", content=stripped_content)
                    )
                    self._event_ch.send_nowait(chunk)

        except InputGuardrailTripwireTriggered as e:
            if self.guardrail_handler:
                final_output = self.guardrail_handler(e, json.dumps(self.chat_ctx.to_dict()))
                if final_output:
                    self.response_text = str(final_output)
                    chunk = ChatChunk(
                        id=shortuuid(),
                        delta=ChoiceDelta(role="assistant", content=self.response_text)
                    )
                    self._event_ch.send_nowait(chunk)
            else:
                raise e
        except Exception as e:
            raise e


class OpenAIAgentAdapter(LLM, AsyncIOEventEmitter):
    """
    Adapter to use an OpenAI Agents Agent with LiveKit.

    Args:
        orchestrator: The OpenAI Agents Agent instance to adapt.
        guardrail_handler: Optional function to handle guardrail trips.
        context: Optional context to provide to the agent.
        streaming: Whether to enable streaming responses. Defaults to False (non-streaming).
    """

    def __init__(self, orchestrator: OpenAIAgent,
                 guardrail_handler: Optional[Callable[[InputGuardrailTripwireTriggered, str], None]] = None,
                 context: Optional[List[Dict[str, Any]]] = None,
                 streaming: bool = False):
        super().__init__()
        self.orchestrator = orchestrator
        self.guardrail_handler = guardrail_handler
        self.context: List[Dict[str, Any]] = context if context is not None else []
        self.message_history: List[Dict[str, Any]] = []
        self.streaming = streaming

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
        
        if self.streaming:
            # Use streaming approach
            streaming_result = Runner.run_streamed(self.orchestrator, generated_ctx_str)
            self.message_history = chat_ctx.to_dict()

            return OpenAIAgentStream(
                self,
                chat_ctx=chat_ctx,
                tools=tools,
                conn_options=conn_options,
                streaming_result=streaming_result,
                guardrail_handler=self.guardrail_handler,
                is_streaming=True
            )
        else:
            # Use original non-streaming approach
            coro = Runner.run(self.orchestrator, generated_ctx_str)
            future = ensure_future(coro)
            self.message_history = chat_ctx.to_dict()

            return OpenAIAgentStream(
                self,
                chat_ctx=chat_ctx,
                tools=tools,
                conn_options=conn_options,
                response_future=future,
                guardrail_handler=self.guardrail_handler,
                is_streaming=False
            )

    async def generate(self, prompt: str, chat_ctx: Optional[ChatContext] = None) -> str:
        """
        Generates a response string from the orchestrator.
        This method always uses non-streaming mode regardless of the streaming parameter.
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

    def set_streaming(self, streaming: bool):
        """
        Sets the streaming mode for future chat calls.
        
        Args:
            streaming: Whether to enable streaming responses.
        """
        self.streaming = streaming

    def is_streaming_enabled(self) -> bool:
        """
        Returns whether streaming is currently enabled.
        
        Returns:
            True if streaming is enabled, False otherwise.
        """
        return self.streaming