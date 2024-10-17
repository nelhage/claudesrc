from typing import Iterable, Literal

import anthropic
from anthropic.types import (
    ContentBlock,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
)
from pydantic import BaseModel, TypeAdapter

from claudesrc import tool
from claudesrc.tool import to_api_block

DEFAULT_MAX_TOKENS = 1024

CONTENT_ADAPTER = TypeAdapter(list[ContentBlock])


class MessageTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[TextBlockParam | ToolUseBlockParam | ToolResultBlockParam]


class Conversation:
    def __init__(
        self,
        client: anthropic.Client,
        *,
        tools: list[tool.Tool] = [],
        **create_kwargs,
    ):
        assert "messages" not in create_kwargs
        self.client = client
        self.create_kwargs = create_kwargs | dict(
            tools=[to_api_block(tool) for tool in tools]
        )
        self.create_kwargs.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
        self.tools = {t.name: t for t in tools}
        self.turns: list[MessageTurn] = []

    def user_prompt(self, prompt) -> Iterable[MessageTurn]:
        self.append_user(prompt)

        return self.pump()

    def append_user(self, prompt):
        self.turns.append(
            MessageTurn(
                role="user",
                content=prompt,
            ),
        )

    def pump(self) -> Iterable[MessageTurn]:
        if len(self.turns) == 0:
            return

        while True:
            last = self.turns[-1]
            if last.role == "user":
                yield self._send_one()
            else:
                turn = self._maybe_use_tool()
                if not turn:
                    return
                yield turn

    def _send_one(self) -> MessageTurn:
        message = self.client.messages.create(
            **self.create_kwargs,
            messages=[t.model_dump() for t in self.turns],  # type: ignore
        )
        turn = MessageTurn(
            role=message.role,
            content=CONTENT_ADAPTER.dump_python(message.content),
        )
        self.turns.append(turn)
        return turn

    def _maybe_use_tool(self) -> MessageTurn | None:
        last = self.turns[-1].content[-1]

        if isinstance(last, str):
            return
        if last["type"] != "tool_use":
            return

        tool = self.tools[last["name"]]
        response = tool.call_tool(last["input"])
        turn = MessageTurn(
            role="user",
            content=[
                ToolResultBlockParam(
                    tool_use_id=last["id"],
                    type="tool_result",
                    content=response,
                    is_error=False,
                )
            ],
        )
        self.turns.append(turn)
        return turn
