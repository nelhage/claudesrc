import anthropic
from anthropic.types import (
    ContentBlock,
    MessageParam,
    ToolResultBlockParam,
    ToolUseBlock,
)
from pydantic import TypeAdapter

from claudesrc import tool
from claudesrc.tool import to_api_block

DEFAULT_MAX_TOKENS = 1024

CONTENT_ADAPTER = TypeAdapter(list[ContentBlock])


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
        self.turns: list[MessageParam] = []

    def user_prompt(self, prompt):
        self.turns.append(
            MessageParam(
                role="user",
                content=prompt,
            ),
        )

        while True:
            message = self.client.messages.create(
                **self.create_kwargs,
                messages=self.turns,
            )
            self.turns.append(
                MessageParam(
                    role=message.role,
                    content=CONTENT_ADAPTER.dump_python(message.content),
                )
            )

            last = message.content[-1]
            if isinstance(last, ToolUseBlock):
                print(f"Tool use: {last.name}: {last.input}")
                tool = self.tools[last.name]
                response = tool.call_tool(last.input)
                self.turns.append(
                    MessageParam(
                        role="user",
                        content=[
                            ToolResultBlockParam(
                                tool_use_id=last.id,
                                type="tool_result",
                                content=response,
                                is_error=False,
                            )
                        ],
                    )
                )

                continue

            return message
