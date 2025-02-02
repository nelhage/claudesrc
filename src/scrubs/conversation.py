from typing import Iterable, Literal, cast

import anthropic
from anthropic.types import (
    DocumentBlockParam,
    ImageBlockParam,
    MessageParam,
    ModelParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from pydantic import BaseModel

from scrubs import tool

from .context import Context
from .objects import (
    DEFAULT_MAX_TOKENS,
    ContentDict,
    ContentObject,
    CreateMessageObject,
    MessageObject,
    ModelOptsObject,
    PromptObject,
    ResponseObject,
    ToolObject,
    ToolResultObject,
    ToolUseObject,
)
from .store import ObjectID
from .tool import to_api_block

RoleType = Literal["user", "assistant"]


class MessageTurn(BaseModel):
    role: RoleType
    content: ContentDict


BlockParam = (
    TextBlockParam
    | ImageBlockParam
    | ToolUseBlockParam
    | ToolResultBlockParam
    | DocumentBlockParam
)


def insert_tool(ctx: Context, tool: tool.Tool) -> ObjectID:
    return ctx.insert(
        ToolObject(
            name=tool.name,
            cache_params=tool.cache_params(),
            input_schema=tool.input_schema,
        )
    )


def flatten_prompt(ctx: Context, prompt: ObjectID | None) -> list[MessageParam]:
    """Flatten a prompt object in order to feed it to the API.

    Args:
        ctx: Context object for accessing stored objects
        prompt: ObjectID of the prompt to flatten, or None

    Returns:
        List of MessageParam objects ready for the API in chronological order
    """
    messages: list[MessageParam] = []
    current_prompt = prompt

    # Build list in reverse order (most recent first)
    while current_prompt is not None:
        prompt_obj = ctx.get_prompt(current_prompt)

        # Get content for current message
        message = prompt_obj.message
        content_obj = ctx.get_content(message.content)
        content = content_obj.to_dict()

        # Add message to list
        messages.append(
            {
                "role": message.role,
                "content": [content],  # type:ignore
            }
        )

        # Move to prefix
        current_prompt = prompt_obj.prefix

    # Reverse to get chronological order
    messages.reverse()
    return messages


class Conversation:
    def __init__(
        self,
        ctx: Context,
        client: anthropic.Client,
        *,
        seed: int = 0,
        tools: list[tool.Tool] = [],
        model: ModelParam = "claude-3-5-sonnet-latest",
        system_prompt: Iterable[str | TextBlockParam] = [],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.ctx = ctx
        self.client = client
        self.system_prompt: tuple[ContentObject, ...] = tuple(
            ContentObject.from_api(p) for p in system_prompt
        )
        self.seed = seed
        self.max_tokens = max_tokens

        self.model = self.ctx.insert(
            ModelOptsObject(
                model=model,
                system=[self.ctx.insert(p) for p in self.system_prompt],
                tools=[insert_tool(ctx, t) for t in tools],
            )
        )

        self.tools = {t.name: t for t in tools}

        self.turns: list[MessageParam] = []
        self.prompt: ObjectID | None = None

    def append_user(self, prompt: str | TextBlockParam):
        self.append_turn(
            role="user",
            content=self.ctx.insert(ContentObject.from_api(prompt)),
        )

    def append_turn(self, role: RoleType, content: ObjectID) -> MessageTurn:
        self.prompt = self.ctx.insert(
            PromptObject(
                prefix=self.prompt,
                message=MessageObject(role=role, content=content),
            )
        )

        block = self.ctx.get_content(content).to_dict()
        turn = MessageParam(role=role, content=[cast(BlockParam, block)])

        self.turns.append(turn)
        return MessageTurn(role=role, content=block)

    def pump(self, seed: int | None = None) -> Iterable[MessageTurn]:
        if seed is None:
            seed = self.seed

        if self.prompt is None:
            return

        while True:
            last = self.ctx.get_prompt(self.prompt).message

            if last.role == "user":
                yield from self._send_user(seed)
            else:
                turn = self._maybe_use_tool()
                if not turn:
                    return
                yield turn

    def _send_user(self, seed: int) -> Iterable[MessageTurn]:
        assert self.prompt is not None

        create = CreateMessageObject(
            model=self.model, prompt=self.prompt, seed=seed, max_tokens=self.max_tokens
        )

        reply_obj = self.ctx.get_cache(self.ctx.insert(create))
        if reply_obj is not None:
            reply = self.ctx.get_response(reply_obj)
        else:
            reply = self._send_api(create)

        for content in reply.content:
            yield self.append_turn("assistant", content)

    def _send_api(self, create: CreateMessageObject) -> ResponseObject:
        create_id = self.ctx.insert(create)
        model = self.ctx.get_model_opts(create.model)

        # TODO: re-serialize the turns and confirm consistency?

        reply = self.client.messages.create(
            messages=self.turns,
            model=model.model,
            system=[p.to_api(TextBlockParam) for p in self.system_prompt],
            tools=[to_api_block(tool) for tool in self.tools.values()],
            max_tokens=create.max_tokens,
        )

        content = [
            self.ctx.insert(ContentObject.from_api(c.model_dump()))
            for c in reply.content
        ]

        response = ResponseObject(
            request=create_id,
            content=content,
            id=reply.id,
            stop_reason=reply.stop_reason,
            usage=reply.usage,
        )

        self.ctx.put_cache(create_id, self.ctx.insert(response))

        return response

    def _maybe_use_tool(self) -> MessageTurn | None:
        if self.prompt is None:
            return None

        last = self.ctx.get_prompt(self.prompt).message
        message = self.ctx.get_content(last.content).to_dict()

        if message["type"] != "tool_use":
            return

        message = cast(ToolUseBlockParam, message)

        tool_id = insert_tool(self.ctx, self.tools[message["name"]])
        assert tool_id in self.ctx.get_model_opts(self.model).tools

        tool_use = ToolUseObject(
            tool=tool_id,
            id=message["id"],
            input=message["input"],  # type: ignore
        )

        tool_use_oid = self.ctx.insert(tool_use)

        result = self.ctx.get_cache(tool_use_oid)
        if result is not None:
            result = self.ctx.get_tool_result(result)
        else:
            tool = self.tools[self.ctx.get_tool(tool_use.tool).name]
            result_content = tool.call_tool(tool_use.input)
            if not isinstance(result_content, list):
                result_content = [result_content]

            content = [
                self.ctx.insert(ContentObject.from_api(c)) for c in result_content
            ]

            result = ToolResultObject(
                tool_use=tool_use_oid,
                response=content,
            )
            self.ctx.put_cache(tool_use_oid, self.ctx.insert(result))

        content = self.ctx.insert(
            ContentObject.from_api(
                ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id=tool_use.id,
                    content=[
                        self.ctx.get_content(c).to_api(TextBlockParam)
                        for c in result.response
                    ],
                    is_error=False,
                )
            )
        )

        return self.append_turn(role="user", content=content)
