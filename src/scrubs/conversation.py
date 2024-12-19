from typing import Iterable, Literal

import anthropic
from anthropic.types import (
    ContentBlock,
    MessageParam,
    ModelParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
)
from pydantic import BaseModel, TypeAdapter

from scrubs import tool
from scrubs.types import ignore_type

from .cache import Cache
from .objects import (
    DEFAULT_MAX_TOKENS,
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

CONTENT_ADAPTER = TypeAdapter(list[ContentBlock])

RoleType = Literal["user", "assistant"]


class MessageTurn(BaseModel):
    role: RoleType
    content: str | dict


def insert_tool(cache: Cache, tool: tool.Tool) -> ObjectID:
    return cache.insert(
        ToolObject(
            name=tool.name,
            cache_params=tool.cache_params(),
            input_schema=tool.input_schema,
        )
    )


class Conversation:
    def __init__(
        self,
        cache: Cache,
        client: anthropic.Client,
        *,
        seed: int = 0,
        tools: list[tool.Tool] = [],
        model: ModelParam = "claude-3-5-sonnet-latest",
        system_prompt: list[str] = [],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.cache = cache
        self.client = client
        self.system_prompt: tuple[TextBlockParam] = tuple(
            TextBlockParam(type="text", text=p) for p in system_prompt
        )
        self.seed = seed
        self.max_tokens = max_tokens

        self.model = self.cache.insert(
            ModelOptsObject(
                model=model,
                system=[
                    self.cache.insert(ContentObject(content=m)) for m in system_prompt
                ],
                tools=[insert_tool(cache, t) for t in tools],
            )
        )

        self.tools = {t.name: t for t in tools}

        self.turns: list[MessageParam] = []
        self.prompt: ObjectID | None = None

    def user_prompt(self, prompt, seed: int | None = None) -> Iterable[MessageTurn]:
        self.append_user(prompt)

        return self.pump(seed)

    def append_user(self, prompt):
        self.append_turn(
            role="user",
            content=self.cache.insert(ContentObject(content=prompt)),
        )

    def append_turn(self, role: RoleType, content: ObjectID) -> MessageTurn:
        self.prompt = self.cache.insert(
            PromptObject(
                prefix=self.prompt,
                message=MessageObject(role=role, content=content),
            )
        )

        block = self.cache.get_content(content).content
        if isinstance(block, dict):
            turn = MessageParam(role=role, content=[ignore_type(block)])
        else:
            turn = MessageParam(role=role, content=block)

        self.turns.append(turn)
        return MessageTurn(role=role, content=block)

    def pump(self, seed: int | None = None) -> Iterable[MessageTurn]:
        if seed is None:
            seed = self.seed

        if self.prompt is None:
            return

        while True:
            last = self.cache.get_prompt(self.prompt).message

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

        reply_obj = self.cache.get_cache(self.cache.insert(create))
        if reply_obj is not None:
            reply = self.cache.get_response(reply_obj)
        else:
            reply = self._send_api(create)

        for content in reply.content:
            yield self.append_turn("assistant", content)

    def _send_api(self, create: CreateMessageObject) -> ResponseObject:
        create_id = self.cache.insert(create)
        model = self.cache.get_model_opts(create.model)

        # TODO: re-serialize the turns and confirm consistency?

        reply = self.client.messages.create(
            messages=self.turns,
            model=model.model,
            system=self.system_prompt,
            tools=[to_api_block(tool) for tool in self.tools.values()],
            max_tokens=create.max_tokens,
        )

        content = [
            self.cache.insert(ContentObject(content=c.model_dump()))
            for c in reply.content
        ]

        response = ResponseObject(
            request=create_id,
            content=content,
            id=reply.id,
            stop_reason=reply.stop_reason,
            usage=reply.usage,
        )

        self.cache.put_cache(create_id, self.cache.insert(response))

        return response

    def _maybe_use_tool(self) -> MessageTurn | None:
        if self.prompt is None:
            return None

        last = self.cache.get_prompt(self.prompt).message
        message = self.cache.get_content(last.content).content

        if not isinstance(message, dict):
            return None

        if message["type"] != "tool_use":
            return

        tool_id = insert_tool(self.cache, self.tools[message["name"]])
        assert tool_id in self.cache.get_model_opts(self.model).tools

        tool_use = ToolUseObject(
            tool=tool_id,
            id=message["id"],
            input=message["input"],
        )

        tool_use_oid = self.cache.insert(tool_use)

        result = self.cache.get_cache(tool_use_oid)
        if result is not None:
            result = self.cache.get_tool_result(result)
        else:
            tool = self.tools[self.cache.get_tool(tool_use.tool).name]
            result_content = tool.call_tool(tool_use.input)
            if not isinstance(result_content, list):
                result_content = [dict(type="text", text=result_content)]

            content = [
                self.cache.insert(ContentObject(content=ignore_type(c)))
                for c in result_content
            ]

            result = ToolResultObject(
                tool_use=tool_use_oid,
                response=content,
            )
            self.cache.put_cache(tool_use_oid, self.cache.insert(result))

        content = self.cache.insert(
            ContentObject(
                content=ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id=tool_use.id,
                    content=[
                        ignore_type(self.cache.get_content(c).content)
                        for c in result.response
                    ],
                    is_error=False,
                )
            )
        )

        return self.append_turn(role="user", content=content)
