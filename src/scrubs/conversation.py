from typing import Iterable, Literal, cast

from pydantic import BaseModel

from anthropic.types import (
    DocumentBlockParam,
    ImageBlockParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from scrubs import tool
from scrubs.llm import LLMInterface

from .context import Context
from .objects import (
    ContentDict,
    ContentObject,
    CreateMessageObject,
    CreateMessageOptsObject,
    MessageObject,
    PromptObject,
    ResponseObject,
    ToolResultObject,
    ToolUseObject,
)
from .store import ObjectID
from .tool import as_tool_object

RoleType = Literal["user", "assistant"]
DEFAULT_MAX_TOKENS = 1024


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


class Conversation:
    def __init__(
        self,
        ctx: Context,
        model: LLMInterface,
        *,
        seed: int = 0,
        tools: dict[str, tool.Tool] = {},
        system_prompt: Iterable[str | TextBlockParam] = [],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.ctx = ctx
        self.model = model
        self.seed = seed
        self.max_tokens = max_tokens

        self.tools = tools
        self.prompt: ObjectID | None = None
        self.system_prompt: tuple[ObjectID, ...] = tuple(
            ctx.insert(ContentObject.from_api(p)) for p in system_prompt
        )

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

        block = self.ctx.content(content).to_dict()
        return MessageTurn(role=role, content=block)

    def pump(self, seed: int | None = None) -> Iterable[MessageTurn]:
        if seed is None:
            seed = self.seed

        if self.prompt is None:
            return

        while True:
            last = self.ctx.prompt(self.prompt).message

            if last.role == "user":
                yield from self._send_user(seed)
            else:
                turn = self._maybe_use_tool()
                if not turn:
                    return
                yield turn

    def _send_user(self, seed: int) -> Iterable[MessageTurn]:
        assert self.prompt is not None

        opts = self.ctx.insert(
            CreateMessageOptsObject(
                model=self.ctx.insert(self.model.model),
                seed=seed,
                max_tokens=self.max_tokens,
                tools=[self.ctx.insert(as_tool_object(t)) for t in self.tools.values()],
                system=list(self.system_prompt),
            )
        )

        create = CreateMessageObject(
            opts=opts,
            prompt=self.prompt,
        )

        reply_obj = self.ctx.get_cache(self.ctx.insert(create))

        if reply_obj is not None:
            reply = self.ctx.response(reply_obj)
        else:
            reply = self._send_api(create)

        for content in reply.content:
            yield self.append_turn("assistant", content)

    def _send_api(self, create: CreateMessageObject) -> ResponseObject:
        response = self.model.create_message(self.ctx, create)

        self.ctx.put_cache(self.ctx.insert(create), self.ctx.insert(response))

        return response

    def _maybe_use_tool(self) -> MessageTurn | None:
        if self.prompt is None:
            return None

        last = self.ctx.prompt(self.prompt).message
        message = self.ctx.content(last.content).to_dict()

        if message["type"] != "tool_use":
            return None

        message = cast(ToolUseBlockParam, message)

        tool_id = self.ctx.insert(as_tool_object(self.tools[message["name"]]))

        tool_use = ToolUseObject(
            tool=tool_id,
            id=message["id"],
            input=message["input"],  # type: ignore
        )

        tool_use_oid = self.ctx.insert(tool_use)

        result = self.ctx.get_cache(tool_use_oid)
        if result is not None:
            result = self.ctx.tool_result(result)
        else:
            tool = self.tools[self.ctx.tool(tool_use.tool).name]
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
                        self.ctx.content(c).to_api(TextBlockParam)
                        for c in result.response
                    ],
                    is_error=False,
                )
            )
        )

        return self.append_turn(role="user", content=content)
