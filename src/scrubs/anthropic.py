from typing import Iterable, Literal, cast

from pydantic import BaseModel

import anthropic
from anthropic import Client
from anthropic.types import (
    DocumentBlockParam,
    ImageBlockParam,
    MessageParam,
    ModelParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from scrubs import tool
from scrubs.llm import LLMInterface
from scrubs.models import MODEL_ALIASES

from .context import Context, flatten_prompt
from .objects import (
    ContentDict,
    ContentObject,
    CreateMessageObject,
    MessageObject,
    ModelObject,
    PromptObject,
    ResponseObject,
    ToolObject,
    ToolResultObject,
    ToolUseObject,
)
from .store import ObjectID
from .tool import Tool


def tool_to_api(ctx: Context, tool_id: ObjectID) -> ToolParam:
    tool = ctx.tool(tool_id)
    return ToolParam(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
    )


class AnthropicModel(LLMInterface):
    def __init__(
        self,
        client: Client,
        model: ModelParam,
    ):
        self.client = client
        self.model = ModelObject(
            provider="anthropic",
            model=MODEL_ALIASES[model],
        )

    def create_message(
        self,
        ctx: Context,
        create: CreateMessageObject,
    ) -> ResponseObject:
        opts = ctx.create_opts(create.opts)
        assert opts.model == ctx.insert(self.model)

        messages = flatten_prompt(ctx, create.prompt)

        reply = self.client.messages.create(
            messages=messages,
            model=self.model.model,
            system=[
                cast(TextBlockParam, ctx.content(p).to_dict()) for p in opts.system
            ],
            tools=[tool_to_api(ctx, tool) for tool in opts.tools or ()],
            max_tokens=opts.max_tokens,
        )

        content = [
            ctx.insert(ContentObject.from_api(c.model_dump())) for c in reply.content
        ]

        return ResponseObject(
            request=ctx.insert(create),
            content=content,
            id=reply.id,
            stop_reason=reply.stop_reason,
            usage=reply.usage,
        )
