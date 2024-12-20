from typing import Any, ClassVar, Literal, Type, get_args

from anthropic.types import Usage
from pydantic import BaseModel, Field

from scrubs.store import ObjectID

DEFAULT_MAX_TOKENS = 1024


class ModelOptsObject(BaseModel):
    object_type: ClassVar[str] = "model_opts"

    model: str
    metadata: dict | None = None
    system: list[ObjectID] = Field(default_factory=list)
    tools: list[ObjectID] = Field(default_factory=list)


class ToolObject(BaseModel):
    object_type: ClassVar[str] = "tool"

    name: str
    input_schema: dict = Field(default_factory=dict)
    cache_params: Any = Field(default_factory=dict)


class ContentObject(BaseModel):
    object_type: ClassVar[str] = "content"
    content: str | dict[str, Any]


class MessageObject(BaseModel):
    content: ObjectID
    role: Literal["user", "assistant"]


class PromptObject(BaseModel):
    object_type: ClassVar[str] = "prompt"

    message: MessageObject
    prefix: ObjectID | None = None


class CreateMessageObject(BaseModel):
    object_type: ClassVar["str"] = "create_message"

    model: ObjectID
    prompt: ObjectID

    seed: int = 0
    max_tokens: int = DEFAULT_MAX_TOKENS


class ResponseObject(BaseModel):
    object_type: ClassVar["str"] = "message_response"
    request: ObjectID  # CreateMessageObject
    content: list[ObjectID]  # ContentObject
    id: str
    stop_reason: str | None
    usage: Usage


class ToolUseObject(BaseModel):
    object_type: ClassVar["str"] = "tool_use"

    id: str
    tool: ObjectID  # ToolObject
    input: dict


class ToolResultObject(BaseModel):
    object_type: ClassVar["str"] = "tool_result"

    tool_use: ObjectID  # ToolUseObject
    response: list[ObjectID]


ObjectType = (
    ModelOptsObject
    | ToolObject
    | ContentObject
    | PromptObject
    | CreateMessageObject
    | ResponseObject
    | ToolUseObject
    | ToolResultObject
)

OBJECT_TYPES: dict[str, Type[ObjectType]] = {
    t.object_type: t for t in get_args(ObjectType)
}
