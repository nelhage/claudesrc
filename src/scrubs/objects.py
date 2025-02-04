from functools import lru_cache
from typing import Annotated, Any, ClassVar, Literal, Type, TypeVar, cast, get_args

from anthropic.types import Usage
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetPydanticSchema,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    model_serializer,
)
from pydantic_core import core_schema
from typing_extensions import ReadOnly, TypedDict

from scrubs.store import ObjectID

DEFAULT_MAX_TOKENS = 1024


class ModelOptsObject(BaseModel):
    __pydantic_config__ = ConfigDict(frozen=True)

    object_type: ClassVar[str] = "model_opts"

    model: str
    metadata: dict | None = None
    system: list[ObjectID] = Field(default_factory=list)
    tools: list[ObjectID] | None = None


class ToolObject(BaseModel):
    __pydantic_config__ = ConfigDict(frozen=True)

    object_type: ClassVar[str] = "tool"

    name: str
    input_schema: dict = Field(default_factory=dict)
    cache_params: Any = Field(default_factory=dict)


class ContentDict(TypedDict):
    __pydantic_config__ = ConfigDict(extra="allow", frozen=True)  # type: ignore

    type: Annotated[
        ReadOnly[str],
        GetPydanticSchema(
            lambda tp, handler: core_schema.no_info_after_validator_function(
                str, handler(str)
            )
        ),
    ]


C = TypeVar("C", bound=ContentDict)

cached_adapter = lru_cache(maxsize=16)(TypeAdapter)


class ContentObject(BaseModel):
    __pydantic_config__ = ConfigDict(frozen=True)

    object_type: ClassVar[str] = "content"

    type: str
    fields: dict[str, Any]

    @classmethod
    def from_api(cls, obj: str | ContentDict | dict[str, Any]) -> "ContentObject":
        if isinstance(obj, str):
            return cls(type="text", fields=dict(text=obj))
        fields = dict(obj)
        type = cast(str, fields.pop("type"))
        return cls(type=type, fields=fields)

    def to_dict(self) -> ContentDict:
        return ContentDict(type=self.type, **self.fields)

    def to_api(self, C: Type[C]) -> C:
        return cached_adapter(C).validate_python(self.to_dict())


class MessageObject(BaseModel):
    __pydantic_config__ = ConfigDict(frozen=True)

    content: ObjectID
    role: Literal["user", "assistant"]


class PromptObject(BaseModel):
    __pydantic_config__ = ConfigDict(frozen=True)

    object_type: ClassVar[str] = "prompt"

    message: MessageObject
    prefix: ObjectID | None = None


class CreateMessageObject(BaseModel):
    __pydantic_config__ = ConfigDict(frozen=True)

    object_type: ClassVar["str"] = "create_message"

    model: ObjectID
    prompt: ObjectID

    tools: list[ObjectID] | None = Field(default=None)

    seed: int = 0
    max_tokens: int = DEFAULT_MAX_TOKENS

    @model_serializer(mode="wrap")
    def _serializer(self, nxt: SerializerFunctionWrapHandler, info: SerializationInfo):
        orig = nxt(self)
        if orig.get("tools") is None:
            orig.pop("tools")
        return orig


class ResponseObject(BaseModel):
    __pydantic_config__ = ConfigDict(frozen=True)

    object_type: ClassVar["str"] = "message_response"
    request: ObjectID  # CreateMessageObject
    content: list[ObjectID]  # ContentObject
    id: str
    stop_reason: str | None
    usage: Usage


class ToolUseObject(BaseModel):
    __pydantic_config__ = ConfigDict(frozen=True)

    object_type: ClassVar["str"] = "tool_use"

    id: str
    tool: ObjectID  # ToolObject
    input: dict


class ToolResultObject(BaseModel):
    __pydantic_config__ = ConfigDict(frozen=True)

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
