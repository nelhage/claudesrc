from functools import lru_cache
from typing import Annotated, Any, ClassVar, Literal, Type, TypeVar, cast, get_args

from anthropic.types import Usage
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetPydanticSchema,
    TypeAdapter,
)
from pydantic_core import core_schema
from typing_extensions import ReadOnly, TypedDict

from scrubs.store import ObjectID


class ModelObject(BaseModel):
    __pydantic_config__ = ConfigDict(frozen=True)

    object_type: ClassVar[str] = "model_ref"

    provider: str
    model: str


class ToolObject(BaseModel):
    __pydantic_config__ = ConfigDict(frozen=True)

    object_type: ClassVar[str] = "tool"

    name: str
    description: str
    input_schema: dict = Field(default_factory=dict)

    params: Any = None


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


class CreateMessageOptsObject(BaseModel):
    __pydantic_config__ = ConfigDict(frozen=True)

    object_type: ClassVar["str"] = "create_message_opts"

    model: ObjectID  # ModelRefObject

    max_tokens: int

    tools: list[ObjectID] = Field(default_factory=list)
    seed: int = 0
    system: list[ObjectID] = Field(default_factory=list)


class CreateMessageObject(BaseModel):
    __pydantic_config__ = ConfigDict(frozen=True)

    object_type: ClassVar["str"] = "create_message"

    opts: ObjectID  # CreateMessageOptsObject
    prompt: ObjectID  # PromptObject


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
    ModelObject
    | ToolObject
    | ContentObject
    | PromptObject
    | CreateMessageOptsObject
    | CreateMessageObject
    | ResponseObject
    | ToolUseObject
    | ToolResultObject
)

OBJECT_TYPES: dict[str, Type[ObjectType]] = {
    t.object_type: t for t in get_args(ObjectType)
}
