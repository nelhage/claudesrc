from abc import abstractmethod
from typing import Any, Generic, Protocol, TypeVar

from anthropic.types.tool_result_block_param import Content
from pydantic import BaseModel

from scrubs.objects import ToolObject


class Tool(Protocol):
    name: str
    description: str

    @property
    def input_schema(self) -> dict: ...

    @abstractmethod
    def serialize_params(self) -> Any: ...

    @abstractmethod
    def call_tool(self, args) -> str | list[Content]: ...


ParamsT = TypeVar("ParamsT", bound=BaseModel)


class PydanticTool(Tool, Generic[ParamsT]):
    """
    Base class for tools that use Pydantic models to define their input schema.
    """

    # This should be ClassVar[type[ParamsT]], but Python doesn't allow
    # that. This should work fine.
    Params: type[ParamsT]

    @property
    def input_schema(self) -> dict:
        return self.Params.model_json_schema()

    @abstractmethod
    def serialize_params(self) -> Any:
        pass

    @abstractmethod
    def call(self, params: ParamsT) -> str | list[Any]:
        """Execute the tool with validated parameters."""
        pass

    def call_tool(self, args: dict) -> str | list[Any]:
        """Implementation of Tool protocol method."""
        params = self.Params.model_validate(args)
        return self.call(params)


def as_tool_object(tool: Tool) -> ToolObject:
    return ToolObject(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
        params=tool.serialize_params(),
    )
