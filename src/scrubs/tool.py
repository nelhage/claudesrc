from abc import abstractmethod
from typing import Any, ClassVar, Generic, Protocol, TypeVar

from anthropic.types import ToolParam
from anthropic.types.tool_result_block_param import Content
from pydantic import BaseModel


class Tool(Protocol):
    name: str
    description: str

    @property
    def input_schema(self) -> dict: ...

    @abstractmethod
    def cache_params(self) -> Any: ...

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
    def cache_params(self) -> Any:
        """Return parameters needed to reconstruct this tool instance."""
        pass

    @abstractmethod
    def call(self, params: ParamsT) -> str | list[Any]:
        """Execute the tool with validated parameters."""
        pass

    def call_tool(self, args: dict) -> str | list[Any]:
        """Implementation of Tool protocol method."""
        params = self.Params.model_validate(args)
        return self.call(params)


def to_api_block(tool: Tool) -> ToolParam:
    return ToolParam(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
    )
