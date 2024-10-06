from abc import abstractmethod
from typing import Protocol

from anthropic.types import ImageBlockParam, TextBlock, ToolParam


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict

    @abstractmethod
    def call_tool(self, args) -> str | list[TextBlock | ImageBlockParam]: ...


def to_api_block(tool: Tool) -> ToolParam:
    return ToolParam(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
    )
