from abc import abstractmethod
from typing import Protocol

from anthropic.types import ToolParam
from anthropic.types.tool_result_block_param import Content


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict

    @abstractmethod
    def call_tool(self, args) -> str | list[Content]: ...


def to_api_block(tool: Tool) -> ToolParam:
    return ToolParam(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
    )
