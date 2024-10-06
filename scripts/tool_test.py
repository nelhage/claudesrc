from textwrap import dedent

import anthropic
from claudesrc import anthropic_api_key, models
from claudesrc.tool import Tool, to_api_block
from pydantic import BaseModel, Field


class TestTool(Tool):
    class Params(BaseModel):
        response: str = Field(description="Your response acknowledging this tool")
        favorite_color: str | None = Field(
            description="Your favorite color, should you choose to respond"
        )

    name = "test_tool"
    description = dedent("""\
    A test tool for trying out the Anthropic API.

    Call this tool with the "response" field set to an acknowledgement
    that you understand how to use the tool, and optionally with your
    favorite color.
    """)

    input_schema = Params.model_json_schema()

    def call_tool(self, args) -> str:
        return "I gotcha!"


def main():
    tool = TestTool()

    client = anthropic.Client(api_key=anthropic_api_key())
    message = client.messages.create(
        # model="claude-3-5-sonnet-20240620",
        model=models.HAIKU,
        max_tokens=1024,
        tools=[to_api_block(tool)],
        messages=[
            {
                "role": "user",
                "content": dedent("""\
            Hello, Claude. I'm trying to learn how you use tools. Can
            you please let me know if I did this right by using the
            tool I've provided?
            """),
            },
        ],
    )
    print(message.content)


if __name__ == "__main__":
    main()
