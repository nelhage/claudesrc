import os
import re
import subprocess
from pathlib import Path
from typing import cast

from anthropic.types import (
    ToolResultBlockParam,
    ToolUseBlockParam,
)

from scrubs.context import Context, walk_prompt_chain
from scrubs.conversation import Conversation
from scrubs.objects import MessageObject
from scrubs.store import ObjectID

USER_SEPARATOR = "# Respond below this line. Delete this header to exit\n"

FILE_CONTENT = re.compile(
    r"""
(?P<header><file [^\n]+>) \n
(?P<content>.*)
(?P<footer></file>) $
""",
    re.M | re.S | re.X,
)


def format_text(role, text) -> str:
    if role != "user":
        return text

    m = FILE_CONTENT.search(text)
    if m is None:
        return text

    header = m.group("header")
    footer = m.group("footer")
    nlines = m.group("content").count("\n")

    return "\n".join([header, f"[{nlines} lines]", footer])


def render_content(ctx: Context, fh, role: str, content_id: ObjectID):
    content = ctx.content(content_id)
    content_dict = content.to_dict()

    match content_dict["type"]:
        case "text":
            text = content_dict.get("text", "")
            print(format_text(role, text), file=fh)
            print(file=fh)
        case "tool_use":
            tool_use = ctx.tool_use(content_id)
            tool = ctx.tool(tool_use.tool)
            print(f"# tool_use tool={tool.name}: {tool_use.input}", file=fh)
            print(file=fh)
        case "tool_result":
            tool_result = ctx.tool_result(content_id)
            nlines = 0
            for resp_id in tool_result.response:
                resp = ctx.content(resp_id)
                if resp.type == "text":
                    nlines += resp.fields.get("text", "").count("\n")

            print(f"# tool_result lines={nlines}", file=fh)
            print(file=fh)
        case _:
            raise AssertionError(f"Unknown content type: {content_dict['type']!r}")


def render_message(ctx: Context, fh, message: MessageObject):
    header = f"# {message.role.title()}"
    print(header, file=fh)
    print(file=fh)

    render_content(ctx, fh, message.role, message.content)


def read_user_turn(ctx: Context, tmpdir: Path, convo: Conversation) -> str | None:
    md_path = tmpdir / "claude.md"

    with md_path.open("w") as fh:
        print("# -*- mode: markdown; mode: visual-line; -*-", file=fh)

        prompt = convo.prompt
        for message in walk_prompt_chain(ctx, prompt):
            render_message(ctx, fh, message)

        print(USER_SEPARATOR, file=fh)

    subprocess.check_call([os.getenv("EDITOR", "vi"), md_path])
    reply = md_path.read_text()
    bits = reply.split(USER_SEPARATOR, 2)
    if len(bits) == 1:
        return None

    if not bits[1].strip():
        return None

    return bits[1]


class StopConversation(Exception):
    pass


def run_conversation(
    conversation: Conversation, handle_user_turn, seed: int | None = None
):
    while True:
        for turn in conversation.pump(seed):
            if isinstance(turn.content, dict):
                type = turn.content.get("type", "<dict>")
            else:
                type = "str"

            print(f"Turn role={turn.role} type={type}: prompt={conversation.prompt}")

            block = turn.content
            if not isinstance(block, dict):
                continue

            if block["type"] == "tool_use":
                block = cast(ToolUseBlockParam, block)
                print(f"Use tool: {block['name']}: {block['input']}")
            elif block["type"] == "tool_result":
                block = cast(ToolResultBlockParam, block)
                content = block.get("content", "")
                if isinstance(content, str):
                    content = [dict(type="text", text=content)]
                lines = sum(
                    block["text"].count("\n") for block in content if "text" in block
                )
                print(f"Tool done: <returned {lines} lines>")
        try:
            handle_user_turn(conversation)
        except StopConversation:
            break
