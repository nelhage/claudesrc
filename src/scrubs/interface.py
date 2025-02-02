import os
import subprocess
from pathlib import Path
from typing import cast

from anthropic.types import (
    MessageParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)

from scrubs.conversation import Conversation

USER_SEPARATOR = "# Respond below this line. Delete this header to exit\n"


def render_turn(fh, turn: MessageParam):
    lines = []
    header = f"# {turn['role'].title()}"
    has_content = False

    if isinstance(turn["content"], str):
        has_content = True
        lines.append(turn["content"])
    else:
        for block in turn["content"]:
            assert isinstance(block, dict)
            match block["type"]:
                case "text":
                    lines.append(block["text"])
                    has_content = True
                case "tool_use":
                    lines.append(f"# tool_use tool={block['name']}: {block['input']}")
                case "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, str):
                        content = [dict(type="text", text=content)]
                    nlines = sum(c.get("text", "").count("\n") for c in content)

                    lines.append(
                        f"# tool_result lines={nlines} error={block.get('is_error', False)}"
                    )
                case _:
                    raise AssertionError(f"Unknown block: {block!r}")

    if has_content:
        lines.insert(0, header)

    for line in lines:
        print(line, file=fh)
        print(file=fh)


class StopConversation(Exception):
    pass


def read_user_turn(tmpdir: Path, convo: Conversation) -> str | None:
    md_path = tmpdir / "claude.md"

    with md_path.open("w") as fh:
        print("# -*- mode: markdown; mode: visual-line; -*-", file=fh)
        for turn in convo.turns:
            render_turn(fh, turn)
        print(USER_SEPARATOR, file=fh)

    subprocess.check_call([os.getenv("EDITOR", "vi"), md_path])
    reply = md_path.read_text()
    bits = reply.split(USER_SEPARATOR, 2)
    if len(bits) == 1:
        return None

    if not bits[1].strip():
        return None

    return bits[1]


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
