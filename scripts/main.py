import os
import subprocess
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from typing import cast

import anthropic
from anthropic.types import (
    MessageParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from scrubs import anthropic_api_key, models
from scrubs.cache import Cache
from scrubs.conversation import Conversation
from scrubs.store import Store
from scrubs.tool import Tool
from scrubs.tools.repo import ListFiles, ReadFiles, SearchFiles

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


def read_user_turn(tmpdir: Path, convo: Conversation) -> str | None:
    # with (tmpdir / "transcript.json").open("w") as fh:
    #    json.dump(convo.turns, fh)

    md_path = tmpdir / "claude.md"

    with md_path.open("w") as fh:
        for turn in convo.turns:
            render_turn(fh, turn)
        print(USER_SEPARATOR, file=fh)

    subprocess.check_call([os.getenv("EDITOR", "vi"), md_path])
    reply = md_path.read_text()
    bits = reply.split(USER_SEPARATOR, 2)
    if len(bits) == 1:
        return None
    return bits[1]


def tools_for(repo: Path) -> list[Tool]:
    return [ListFiles(repo), ReadFiles(repo), SearchFiles(repo)]


def begin_conversation(cache: Cache, client: anthropic.Client) -> Conversation:
    repo_name = "The Linux Kernel"
    root = Path("~/code/linux/").expanduser()

    SYSTEM_PROMPT = dedent("""\
    You are an agent who helps experienced software engineers
    understand and learn about large and complex codebases. You have
    access to a git checkout of a source code repository, and tools
    for exploring it. Your job is to answer the user's questions based
    on reference to the source.

    You will mention specific source files and functions in your
    answers, where appropriate. You will answer questions at a high-
    level conceptual and architectural level by default, but be
    willing to explain specific implementation details with reference
    to the source when useful.

    In general you work in repositories too large for a human to read
    or to fit in your context window; you will need to use search
    tools to discover and read the relevant files.
    """)

    system = [
        SYSTEM_PROMPT,
        f"Today, you are working in {repo_name} ({root.name}.git)",
    ]

    convo = Conversation(
        cache=cache,
        client=client,
        model=models.SONNET_3_5,
        system_prompt=system,
        tools=tools_for(root),
    )
    return convo


@contextmanager
def breakpoint_on_exception():
    import pdb

    try:
        yield
    except Exception as ex:
        traceback.print_exception(ex)

        pdb.post_mortem(ex.__traceback__)

        raise


CACHE_DIR = Path("~/.cache/scrubs").expanduser()


@breakpoint_on_exception()
def main():
    client = anthropic.Client(api_key=anthropic_api_key())

    CACHE_DIR.mkdir(exist_ok=True, parents=True)

    store = Store(str(CACHE_DIR / "cache.sqlite"))
    cache = Cache(store)

    conversation = begin_conversation(cache, client)

    query = """\
What is a Maple tree? Where is the data structure defined?
"""

    conversation.user_prompt(query)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        while True:
            for turn in conversation.pump():
                if isinstance(turn.content, dict):
                    type = turn.content.get("type", "<dict>")
                else:
                    type = "str"

                print(
                    f"Turn role={turn.role} type={type}: prompt={conversation.prompt}"
                )

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
                        block["text"].count("\n")
                        for block in content
                        if "text" in block
                    )
                    print(f"Tool done: <returned {lines} lines>")

            user_turn = read_user_turn(td, conversation)
            if not user_turn:
                break
            conversation.append_user(user_turn)


if __name__ == "__main__":
    main()
