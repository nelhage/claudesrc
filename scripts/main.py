import html
import io
import json
import os
import subprocess
import tempfile
import traceback
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from textwrap import dedent

import anthropic
from anthropic.types import (
    TextBlockParam,
)
from pydantic import BaseModel, Field
from scrubs import anthropic_api_key, models
from scrubs.cache import Cache
from scrubs.conversation import Conversation, MessageTurn
from scrubs.store import Store
from scrubs.tool import Tool

# class Project:
#     respositories: dict[str, Path]


class ListFiles(Tool):
    def __init__(self, root: Path):
        self.root = root

    def cache_params(self) -> dict:
        return dict(root=self.root)

    class Params(BaseModel):
        path: str | list[str] = Field(
            description="The filesystem path or paths you want to list"
        )

    name = "list_paths"
    description = dedent("""\
    List files under one or more directories.

    You may specify one or more paths, relative to the root of the
    repository you are working in. The output will contain the names
    of files in those directories, as well as their sizes and number
    of lines.
    """)

    input_schema = Params.model_json_schema()

    def call_tool(self, raw: dict) -> str:
        out = io.StringIO()
        write = partial(print, file=out)

        args = self.Params.model_validate(raw)

        paths = args.path if isinstance(args.path, list) else [args.path]
        for rel in paths:
            path = self.root / rel
            if not path.is_dir():
                if path.exists():
                    write(f"Not a directory: {rel}\n")
                else:
                    write(f"No such file or directory: {rel}\n")
                continue

            write(f"Directory listing: {rel}/")
            write("BYTES\tLINES\tNAME")

            for ent in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
                if ent.is_dir():
                    write(f"DIR\t\t{ent.name}")
                elif ent.is_file():
                    bytes = ent.stat().st_size
                    lines = ent.read_bytes().count(b"\n")
                    write(f"{bytes}\t{lines}\t{ent.name}")

        return out.getvalue()


class ReadFiles(Tool):
    def __init__(self, root: Path):
        self.root = root

    def cache_params(self) -> dict:
        return dict(root=self.root)

    class Params(BaseModel):
        path: str | list[str] = Field(description="The files you want to read")

    name = "read_files"
    description = dedent("""\
    Read one or more files.

    You may specify one or more paths, relative to the root of the
    repository.
    """)

    input_schema = Params.model_json_schema()

    def call_tool(self, raw: dict) -> str:
        out = io.StringIO()
        write = partial(print, file=out)

        args = self.Params.model_validate(raw)

        paths = args.path if isinstance(args.path, list) else [args.path]
        for rel in paths:
            path = self.root / rel
            if not path.is_file():
                if path.exists():
                    write(f"Not a file: {rel}\n")
                else:
                    write(f"No such file or directory: {rel}\n")
                continue

            write(f"<file-contents path='{html.escape(rel)}'>")
            body = path.read_text()
            out.write(body)
            if not body.endswith("\n"):
                write()
            write("</file-contents>")

        return out.getvalue()


class SearchFiles(Tool):
    def __init__(self, root: Path):
        self.root = root

    def cache_params(self) -> dict:
        return dict(root=self.root)

    class Params(BaseModel):
        pattern: str = Field(
            description="A regular expression to search for. Uses Perl-style regex syntax but without support for backreferences."
        )
        path: str | None = Field(
            description="Only search files under a given directory",
            default=None,
        )
        glob: list[str] | None = Field(
            description="Only search files whose names match any of the given glob patterns (e.g. '*.h')",
            default=None,
        )

    MAX_RESULTS = 1000

    name = "search_files"
    description = dedent(f"""\
    Search for files matching a regular expression.

    You will receive a list of files which match the provided regular
    expression, including the contents of the matching line(s).

    You may limit your search to a given subtree or to files matching
    a given pattern.

    If your search returns more than {MAX_RESULTS} lines, the result
    will be truncated.
    """)

    input_schema = Params.model_json_schema()

    def call_tool(self, raw: dict) -> str:
        args = self.Params.model_validate(raw)

        cmd = [
            "rg",
            "-e",
            args.pattern,
            "-n",
            "-M",
            "200",
            "--max-columns-preview",
            "-H",
            "--no-heading",
        ]
        if args.glob:
            for pat in args.glob:
                cmd.extend(["-g", pat])
        if args.path:
            cmd.append(args.path)

        try:
            out = subprocess.check_output(cmd, cwd=self.root, text=True)
            nmatch = out.count("\n")
            if nmatch > self.MAX_RESULTS:
                out = "\n".join(out.split("\n")[: self.MAX_RESULTS])
                out += (
                    f"\n** OUTPUT TRUNCATED: {nmatch-self.MAX_RESULTS} matches hidden\n"
                )
            return out
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 1:
                return "<no matches>"
            else:
                raise exc


def test_list():
    lst = ListFiles(root=Path("~/code/linux/").expanduser())

    for args in [
        ".",
        ["lib", "include/linux"],
        ["enoent", "fs"],
    ]:
        print(f"LIST paths={args=}")
        result = lst.call_tool(dict(path=args))
        print(result)
        print()


def test_read_file():
    cmd = ReadFiles(root=Path("~/code/linux/").expanduser())

    for args in [
        ".",
        "fs/namei.c",
        ["lib", "enoent", "include/linux/compiler.h"],
    ]:
        print(f"READFILES paths={args=}")
        result = cmd.call_tool(dict(path=args))
        print(result)
        print()


def test_search():
    cmd = SearchFiles(root=Path("~/code/linux/").expanduser())

    for args in [
        SearchFiles.Params(
            pattern="printk",
        ),
        SearchFiles.Params(
            pattern="no such rhino",
        ),
        SearchFiles.Params(
            pattern="dma_alloc_coherent",
            path="Documentation",
            glob=["*.txt"],
        ),
    ]:
        print(f"SEARCH {args=}")
        result = cmd.call_tool(args.model_dump())
        print(result)
        print()


def selftest():
    test_list()
    test_read_file()
    test_search()


USER_SEPARATOR = "# Respond below this line. Delete this header to exit\n"


def render_turn(fh, turn: MessageTurn):
    lines = []
    header = f"# {turn.role.title()}"
    has_content = False

    if isinstance(turn.content, str):
        has_content = True
        lines.append(turn.content)
    else:
        for block in turn.content:
            assert isinstance(block, dict)
            match block["type"]:
                case "text":
                    lines.append(block["text"])
                    has_content = True
                case "tool_use":
                    lines.append(f"# tool_use tool={block['name']}: {block['input']}")
                case "tool_result":
                    if "content" in block:
                        nlines = block["content"].count("\n")
                    else:
                        nlines = 0
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
    import sys

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
                    print(f"Use tool: {block['name']}: {block['input']}")
                elif block["type"] == "tool_result":
                    content = block["content"]
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
