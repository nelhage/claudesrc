import html
import io
import json
import subprocess
from functools import partial
from pathlib import Path
from textwrap import dedent

import anthropic
from anthropic.types import (
    ContentBlock,
    Message,
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
)
from claudesrc import anthropic_api_key, models
from claudesrc.conversation import Conversation
from claudesrc.tool import Tool, to_api_block
from pydantic import BaseModel, Field, TypeAdapter

# class Project:
#     respositories: dict[str, Path]


class ListFiles(Tool):
    def __init__(self, root: Path):
        self.root = root

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
            write(f"BYTES\tLINES\tNAME")

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
            write(f"</file-contents>")

        return out.getvalue()


class SearchFiles(Tool):
    def __init__(self, root: Path):
        self.root = root

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


def main():
    repo_name = "The Linux Kernel"
    root = Path("~/code/linux/").expanduser()

    tools: list[Tool] = [ListFiles(root), ReadFiles(root), SearchFiles(root)]

    client = anthropic.Client(api_key=anthropic_api_key())

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
        TextBlockParam(type="text", text=SYSTEM_PROMPT),
        TextBlockParam(
            type="text", text=f"Today, you are working in {repo_name} ({root.name}.git)"
        ),
    ]

    transcript = Path("transcript.json")

    convo = Conversation(
        client=client,
        model=models.SONNET_3_5,
        system=system,
        max_tokens=1024,
        tools=tools,
    )

    if transcript.exists():
        with transcript.open("r") as fh:
            messages = json.load(fh)
        convo.turns = messages

    reply = convo.user_prompt("What is the VFS?")
    print(reply)

    with open(transcript, "w") as fh:
        json.dump(convo.turns, fh)


if __name__ == "__main__":
    main()
