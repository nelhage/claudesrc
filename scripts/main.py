import html
import io
from functools import partial
from pathlib import Path
from textwrap import dedent

import anthropic
from anthropic.types import MessageParam, ToolParam
from claudesrc import anthropic_api_key, models
from claudesrc.tool import Tool, to_api_block
from pydantic import BaseModel, Field

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


def main():
    test_list()
    test_read_file()


if __name__ == "__main__":
    main()
