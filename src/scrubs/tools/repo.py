import html
import io
import subprocess
from functools import partial
from pathlib import Path
from textwrap import dedent

from pydantic import BaseModel, Field
from scrubs.tool import Tool


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

    def call_tool(self, args: dict) -> str:
        out = io.StringIO()
        write = partial(print, file=out)

        params = self.Params.model_validate(args)

        paths = params.path if isinstance(params.path, list) else [params.path]
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

    def call_tool(self, args: dict) -> str:
        out = io.StringIO()
        write = partial(print, file=out)

        params = self.Params.model_validate(args)

        paths = params.path if isinstance(params.path, list) else [params.path]
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

    def call_tool(self, args: dict) -> str:
        params = self.Params.model_validate(args)

        cmd = [
            "rg",
            "-e",
            params.pattern,
            "-n",
            "-M",
            "200",
            "--max-columns-preview",
            "-H",
            "--no-heading",
        ]
        if params.glob:
            for pat in params.glob:
                cmd.extend(["-g", pat])
        if params.path:
            cmd.append(params.path)

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
