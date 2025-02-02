from pathlib import Path

from scrubs.tools.repo import ListFiles, ReadFiles, SearchFiles

# TODO: actually make assertions
# TODO: flag as slow/integration


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
