import traceback
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent

import anthropic
import click

from scrubs import anthropic_api_key, models
from scrubs.cache import Cache
from scrubs.conversation import Conversation
from scrubs.interface import run_interactive_conversation
from scrubs.store import Store
from scrubs.tool import Tool
from scrubs.tools.repo import ListFiles, ReadFiles, SearchFiles


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


@click.group()
def main():
    pass


@main.command()
def sourcetool():
    client = anthropic.Client(api_key=anthropic_api_key())

    CACHE_DIR.mkdir(exist_ok=True, parents=True)

    store = Store(str(CACHE_DIR / "cache.sqlite"))
    cache = Cache(store)

    conversation = begin_conversation(cache, client)

    query = """\
What is a Maple tree? Where is the data structure defined?
"""

    conversation.append_user(query)
    run_interactive_conversation(conversation)


@main.command()
@click.option(
    "--model",
    default=models.SONNET_3_5,
    type=click.Choice(tuple(models.MODEL_ALIASES.keys())),
    help="Model to use",
)
@click.option("--system", default=(), type=tuple[str, ...], multiple=True)
@click.option("--seed", default=1, type=int, help="Seed for caching responses")
@click.argument("query", default=None, type=str, required=False)
def query(
    query: str | None = None,
    model: str = models.SONNET_3_5,
    system: list[str] = [],
    seed: int = 0,
):
    client = anthropic.Client(api_key=anthropic_api_key())

    CACHE_DIR.mkdir(exist_ok=True, parents=True)

    store = Store(str(CACHE_DIR / "cache.sqlite"))
    cache = Cache(store)

    conversation = Conversation(
        cache=cache,
        client=client,
        model=models.SONNET_3_5,
        system_prompt=system,
        seed=seed,
    )

    if query is not None:
        conversation.append_user(query)

    run_interactive_conversation(conversation)
