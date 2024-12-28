import traceback
from contextlib import contextmanager
from dataclasses import dataclass
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


@dataclass
class State:
    cache_dir: str
    cache: Cache
    client: anthropic.Client


@click.group()
@click.option(
    "--cache-dir",
    type=Path,
    default=CACHE_DIR,
    metavar="DIR",
    help="Path to persistent cache",
)
@click.pass_context
def main(ctx: click.Context, cache_dir):
    client = anthropic.Client(api_key=anthropic_api_key())

    cache_dir.mkdir(exist_ok=True, parents=True)

    store = Store(str(cache_dir / "cache.sqlite"))
    cache = Cache(store)

    ctx.obj = State(
        cache_dir=cache_dir,
        cache=cache,
        client=client,
    )
    pass


@main.command()
@click.pass_context
def sourcetool(ctx: click.Context):
    state = ctx.find_object(State)
    assert state is not None

    repo_name = "The Linux Kernel"
    repo = Path("~/code/linux/").expanduser()

    tools = [ListFiles(repo), ReadFiles(repo), SearchFiles(repo)]

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
        f"Today, you are working in {repo_name} ({repo.name}.git)",
    ]

    conversation = Conversation(
        cache=state.cache,
        client=state.client,
        model=models.SONNET_3_5,
        system_prompt=system,
        tools=tools,
    )

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
@click.option(
    "--system",
    default=(),
    type=tuple[str, ...],
    multiple=True,
    metavar="PROMPT",
    help="System prompt",
)
@click.option("--seed", default=1, type=int, help="Seed for caching responses")
@click.argument("query", default=None, type=str, required=False)
@click.pass_context
def query(
    ctx: click.Context,
    query: str | None = None,
    model: str = models.SONNET_3_5,
    system: list[str] = [],
    seed: int = 0,
):
    state = ctx.find_object(State)
    assert state is not None

    conversation = Conversation(
        cache=state.cache,
        client=state.client,
        model=models.SONNET_3_5,
        system_prompt=system,
        seed=seed,
    )

    if query is not None:
        conversation.append_user(query)

    run_interactive_conversation(conversation)
