import sys
import tempfile
import traceback
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from textwrap import dedent

import anthropic
import click

from scrubs import anthropic_api_key, models, prompts
from scrubs.anthropic import AnthropicModel
from scrubs.context import Context
from scrubs.conversation import DEFAULT_MAX_TOKENS, Conversation
from scrubs.interface import StopConversation, read_user_turn, run_conversation
from scrubs.store import Store
from scrubs.tools.repo import ListFiles, ReadFiles, SearchFiles

from . import objects
from .state import State


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
    cache = Context(store)

    ctx.obj = State(
        cache_dir=cache_dir,
        ctx=cache,
        client=client,
    )
    pass


modelarg = click.option(
    "--model",
    default=models.SONNET_3_5,
    type=click.Choice(tuple(models.MODEL_ALIASES.keys())),
    help="Model to use",
)


def do_user_turn(ctx: Context, td: Path, convo: Conversation):
    turn = read_user_turn(ctx, td, convo)
    if turn is None:
        raise StopConversation()
    convo.append_user(turn)


def cli_conversation(
    ctx: Context, conversation: Conversation, force_user_start: bool = False
):
    with tempfile.TemporaryDirectory() as td:
        handle_user = partial(do_user_turn, ctx, Path(td))

        if force_user_start:
            handle_user(conversation)

        run_conversation(conversation, handle_user)


@main.command()
@modelarg
@click.pass_context
def sourcetool(ctx: click.Context, model: str):
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
        ctx=state.ctx,
        model=AnthropicModel(client=state.client, model=model),
        system_prompt=system,
        tools={t.name: t for t in tools},
    )

    query = """\
What is a Maple tree? Where is the data structure defined?
"""

    conversation.append_user(query)
    cli_conversation(state.ctx, conversation)


@main.command()
@modelarg
@click.option(
    "--system",
    default=(),
    type=str,
    multiple=True,
    metavar="PROMPT",
    help="System prompt",
)
@click.option(
    "--repo",
    default=None,
    type=str,
    required=None,
    metavar="PATH",
    help="Include tools for accessing a git repository",
)
@click.option(
    "--file",
    default=[],
    type=str,
    required=False,
    multiple=True,
    metavar="PATH",
    help="Include one or more files in the context",
)
@click.option("--seed", default=1, type=int, help="Seed for caching responses")
@click.option(
    "--max-tokens",
    default=DEFAULT_MAX_TOKENS,
    type=int,
    help="Maximum tokens per model output turn",
)
@click.argument("query", default=None, type=str, required=False)
@click.pass_context
def query(
    ctx: click.Context,
    query: str | None = None,
    repo: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model: str = models.SONNET_3_5,
    system: tuple[str, ...] = (),
    file: list[str] = [],
    seed: int = 0,
):
    state = ctx.find_object(State)
    assert state is not None

    if repo is not None:
        repopath: Path = Path(repo)
        system = system + (
            f"You are answering questions in the context of the git repository `{repopath.name}.git`."
            " You have access to tools to search and read files in this repository."
            " Use them as appropriate to answer the user's questions.",
        )

        tools = [ListFiles(repopath), ReadFiles(repopath), SearchFiles(repopath)]
    else:
        tools = []

    conversation = Conversation(
        ctx=state.ctx,
        model=AnthropicModel(client=state.client, model=model),
        system_prompt=system,
        tools={t.name: t for t in tools},
        max_tokens=max_tokens,
        seed=seed,
    )

    for f in file:
        p = Path(f)
        conversation.append_user(prompts.file_contents(p.name, p.read_text()))

    if query == "-":
        query = sys.stdin.read()

    if query is not None:
        conversation.append_user(query)

    cli_conversation(state.ctx, conversation, force_user_start=query is None)


@main.command()
@click.option(
    "--model",
    default=models.SONNET_3_5,
    type=click.Choice(tuple(models.MODEL_ALIASES.keys())),
    help="Model to use",
)
@click.argument("query", type=str, required=False, default=None)
@click.pass_context
def count_tokens(
    ctx: click.Context,
    query: str | None = None,
    model: str = models.SONNET_3_5,
):
    state = ctx.find_object(State)
    assert state is not None

    text = query
    if text is None:
        text = sys.stdin.read()

    resp = state.client.messages.count_tokens(
        model=model,
        messages=[
            {
                "role": "user",
                "content": text,
            }
        ],
    )

    print(resp.input_tokens)


@main.command()
@click.argument("path", type=str, required=False, default=None)
@click.option("--components", type=int, required=False, default=1)
@click.option("--relative-to", type=str, required=False, default=None)
def format_file(
    path: str,
    components: int,
    relative_to: str | None = None,
):
    pobj = Path(path)

    contents = pobj.read_text()

    if relative_to is not None:
        relpath = str(pobj.relative_to(relative_to))
    else:
        relpath = "/".join(pobj.parts[-components:])

    print(prompts.file_contents(relpath, contents))


objects.register_commands(main)
