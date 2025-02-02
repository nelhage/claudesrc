import sys

import click

from scrubs.objects import PromptObject

from .state import State


@click.command
@click.pass_context
@click.option(
    "-t",
    "show_type",
    is_flag=True,
    type=bool,
    default=False,
    help="Show the object's type",
)
@click.argument("object", type=str)
def cat_object(ctx: click.Context, show_type: bool, object: str):
    state = ctx.find_object(State)
    assert state is not None

    raw = state.ctx.store.get(object)
    if raw is None:
        sys.exit(1)

    if show_type:
        print(raw.type)
        return

    print(raw.object)


@click.command
@click.pass_context
@click.argument("object", type=str)
def show(ctx: click.Context, object: str):
    state = ctx.find_object(State)
    assert state is not None

    raw = state.ctx.get(object)
    if raw is None:
        sys.exit(1)

    # TODO fill me in, format result
    print(raw)
    # if isinstance(raw, PromptObject):
    #    pass


def register_commands(main: click.Group):
    main.add_command(cat_object)
    main.add_command(show)
