import sys

import click

from scrubs.context import Context
from scrubs.interface import render_prompt
from scrubs.objects import PromptObject
from scrubs.store import ObjectID

from .state import State


def resolve_object(ctx: Context, object: str) -> ObjectID:
    try:
        return ctx.store.resolve_id(object)
    except KeyError:
        raise click.UsageError(f"Unknown object: {object}")
    except ValueError:
        raise click.UsageError(f"Ambiguous object ID: {object}")


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

    raw = state.ctx.store.get(resolve_object(state.ctx, object))
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

    oid = resolve_object(state.ctx, object)
    raw = state.ctx.get(oid)
    if raw is None:
        sys.exit(1)

    if isinstance(raw, PromptObject):
        render_prompt(state.ctx, oid, sys.stdout)
        return

    # TODO fill me in, format result
    print(raw)
    # if isinstance(raw, PromptObject):
    #    pass


def register_commands(main: click.Group):
    main.add_command(cat_object)
    main.add_command(show)
