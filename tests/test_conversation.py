import pytest
from scrubs.context import Context
from scrubs.conversation import get_cached_create
from scrubs.objects import (
    ContentObject,
    CreateMessageObject,
    MessageObject,
    ModelOptsObject,
    PromptObject,
    ToolObject,
)
from scrubs.store import Store


def test_tools_migration_cache_compat():
    # Setup
    store = Store(":memory:")
    ctx = Context(store)

    # Create some test objects
    tool = ToolObject(name="test_tool")
    tool_id = ctx.insert(tool)

    # Create old-style ModelOpts with tools
    model_opts = ModelOptsObject(model="test-model", tools=[tool_id])
    model_id = ctx.insert(model_opts)

    # Create message content and prompt
    content = ContentObject(type="text", fields={"text": "test message"})
    content_id = ctx.insert(content)

    message = MessageObject(content=content_id, role="user")
    prompt = PromptObject(message=message)
    prompt_id = ctx.insert(prompt)

    # Create old-style create message (tools=None)
    old_create = CreateMessageObject(model=model_id, prompt=prompt_id, tools=None)
    old_create_id = ctx.insert(old_create)

    # Put a fake response in cache
    fake_response_id = "response123"
    ctx.put_cache(old_create_id, fake_response_id)

    # Create new-style create message (tools moved from ModelOpts)
    new_create = CreateMessageObject(
        model=model_id,
        prompt=prompt_id,
        tools=[tool_id],  # Same tools that were in ModelOpts
    )

    # Test
    result = get_cached_create(ctx, new_create)

    # Should find the cached response despite tools being in different place
    assert result == fake_response_id
