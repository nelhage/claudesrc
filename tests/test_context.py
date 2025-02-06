import pytest
from scrubs.context import Context
from scrubs.objects import ContentObject, ModelObject, ToolObject
from scrubs.store import ObjectID, Store


@pytest.fixture
def store():
    return Store(":memory:")


@pytest.fixture
def ctx(store):
    return Context(store)


def test_insert_and_retrieve(ctx):
    content = ContentObject(type="text", fields={"text": "hello"})
    id = ctx.insert(content)

    retrieved = ctx.get(id)
    assert isinstance(retrieved, ContentObject)
    assert retrieved.type == "text"
    assert retrieved.fields["text"] == "hello"


def test_get_type_correct(ctx):
    content = ContentObject(type="text", fields={"text": "hello"})
    id = ctx.insert(content)

    retrieved = ctx.get_type(id, ContentObject)
    assert isinstance(retrieved, ContentObject)
    assert retrieved.type == "text"


def test_get_type_incorrect(ctx):
    content = ContentObject(type="text", fields={"text": "hello"})
    id = ctx.insert(content)

    with pytest.raises(
        Exception
    ):  # The exact exception type would depend on implementation
        ctx.get_type(id, ToolObject)


def test_insert_get_preserves_identity(ctx):
    content = ContentObject(type="text", fields={"text": "hello"})
    id = ctx.insert(content)

    obj1 = ctx.get(id)
    assert obj1 is content  # Should be the exact same object in memory


def test_insert_within_scope(ctx: Context):
    content = ContentObject(type="text", fields={"text": "hello"})
    with ctx.cache_scope():
        id = ctx.insert(content)

    obj1 = ctx.get(id)
    assert obj1 is not content


def test_cache_identity_within_scope(ctx):
    c1 = Context(ctx.store)
    content = ContentObject(type="text", fields={"text": "hello"})
    id = c1.insert(content)

    with ctx.cache_scope():
        obj1 = ctx.get(id)
        obj2 = ctx.get(id)
        assert obj1 is obj2  # Should be the exact same object in memory


def test_cache_across_scopes(ctx):
    content = ContentObject(type="text", fields={"text": "hello"})
    id = ctx.insert(content)

    obj1 = ctx.get(id)
    with ctx.cache_scope():
        obj2 = ctx.get(id)
        assert obj1 is obj2  # Should still be the same object


def test_cache_scope_isolation(ctx: Context):
    c1 = Context(ctx.store)
    content = ContentObject(type="text", fields={"text": "hello"})
    id = c1.insert(content)

    with ctx.cache_scope():
        obj1 = ctx.get(id)

    obj2 = ctx.get(id)
    assert obj1 is not obj2  # Should be different objects after scope exit


def test_specialized_getters(ctx: Context):
    model_opts = ModelObject(provider="dummy", model="test-model")
    id = ctx.insert(model_opts)

    retrieved = ctx.get_model(id)
    assert isinstance(retrieved, ModelObject)
    assert retrieved.model == "test-model"


def test_nonexistent_object(ctx):
    fake_id = ObjectID("nonexistent")
    assert ctx.get(fake_id) is None


def test_nested_cache_scopes(ctx):
    content = ContentObject(type="text", fields={"text": "hello"})
    id = ctx.insert(content)

    with ctx.cache_scope():
        obj1 = ctx.get(id)
        with ctx.cache_scope():
            obj2 = ctx.get(id)
            assert obj1 is obj2  # Should be the same object in nested scopes
        obj3 = ctx.get(id)
        assert obj1 is obj3  # Should still be the same object in outer scope


def test_duplicate_insert(ctx):
    content = ContentObject(type="text", fields={"text": "hello"})
    id1 = ctx.insert(content)
    id2 = ctx.insert(content)

    assert id1 == id2  # Should return same ID for identical objects
