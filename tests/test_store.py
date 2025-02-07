import json
import os

import pytest
from pydantic import BaseModel
from scrubs.store import Store


class Object(BaseModel):
    name: str
    words: list[str]

    @classmethod
    def object_type(cls) -> str:
        return "dummy"


def test_store():
    store = Store(":memory:")

    o1 = Object(name="bob", words=["sam", "frodo"])
    o2 = Object(name="jonas", words=["apples"])

    id1 = store.insert("dummy", o1.model_dump_json())
    id2 = store.insert("dummy", o2.model_dump_json())

    assert store.insert("dummy", o1.model_dump_json()) == id1

    get1 = store.get(id1)
    assert get1 is not None
    assert Object.model_validate_json(get1.object).words == ["sam", "frodo"]

    assert json.loads(store.get(id2).object)["words"] == ["apples"]  # type: ignore

    assert store.object_count() == 2


@pytest.fixture
def store():
    """Fixture providing a fresh in-memory Store"""
    return Store(":memory:")


@pytest.fixture
def basic_object(store):
    """Fixture providing a single test object ID"""
    return store.insert("test", '{"name":"object1"}')


@pytest.fixture
def similar_objects(store):
    """Fixture providing two objects with similar IDs"""
    id1 = store.insert("test", '{"name":"similar1"}')
    id2 = store.insert("test", '{"name":"similar2"}')
    return id1, id2


def test_resolve_exact_id(store, basic_object):
    """Test resolving with a complete ID"""
    resolved = store.resolve_id(basic_object)
    assert resolved == basic_object


def test_resolve_unique_prefix(store, basic_object):
    """Test resolving with a unique prefix"""
    prefix = basic_object[:8]
    resolved = store.resolve_id(prefix)
    assert resolved == basic_object


def test_resolve_ambiguous_prefix(store, similar_objects):
    """Test that ambiguous prefixes raise ValueError"""
    id1, id2 = similar_objects
    common_prefix = os.path.commonprefix([id1, id2])

    with pytest.raises(ValueError) as exc_info:
        store.resolve_id(common_prefix)
    assert "Ambiguous prefix" in str(exc_info.value)
    assert id1 in str(exc_info.value)
    assert id2 in str(exc_info.value)


def test_resolve_nonexistent_prefix(store):
    """Test that non-existent prefixes raise KeyError"""
    with pytest.raises(KeyError) as exc_info:
        store.resolve_id("nonexistent")
    assert "No object found" in str(exc_info.value)


def test_resolve_empty_prefix(store, basic_object, similar_objects):
    """Test that empty prefix raises ValueError due to ambiguity"""
    with pytest.raises(ValueError) as exc_info:
        store.resolve_id("")
    assert "Ambiguous prefix" in str(exc_info.value)
