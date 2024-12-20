import json

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
