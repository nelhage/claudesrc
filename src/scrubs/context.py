from typing import Type, TypeVar

from .objects import (
    OBJECT_TYPES,
    ContentObject,
    ModelOptsObject,
    ObjectType,
    PromptObject,
    ResponseObject,
    ToolObject,
    ToolResultObject,
)
from .store import ObjectID, Store

Obj = TypeVar("Obj", bound=ObjectType)


def dump_object(obj: ObjectType) -> str:
    return obj.model_dump_json()


class Context:
    def __init__(self, store: Store):
        self.store = store

    def insert(self, obj: ObjectType) -> ObjectID:
        return self.store.insert(obj.object_type, dump_object(obj))

    def get(self, id: ObjectID) -> ObjectType | None:
        got = self.store.get(id)
        if got is None:
            return None
        return OBJECT_TYPES[got.type].model_validate_json(got.object)

    def get_type(self, id: ObjectID, ty: Type[Obj]) -> Obj:
        got = self.store.fetch(id, ty.object_type)
        return ty.model_validate_json(got.object)

    def get_model_opts(self, id: ObjectID) -> ModelOptsObject:
        return self.get_type(id, ModelOptsObject)

    def get_tool(self, id: ObjectID) -> ToolObject:
        return self.get_type(id, ToolObject)

    def get_content(self, id: ObjectID) -> ContentObject:
        return self.get_type(id, ContentObject)

    def get_prompt(self, id: ObjectID) -> PromptObject:
        return self.get_type(id, PromptObject)

    def get_response(self, id: ObjectID) -> ResponseObject:
        return self.get_type(id, ResponseObject)

    def get_tool_result(self, id: ObjectID) -> ToolResultObject:
        return self.get_type(id, ToolResultObject)

    # API Cache

    def put_cache(self, query: ObjectID, response: ObjectID):
        self.store.put_cache(query, response)

    def get_cache(self, query: ObjectID) -> ObjectID | None:
        return self.store.get_cache(query)
