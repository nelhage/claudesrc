from contextlib import contextmanager
from typing import Type, TypeVar

from anthropic.types import MessageParam

from .objects import (
    OBJECT_TYPES,
    ContentObject,
    MessageObject,
    ModelOptsObject,
    ObjectType,
    PromptObject,
    ResponseObject,
    ToolObject,
    ToolResultObject,
    ToolUseObject,
)
from .store import ObjectID, Store

Obj = TypeVar("Obj", bound=ObjectType)


def dump_object(obj: ObjectType) -> str:
    return obj.model_dump_json()


class DictStack:
    def __init__(self):
        """Initialize an empty stack with one frame."""
        self.frames = [{}]

    def get(self, key):
        """
        Search for key from top to bottom of the stack.
        Returns the first value found or None if not found.
        """
        for frame in reversed(self.frames):
            if key in frame:
                return frame[key]
        return None

    def __contains__(self, key):
        """
        Implement the 'in' operator to check if a key exists in any frame.
        Returns True if the key exists, False otherwise.
        """
        return self.get(key) is not None

    def set(self, key, value):
        """
        Set key-value pair in the topmost frame.
        Raises ValueError if key exists anywhere in the stack.
        """
        if self.get(key) is not None:
            raise ValueError(f"Key '{key}' already exists in the stack")
        self.frames[-1][key] = value

    def push_frame(self):
        """Add a new empty dictionary frame to the top of the stack."""
        self.frames.append({})

    def pop_frame(self):
        """
        Remove and return the topmost frame.
        Raises IndexError if attempting to pop the last frame.
        """
        if len(self.frames) <= 1:
            raise IndexError("Cannot pop the last frame")
        return self.frames.pop()

    def __str__(self):
        """Return a string representation of the stack."""
        return "\n".join(f"Frame {i}: {frame}" for i, frame in enumerate(self.frames))


class Context:
    def __init__(self, store: Store):
        self.store = store
        self.object_cache = DictStack()

    @contextmanager
    def cache_scope(self):
        try:
            self.object_cache.push_frame()
            yield
        finally:
            self.object_cache.pop_frame()

    def hash_object(self, obj: ObjectType) -> ObjectID:
        return self.store.hash_object(dump_object(obj))

    def insert(self, obj: ObjectType) -> ObjectID:
        flat = dump_object(obj)
        id = self.store.hash_object(flat)
        if id in self.object_cache:
            return id
        id = self.store.insert(obj.object_type, flat)
        self.object_cache.set(id, obj)
        return id

    def get(self, id: ObjectID) -> ObjectType | None:
        if inmem := self.object_cache.get(id):
            return inmem

        raw = self.store.get(id)
        if raw is None:
            return None
        inmem = OBJECT_TYPES[raw.type].model_validate_json(raw.object)
        self.object_cache.set(id, inmem)
        return inmem

    def get_type(self, id: ObjectID, ty: Type[Obj]) -> Obj:
        if inmem := self.object_cache.get(id):
            assert isinstance(inmem, ty)
            return inmem

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

    def get_tool_use(self, id: ObjectID) -> ToolUseObject:
        return self.get_type(id, ToolUseObject)

    def get_tool_result(self, id: ObjectID) -> ToolResultObject:
        return self.get_type(id, ToolResultObject)

    # API Cache

    def put_cache(self, query: ObjectID, response: ObjectID):
        self.store.put_cache(query, response)

    def get_cache(self, query: ObjectID) -> ObjectID | None:
        return self.store.get_cache(query)


def walk_prompt_chain(ctx: Context, prompt_id: ObjectID | None) -> list[MessageObject]:
    """Walk backwards through the prompt chain"""
    messages = []
    current = prompt_id

    while current is not None:
        prompt = ctx.get_prompt(current)
        messages.append(prompt.message)
        current = prompt.prefix

    return list(reversed(messages))


def flatten_prompt(ctx: Context, prompt: ObjectID | None) -> list[MessageParam]:
    """Flatten a prompt object in order to feed it to the API.

    Args:
        ctx: Context object for accessing stored objects
        prompt: ObjectID of the prompt to flatten, or None

    Returns:
        List of MessageParam objects ready for the API in chronological order
    """
    messages: list[MessageParam] = []
    current_prompt = prompt

    # Build list in reverse order (most recent first)
    while current_prompt is not None:
        prompt_obj = ctx.get_prompt(current_prompt)

        # Get content for current message
        message = prompt_obj.message
        content_obj = ctx.get_content(message.content)
        content = content_obj.to_dict()

        # Add message to list
        messages.append(
            {
                "role": message.role,
                "content": [content],  # type:ignore
            }
        )

        # Move to prefix
        current_prompt = prompt_obj.prefix

    # Reverse to get chronological order
    messages.reverse()
    return messages
