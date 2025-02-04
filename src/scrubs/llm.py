from abc import abstractmethod
from typing import Iterable, Protocol

from .context import Context
from .objects import (
    ContentObject,
    CreateMessageObject,
    ModelObject,
    ResponseObject,
)
from .tool import Tool


class LLMInterface(Protocol):
    model: ModelObject

    def create_message(
        self,
        ctx: Context,
        create: CreateMessageObject,
    ) -> ResponseObject: ...
