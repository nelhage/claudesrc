from typing import Protocol

from .context import Context
from .objects import (
    CreateMessageObject,
    ModelObject,
    ResponseObject,
)


class LLMInterface(Protocol):
    model: ModelObject

    def create_message(
        self,
        ctx: Context,
        create: CreateMessageObject,
    ) -> ResponseObject: ...
