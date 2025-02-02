from dataclasses import dataclass
from pathlib import Path

import anthropic

from scrubs.context import Context


@dataclass
class State:
    cache_dir: Path
    cache: Context
    client: anthropic.Client
