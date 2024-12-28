from dataclasses import dataclass
from pathlib import Path

import anthropic
from scrubs.cache import Cache


@dataclass
class State:
    cache_dir: Path
    cache: Cache
    client: anthropic.Client
