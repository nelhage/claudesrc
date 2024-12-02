import netrc
import os
from functools import lru_cache


@lru_cache
def anthropic_api_key() -> str:
    if "ANTHROPIC_API_KEY" in os.environ:
        return os.environ["ANTHROPIC_API_KEY"]

    creds = netrc.netrc().authenticators("api.anthropic.com")
    if creds is None:
        raise ValueError("No credentials found for api.anthropic.com")
    return creds[-1]
