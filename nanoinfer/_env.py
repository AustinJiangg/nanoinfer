"""Environment-compatibility shims, applied before we touch the HF Hub.

Why this exists: in some environments (notably WSL2 here) the `no_proxy` /
`NO_PROXY` env vars carry IPv6 loopback entries like `::1` and `[::1]`. The
httpx version huggingface_hub builds its client from cannot parse those tokens
and raises `httpx.InvalidURL: Invalid port: ':1]'` *before any network call* —
so even loading a fully-cached model dies the moment HF reads the proxy env.

The HTTP(S) proxy itself is fine; only those IPv6 no_proxy tokens are toxic, so
we strip just those (keeping the rest of the no_proxy list intact). This is
called from nanoinfer/__init__.py so it runs the instant the package is imported
— which is before any code path reaches huggingface_hub.
"""

import os


def sanitize_proxy_env() -> None:
    """Drop IPv6 entries from no_proxy/NO_PROXY that httpx cannot parse."""
    for var in ("no_proxy", "NO_PROXY"):
        value = os.environ.get(var)
        if not value:
            continue
        kept = [
            tok
            for tok in value.split(",")
            if "::" not in tok and "[" not in tok and "]" not in tok
        ]
        os.environ[var] = ",".join(kept)
