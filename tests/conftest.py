"""Test-suite setup that runs before anything imports huggingface_hub.

Importing nanoinfer triggers nanoinfer._env.sanitize_proxy_env(), which strips
the IPv6 no_proxy tokens that crash httpx (see nanoinfer/_env.py for the why).
conftest.py is imported by pytest before test collection — i.e. before any HF
import — so a bare `pytest` works without the manual `no_proxy=...` prefix.
"""

import nanoinfer  # noqa: F401  (import for its env-sanitizing side effect)
