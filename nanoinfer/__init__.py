"""nanoinfer — a from-scratch LLM inference engine, built for learning."""

from ._env import sanitize_proxy_env

# Run before anything in the package can reach the HF Hub (the CLI's load_model,
# library imports, the parity tests). See nanoinfer/_env.py for the why.
sanitize_proxy_env()

__version__ = "0.1.0"
