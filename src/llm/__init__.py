from loguru import logger


def call_llm(backend, prompt, kwargs):
    """Dispatch a prompt to one of the supported backends.

    Backends are imported lazily inside each branch so that importing
    ``src.llm`` never pulls in optional/heavy or platform-specific dependencies
    that a given backend needs (e.g. the Gemini backend imports ``google.genai``
    and the POSIX-only ``fcntl`` module). This lets the offline ``mock`` backend
    run on any machine — including Windows and nodes without ``google-genai``
    installed.
    """
    if backend == "vllm":
        from src.llm.vllm.utils import call_vllm_server

        logger.info(f"Calling vLLM server with kwargs: {kwargs}")
        return call_vllm_server(prompt, **kwargs)
    elif backend == "gemini":
        from src.llm.gemini.utils import call_llm_gemini_default

        return call_llm_gemini_default(prompt, **kwargs)
    elif backend == "mock":
        from src.llm.mock.utils import call_mock

        return call_mock(prompt, **kwargs)
    else:
        raise ValueError(f"Backend {backend} not supported")
