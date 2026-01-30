from loguru import logger

from src.llm.gemini.utils import call_llm_gemini_default
from src.llm.vllm.utils import call_vllm_server

def call_llm(backend, prompt, kwargs):
    if backend == "vllm":
        logger.info(f"Calling vLLM server with kwargs: {kwargs}")
        return call_vllm_server(prompt, **kwargs)
    elif backend == "gemini":
        # logger.warning("Ignoring stream argument for Gemini")
        return call_llm_gemini_default(prompt, **kwargs)
    else:
        raise ValueError(f"Backend {backend} not supported")