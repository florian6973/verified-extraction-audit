import os

from google import genai
from google.genai.types import HttpOptions
from loguru import logger
import fcntl
from datetime import datetime
from google.genai.types import GenerateContentConfig, ThinkingConfig

# Setup Gemini API: set GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_GENAI_USE_VERTEXAI in env
if "GOOGLE_CLOUD_PROJECT" not in os.environ:
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "")
if "GOOGLE_CLOUD_LOCATION" not in os.environ:
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
if "GOOGLE_GENAI_USE_VERTEXAI" not in os.environ:
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")



def setup_gemini_client():
    """Initialize and return the Gemini client."""
    return genai.Client(http_options=HttpOptions(api_version="v1"))

def call_gemini(client, note, prompt, model, task):
    logger.info(f"Note {note}, task {task}, calling Gemini API with model {model}")
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=GenerateContentConfig(
        **(dict(thinking_config=ThinkingConfig(
            thinking_budget=0,  # Use `0` to turn off thinking
        )) if "flash" in model else {})
        )
    )
    logger.info(f"Note {note}, task {task}, length {len(response.text)}, prompt tokens {response.usage_metadata.prompt_token_count}, candidates tokens {response.usage_metadata.candidates_token_count}, thoughts tokens {response.usage_metadata.thoughts_token_count}, total tokens {response.usage_metadata.total_token_count}, with model {model}")
    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    _usage_log = os.environ.get("GEMINI_USAGE_LOG", os.path.join(_repo, "outputs", "gemini_usage.txt"))
    os.makedirs(os.path.dirname(_usage_log) or ".", exist_ok=True)
    with open(_usage_log, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(f"{datetime.now()},{model},{response.usage_metadata.candidates_token_count},{response.usage_metadata.prompt_token_count},{response.usage_metadata.thoughts_token_count},{response.usage_metadata.total_token_count}\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return response.text

def call_llm_gemini_default(prompt, stream=None, task="<unspecified>", port=None, model="gemini-2.5-pro-preview-05-06", note="<unspecified>"):
    client = setup_gemini_client()
    return call_gemini(client, note, prompt, model, task)