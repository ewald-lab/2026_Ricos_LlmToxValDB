from langchain.chat_models import init_chat_model
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_model(
    model: str,
    model_provider: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs: Any
) -> Any:
    """
    Description:
    Initializes a chat model using LangChain's init_chat_model.

    Args:
        model (str): Name of the model to initialize.
        model_provider (str): Provider of the model (e.g. 'openai').
        base_url (Optional[str]): Optional custom base URL for the API endpoint. Defaults to None.
        api_key (Optional[str]): Optional custom API key. Defaults to None.
        temperature (float): Sampling temperature. Defaults to 0.0.
        **kwargs (Any): Additional keyword arguments passed to the model constructor.

    Returns:
        Any: The initialized LangChain chat model instance.
    """
    logger.info(f"Initializing model: {model} (provider: {model_provider})")

    init_args = {
        "model": model,
        "model_provider": model_provider,
        "temperature": temperature,
    }

    if base_url:
        init_args["base_url"] = base_url
    if api_key:
        init_args["api_key"] = api_key

    init_args.update(kwargs)

    return init_chat_model(**init_args)


def test_model(model: Any, max_retries: int = 10, delay: int = 10) -> bool:
    """
    Description:
    Sends a basic query to the model to verify connectivity and response,
    filtering out the reasoning tags (<think>...</think>).

    Args:
        model (Any): The chat model instance to test.
        max_retries (int): Maximum number of connection attempts. Defaults to 10.
        delay (int): Delay in seconds between retry attempts. Defaults to 10.

    Returns:
        bool: True if the model successfully responds, False otherwise.
    """
    base_url = getattr(model, "base_url", None) or getattr(model, "openai_api_base", "Unknown URL")
    logger.info(f"Testing model connectivity to: {base_url}")

    for attempt in range(max_retries):
        try:
            response = model.invoke("Hello")
            raw_content = response.content

            if "</think>" in raw_content:
                clean_content = raw_content.split("</think>")[-1].strip()
            else:
                clean_content = raw_content.strip()

            logger.info(f"Model test successful: {clean_content}")
            return True

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed connecting to {base_url}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)

    logger.error("All model connectivity tests failed.")
    return False
