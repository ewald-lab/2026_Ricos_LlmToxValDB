from langchain.chat_models import init_chat_model
import logging

logger = logging.getLogger(__name__)

def get_model(model: str, model_provider: str, base_url: str = None, api_key: str = None, temperature: float = 0.0,
            **kwargs) -> object:
    """
    Initializes a chat model using LangChain's init_chat_model.
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
