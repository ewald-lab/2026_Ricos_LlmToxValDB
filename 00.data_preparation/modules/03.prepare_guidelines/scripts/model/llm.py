from langchain.chat_models import init_chat_model
import logging
import time

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
    
    # Default to "EMPTY" for local OpenAI-compatible servers if no key is provided
    if not api_key and model_provider == "openai":
        api_key = "EMPTY"

    if api_key:
        init_args["api_key"] = api_key
        
    init_args.update(kwargs)
    
    return init_chat_model(**init_args)

def test_model(model, max_retries=10, delay=10):
    """
    Sends a basic query to the model to verify connectivity and response,
    filtering out the reasoning tags (<think>...</think>).
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
