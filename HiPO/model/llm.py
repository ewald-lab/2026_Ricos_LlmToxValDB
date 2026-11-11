import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain.chat_models import init_chat_model
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
import openai

logger = logging.getLogger(__name__)

_CLOSED_REASONING_RE = re.compile(
    r"<(think|thought|reasoning)\b[^>]*>(.*?)</\1>",
    flags=re.DOTALL | re.IGNORECASE,
)
_BARE_CLOSE_REASONING_RE = re.compile(
    r"</(think|thought|reasoning)\s*>",
    flags=re.IGNORECASE,
)
_UNCLOSED_OPEN_REASONING_RE = re.compile(
    r"<(think|thought|reasoning)\b[^>]*>(.*)$",
    flags=re.DOTALL | re.IGNORECASE,
)
_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")


class ReasoningChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI subclass that preserves vLLM / OpenAI-compatible reasoning channels
    (`reasoning` or `reasoning_content`) in `AIMessage.additional_kwargs["reasoning_content"]`.
    """

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: Optional[dict] = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info=generation_info)
        response_dict = response if isinstance(response, dict) else response.model_dump()
        choices = response_dict.get("choices") or []
        raw_choices_objs = getattr(response, "choices", None) if not isinstance(response, dict) else None

        for idx, gen in enumerate(result.generations):
            raw_choice = choices[idx] if idx < len(choices) else None
            msg_dict = (raw_choice.get("message") or {}) if isinstance(raw_choice, dict) else {}
            reasoning = msg_dict.get("reasoning_content") or msg_dict.get("reasoning")
            if not reasoning and raw_choices_objs and idx < len(raw_choices_objs):
                msg_obj = getattr(raw_choices_objs[idx], "message", None)
                reasoning = getattr(msg_obj, "reasoning_content", None) or getattr(msg_obj, "reasoning", None)
            if isinstance(reasoning, str) and reasoning.strip():
                gen.message.additional_kwargs.setdefault("reasoning_content", reasoning)
        return result


def get_finish_reason(message: Any) -> Optional[str]:
    """Returns `finish_reason` from LangChain `response_metadata` if available."""
    meta = getattr(message, "response_metadata", None)
    if isinstance(meta, dict):
        reason = meta.get("finish_reason")
        if reason is not None:
            return str(reason)
    return None


def _coerce_block_text(block: Any) -> Tuple[str, bool]:
    """
    Extracts text from a structured content block and returns `(text, is_reasoning)`.
    """
    if isinstance(block, str):
        return block, False
    if not isinstance(block, dict):
        return str(block), False

    block_type = str(block.get("type", "")).lower()
    if block_type in ("thinking", "reasoning", "reasoning_content", "thought"):
        val = (
            block.get("thinking")
            or block.get("reasoning")
            or block.get("reasoning_content")
            or block.get("text")
            or ""
        )
        return str(val), True

    if "text" in block:
        return str(block["text"]), False
    if "content" in block and isinstance(block["content"], str):
        return block["content"], False
    return "", False


def extract_reasoning_and_content(message: Any) -> Tuple[str, str]:
    """
    Description:
    Extracts `(reasoning_text, final_content)` across provider/server conventions:
    - `additional_kwargs["reasoning_content"]` / `"reasoning"` / `"thinking"`
    - structured content block lists (Anthropic/Google/OpenAI Responses API)
    - inline `<think>...</think>`, `<thought>...</thought>`, `<reasoning>...</reasoning>` tags,
      prefill-opened `...</think>`, and truncated unclosed `<think>...` blocks.

    Args:
        message (Any): LangChain message object or raw string.

    Returns:
        Tuple[str, str]: `(reasoning_text, cleaned_final_content)`.
    """
    reasoning_parts: List[str] = []
    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        for key in ("reasoning_content", "reasoning", "thinking"):
            val = additional_kwargs.get(key)
            if isinstance(val, str) and val.strip():
                reasoning_parts.append(val.strip())
            elif isinstance(val, list):
                for item in val:
                    t, _ = _coerce_block_text(item)
                    if t.strip():
                        reasoning_parts.append(t.strip())

    raw_content = message.content if hasattr(message, "content") else message
    if isinstance(raw_content, list):
        text_chunks: List[str] = []
        for blk in raw_content:
            blk_text, is_reasoning = _coerce_block_text(blk)
            if is_reasoning:
                if blk_text.strip():
                    reasoning_parts.append(blk_text.strip())
            elif blk_text:
                text_chunks.append(blk_text)
        text = "".join(text_chunks)
    elif raw_content is None:
        text = ""
    else:
        text = raw_content if isinstance(raw_content, str) else str(raw_content)

    # 1. Paired <think>...</think> / <thought>...</thought> / <reasoning>...</reasoning>
    def _collect_paired(match: re.Match) -> str:
        inner = (match.group(2) or "").strip()
        if inner:
            reasoning_parts.append(inner)
        return ""

    text = _CLOSED_REASONING_RE.sub(_collect_paired, text)

    # 2. Bare closing tag (assistant prefill already injected opening <think>)
    close_matches = list(_BARE_CLOSE_REASONING_RE.finditer(text))
    if close_matches:
        last_close = close_matches[-1]
        prefix_thought = text[: last_close.start()].strip()
        if prefix_thought:
            reasoning_parts.append(prefix_thought)
        text = text[last_close.end() :]

    # 3. Unclosed opening tag due to truncation mid-reasoning
    unclosed = _UNCLOSED_OPEN_REASONING_RE.search(text)
    if unclosed:
        trailing_thought = (unclosed.group(2) or "").strip()
        if trailing_thought:
            reasoning_parts.append(trailing_thought)
        text = text[: unclosed.start()]

    # Deduplicate identical reasoning strings while keeping order
    deduped_reasoning: List[str] = []
    for part in reasoning_parts:
        if part and part not in deduped_reasoning:
            deduped_reasoning.append(part)

    return "\n\n".join(deduped_reasoning).strip(), text.strip()


def _repair_unescaped_quotes(text: str) -> str:
    """
    Description:
    Character-level scanner that escapes unescaped double quotes inside JSON string values.

    Args:
        text (str): JSON string candidate that may contain unescaped interior double quotes.

    Returns:
        str: Repaired JSON string candidate.
    """
    result = []
    n = len(text)
    in_string = False
    i = 0
    while i < n:
        ch = text[i]
        if ch == "\\" and in_string:
            result.append(ch)
            if i + 1 < n:
                result.append(text[i + 1])
                i += 2
            else:
                i += 1
            continue

        if ch == '"':
            if not in_string:
                in_string = True
                result.append(ch)
            else:
                j = i + 1
                while j < n and text[j] in (" ", "\t", "\r", "\n"):
                    j += 1
                next_structural = text[j] if j < n else ""
                if next_structural in (",", "}", "]", ":"):
                    in_string = False
                    result.append(ch)
                else:
                    result.append('\\"')
        elif ch == "\n" and in_string:
            result.append("\\n")
        elif ch == "\r" and in_string:
            result.append("\\r")
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def _extract_balanced_json_candidates(text: str) -> List[str]:
    """Finds top-level balanced `{...}` slices in `text` using a quote/escape-aware scanner."""
    candidates: List[str] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        start = i
        depth = 0
        in_string = False
        escaped = False
        while i < n:
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : i + 1])
                        i += 1
                        break
            i += 1
        else:
            break
    return candidates


def _repair_eval_dict(eval_dict: Any) -> Any:
    """Normalizes common structural anomalies in LLM evaluation dictionaries."""
    if not isinstance(eval_dict, dict):
        return eval_dict

    if "decision" not in eval_dict or eval_dict["decision"] not in ("ACCEPTED", "REJECTED"):
        for k, v in list(eval_dict.items()):
            if isinstance(v, str) and v.strip().upper() in ("ACCEPTED", "REJECTED"):
                eval_dict["decision"] = v.strip().upper()
                if k != "decision":
                    del eval_dict[k]
                break

    if "decision" not in eval_dict or eval_dict["decision"] not in ("ACCEPTED", "REJECTED"):
        reason = eval_dict.get("reason", "")
        if isinstance(reason, str):
            upper = reason.upper().strip()
            if upper.startswith("ACCEPTED"):
                eval_dict["decision"] = "ACCEPTED"
                eval_dict["reason"] = re.sub(r"^accepted[\s.,:-]*", "", reason, flags=re.IGNORECASE).strip()
            elif upper.startswith("REJECTED"):
                eval_dict["decision"] = "REJECTED"
                eval_dict["reason"] = re.sub(r"^rejected[\s.,:-]*", "", reason, flags=re.IGNORECASE).strip()

    if "decision" in eval_dict and isinstance(eval_dict["decision"], str):
        eval_dict["decision"] = eval_dict["decision"].upper()

    if "reason" not in eval_dict or not eval_dict["reason"]:
        for k, v in list(eval_dict.items()):
            if k not in ("candidate_name", "decision") and isinstance(v, str):
                eval_dict["reason"] = v
                if k != "reason":
                    del eval_dict[k]
                break
    return eval_dict


def _try_parse_and_normalize_json(raw: str) -> Optional[str]:
    """Attempts to parse a JSON string (with trailing-comma and quote repair) and normalize eval dicts."""
    candidate = _TRAILING_COMMA_RE.sub(r"\1", raw.strip())
    data: Any = None
    parsed = False
    for attempt_text, strict in (
        (candidate, True),
        (candidate, False),
        (_repair_unescaped_quotes(candidate), False),
    ):
        try:
            data = json.loads(attempt_text, strict=strict)
            parsed = True
            break
        except Exception:
            continue

    if not parsed:
        return None

    if isinstance(data, dict):
        if "evaluation" in data and isinstance(data["evaluation"], list):
            data["evaluation"] = [_repair_eval_dict(item) for item in data["evaluation"]]
        else:
            data = _repair_eval_dict(data)
    return json.dumps(data, ensure_ascii=False)


def clean_llm_output(message: Any, *, allow_empty: bool = False) -> str:
    """
    Description:
    Cleans raw LLM responses across thinking and non-thinking model families by:
    - Extracting and separating reasoning (`additional_kwargs["reasoning_content"]` /
      `"reasoning"`, structured thinking blocks, or `<think>`/`<thought>`/`<reasoning>` tags,
      including unclosed truncated tags)
    - Detecting `finish_reason == "length"` truncation and failing loudly on empty outputs
    - Unwrapping markdown code fences and embedded balanced JSON payloads when present
    - Repairing minor JSON formatting glitches without altering plain-text responses.

    Args:
        message (Any): LangChain message object or raw string returned from an LLM.
        allow_empty (bool): If True, return `""` instead of raising ValueError on empty final output.

    Returns:
        str: Cleaned JSON string (when JSON is present) or cleaned plain text.

    Raises:
        ValueError: When final content after stripping reasoning is empty and `allow_empty=False`.
    """
    reasoning_text, text = extract_reasoning_and_content(message)
    finish_reason = get_finish_reason(message)

    if finish_reason == "length":
        logger.warning(
            "LLM output truncated (finish_reason='length'; reasoning_chars=%d, content_chars=%d). "
            "Consider increasing max_tokens / --max-model-len or tuning presence_penalty.",
            len(reasoning_text),
            len(text),
        )

    if not text:
        if allow_empty:
            return ""
        raise ValueError(
            f"Empty final output from LLM (finish_reason={finish_reason!r}, reasoning_chars={len(reasoning_text)}). "
            "Thinking may have exhausted max_tokens or entered a reasoning loop."
        )

    # Check fenced markdown blocks (either full response or embedded inside conversational wrapper)
    fences = [m.group(1).strip() for m in _FENCED_BLOCK_RE.finditer(text) if m.group(1).strip()]
    for fenced in fences:
        normalized = _try_parse_and_normalize_json(fenced)
        if normalized is not None:
            return normalized
        for balanced in _extract_balanced_json_candidates(fenced):
            normalized = _try_parse_and_normalize_json(balanced)
            if normalized is not None:
                return normalized

    stripped = text
    if stripped.startswith("```json"):
        stripped = stripped[7:].strip()
    elif stripped.startswith("```"):
        stripped = stripped[3:].strip()
    if stripped.endswith("```"):
        stripped = stripped[:-3].strip()

    normalized = _try_parse_and_normalize_json(stripped)
    if normalized is not None:
        return normalized

    for balanced in _extract_balanced_json_candidates(stripped):
        normalized = _try_parse_and_normalize_json(balanced)
        if normalized is not None:
            return normalized

    if fences and len(fences) == 1 and stripped.startswith("```"):
        return fences[0]
    return stripped


def inspect_served_model_info(
    base_url: Optional[str], model_name: Optional[str] = None, timeout: float = 3.0
) -> Optional[Dict[str, Any]]:
    """Best-effort query of OpenAI/vLLM `/models` endpoint to discover `max_model_len`."""
    if not base_url or not str(base_url).startswith(("http://", "https://")):
        return None
    url = str(base_url).rstrip("/") + "/models"
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        for entry in payload.get("data", []):
            if model_name is None or entry.get("id") == model_name:
                return entry
        data = payload.get("data", [])
        return data[0] if data else None
    except (HTTPError, URLError, TimeoutError, Exception):
        return None


def _warn_if_thinking_config_risky(
    model: str, temperature: float, kwargs: Dict[str, Any]
) -> None:
    """Logs diagnostic warnings if thinking appears enabled with loop/truncation-prone settings."""
    extra_body = kwargs.get("extra_body") or {}
    chat_kwargs = extra_body.get("chat_template_kwargs") or {}
    thinking_enabled = chat_kwargs.get("enable_thinking") is True or (
        isinstance(chat_kwargs.get("thinking_budget"), (int, float))
        and chat_kwargs.get("thinking_budget", 0) > 0
    )
    if not thinking_enabled:
        return

    max_tokens = kwargs.get("max_tokens")
    if temperature < 0.3:
        logger.warning(
            "Model '%s' has thinking enabled with low temperature=%s; low temperatures can cause "
            "deterministic reasoning loops (recommend temperature=0.6 and presence_penalty>=0.5).",
            model,
            temperature,
        )
    if isinstance(max_tokens, int) and max_tokens < 4096:
        logger.warning(
            "Model '%s' has thinking enabled with small max_tokens=%d; complex ontology prompts "
            "may truncate before emitting final JSON (recommend max_tokens>=8192 and larger --max-model-len).",
            model,
            max_tokens,
        )


def get_model(
    model: str,
    model_provider: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> Any:
    """
    Description:
    Initializes a chat model using `ReasoningChatOpenAI` for OpenAI-compatible endpoints
    (preserving `reasoning`/`reasoning_content` fields from vLLM `--reasoning-parser`)
    or LangChain's `init_chat_model` for other providers.

    Args:
        model (str): Name of the model to initialize.
        model_provider (str): Provider of the model (e.g. `'openai'`, `'anthropic'`, `'google_genai'`).
        base_url (Optional[str]): Optional custom base URL for the API endpoint. Defaults to None.
        api_key (Optional[str]): Optional custom API key. Defaults to None.
        temperature (float): Sampling temperature. Defaults to 0.0.
        **kwargs (Any): Additional keyword arguments passed to the model constructor.

    Returns:
        Any: The initialized LangChain chat model instance.
    """
    logger.info(f"Initializing model: {model} (provider: {model_provider})")
    _warn_if_thinking_config_risky(model, temperature, kwargs)

    clean_kwargs = {k: v for k, v in kwargs.items() if k != "max_concurrency" and not k.startswith("_")}

    if model_provider == "openai":
        openai_args: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            **clean_kwargs,
        }
        if base_url:
            openai_args["base_url"] = base_url
        if api_key:
            openai_args["api_key"] = api_key
        return ReasoningChatOpenAI(**openai_args)

    init_args: Dict[str, Any] = {
        "model": model,
        "model_provider": model_provider,
        "temperature": temperature,
        **clean_kwargs,
    }
    if base_url:
        init_args["base_url"] = base_url
    if api_key:
        init_args["api_key"] = api_key

    return init_chat_model(**init_args)


def test_model(model: Any, max_retries: int = 10, delay: int = 10) -> bool:
    """
    Description:
    Sends a connectivity check prompt to the model, verifies non-empty final output
    outside any reasoning channel (`reasoning_content`, structured blocks, or `<think>` tags),
    and inspects vLLM `max_model_len` headroom when available.

    Args:
        model (Any): Initialized LangChain chat model client.
        max_retries (int): Maximum number of retry attempts. Defaults to 10.
        delay (int): Seconds to wait between retries. Defaults to 10.

    Returns:
        bool: True if connectivity and valid final output succeeded, False otherwise.
    """
    base_url = getattr(model, "base_url", None) or getattr(model, "openai_api_base", "Unknown URL")
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None)
    logger.info(f"Testing model connectivity to: {base_url}")

    served_info = inspect_served_model_info(str(base_url), model_name=model_name)
    if served_info and "max_model_len" in served_info:
        max_model_len = served_info["max_model_len"]
        req_max_tokens = getattr(model, "max_tokens", None)
        logger.info(f"Server reports '{served_info.get('id', model_name)}' max_model_len={max_model_len}")
        if isinstance(req_max_tokens, int) and isinstance(max_model_len, int) and req_max_tokens >= max_model_len:
            logger.warning(
                "Configured max_tokens (%d) >= server max_model_len (%d); long prompts will fail unless "
                "prompt + max_tokens <= max_model_len (increase vLLM --max-model-len or reduce max_tokens).",
                req_max_tokens,
                max_model_len,
            )

    for attempt in range(max_retries):
        try:
            response = model.invoke("Hello")
            reasoning_text, clean_content = extract_reasoning_and_content(response)
            finish_reason = get_finish_reason(response)

            if finish_reason == "length":
                logger.warning(
                    "Model test hit finish_reason='length' (reasoning_chars=%d, content_chars=%d).",
                    len(reasoning_text),
                    len(clean_content),
                )

            if not clean_content:
                raise ValueError(
                    f"Empty final content from model (finish_reason={finish_reason!r}, "
                    f"reasoning_chars={len(reasoning_text)})."
                )

            if reasoning_text:
                logger.info(f"Model test successful: {clean_content} (reasoning: {len(reasoning_text)} chars)")
            else:
                logger.info(f"Model test successful: {clean_content}")
            return True
        except Exception as e:
            logger.warning(f"Connection test attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)

    logger.error("Could not connect cleanly to the LLM model after multiple attempts.")
    return False
