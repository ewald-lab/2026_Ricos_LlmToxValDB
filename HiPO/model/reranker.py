import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_SYSTEM_INSTRUCTION = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".'
)

# Explicit registry of supported causal LM yes/no reranker families.
# Models whose `model_type` is an encoder/sequence-classifier or lacks a compatible
# yes/no chat template are rejected explicitly with ValueError rather than
# falling back silently to Qwen3 ChatML tags.
RERANKER_FAMILY_SPECS: Dict[str, Dict[str, Any]] = {
    "qwen3": {"true_token": "yes", "false_token": "no", "prefer_three_role": True},
    "qwen2": {"true_token": "yes", "false_token": "no", "prefer_three_role": True},
    "qwen3_5": {"true_token": "yes", "false_token": "no", "prefer_three_role": True},
    "qwen3_5_text": {"true_token": "yes", "false_token": "no", "prefer_three_role": True},
    "llama": {"true_token": "yes", "false_token": "no", "prefer_three_role": False},
    "mistral": {"true_token": "yes", "false_token": "no", "prefer_three_role": False},
    "gemma2": {"true_token": "yes", "false_token": "no", "prefer_three_role": False},
    "gemma3": {"true_token": "yes", "false_token": "no", "prefer_three_role": False},
    "gemma3_text": {"true_token": "yes", "false_token": "no", "prefer_three_role": False},
    "gemma4": {"true_token": "yes", "false_token": "no", "prefer_three_role": False},
    "nemotron": {"true_token": "yes", "false_token": "no", "prefer_three_role": False},
    "nemotron_h": {"true_token": "yes", "false_token": "no", "prefer_three_role": False},
}

UNSUPPORTED_ENCODER_MODEL_TYPES = frozenset({
    "albert",
    "bert",
    "camembert",
    "deberta",
    "deberta-v2",
    "distilbert",
    "electra",
    "mpnet",
    "roberta",
    "xlm-roberta",
})

_SENTINEL_INST_A = "__HIPO_INST_PROBE_A__"
_SENTINEL_QUERY_A = "__HIPO_QUERY_PROBE_A__"
_SENTINEL_DOC_A = "__HIPO_DOC_PROBE_A__"
_SENTINEL_INST_B = "__HIPO_INST_PROBE_B__"
_SENTINEL_QUERY_B = "__HIPO_QUERY_PROBE_B__"
_SENTINEL_DOC_B = "__HIPO_DOC_PROBE_B__"
_SENTINEL_BODY_A = "__HIPO_BODY_PROBE_A__"
_SENTINEL_BODY_B = "__HIPO_BODY_PROBE_B__"


@dataclass(frozen=True)
class RerankerPromptSpec:
    """Validated prompt rendering specification derived from a tokenizer's native chat template."""

    schema: str  # "three_role" (system, query, document) or "two_role" (system, user)
    system_instruction: str
    suffix: str
    suffix_tokens: Tuple[int, ...]
    token_true_id: int
    token_false_id: int
    max_content_len: int
    pass_disable_thinking: bool


def validate_reranker_architecture(config: Any, model_name: str = "") -> Dict[str, Any]:
    """
    Description:
    Validates that a Hugging Face model config represents a supported causal yes/no reranker family.

    Args:
        config (Any): Loaded Hugging Face PretrainedConfig object.
        model_name (str): Model identifier for clear error messages.

    Returns:
        Dict[str, Any]: Registered family spec containing expected yes/no tokens.

    Raises:
        ValueError: If the model is a sequence-classification/encoder model or unregistered family.
    """
    label = model_name or getattr(config, "_name_or_path", "<unknown>")
    architectures = list(getattr(config, "architectures", None) or [])
    model_type = (getattr(config, "model_type", None) or "").lower()

    if any("SequenceClassification" in arch or "MaskedLM" in arch or "TokenClassification" in arch for arch in architectures):
        raise ValueError(
            f"Unsupported reranker architecture {architectures} for '{label}'. "
            "CausalReranker requires a supported causal LM reranker with a chat template and 'yes'/'no' tokens."
        )

    if model_type in UNSUPPORTED_ENCODER_MODEL_TYPES:
        raise ValueError(
            f"Unsupported encoder reranker model_type '{model_type}' for '{label}'. "
            "Cross-encoder classification models cannot use causal yes/no logit scoring."
        )

    spec = RERANKER_FAMILY_SPECS.get(model_type)
    if spec is None:
        supported = ", ".join(sorted(RERANKER_FAMILY_SPECS))
        raise ValueError(
            f"Unsupported reranker model family '{model_type or '<unknown>'}' for '{label}' "
            f"(architectures={architectures}). Registered families: {supported}."
        )
    return spec


def _resolve_single_token_id(tokenizer: Any, token: str, model_name: str = "") -> int:
    """Validates and returns a single integer vocabulary ID for `token`."""
    label = model_name or getattr(tokenizer, "name_or_path", "<tokenizer>")
    tok_id = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if not isinstance(tok_id, int) or tok_id < 0 or (unk_id is not None and tok_id == unk_id):
        raise ValueError(
            f"Reranker tokenizer '{label}' does not map target answer token '{token}' to a valid token ID "
            f"(got {tok_id!r}, unk_token_id={unk_id!r})."
        )
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if encoded != [tok_id]:
        raise ValueError(
            f"Reranker tokenizer '{label}' encodes target answer token '{token}' as {encoded} "
            f"instead of single token [{tok_id}]."
        )
    return tok_id


def _apply_chat_template_text(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    *,
    enable_thinking: Optional[bool] = None,
) -> Tuple[str, bool]:
    """
    Renders messages with `tokenizer.apply_chat_template`.
    Returns `(rendered_text, used_enable_thinking_kwarg)`.
    """
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        try:
            rendered = tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
            return rendered, True
        except TypeError:
            pass
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    return rendered, False


def _probe_three_role_template(tokenizer: Any) -> Optional[str]:
    """
    Probes whether `tokenizer.chat_template` natively renders `[system, query, document]`.
    Requires the Jinja template source itself to reference both `query` and `document` roles
    so generic message loops are not misclassified as 3-role reranker templates.
    Returns the constant generation suffix string if valid, else None.
    """
    tmpl_src = getattr(tokenizer, "chat_template", "") or ""
    if not isinstance(tmpl_src, str) or "query" not in tmpl_src or "document" not in tmpl_src:
        return None
    try:
        rendered_a, _ = _apply_chat_template_text(
            tokenizer,
            [
                {"role": "system", "content": _SENTINEL_INST_A},
                {"role": "query", "content": _SENTINEL_QUERY_A},
                {"role": "document", "content": _SENTINEL_DOC_A},
            ],
        )
        rendered_b, _ = _apply_chat_template_text(
            tokenizer,
            [
                {"role": "system", "content": _SENTINEL_INST_B},
                {"role": "query", "content": _SENTINEL_QUERY_B},
                {"role": "document", "content": _SENTINEL_DOC_B},
            ],
        )
    except Exception:
        return None

    required_a = (_SENTINEL_INST_A, _SENTINEL_QUERY_A, _SENTINEL_DOC_A)
    required_b = (_SENTINEL_INST_B, _SENTINEL_QUERY_B, _SENTINEL_DOC_B)
    if not (all(s in rendered_a for s in required_a) and all(s in rendered_b for s in required_b)):
        return None

    # Ensure sentinels appear in system -> query -> document order
    idx_ia = rendered_a.index(_SENTINEL_INST_A)
    idx_qa = rendered_a.index(_SENTINEL_QUERY_A)
    idx_da = rendered_a.index(_SENTINEL_DOC_A)
    if not (idx_ia < idx_qa < idx_da):
        return None

    suffix_a = rendered_a.split(_SENTINEL_DOC_A, 1)[1]
    suffix_b = rendered_b.split(_SENTINEL_DOC_B, 1)[1]
    if suffix_a != suffix_b:
        return None
    return suffix_a


def _probe_two_role_template(
    tokenizer: Any, system_instruction: str
) -> Optional[Tuple[str, bool]]:
    """
    Probes standard `[system, user]` chat templates with thinking disabled when supported.
    Returns `(suffix, used_enable_thinking_kwarg)` if valid, else None.
    """
    try:
        rendered_a, used_kw = _apply_chat_template_text(
            tokenizer,
            [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": _SENTINEL_BODY_A},
            ],
            enable_thinking=False,
        )
        rendered_b, _ = _apply_chat_template_text(
            tokenizer,
            [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": _SENTINEL_BODY_B},
            ],
            enable_thinking=False if used_kw else None,
        )
    except Exception:
        return None

    if system_instruction not in rendered_a or _SENTINEL_BODY_A not in rendered_a or _SENTINEL_BODY_B not in rendered_b:
        return None

    suffix_a = rendered_a.split(_SENTINEL_BODY_A, 1)[1]
    suffix_b = rendered_b.split(_SENTINEL_BODY_B, 1)[1]
    if suffix_a != suffix_b:
        return None
    return suffix_a, used_kw


def build_reranker_prompt_spec(
    tokenizer: Any,
    *,
    model_name: str = "",
    system_instruction: str = DEFAULT_RERANKER_SYSTEM_INSTRUCTION,
    true_token: str = "yes",
    false_token: str = "no",
    max_length: int = 1024,
) -> RerankerPromptSpec:
    """
    Description:
    Derives prompt schema and suffix tokens dynamically using `tokenizer.apply_chat_template`,
    failing loudly with ValueError on missing templates or incompatible tokenizers.

    Args:
        tokenizer (Any): Hugging Face tokenizer configured with `chat_template`.
        model_name (str): Model name for diagnostic error messages.
        system_instruction (str): System instruction for standard `[system, user]` chat templates.
        true_token (str): Positive class token (default `"yes"`).
        false_token (str): Negative class token (default `"no"`).
        max_length (int): Total sequence token cap (prompt + suffix).

    Returns:
        RerankerPromptSpec: Validated prompt specification for `CausalReranker`.

    Raises:
        ValueError: If `chat_template` is absent, template rendering fails/drops roles,
            or `true_token`/`false_token` cannot be resolved cleanly.
    """
    label = model_name or getattr(tokenizer, "name_or_path", "<tokenizer>")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"Reranker tokenizer '{label}' does not define a `chat_template`. "
            "Refusing to fall back to hardcoded ChatML tags."
        )

    token_true_id = _resolve_single_token_id(tokenizer, true_token, model_name=label)
    token_false_id = _resolve_single_token_id(tokenizer, false_token, model_name=label)

    three_role_suffix = _probe_three_role_template(tokenizer)
    if three_role_suffix is not None:
        schema = "three_role"
        suffix = three_role_suffix
        pass_disable_thinking = False
    else:
        two_role_probe = _probe_two_role_template(tokenizer, system_instruction)
        if two_role_probe is None:
            raise ValueError(
                f"Reranker tokenizer '{label}' chat_template is incompatible with both "
                "3-role [system, query, document] and 2-role [system, user] yes/no prompting."
            )
        schema = "two_role"
        suffix, pass_disable_thinking = two_role_probe

    suffix_tokens = tuple(tokenizer.encode(suffix, add_special_tokens=False)) if suffix else ()
    max_content_len = max(1, max_length - len(suffix_tokens))

    return RerankerPromptSpec(
        schema=schema,
        system_instruction=system_instruction,
        suffix=suffix,
        suffix_tokens=suffix_tokens,
        token_true_id=token_true_id,
        token_false_id=token_false_id,
        max_content_len=max_content_len,
        pass_disable_thinking=pass_disable_thinking,
    )


class CausalReranker:
    """Shared yes/no causal LM reranker powered by each model's native chat template."""

    def __init__(
        self,
        model_name: str,
        instruction: Optional[str] = None,
        batch_size: int = 100,
        max_length: int = 1024,
        torch_dtype: Any = torch.bfloat16,
        device_map: Any = "auto",
    ) -> None:
        """
        Description:
        Initializes a validated causal LM reranker without hardcoding model-specific chat tokens.

        Args:
            model_name (str): Hugging Face model ID or local path.
            instruction (Optional[str]): Default ranking instruction text.
            batch_size (int): Default batch size for scoring pairs.
            max_length (int): Maximum sequence length passed to the causal model.
            torch_dtype (Any): Weight dtype on CUDA devices.
            device_map (Any): Hugging Face device placement strategy on CUDA.

        Returns:
            None
        """
        self.model_name = model_name
        self.instruction = (
            instruction.strip()
            if instruction
            else "Given a web search query, retrieve relevant passages that answer the query."
        )
        self.batch_size = batch_size
        self.max_length = max_length
        self._lock = threading.Lock()

        config = AutoConfig.from_pretrained(model_name)
        family_spec = validate_reranker_architecture(config, model_name=model_name)

        logger.info(f"Loading Reranker Model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.prompt_spec = build_reranker_prompt_spec(
            self.tokenizer,
            model_name=model_name,
            system_instruction=DEFAULT_RERANKER_SYSTEM_INSTRUCTION,
            true_token=family_spec["true_token"],
            false_token=family_spec["false_token"],
            max_length=max_length,
        )

        use_cuda = torch.cuda.is_available()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype if use_cuda else torch.float32,
            device_map=device_map if use_cuda else None,
        ).eval()
        self.device = self.model.device

    def format_left_prompt(self, query: str, document: str, instruction: Optional[str] = None) -> str:
        """Renders the pair prompt up to (excluding) the trailing assistant generation suffix."""
        inst = self.instruction if instruction is None else instruction
        spec = self.prompt_spec
        if spec.schema == "three_role":
            messages = [
                {"role": "system", "content": inst},
                {"role": "query", "content": query},
                {"role": "document", "content": document},
            ]
            rendered, _ = _apply_chat_template_text(self.tokenizer, messages)
        else:
            body = f"<Instruct>: {inst}\n<Query>: {query}\n<Document>: {document}"
            messages = [
                {"role": "system", "content": spec.system_instruction},
                {"role": "user", "content": body},
            ]
            rendered, _ = _apply_chat_template_text(
                self.tokenizer,
                messages,
                enable_thinking=False if spec.pass_disable_thinking else None,
            )

        if spec.suffix:
            if not rendered.endswith(spec.suffix):
                raise ValueError(
                    f"Rendered prompt for '{self.model_name}' unexpectedly did not end with derived suffix."
                )
            return rendered[: len(rendered) - len(spec.suffix)]
        return rendered

    def encode_pairs(
        self, pairs: Sequence[Tuple[str, str]], instruction: Optional[str] = None
    ) -> Dict[str, List[List[int]]]:
        """Encodes `(query, document)` pairs using the derived chat template and suffix tokens."""
        left_prompts = [self.format_left_prompt(q, d, instruction=instruction) for q, d in pairs]
        inputs = self.tokenizer(
            left_prompts,
            add_special_tokens=False,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.prompt_spec.max_content_len,
        )
        suffix_list = list(self.prompt_spec.suffix_tokens)
        if suffix_list:
            inputs["input_ids"] = [ids + suffix_list for ids in inputs["input_ids"]]
        return inputs

    def score_pairs(
        self,
        pairs: Sequence[Tuple[str, str]],
        instruction: Optional[str] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[float]:
        """
        Description:
        Scores a sequence of `(query, document)` pairs returning positive-class probabilities in `[0, 1]`.

        Args:
            pairs (Sequence[Tuple[str, str]]): Pairs of `(query, document)` strings.
            instruction (Optional[str]): Optional override for ranking instruction.
            batch_size (Optional[int]): Optional override for batch size.
            show_progress (bool): Whether to display a tqdm progress bar.

        Returns:
            List[float]: Yes-probabilities aligned with `pairs`.
        """
        if not pairs:
            return []

        effective_batch = batch_size or self.batch_size
        scores: List[float] = []
        indices = range(0, len(pairs), effective_batch)
        if show_progress:
            indices = tqdm(indices, desc="Reranking", smoothing=0.0)

        with self._lock:
            with torch.no_grad():
                for start in indices:
                    batch_pairs = pairs[start : start + effective_batch]
                    encoded = self.encode_pairs(batch_pairs, instruction=instruction)
                    padded = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
                    padded = {k: v.to(self.device) for k, v in padded.items()}

                    logits = self.model(**padded).logits[:, -1, :]
                    true_vec = logits[:, self.prompt_spec.token_true_id]
                    false_vec = logits[:, self.prompt_spec.token_false_id]
                    pair_logits = torch.stack([false_vec, true_vec], dim=1)
                    probs = torch.nn.functional.log_softmax(pair_logits, dim=1)[:, 1].exp().tolist()
                    scores.extend(probs)
        return scores

    def close(self) -> None:
        """Releases model/tokenizer references when done."""
        self.model = None
        self.tokenizer = None
