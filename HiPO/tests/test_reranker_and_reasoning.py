import glob
import json
import os
from types import SimpleNamespace
import unittest

from jinja2 import Environment
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from HiPO.model.llm import (
    ReasoningChatOpenAI,
    clean_llm_output,
    extract_reasoning_and_content,
    test_model,
)
from HiPO.model.reranker import (
    DEFAULT_RERANKER_SYSTEM_INSTRUCTION,
    build_reranker_prompt_spec,
    validate_reranker_architecture,
)
from HiPO.src.graph import TextToOntoGraph


class _DummyTokenizer:
    """Minimal Hugging Face-like tokenizer backed by Jinja2 for unit testing chat templates."""

    def __init__(
        self,
        chat_template,
        vocab=None,
        unk_token_id=0,
        name_or_path="dummy/tokenizer",
        accept_enable_thinking=False,
    ):
        self.chat_template = chat_template
        self.vocab = vocab if vocab is not None else {"yes": 9693, "no": 2152}
        self.unk_token_id = unk_token_id
        self.name_or_path = name_or_path
        self.accept_enable_thinking = accept_enable_thinking
        self._jinja = Environment()

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        if text in self.vocab:
            return [self.vocab[text]]
        # Deterministic byte encoding shifted away from unk/special IDs
        return [100 + ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        if "enable_thinking" in kwargs and not self.accept_enable_thinking:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        tmpl = self._jinja.from_string(self.chat_template)
        rendered = tmpl.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=kwargs.get("enable_thinking", True),
        )
        if tokenize:
            ids = self.encode(rendered)
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return rendered


class RerankerTemplateTests(unittest.TestCase):
    def test_three_role_qwen3_style_template_derived_dynamically(self):
        jinja = (
            "{%- set inst = messages | selectattr('role','eq','system') | map(attribute='content') | first -%}"
            "{%- set q = messages | selectattr('role','eq','query') | map(attribute='content') | first -%}"
            "{%- set d = messages | selectattr('role','eq','document') | map(attribute='content') | first -%}"
            "<SYS>{{ inst }}<Q>{{ q }}<D>{{ d }}<END><THINK></THINK>"
        )
        tok = _DummyTokenizer(jinja)
        spec = build_reranker_prompt_spec(tok)
        self.assertEqual(spec.schema, "three_role")
        self.assertEqual(spec.suffix, "<END><THINK></THINK>")
        self.assertEqual(spec.token_true_id, 9693)
        self.assertEqual(spec.token_false_id, 2152)

    def test_two_role_template_with_enable_thinking_toggle(self):
        jinja = (
            "{%- for m in messages -%}[{{ m.role }}:{{ m.content }}]{%- endfor -%}"
            "{%- if add_generation_prompt -%}[assistant:{% if enable_thinking %}<think>{% endif %}]{%- endif -%}"
        )
        tok = _DummyTokenizer(jinja, accept_enable_thinking=True)
        spec = build_reranker_prompt_spec(tok)
        self.assertEqual(spec.schema, "two_role")
        self.assertTrue(spec.pass_disable_thinking)
        self.assertEqual(spec.suffix, "][assistant:]")
        self.assertIn(DEFAULT_RERANKER_SYSTEM_INSTRUCTION[:20], tok.apply_chat_template(
            [{"role": "system", "content": DEFAULT_RERANKER_SYSTEM_INSTRUCTION}, {"role": "user", "content": "x"}],
            tokenize=False,
            enable_thinking=False,
        ))

    def test_missing_chat_template_raises_value_error(self):
        tok = _DummyTokenizer(chat_template=None)
        with self.assertRaises(ValueError):
            build_reranker_prompt_spec(tok)

    def test_missing_yes_no_token_raises_value_error(self):
        jinja = "{% for m in messages %}{{ m.content }}{% endfor %}::OUT::"
        tok = _DummyTokenizer(jinja, vocab={"yes": 10}, unk_token_id=0)
        with self.assertRaises(ValueError):
            build_reranker_prompt_spec(tok)

    def test_unsupported_architecture_and_family_rejected(self):
        enc_cfg = SimpleNamespace(
            _name_or_path="BAAI/bge-reranker-v2-m3",
            architectures=["XLMRobertaForSequenceClassification"],
            model_type="xlm-roberta",
        )
        with self.assertRaises(ValueError):
            validate_reranker_architecture(enc_cfg)

        unknown_cfg = SimpleNamespace(
            _name_or_path="custom/unknown-reranker",
            architectures=["CustomForCausalLM"],
            model_type="unknown_family",
        )
        with self.assertRaises(ValueError):
            validate_reranker_architecture(unknown_cfg)

    def test_cached_qwen3_rerankers_match_chat_template_when_present(self):
        hub = "/nfs/research/ewald/ricos/huggingface_cache/hub"
        if not os.path.isdir(hub):
            self.skipTest("Cluster HF cache directory not present locally.")
        from transformers import AutoConfig, AutoTokenizer

        checked = 0
        for name in ("Qwen--Qwen3-Reranker-0.6B", "Qwen--Qwen3-Reranker-4B", "Qwen--Qwen3-Reranker-8B"):
            snaps = sorted(glob.glob(f"{hub}/models--{name}/snapshots/*"))
            if not snaps:
                continue
            cfg = AutoConfig.from_pretrained(snaps[-1])
            spec_family = validate_reranker_architecture(cfg, model_name=name)
            tok = AutoTokenizer.from_pretrained(snaps[-1], padding_side="left")
            spec = build_reranker_prompt_spec(
                tok,
                model_name=name,
                true_token=spec_family["true_token"],
                false_token=spec_family["false_token"],
            )
            self.assertEqual(spec.schema, "three_role")
            self.assertEqual(spec.token_true_id, 9693)
            self.assertEqual(spec.token_false_id, 2152)
            checked += 1
        self.assertGreater(checked, 0)


class ReasoningAndCleanerTests(unittest.TestCase):
    def test_extract_additional_kwargs_reasoning_content(self):
        msg = AIMessage(
            content='\n\n{"candidate_name": "hepatomegaly", "reason": "Matches directly", "decision": "ACCEPTED"}',
            additional_kwargs={"reasoning_content": "Step 1: compare liver enlargement."},
            response_metadata={"finish_reason": "stop"},
        )
        reasoning, content = extract_reasoning_and_content(msg)
        self.assertEqual(reasoning, "Step 1: compare liver enlargement.")
        self.assertTrue(content.startswith("{"))

        cleaned = json.loads(clean_llm_output(msg))
        self.assertEqual(cleaned["decision"], "ACCEPTED")

    def test_reasoning_chat_openai_preserves_vllm_reasoning_field(self):
        llm = ReasoningChatOpenAI(model="Qwen3.6-35B-A3B-FP8", api_key="EMPTY", base_url="http://localhost:9999/v1")
        raw_response = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "Qwen3.6-35B-A3B-FP8",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '{"ok": true}',
                        "reasoning": "Analyzed schema and verified.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        result = llm._create_chat_result(raw_response)
        ai_msg = result.generations[0].message
        self.assertEqual(ai_msg.additional_kwargs.get("reasoning_content"), "Analyzed schema and verified.")

    def test_extract_inline_unclosed_and_prefill_thinking(self):
        # Unclosed <think> truncation
        trunc = AIMessage(content="<think>looping forever without end", response_metadata={"finish_reason": "length"})
        reasoning, final = extract_reasoning_and_content(trunc)
        self.assertEqual(reasoning, "looping forever without end")
        self.assertEqual(final, "")
        with self.assertRaises(ValueError):
            clean_llm_output(trunc)

        # Bare closing tag from prefill-opened <think>
        prefill = AIMessage(content="thought process here\n</think>\n```json\n{\"decision\": \"accepted\", \"reason\": \"ok\",}\n```")
        reasoning, _ = extract_reasoning_and_content(prefill)
        self.assertEqual(reasoning, "thought process here")
        parsed = json.loads(clean_llm_output(prefill))
        self.assertEqual(parsed["decision"], "ACCEPTED")

    def test_structured_content_blocks_and_prose_json_extraction(self):
        msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "Comparing organ systems"},
                {"type": "text", "text": 'Sure! Result:\n{"candidate_name": "rash", "reason": "Quoted \\"skin\\" match", "decision": "ACCEPTED"}\nNote: {ignore me}.'},
            ]
        )
        reasoning, _ = extract_reasoning_and_content(msg)
        self.assertEqual(reasoning, "Comparing organ systems")
        parsed = json.loads(clean_llm_output(msg))
        self.assertEqual(parsed["candidate_name"], "rash")

    def test_plain_text_description_analysis_not_corrupted_by_braces(self):
        msg = AIMessage(
            content="Describes hepatic lesion {specificity: organ level}.",
            additional_kwargs={"reasoning_content": "Evaluated specificity."},
        )
        cleaned = clean_llm_output(msg)
        self.assertEqual(cleaned, "Describes hepatic lesion {specificity: organ level}.")

    def test_test_model_rejects_empty_final_output(self):
        class _TruncatedLLM:
            base_url = "mock://local"
            model_name = "mock-thinking"

            def invoke(self, _prompt):
                return AIMessage(
                    content="",
                    additional_kwargs={"reasoning_content": "Still thinking..."},
                    response_metadata={"finish_reason": "length"},
                )

        self.assertFalse(test_model(_TruncatedLLM(), max_retries=2, delay=0))

    def test_graph_retry_clamps_context_and_breaks_low_temp_loop(self):
        graph = object.__new__(TextToOntoGraph)
        graph.max_retries = 3
        graph.max_concurrency = 2

        bound_calls = []

        class _MockLLM:
            temperature = 0.0
            extra_body = {"chat_template_kwargs": {"enable_thinking": True}}

            def bind(self, **kwargs):
                bound_calls.append(kwargs)
                return self

        err_ctx = RuntimeError(
            "This model's maximum context length is 16384 tokens. However, you requested 18000 tokens (2000 in the messages, 16000 in the completion)."
        )
        graph._build_retry_llm(_MockLLM(), err_ctx, retry_idx=0)
        self.assertEqual(bound_calls[0]["max_tokens"], 16384 - 2000 - 64)
        self.assertGreaterEqual(bound_calls[0]["temperature"], 0.6)

        err_len = ValueError("Empty final output from LLM (finish_reason='length', reasoning_chars=9000)")
        graph._build_retry_llm(_MockLLM(), err_len, retry_idx=2)
        self.assertFalse(bound_calls[1]["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_graph_mode2_preserves_thinking_trace_in_reasoning(self):
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from HiPO.src.graph import BatchVerificationResult

        graph = object.__new__(TextToOntoGraph)
        graph.max_retries = 2
        graph.max_concurrency = 2
        graph.consensus_runs = 1
        graph.consensus_threshold = 1
        graph.ontology_definitions = False
        graph.prefix = "MP:"
        graph.categories = {"MP:n1"}
        graph.include_superiors = False
        graph.prompt_map = {}
        graph.nodes = {"n1": {"sanitized_name": "liver necrosis", "def": ""}}
        graph.parent_map = {}
        graph.children_map = {}
        graph._mode2_prompt = ChatPromptTemplate.from_messages([
            ("system", "Evaluate {format_instructions}"),
            ("user", "{description}\n{candidates}"),
        ])
        graph._mode2_parser = PydanticOutputParser(pydantic_object=BatchVerificationResult)
        graph._mode2_format_instructions = graph._mode2_parser.get_format_instructions()

        payload = json.dumps({
            "analysis": "Specific liver cell death.",
            "evaluation": [{"candidate_name": "liver necrosis", "reason": "Direct match.", "decision": "ACCEPTED"}],
            "selected_candidates": ["liver necrosis"],
        })
        fake_llm = RunnableLambda(
            lambda _in: AIMessage(
                content=payload,
                additional_kwargs={"reasoning_content": "Checked hierarchy and synonyms carefully."},
                response_metadata={"finish_reason": "stop"},
            )
        )
        results, trace, hits = graph._execute_mode_2(
            fake_llm,
            "liver necrosis",
            [{"name": "n1", "sanitized_name": "liver necrosis", "def": "", "source": "dense_rag"}],
        )
        self.assertEqual(hits, ["MP:n1"])
        self.assertEqual(results, {"MP:n1": 1})
        self.assertEqual(trace[0]["runs"][0]["thinking"], "Checked hierarchy and synonyms carefully.")

    def test_hipo_cli_mode_overrides_choice_llm_and_thinking_preset_keeps_served_model(self):
        import argparse
        import tempfile
        from unittest.mock import patch
        from HiPO import hipo

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump({"model": "Qwen3.6-35B-A3B-FP8", "mode": 2, "consensus_runs": 1, "ontology_definitions": True}, tf)
            llm_cfg_path = tf.name

        try:
            args_override = argparse.Namespace(
                llm_config=llm_cfg_path,
                model="Qwen3.6-27B-FP8-thinking",
                mode=1,
                consensus_runs=None,
                ontology_definitions=None,
                output_file="/tmp/out_{MODEL}.csv",
                reasoning_file="/tmp/reason_{MODEL}.json",
                system_prompt_file="/tmp/system_prompt_{MODE}.txt",
            )
            hipo.load_llm_config(args_override)
            self.assertEqual(args_override.mode, 1)
            self.assertEqual(args_override.consensus_runs, 1)
            self.assertTrue(args_override.ontology_definitions)
            self.assertEqual(args_override.system_prompt_file, "/tmp/system_prompt_1.txt")

            args_default = argparse.Namespace(
                llm_config=llm_cfg_path,
                model=None,
                mode=None,
                consensus_runs=None,
                ontology_definitions=None,
                output_file="/tmp/out_{MODEL}.csv",
                reasoning_file="/tmp/reason_{MODEL}.json",
                system_prompt_file="/tmp/system_prompt_{MODE}.txt",
            )
            hipo.load_llm_config(args_default)
            self.assertEqual(args_default.mode, 2)
            self.assertEqual(args_default.model, "Qwen3.6-35B-A3B-FP8")
        finally:
            os.remove(llm_cfg_path)

        init_args = argparse.Namespace(
            model="Qwen3.6-27B-FP8-thinking",
            model_config=None,
            base_url="http://codon-gpu-013:8009/v1",
        )
        captured = {}

        def fake_get_model(**kwargs):
            captured.update(kwargs)
            return object()

        with patch.object(hipo, "get_model", side_effect=fake_get_model), patch.object(hipo, "test_model", return_value=True):
            hipo.init_llm(init_args)
        self.assertEqual(captured["model"], "Qwen3.6-27B-FP8")
        self.assertEqual(captured["extra_body"]["chat_template_kwargs"]["enable_thinking"], True)


if __name__ == "__main__":
    unittest.main()
