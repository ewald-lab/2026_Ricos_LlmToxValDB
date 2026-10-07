from __future__ import annotations
import argparse
import json
import os
import re
import pandas as pd
import logging
import time
import asyncio
import sys
from typing import Any

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from tqdm import tqdm
from pathlib import Path
from model.llm import get_model, test_model
from src.utils import (
    setup_logging,
    load_pipeline_prompts,
    load_system_prompt,
    clean_llm_response,
    parse_llm_result,
    process_clean_exceptions_batch,
    needs_cleanup,
)

logger = logging.getLogger(__name__)

def arg_parser() -> argparse.Namespace:
    """
    Description:
    Parses command line arguments for the negatives-focused LLM cleaning script.

    Args:
        None

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="LLM-based cleaning of toxicological effects: focused on NEGATIVE effects (no-observed-effects)."
    )
    parser.add_argument("--input", default="ToxValDB_v9.7.0_Cleaned_01.tsv", help="Path to input TSV file.")
    parser.add_argument("--output", default="ToxValDB_v9.7.0_Cleaned_02_negatives.tsv", help="Path to output TSV file.")
    parser.add_argument("--reasoning_file", default="llm_clean_reasoning_negatives.json", help="Path to save LLM reasoning traces.")
    parser.add_argument("--model_config", required=True, help="Path to model config JSON file.")
    parser.add_argument("--base_url", default=None, help="Base URL for the LLM API. Overrides config value.")
    parser.add_argument(
        "--pipeline_dir",
        default=None,
        help="Path to directory containing sequential pipeline prompts.",
    )
    parser.add_argument(
        "--system_prompt_positivize_file",
        default=None,
        help="Path to system prompt for converting negative effects to positive.",
    )
    parser.add_argument(
        "--system_prompt_exceptions_file",
        default=None,
        help="Path to system prompt for clean-exceptions phase.",
    )
    parser.add_argument("--target_column", default="TOXICOLOGICAL_EFFECT", help="Column to process.")
    parser.add_argument("--max_concurrency", type=int, default=20, help="Max concurrent LLM calls.")
    parser.add_argument("--debug", action="store_true", help="If set, process only 100 random unique descriptions.")
    parser.add_argument("--debug_rows", type=int, nargs="+", default=None, help="Row indices to process for debugging.")
    parser.add_argument("--keep-non-tox", action="store_true", help="If set, do not remove entries labeled as is_negative_tox_effect: false.")
    return parser.parse_args()

async def process_batch(
    llm: Any, prompts: list[tuple[str, str]], descriptions: list[str], max_concurrency: int
) -> dict[str, Any]:
    """
    Description:
    Runs the sequential prompts cleaning pipeline on a batch of negative effect descriptions concurrently.

    Args:
        llm (Any): The LLM engine wrapper.
        prompts (list[tuple[str, str]]): List of prompt tuples (filename, content) in the pipeline.
        descriptions (list[str]): List of raw descriptions to clean.
        max_concurrency (int): Maximum concurrent LLM requests.

    Returns:
        dict[str, Any]: A dictionary mapping original descriptions to their cleaned negative effects and metadata.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    cache = {}
    counter = 0

    async def process_one(desc: str) -> None:
        """
        Description:
        Helper coroutine to process a single description through the negative effects pipeline.

        Args:
            desc (str): The raw description string.

        Returns:
            None
        """
        nonlocal counter
        async with semaphore:
            if desc.strip() == "-":
                cache[desc] = {
                    "effects": ["-"],
                    "is_negative_tox_effect": True,
                    "reasoning": "Directly accepted '-' as requested."
                }
                counter += 1
                return

            current_effects = [desc]
            full_reasoning = {}
            is_negative_tox = True

            for i, (fname, sys_prompt) in enumerate(prompts):
                user_message = (
                    f'Original entry: "{desc}"\n'
                    f'Current cleaned effects: {current_effects}'
                )

                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_message},
                ]

                success = False
                for attempt in range(5):
                    try:
                        response = await llm.ainvoke(messages)
                        result = parse_llm_result(response.content)

                        if result.get("parse_error"):
                            raise ValueError(result.get("reasoning", "Parse error"))

                        step_name = fname.replace("system_prompt_clean_", "").replace(".txt", "").split("_", 1)[-1]
                        step_data = {"reasoning": result.get("reasoning")}
                        if "effects" in result:
                            step_data["effects"] = result.get("effects")
                        full_reasoning[step_name] = step_data

                        if "effects" in result:
                            current_effects = result.get("effects", [])

                        print(f"[{counter}] Step: {step_name} | Effects: {current_effects} | Reasoning: {result.get('reasoning')}", flush=True)

                        if "is_negative_tox_effect" in result:
                            is_negative_tox = result.get("is_negative_tox_effect", True)
                            if not is_negative_tox:
                                current_effects = []
                                break

                        if i > 0 and len(current_effects) == 0:
                            is_negative_tox = False
                            break

                        success = True
                        break
                    except Exception as e:
                        if attempt == 4:
                            logger.error(f"LLM error for '{desc[:30]}' at step {fname}: {e}")
                            break
                        await asyncio.sleep(1)

                if not is_negative_tox or not success:
                    break

            final_result = {
                "effects": current_effects,
                "is_negative_tox_effect": is_negative_tox,
                "reasoning": full_reasoning
            }
            cache[desc] = final_result
            counter += 1
            status_tag = "NOT-TOX" if not is_negative_tox else "Cleaned"
            print(f"{counter}: LLM Pipeline | Orig: '{desc}' -> {status_tag}: {current_effects}", flush=True)

    tasks = [process_one(d) for d in descriptions]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM Pipeline Processing", smoothing=0.0):
        await f

    return cache

async def process_positivize_batch(
    llm: Any, system_prompt: str, effects: list[str], max_concurrency: int
) -> dict[str, Any]:
    """
    Description:
    Converts a batch of negative effects to their positive counterparts using the LLM.

    Args:
        llm (Any): The LLM engine wrapper.
        system_prompt (str): The positivization system prompt instructions.
        effects (list[str]): List of negative effect terms to positivize.
        max_concurrency (int): Maximum concurrent LLM requests.

    Returns:
        dict[str, Any]: A dictionary mapping negative effect terms to positive names.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    cache = {}
    counter = 0

    async def process_one(effect: str) -> None:
        """
        Description:
        Helper coroutine to positivize a single negative effect.

        Args:
            effect (str): The negative effect term.

        Returns:
            None
        """
        nonlocal counter
        async with semaphore:
            if effect == "-":
                cache[effect] = {"positive_name": "-", "reasoning": "Preserved '-'"}
                counter += 1
                return

            user_message = f'Remove the negation from this toxicological effect to make it a positive finding:\n\n"{effect}"'
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            for attempt in range(5):
                try:
                    response = await llm.ainvoke(messages)
                    cleaned = clean_llm_response(response.content)
                    try:
                        result = json.loads(cleaned)
                    except json.JSONDecodeError:
                        raise ValueError("Invalid JSON returned by LLM")

                    if "positive_name" not in result:
                        raise ValueError("Missing 'positive_name' in JSON")

                    cache[effect] = result
                    counter += 1
                    print(f"{counter}: LLM Positivize | Neg: '{effect}' -> Pos: {result['positive_name']}", flush=True)
                    return
                except Exception as e:
                    if attempt == 4:
                        logger.error(f"LLM positivize error for '{effect[:50]}...' after 5 attempts: {e}")
                        cache[effect] = {
                            "positive_name": effect,
                            "reasoning": f"LLM error: {e}",
                        }
                    else:
                        await asyncio.sleep(1)

    tasks = [process_one(e) for e in effects]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM Positivizing", smoothing=0.0):
        await f
    return cache

def main() -> None:
    """
    Description:
    Main entry point for the negatives-focused LLM cleaning script. It loads input data,
    runs the negative pipeline, positivizes negative effects, standardizes, and writes output.

    Args:
        None

    Returns:
        None
    """
    args = arg_parser()

    global logger
    logger = setup_logging(level=logging.INFO, stream=sys.stdout, fmt="%(asctime)s - %(levelname)s - %(message)s")

    start_time = time.time()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # --- Setup default paths for prompts ---
    if args.pipeline_dir is None:
        args.pipeline_dir = str(project_root / "config" / "clean_prompts_negatives" / "pipeline")

    if args.system_prompt_positivize_file is None:
        args.system_prompt_positivize_file = str(project_root / "config" / "clean_prompts_negatives" / "system_prompt_positivize.txt")

    if args.system_prompt_exceptions_file is None:
        args.system_prompt_exceptions_file = str(project_root / "config" / "clean_prompts_negatives" / "system_prompt_clean_exceptions.txt")

    # --- Model ---
    with open(args.model_config, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    max_concurrency = args.max_concurrency or model_config.pop("max_concurrency", 10)
    model_config.pop("max_concurrency", None)

    if args.base_url:
        model_config["base_url"] = args.base_url

    logger.info(f"Loading model from config: {args.model_config}")
    llm = get_model(**model_config)

    if not test_model(llm, max_retries=3):
        raise RuntimeError("Model connectivity test failed.")

    # --- Data ---
    logger.info(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input, sep="\t", low_memory=False)
    if len(df.columns) == 1 and "," in df.columns[0]:
        df = pd.read_csv(args.input, sep=",", low_memory=False)
    logger.info(f"Input shape: {df.shape}")

    col = args.target_column
    unique_descs = df[col].dropna().unique().tolist()
    unique_descs = [d for d in unique_descs if isinstance(d, str) and (len(d.strip()) >= 4 or d.strip() == "-")]

    if args.debug:
        import random
        random.seed(45)
        random.shuffle(unique_descs)
        unique_descs = unique_descs[:100]
        logger.info("DEBUG MODE: Selected 100 random unique descriptions.")

    # --- Phase 1: Pipeline Cleaning (Negatives) ---
    prompts = load_pipeline_prompts(args.pipeline_dir)
    if not prompts:
        logger.error(f"No prompts found in pipeline directory: {args.pipeline_dir}")
        return

    cache = asyncio.run(process_batch(llm, prompts, unique_descs, max_concurrency))

    # --- Apply Phase 1 results ---
    new_rows = []
    reasoning_output = {}
    temp_neg_col = '_TEMP_CLEANED_NEGATIVE'

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Applying LLM Results"):
        desc = row.get(col)
        if not isinstance(desc, str) or (len(desc.strip()) < 4 and desc.strip() != "-"):
            row_dict = row.to_dict()
            row_dict[temp_neg_col] = desc
            new_rows.append(row_dict)
            continue

        result = cache.get(desc)
        if result is None:
            if args.debug: continue
            row_dict = row.to_dict()
            row_dict[temp_neg_col] = desc
            new_rows.append(row_dict)
            continue

        reasoning_output[desc] = {
            "reasoning": result.get("reasoning", ""),
            "effects": result.get("effects", []),
        }

        if not args.keep_non_tox and not result.get("is_negative_tox_effect", True):
            continue

        effects = result.get("effects", [])
        if not effects:
            row_dict = row.to_dict()
            row_dict[temp_neg_col] = desc
            new_rows.append(row_dict)
            continue

        for eff in effects:
            eff = eff.strip()
            if eff:
                row_dict = row.to_dict()
                row_dict[temp_neg_col] = eff
                new_rows.append(row_dict)

    result_df = pd.DataFrame(new_rows)

    # --- Phase 1.9: Exceptions ---
    if os.path.exists(args.system_prompt_exceptions_file):
        sys_prompt_exceptions = load_system_prompt(args.system_prompt_exceptions_file)



        unique_to_clean = result_df[temp_neg_col].dropna().unique().tolist()
        effects_to_fix = [e for e in unique_to_clean if needs_cleanup(e)]
        if effects_to_fix:
            exceptions_cache = asyncio.run(process_clean_exceptions_batch(llm, sys_prompt_exceptions, effects_to_fix, max_concurrency))
            result_df[temp_neg_col] = result_df[temp_neg_col].apply(lambda e: exceptions_cache.get(e, {"corrected": e})["corrected"] if isinstance(e, str) else e)

    # --- Phase 2: Positivize ---
    logger.info("Starting Phase 2: Positivizing cleaned negative effects...")
    if not os.path.exists(args.system_prompt_positivize_file):
        logger.error(f"Positivize system prompt not found: {args.system_prompt_positivize_file}")
        result_df['TOXICOLOGICAL_EFFECT_LLM_CLEANED'] = result_df[temp_neg_col]
    else:
        sys_prompt_pos = load_system_prompt(args.system_prompt_positivize_file)
        cleaned_neg_effects = result_df[temp_neg_col].dropna().unique().tolist()
        pos_cache = asyncio.run(process_positivize_batch(llm, sys_prompt_pos, cleaned_neg_effects, max_concurrency))

        result_df['TOXICOLOGICAL_EFFECT_LLM_CLEANED'] = result_df[temp_neg_col].apply(
            lambda e: pos_cache.get(e, {"positive_name": e})["positive_name"] if isinstance(e, str) else e
        )

    result_df['NEGATIVE'] = True

    # --- Save ---
    logger.info(f"Saving results to: {args.output}")


    target_cols = ['TOXICOLOGICAL_EFFECT', 'NEGATIVE', 'TOXICOLOGICAL_EFFECT_LLM_CLEANED']
    cols_to_keep = [c for c in result_df.columns if c not in target_cols and c != temp_neg_col]
    final_cols = target_cols + cols_to_keep

    # Check if columns exist
    existing_cols = [c for c in final_cols if c in result_df.columns]
    result_df[existing_cols].to_csv(args.output, sep="\t", index=False)

    with open(args.reasoning_file, "w", encoding="utf-8") as f:
        json.dump(reasoning_output, f, indent=2, ensure_ascii=False)

    logger.info(f"Total execution time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
