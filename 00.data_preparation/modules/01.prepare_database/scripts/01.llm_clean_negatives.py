import argparse
import json
import os
import re
import pandas as pd
import logging
import time
import asyncio
import sys

os.environ["HF_HOME"] = "/nfs/research/ewald/ricos/huggingface_cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from tqdm import tqdm
from pathlib import Path
from model.llm import get_model, test_model
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def arg_parser():
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
        "--system_prompt_standardize_file",
        default=None,
        help="Path to system prompt for standardization phase.",
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


def load_pipeline_prompts(directory_path: str) -> list:
    prompts = []
    p_dir = Path(directory_path)
    if not p_dir.is_dir():
        logger.warning(f"Pipeline directory {directory_path} not found.")
        return []

    files = sorted([f for f in p_dir.glob("*.txt") if f.is_file()])
    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            prompts.append((f.name, file.read().strip()))
    logger.info(f"Loaded {len(prompts)} pipeline prompts from {directory_path}")
    return prompts


def load_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def clean_llm_response(raw: str) -> str:
    text = raw
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def parse_llm_result(raw_content: str) -> dict:
    cleaned = clean_llm_response(raw_content)
    try:
        result = json.loads(cleaned)
        return result
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM response: {e}\nRaw: {raw_content[:200]}")
        return {"effects": [], "reasoning": f"Parse error: {e}", "parse_error": True}


async def process_batch(llm, prompts: list, descriptions: list, max_concurrency: int) -> dict:
    semaphore = asyncio.Semaphore(max_concurrency)
    cache = {}
    counter = 0

    async def process_one(desc: str):
        nonlocal counter
        async with semaphore:
            # Handle "-" directly as requested
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

                        # Step-by-step logging as requested
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

async def process_positivize_batch(llm, system_prompt: str, effects: list, max_concurrency: int) -> dict:
    """Process a batch of negative effects to generate their positive counterparts."""
    semaphore = asyncio.Semaphore(max_concurrency)
    cache = {}
    counter = 0

    async def process_one(effect: str):
        nonlocal counter
        async with semaphore:
            # Handle "-" separately
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

async def process_standardize_batch(llm, system_prompt: str, effects: list, max_concurrency: int) -> dict:
    semaphore = asyncio.Semaphore(max_concurrency)
    cache = {}
    counter = 0

    async def process_one(effect: str):
        nonlocal counter
        async with semaphore:
            if effect == "-":
                cache[effect] = {"standard_name": "-", "reasoning": "Preserved '-'"}
                counter += 1
                return

            user_message = f'Provide a standard name for this toxicological effect:\n\n"{effect}"'
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

                    if "standard_name" not in result:
                        raise ValueError("Missing 'standard_name' in JSON")

                    cache[effect] = result
                    counter += 1
                    print(f"{counter}: LLM Std | Orig: '{effect}' -> Pos: {result['standard_name']}", flush=True)
                    return
                except Exception as e:
                    if attempt == 4:
                        logger.error(f"LLM standardize error for '{effect[:50]}...' after 5 attempts: {e}")
                        cache[effect] = {
                            "standard_name": effect,
                            "reasoning": f"LLM error: {e}",
                        }
                    else:
                        await asyncio.sleep(1)

    tasks = [process_one(e) for e in effects]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM Standardizing", smoothing=0.0):
        await f
    return cache

async def process_clean_exceptions_batch(llm, system_prompt: str, effects: list, max_concurrency: int) -> dict:
    semaphore = asyncio.Semaphore(max_concurrency)
    cache = {}
    counter = 0

    async def process_one(effect: str):
        nonlocal counter
        async with semaphore:
            if effect == "-":
                cache[effect] = {"corrected": "-", "reasoning": "Preserved '-'"}
                counter += 1
                return

            user_message = f'Fix and simplify this effect:\n\n"{effect}"'
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

                    if "corrected" not in result:
                        raise ValueError("Missing 'corrected' in JSON")

                    cache[effect] = {"corrected": result["corrected"], "reasoning": result.get("reasoning", "")}
                    counter += 1
                    print(f"{counter}: Exceptions | '{effect}' -> '{result['corrected']}'", flush=True)
                    return
                except Exception as e:
                    if attempt == 4:
                        logger.error(f"Exceptions error for '{effect[:50]}...' after 5 attempts: {e}")
                        cache[effect] = {"corrected": effect, "reasoning": f"Error: {e}"}
                    else:
                        await asyncio.sleep(1)

    tasks = [process_one(e) for e in effects]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Cleaning Exceptions", smoothing=0.0):
        await f
    return cache


def main():
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

    if args.system_prompt_standardize_file is None:
        args.system_prompt_standardize_file = str(project_root / "config" / "clean_prompts_negatives" / "system_prompt_standardize.txt")

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
    # Allow "-" as a valid description for this specific script
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

        def needs_cleanup(effect: str) -> bool:
            if not isinstance(effect, str) or effect == "-": return False
            if effect.count(":") > 1: return True
            if ":" in effect:
                parts = effect.split(":", 1)
                ctx = parts[0].strip().lower()
                eff = parts[1].strip().lower()
                ctx_words = set(ctx.replace("/", " ").replace("-", " ").split())
                eff_words = set(eff.replace("/", " ").replace("-", " ").split())
                if any(len(w) > 3 and w in eff_words for w in ctx_words): return True
            return False

        unique_to_clean = result_df[temp_neg_col].dropna().unique().tolist()
        effects_to_fix = [e for e in unique_to_clean if needs_cleanup(e)]
        if effects_to_fix:
            exceptions_cache = asyncio.run(process_clean_exceptions_batch(llm, sys_prompt_exceptions, effects_to_fix, max_concurrency))
            result_df[temp_neg_col] = result_df[temp_neg_col].apply(lambda e: exceptions_cache.get(e, {"corrected": e})["corrected"] if isinstance(e, str) else e)

    # --- Phase 2: Positivize ---
    logger.info("Starting Phase 2: Positivizing cleaned negative effects...")
    if not os.path.exists(args.system_prompt_positivize_file):
        logger.error(f"Positivize system prompt not found: {args.system_prompt_positivize_file}")
        # Use negative text as cleaned text if positivize fails
        result_df['TOXICOLOGICAL_EFFECT_LLM_CLEANED'] = result_df[temp_neg_col]
    else:
        sys_prompt_pos = load_system_prompt(args.system_prompt_positivize_file)
        cleaned_neg_effects = result_df[temp_neg_col].dropna().unique().tolist()
        pos_cache = asyncio.run(process_positivize_batch(llm, sys_prompt_pos, cleaned_neg_effects, max_concurrency))

        result_df['TOXICOLOGICAL_EFFECT_LLM_CLEANED'] = result_df[temp_neg_col].apply(
            lambda e: pos_cache.get(e, {"positive_name": e})["positive_name"] if isinstance(e, str) else e
        )

    # --- Phase 3: Standardization ---
    logger.info("Starting Phase 3: Standardization of positive effects...")
    if not os.path.exists(args.system_prompt_standardize_file):
        logger.error(f"Standardize system prompt not found: {args.system_prompt_standardize_file}")
        result_df['TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED'] = result_df['TOXICOLOGICAL_EFFECT_LLM_CLEANED']
    else:
        sys_prompt_std = load_system_prompt(args.system_prompt_standardize_file)
        positivized_effects = result_df['TOXICOLOGICAL_EFFECT_LLM_CLEANED'].dropna().unique().tolist()
        std_cache = asyncio.run(process_standardize_batch(llm, sys_prompt_std, positivized_effects, max_concurrency))

        result_df['TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED'] = result_df['TOXICOLOGICAL_EFFECT_LLM_CLEANED'].apply(
            lambda e: std_cache.get(e, {"standard_name": e})["standard_name"] if isinstance(e, str) else e
        )

    # Add NEGATIVE column as True for all rows in this script
    result_df['NEGATIVE'] = True

    # --- Save ---
    logger.info(f"Saving results to: {args.output}")
    # Final column mapping as requested by user:
    # 1. TOXICOLOGICAL_EFFECT (Original)
    # 2. NEGATIVE (Boolean True)
    # 3. TOXICOLOGICAL_EFFECT_LLM_CLEANED (Positivized text)
    # 4. TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED (Standardized text)

    target_cols = ['TOXICOLOGICAL_EFFECT', 'NEGATIVE', 'TOXICOLOGICAL_EFFECT_LLM_CLEANED', 'TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED']
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
