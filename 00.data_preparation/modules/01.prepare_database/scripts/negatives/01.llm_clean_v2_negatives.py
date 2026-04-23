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
        description="LLM-based cleaning of toxicological effects: validates and splits composite entries (Negatives Version)."
    )
    parser.add_argument("--input", required=True, help="Path to input TSV file.")
    parser.add_argument("--output", required=True, help="Path to output TSV file.")
    parser.add_argument("--reasoning_file", required=True, help="Path to save LLM reasoning traces.")
    parser.add_argument("--model_config", required=True, help="Path to model config JSON file.")
    parser.add_argument("--base_url", default=None, help="Base URL for the LLM API. Overrides config value.")
    parser.add_argument(
        "--pipeline_dir",
        default=None,
        help="Path to directory containing sequential pipeline prompts.",
    )
    parser.add_argument(
        "--system_prompt_file",
        default=None,
        help="Path to single system prompt (alternative to --pipeline_dir).",
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
    parser.add_argument("--max_concurrency", type=int, default=20, help="Max concurrent LLM calls (overrides config).")
    parser.add_argument("--debug", action="store_true", help="If set, process only 100 random unique descriptions for validation.")
    parser.add_argument("--debug_rows", type=int, nargs="+", default=None, help="Row indices to process for debugging.")
    parser.add_argument("--keep-non-tox", action="store_true", default=True, help="If set, do not remove entries labeled as is_tox_effect: false. (Default: True for Negatives pipeline)")
    return parser.parse_args()


def load_pipeline_prompts(directory_path: str) -> list:
    """Load all .txt files from the directory, sorted by name, and return (filename, content)."""
    prompts = []
    p_dir = Path(directory_path)
    if not p_dir.is_dir():
        logger.warning(f"Pipeline directory {directory_path} not found.")
        return []
    
    # Get all .txt files, sorted by name
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
    """Strip thinking tags and extract JSON from LLM response."""
    text = raw
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    # Try to extract JSON from markdown code blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    # Try to extract raw JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def parse_llm_result(raw_content: str) -> dict:
    """Parse the LLM response into a structured result."""
    cleaned = clean_llm_response(raw_content)
    try:
        result = json.loads(cleaned)
        return result
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM response: {e}\nRaw: {raw_content[:200]}")
        return {"effects": [], "reasoning": f"Parse error: {e}", "parse_error": True}


async def process_batch(llm, prompts: list, descriptions: list, max_concurrency: int) -> dict:
    """Process a batch of descriptions concurrently using a sequence of LLM prompts."""
    semaphore = asyncio.Semaphore(max_concurrency)
    cache = {}
    counter = 0

    async def process_one(desc: str):
        nonlocal counter
        async with semaphore:
            current_effects = [desc]
            full_reasoning = {}
            is_tox = True
            
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
                            
                        # print(f"[{counter}] Step: {step_name} | Effects: {current_effects}", flush=True)
                        
                        if "is_tox_effect" in result:
                            is_tox = result.get("is_tox_effect", True)
                            if not is_tox:
                                break
                                
                        if i > 0 and len(current_effects) == 0:
                            is_tox = False
                            break
                        
                        success = True
                        break
                    except Exception as e:
                        if attempt == 4:
                            logger.error(f"LLM error for '{desc[:30]}' at step {fname}: {e}")
                            break
                        await asyncio.sleep(1)
                
                if not is_tox or not success:
                    break

            final_result = {
                "effects": current_effects,
                "is_tox_effect": is_tox,
                "reasoning": full_reasoning
            }
            cache[desc] = final_result
            
            counter += 1
            status_tag = "NOT-TOX" if not is_tox else "Cleaned"
            print(f"{counter}: LLM Pipeline | Orig: '{desc[:50]}...' -> {status_tag}: {current_effects}", flush=True)

    tasks = [process_one(d) for d in descriptions]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM Pipeline Processing", smoothing=0.0):
        await f

    return cache

async def process_standardize_batch(llm, system_prompt: str, effects: list, max_concurrency: int) -> dict:
    semaphore = asyncio.Semaphore(max_concurrency)
    cache = {}
    counter = 0

    async def process_one(effect: str):
        nonlocal counter
        async with semaphore:
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
                    return
                except Exception as e:
                    if attempt == 4:
                        logger.error(f"LLM standardize error for '{effect[:50]}...' after 5 attempts: {e}")
                        cache[effect] = {"standard_name": effect, "reasoning": f"LLM error: {e}"}
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

                    corrected = result["corrected"]
                    reasoning = result.get("reasoning", "")
                    cache[effect] = {"corrected": corrected, "reasoning": reasoning}

                    counter += 1
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

    if not args.pipeline_dir:
        logger.error("No pipeline directory provided (--pipeline_dir is required).")
        return

    prompts = load_pipeline_prompts(args.pipeline_dir)
    if not prompts:
        logger.error(f"No prompts found in pipeline directory: {args.pipeline_dir}")
        return

    # --- Model ---
    with open(args.model_config, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    max_concurrency = args.max_concurrency or model_config.pop("max_concurrency", 10)
    model_config.pop("max_concurrency", None)

    if args.base_url:
        model_config["base_url"] = args.base_url

    logger.info(f"Loading model from config: {args.model_config}")
    llm = get_model(**model_config)

    # --- Data ---
    logger.info(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input, sep="\t", low_memory=False)
    
    # --- Collect unique descriptions ---
    col = args.target_column
    unique_descs = df[col].dropna().unique().tolist()
    unique_descs = [d for d in unique_descs if isinstance(d, str) and len(d.strip()) >= 2]
    
    if args.debug:
        import random
        random.seed(45)
        random.shuffle(unique_descs)
        unique_descs = unique_descs[:100]

    # --- LLM Processing ---
    cache = asyncio.run(process_batch(llm, prompts, unique_descs, args.max_concurrency))

    # --- Apply results ---
    logger.info("Applying LLM results to dataset...")
    new_rows = []
    reasoning_output = {}

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Applying LLM Results", file=sys.stderr, smoothing=0.0):
        desc = row.get(col)
        if not isinstance(desc, str) or len(desc.strip()) < 2:
            row_dict = row.to_dict()
            row_dict['TOXICOLOGICAL_EFFECT_LLM_CLEANED'] = desc
            new_rows.append(row_dict)
            continue

        result = cache.get(desc)
        if result is None:
            if args.debug: continue
            row_dict = row.to_dict()
            row_dict['TOXICOLOGICAL_EFFECT_LLM_CLEANED'] = desc
            new_rows.append(row_dict)
            continue

        reasoning_output[desc] = {
            "reasoning": result.get("reasoning", ""),
            "effects": result.get("effects", []),
        }

        if not args.keep_non_tox and not result.get("is_tox_effect", True):
            continue

        effects = result.get("effects", [])
        if not effects:
            row_dict = row.to_dict()
            row_dict['TOXICOLOGICAL_EFFECT_LLM_CLEANED'] = desc
            new_rows.append(row_dict)
            continue

        for eff in effects:
            eff = eff.strip()
            if eff:
                row_dict = row.to_dict()
                row_dict['TOXICOLOGICAL_EFFECT_LLM_CLEANED'] = eff
                new_rows.append(row_dict)

    result_df = pd.DataFrame(new_rows)

    # --- Phase 1.9: Final Exception Polish ---
    if args.system_prompt_exceptions_file and os.path.exists(args.system_prompt_exceptions_file):
        sys_prompt_exceptions = load_system_prompt(args.system_prompt_exceptions_file)
        cleaned_col = 'TOXICOLOGICAL_EFFECT_LLM_CLEANED'
        all_cleaned = result_df[cleaned_col].dropna().unique().tolist()
        # simplified needs_cleanup for negatives
        effects_to_fix = [e for e in all_cleaned if isinstance(e, str) and e.count(":") > 1]
        
        if effects_to_fix:
            exceptions_cache = asyncio.run(process_clean_exceptions_batch(llm, sys_prompt_exceptions, effects_to_fix, max_concurrency))
            result_df[cleaned_col] = result_df[cleaned_col].apply(lambda x: exceptions_cache.get(x, {"corrected": x})["corrected"])

    # --- Phase 2: Standardization ---
    if args.system_prompt_standardize_file and os.path.exists(args.system_prompt_standardize_file):
        sys_prompt_std = load_system_prompt(args.system_prompt_standardize_file)
        cleaned_effects = result_df['TOXICOLOGICAL_EFFECT_LLM_CLEANED'].dropna().unique().tolist()
        std_cache = asyncio.run(process_standardize_batch(llm, sys_prompt_std, cleaned_effects, max_concurrency))
        result_df["TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED"] = result_df['TOXICOLOGICAL_EFFECT_LLM_CLEANED'].apply(lambda x: std_cache.get(x, {"standard_name": x})["standard_name"])

    # --- Final Save ---
    with open(args.reasoning_file, "w", encoding="utf-8") as f:
        json.dump(reasoning_output, f, indent=2, ensure_ascii=False)

    result_df.to_csv(args.output, sep="\t", index=False)
    logger.info(f"Total execution time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
