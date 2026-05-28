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

os.environ["HF_HOME"] = "/nfs/research/ewald/ricos/huggingface_cache"
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
    process_standardize_batch,
    process_clean_exceptions_batch,
    needs_cleanup,
)

logger = logging.getLogger(__name__)

def arg_parser() -> argparse.Namespace:
    """
    Description:
    Parses command line arguments for the LLM cleaning script.

    Args:
        None

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="LLM-based cleaning of toxicological effects: validates and splits composite entries."
    )
    parser.add_argument("--input", default="ToxValDB_v9.7.0_Cleaned_01.tsv", help="Path to input TSV file.")
    parser.add_argument("--output", default="ToxValDB_v9.7.0_Cleaned_02.tsv", help="Path to output TSV file.")
    parser.add_argument("--reasoning_file", default="llm_clean_reasoning.json", help="Path to save LLM reasoning traces.")
    parser.add_argument("--model_config", required=True, help="Path to model config JSON file.")
    parser.add_argument("--base_url", default=None, help="Base URL for the LLM API. Overrides config value.")
    parser.add_argument(
        "--pipeline_dir",
        default=None,
        help="Path to directory containing sequential pipeline prompts (system_prompt_clean_1.txt, etc.).",
    )
    parser.add_argument(
        "--system_prompt_file",
        default=None,
        help="Path to single system prompt (alternative to --pipeline_dir).",
    )
    parser.add_argument(
        "--system_prompt_standardize_file",
        default=None,
        help="Path to system prompt for standardization phase (default: config/system_prompt_standardize.txt).",
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
    parser.add_argument("--keep-non-tox", action="store_true", help="If set, do not remove entries labeled as is_tox_effect: false. By default they are removed.")
    return parser.parse_args()

async def process_batch(
    llm: Any, prompts: list[tuple[str, str]], descriptions: list[str], max_concurrency: int
) -> dict[str, Any]:
    """
    Description:
    Runs the sequential prompts cleaning pipeline on a batch of toxicological effect descriptions concurrently.

    Args:
        llm (Any): The LLM engine wrapper.
        prompts (list[tuple[str, str]]): List of prompt tuples (filename, content) in the pipeline.
        descriptions (list[str]): List of raw descriptions to clean.
        max_concurrency (int): Maximum concurrent LLM requests.

    Returns:
        dict[str, Any]: A dictionary mapping original descriptions to their cleaned effects and metadata.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    cache = {}
    counter = 0

    async def process_one(desc: str) -> None:
        """
        Description:
        Helper coroutine to process a single description through the pipeline.

        Args:
            desc (str): The raw description string.

        Returns:
            None
        """
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

                        # Store step results using descriptive part of filename
                        step_name = fname.replace("system_prompt_clean_", "").replace(".txt", "").split("_", 1)[-1]
                        step_data = {"reasoning": result.get("reasoning")}
                        if "effects" in result:
                            step_data["effects"] = result.get("effects")
                        full_reasoning[step_name] = step_data

                        # Update state for next step ONLY if effects are provided
                        if "effects" in result:
                            current_effects = result.get("effects", [])

                        print(f"[{counter}] Step: {step_name} | Effects: {current_effects} | Reasoning: {result.get('reasoning')}", flush=True)

                        # Early exit if a step determines it's not a tox effect
                        if "is_tox_effect" in result:
                            is_tox = result.get("is_tox_effect", True)
                            if not is_tox:
                                current_effects = []
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

            # Final result for the cache
            final_result = {
                "effects": current_effects,
                "is_tox_effect": is_tox,
                "reasoning": full_reasoning
            }
            cache[desc] = final_result

            counter += 1
            status_tag = "NOT-TOX" if not is_tox else "Cleaned"
            print(f"{counter}: LLM Pipeline | Orig: '{desc}' -> {status_tag}: {current_effects}", flush=True)

    tasks = [process_one(d) for d in descriptions]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM Pipeline Processing", smoothing=0.0):
        await f

    return cache

def main() -> None:
    """
    Description:
    Main entry point for the LLM cleaning script. It loads input data, initializes the model,
    runs the cleaning pipeline, standardizes cleaned terms, corrects exceptions, and saves the final result.

    Args:
        None

    Returns:
        None
    """
    args = arg_parser()

    global logger
    logger = setup_logging(level=logging.INFO, stream=sys.stdout, fmt="%(asctime)s - %(levelname)s - %(message)s")

    start_time = time.time()

    # --- System prompt / Pipeline ---
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if args.pipeline_dir is None and args.system_prompt_file is None:
        default_pipeline = str(project_root / "inputs" / "clean_prompts" / "pipeline")
        if os.path.exists(default_pipeline) and os.path.isdir(default_pipeline):
            args.pipeline_dir = default_pipeline
            logger.info(f"Using default pipeline directory: {args.pipeline_dir}")
        else:
            args.system_prompt_file = str(project_root / "inputs" / "clean_prompts" / "system_prompt_clean.txt")
            logger.info(f"Using default system prompt file: {args.system_prompt_file}")

    prompts = []
    if args.pipeline_dir:
        prompts = load_pipeline_prompts(args.pipeline_dir)
    elif args.system_prompt_file:
        if not os.path.exists(args.system_prompt_file):
            logger.error(f"System prompt not found: {args.system_prompt_file}")
            return
        prompts = [load_system_prompt(args.system_prompt_file)]
        logger.info(f"Loaded single system prompt from: {args.system_prompt_file}")

    if not prompts:
        logger.error("No prompts loaded. Execution aborted.")
        return

    if args.system_prompt_standardize_file is None:
        args.system_prompt_standardize_file = str(project_root / "inputs" / "clean_prompts" / "system_prompt_standardize.txt")

    if args.system_prompt_exceptions_file is None:
        args.system_prompt_exceptions_file = str(project_root / "inputs" / "clean_prompts" / "system_prompt_clean_exceptions.txt")

    # --- Model ---
    with open(args.model_config, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    max_concurrency = args.max_concurrency or model_config.pop("max_concurrency", 10)
    model_config.pop("max_concurrency", None)

    if args.base_url:
        model_config["base_url"] = args.base_url
        logger.info(f"Overriding base_url from CLI: {args.base_url}")

    logger.info(f"Loading model from config: {args.model_config}")
    llm = get_model(**model_config)

    test_retries = model_config.get("max_retries", 3)
    if not test_model(llm, max_retries=test_retries):
        raise RuntimeError("Model connectivity test failed.")

    # --- Data ---
    logger.info(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input, sep="\t", low_memory=False)

    # Fallback to comma if the file has a .tsv extension but is actually comma-separated
    if len(df.columns) == 1 and "," in df.columns[0]:
        logger.warning(f"File {args.input} appears to be comma-separated. Retrying with sep=','")
        df = pd.read_csv(args.input, sep=",", low_memory=False)

    logger.info(f"Input shape: {df.shape}")

    if args.debug_rows:
        df = df.iloc[args.debug_rows].copy()
        logger.info(f"Debug mode: processing {len(df)} rows")

    # --- Collect unique descriptions ---
    col = args.target_column
    unique_descs = df[col].dropna().unique().tolist()
    unique_descs = [d for d in unique_descs if isinstance(d, str) and len(d.strip()) >= 4]

    if args.debug:
        import random
        random.seed(45)
        random.shuffle(unique_descs)
        unique_descs = unique_descs[:100]
        logger.info("DEBUG MODE: Selected 100 random unique descriptions to evaluate.")
    else:
        logger.info(f"Found {len(unique_descs)} unique descriptions to evaluate.")

    # --- Load Pipeline Prompts ---
    if args.pipeline_dir:
        prompts = load_pipeline_prompts(args.pipeline_dir)
        if not prompts:
            logger.error(f"No prompts found in pipeline directory: {args.pipeline_dir}")
            return
    else:
        logger.error("No pipeline directory provided (--pipeline_dir is required for modular cleaning).")
        return

    # --- LLM Processing ---
    cache = asyncio.run(process_batch(llm, prompts, unique_descs, args.max_concurrency))

    # --- Apply results ---
    logger.info("Applying LLM results to dataset...")
    new_rows = []
    reasoning_output = {}

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Applying LLM Results", file=sys.stderr, smoothing=0.0):
        desc = row.get(col)
        if not isinstance(desc, str) or len(desc.strip()) < 4:
            # Keep as-is if too short or invalid
            logger.info(f"Row {idx} | Orig: '{desc}' -> [SKIPPED - Too Short/Invalid]")
            row_dict = row.to_dict()
            row_dict['TOXICOLOGICAL_EFFECT_LLM_CLEANED'] = desc
            new_rows.append(row_dict)
            continue

        result = cache.get(desc)
        if result is None:
            if args.debug:
                # In debug mode, skip rows not in our 100 unique sample
                continue
            logger.info(f"Row {idx} | Orig: '{desc}' -> [SKIPPED - No Cache]")
            row_dict = row.to_dict()
            row_dict['TOXICOLOGICAL_EFFECT_LLM_CLEANED'] = desc
            new_rows.append(row_dict)
            continue

        reasoning_output[desc] = {
            "reasoning": result.get("reasoning", ""),
            "effects": result.get("effects", []),
        }

        # Filter out non-tox effects if requested
        if not args.keep_non_tox and not result.get("is_tox_effect", True):
            logger.info(f"Row {idx} | Orig: '{desc}' -> [REMOVED - Non-Tox Effect]")
            continue

        effects = result.get("effects", [])
        if not effects:
            # Parse error or empty: keep original
            logger.info(f"Row {idx} | Orig: '{desc}' -> [KEPT AS IS - Parse Error or Empty]")
            row_dict = row.to_dict()
            row_dict['TOXICOLOGICAL_EFFECT_LLM_CLEANED'] = desc
            new_rows.append(row_dict)
            continue

        logger.info(f"Row {idx} | Orig: '{desc}' -> Cleaned: {effects}")

        if len(effects) == 1:
            row_dict = row.to_dict()
            row_dict['TOXICOLOGICAL_EFFECT_LLM_CLEANED'] = effects[0].strip()
            new_rows.append(row_dict)
        else:
            # Split into multiple rows
            for eff in effects:
                eff = eff.strip()
                if eff:
                    row_dict = row.to_dict()
                    row_dict['TOXICOLOGICAL_EFFECT_LLM_CLEANED'] = eff
                    new_rows.append(row_dict)

    result_df = pd.DataFrame(new_rows)

    # --- Phase 1.9: Final Exception Polish ---
    if os.path.exists(args.system_prompt_exceptions_file):
        logger.info(f"Loading exceptions system prompt from: {args.system_prompt_exceptions_file}")
        sys_prompt_exceptions = load_system_prompt(args.system_prompt_exceptions_file)



        cleaned_col = 'TOXICOLOGICAL_EFFECT_LLM_CLEANED'
        all_cleaned = result_df[cleaned_col].dropna().unique().tolist()
        effects_to_fix = [e for e in all_cleaned if needs_cleanup(e)]
        logger.info(f"Phase 1.9: Found {len(effects_to_fix)} unique effects needing structural fix.")

        if effects_to_fix:
            exceptions_cache = asyncio.run(process_clean_exceptions_batch(llm, sys_prompt_exceptions, effects_to_fix, max_concurrency))

            def apply_exceptions(effect: Any) -> Any:
                """
                Description:
                Helper mapping function to apply corrections from the exceptions cache.

                Args:
                    effect (Any): The original term value.

                Returns:
                    Any: The corrected term value or the original if unchanged.
                """
                if not isinstance(effect, str): return effect
                res = exceptions_cache.get(effect)
                return res["corrected"] if res else effect

            result_df[cleaned_col] = result_df[cleaned_col].apply(apply_exceptions)

            # Update reasoning JSON with fixed effects
            logger.info("Updating reasoning output with structural corrections...")
            for key in reasoning_output:
                old_effects = reasoning_output[key].get("effects", [])
                new_effects = []
                exp_reasonings = []
                for e in old_effects:
                    res = exceptions_cache.get(e)
                    if res and res["corrected"] != e:
                        new_effects.append(res["corrected"])
                        exp_reasonings.append({
                            "original_part": e,
                            "corrected_part": res["corrected"],
                            "reasoning": res["reasoning"]
                        })
                    else:
                        new_effects.append(e)
                reasoning_output[key]["effects"] = new_effects
                if exp_reasonings:
                    reasoning_output[key]["reasoning_exceptions"] = exp_reasonings

    # --- Phase 2: Standardization ---
    logger.info("Starting Phase 2: Standardization of cleaned effects...")

    if not os.path.exists(args.system_prompt_standardize_file):
        logger.error(f"Standardize system prompt not found: {args.system_prompt_standardize_file}")
        return

    sys_prompt_std = load_system_prompt(args.system_prompt_standardize_file)
    logger.info(f"Loaded standardize system prompt from: {args.system_prompt_standardize_file}")

    cleaned_effects = result_df['TOXICOLOGICAL_EFFECT_LLM_CLEANED'].dropna().unique().tolist()
    cleaned_effects = [e for e in cleaned_effects if isinstance(e, str) and len(e.strip()) >= 2]
    logger.info(f"Found {len(cleaned_effects)} unique cleaned effects for standardization.")

    std_cache = asyncio.run(process_standardize_batch(llm, sys_prompt_std, cleaned_effects, max_concurrency))

    std_col = "TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED"
    def get_std_name(effect: Any) -> Any:
        """
        Description:
        Helper mapping function to retrieve standard names from the standardization cache.

        Args:
            effect (Any): The cleaned term value.

        Returns:
            Any: The standardized term name.
        """
        if not isinstance(effect, str): return effect
        res = std_cache.get(effect)
        return res.get("standard_name", effect) if res else effect

    result_df[std_col] = result_df['TOXICOLOGICAL_EFFECT_LLM_CLEANED'].apply(get_std_name)

    # Merge standardization reasoning into main reasoning JSON
    for key in reasoning_output:
        std_entries = []
        for e in reasoning_output[key]["effects"]:
            res = std_cache.get(e)
            if res:
                std_entries.append({
                    "cleaned": e,
                    "standard_name": res.get("standard_name", e),
                    "reasoning": res.get("reasoning", ""),
                })
        if std_entries:
            reasoning_output[key]["reasoning_standardization"] = std_entries

    # --- Final Save ---
    logger.info(f"Saving all cleaning reasoning to: {args.reasoning_file}")
    with open(args.reasoning_file, "w", encoding="utf-8") as f:
        json.dump(reasoning_output, f, indent=2, ensure_ascii=False)

    # --- Save ---
    logger.info(f"Output shape: {result_df.shape} (from input {df.shape})")
    logger.info(f"Saving to: {args.output}")
    result_df.to_csv(args.output, sep="\t", index=False)

    # Removed rows logic isn't as relevant since we keep non-tox now.
    removed = df.shape[0] - result_df.shape[0]
    if removed > 0:
        logger.info(f"Removed {removed} invalid rows.")
    added = result_df.shape[0] - df.shape[0] + removed
    if added > 0:
        logger.info(f"Added {added} rows from splitting composite effects.")

    logger.info(f"Total execution time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
