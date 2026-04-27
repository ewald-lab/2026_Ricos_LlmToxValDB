import os
import sys
import json
import pandas as pd
import asyncio
import logging
import argparse
from tqdm.asyncio import tqdm
from pathlib import Path
from langchain_openai import ChatOpenAI

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def arg_parser():
    parser = argparse.ArgumentParser(description="LLM-based Toxicological Cleaning Pipeline")
    parser.add_argument("--input", required=True, help="Input TSV file path.")
    parser.add_argument("--output", required=True, help="Output TSV file path.")
    parser.add_argument("--reasoning_file", required=True, help="Path to save cleaning reasoning.")
    parser.add_argument("--model_config", required=True, help="Path to model configuration JSON.")
    parser.add_argument("--pipeline_dir", required=True, help="Directory containing pipeline system prompts.")
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
    parser.add_argument("--base_url", default=None, help="Base URL for LLM API.")
    parser.add_argument("--api_key", default="EMPTY", help="API Key for LLM API.")
    parser.add_argument("--keep_non_tox", action="store_true", help="If set, keep rows that are not identified as tox effects.")
    parser.add_argument("--id_col", default=None, help="Column name for ID (e.g. TOXVAL_ID). If not provided, it detects it.")
    return parser.parse_args()

def load_system_prompt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def clean_llm_response(content: str) -> str:
    """Clean LLM response to ensure it's a valid JSON string."""
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()

async def process_batch(llm, system_prompts: list, descriptions: list, max_concurrency: int) -> dict:
    semaphore = asyncio.Semaphore(max_concurrency)
    cache = {}
    counter = 0

    async def process_one(desc: str):
        nonlocal counter
        async with semaphore:
            current_input = desc
            all_reasoning = {}
            current_effects = [desc]
            is_tox = True

            for i, sys_prompt in enumerate(system_prompts):
                step_name = f"Step_{i+1}"
                user_message = f"Original entry: \"{desc}\"\nCurrent cleaned effects: {current_effects}"
                
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_message},
                ]

                for attempt in range(5):
                    try:
                        response = await llm.ainvoke(messages)
                        cleaned = clean_llm_response(response.content)
                        try:
                            result = json.loads(cleaned)
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid JSON in {step_name}")

                        all_reasoning[step_name] = result.get("reasoning", "")
                        current_effects = result.get("effects", [])
                        if "is_tox_effect" in result:
                            is_tox = result["is_tox_effect"]
                            
                        # Speedy exit: if the effects list becomes empty mid-pipeline, there's no point running the rest
                        if i > 0 and len(current_effects) == 0:
                            is_tox = False
                            
                        break
                    except Exception as e:
                        if attempt == 4:
                            logger.error(f"Pipeline error for '{desc[:50]}...' at step {step_name}: {e}")
                            cache[desc] = {"is_tox_effect": True, "reasoning": f"Error at {step_name}: {e}", "effects": [desc]}
                            return
                        await asyncio.sleep(1)
                
                if not is_tox:
                    break

            cache[desc] = {
                "is_tox_effect": is_tox,
                "reasoning": all_reasoning,
                "effects": current_effects
            }
            counter += 1
            print(f"{counter}: Processed | '{desc[:50]}...'", flush=True)

    tasks = [process_one(d) for d in descriptions]
    await tqdm.gather(*tasks, desc="LLM Pipeline Processing", file=sys.stderr, smoothing=0.0)
    return cache

async def process_standardize_batch(llm, system_prompt: str, effects: list, max_concurrency: int) -> dict:
    semaphore = asyncio.Semaphore(max_concurrency)
    cache = {}
    counter = 0

    async def process_one(effect: str):
        nonlocal counter
        async with semaphore:
            user_message = f'Standardize this effect name:\n\n"{effect}"'
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
                    is_changed = result["standard_name"] != effect
                    status = "Changed" if is_changed else "Not Changed"
                    print(f"{counter}: LLM Std | Orig: '{effect}' -> ({status}): '{result['standard_name']}'", flush=True)
                    return
                except Exception as e:
                    if attempt == 4:
                        logger.error(f"Standardize error for '{effect[:50]}...' after 5 attempts: {e}")
                        cache[effect] = {"standard_name": effect, "reasoning": f"Error: {e}"}
                    else:
                        await asyncio.sleep(1)

    tasks = [process_one(e) for e in effects]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM Standardizing", smoothing=0.0):
        await f

    return cache

async def process_clean_exceptions_batch(llm, system_prompt: str, effects: list, max_concurrency: int) -> dict:
    """Process effects with clean exceptions: multiple colons and redundant repetitions."""
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
                    print(f"{counter}: Exceptions | '{effect}' -> '{corrected}'", flush=True)
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
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent

    if args.system_prompt_standardize_file is None:
        args.system_prompt_standardize_file = str(project_root / "config" / "prompts" / "clean" / "system_prompt_standardize.txt")

    if args.system_prompt_exceptions_file is None:
        args.system_prompt_exceptions_file = str(project_root / "config" / "prompts" / "clean" / "system_prompt_clean_exceptions.txt")
    
    # --- Model ---
    with open(args.model_config, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    max_concurrency = args.max_concurrency or model_config.pop("max_concurrency", 10)
    model_config.pop("max_concurrency", None)

    if args.base_url:
        model_config["base_url"] = args.base_url
    if args.api_key:
        model_config["api_key"] = args.api_key

    llm = ChatOpenAI(**model_config)

    # --- Pipeline Selection ---
    prompts = [
        "system_prompt_clean_1_ToxRelevance.txt",
        "system_prompt_clean_2_Concepts.txt",
        "system_prompt_clean_3_Stripping.txt",
        "system_prompt_clean_4_PrefixPropagation.txt",
        "system_prompt_clean_5_Standardization.txt",
        "system_prompt_clean_6_ImplicitAlteration.txt",
        "system_prompt_clean_7_FinalFormatting.txt",
        "system_prompt_clean_8_FinalRelevance.txt"
    ]
    system_prompts = [load_system_prompt(os.path.join(args.pipeline_dir, p)) for p in prompts]

    # --- Data Support ---
    df = pd.read_csv(args.input, sep="\t", low_memory=False)
    col = args.target_column
    unique_descs = df[col].dropna().unique().tolist()
    unique_descs = [d for d in unique_descs if isinstance(d, str) and len(d.strip()) >= 4]

    if args.debug:
        import random
        random.seed(45)
        random.shuffle(unique_descs)
        unique_descs = unique_descs[:100]
        logger.info(f"DEBUG MODE: Selected 100 random unique descriptions.")

    # --- Step 1: Initial LLM Cleaning ---
    cache = asyncio.run(process_batch(llm, system_prompts, unique_descs, max_concurrency))

    # --- Apply results ---
    logger.info("Applying LLM results to dataset...")
    new_rows = []
    reasoning_output = {}

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Applying LLM Results", file=sys.stderr, smoothing=0.0):
        desc = row.get(col)
        if not isinstance(desc, str) or len(desc.strip()) < 4:
            if args.debug: continue
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
    if os.path.exists(args.system_prompt_exceptions_file):
        logger.info(f"Loading exceptions system prompt from: {args.system_prompt_exceptions_file}")
        sys_prompt_exceptions = load_system_prompt(args.system_prompt_exceptions_file)

        def needs_cleanup(effect: str) -> bool:
            if not isinstance(effect, str): return False
            if effect.count(":") > 1: return True
            if ":" in effect:
                parts = effect.split(":", 1)
                ctx = parts[0].strip().lower()
                eff = parts[1].strip().lower()
                ctx_words = set(ctx.replace("/", " ").replace("-", " ").split())
                eff_words = set(eff.replace("/", " ").replace("-", " ").split())
                if any(len(w) > 3 and w in eff_words for w in ctx_words):
                    return True
            return False

        cleaned_col = 'TOXICOLOGICAL_EFFECT_LLM_CLEANED'
        all_cleaned = result_df[cleaned_col].dropna().unique().tolist()
        effects_to_fix = [e for e in all_cleaned if needs_cleanup(e)]
        logger.info(f"Phase 1.9: Found {len(effects_to_fix)} unique effects needing structural fix.")

        if effects_to_fix:
            exceptions_cache = asyncio.run(process_clean_exceptions_batch(llm, sys_prompt_exceptions, effects_to_fix, max_concurrency))
            
            def apply_exceptions(effect):
                if not isinstance(effect, str): return effect
                res = exceptions_cache.get(effect)
                return res["corrected"] if res else effect
            
            result_df[cleaned_col] = result_df[cleaned_col].apply(apply_exceptions)

            # Update reasoning JSON
            for key in reasoning_output:
                old_effs = reasoning_output[key].get("effects", [])
                new_effs = []
                exp_reasonings = []
                for e in old_effs:
                    res = exceptions_cache.get(e)
                    if res and res["corrected"] != e:
                        new_effs.append(res["corrected"])
                        exp_reasonings.append({
                            "original_part": e,
                            "corrected_part": res["corrected"],
                            "reasoning": res["reasoning"]
                        })
                    else:
                        new_effs.append(e)
                reasoning_output[key]["effects"] = new_effs
                if exp_reasonings:
                    reasoning_output[key]["reasoning_exceptions"] = exp_reasonings

    # --- Phase 2: Standardization ---
    logger.info("Starting Phase 2: Standardization of cleaned effects...")
    if os.path.exists(args.system_prompt_standardize_file):
        sys_prompt_std = load_system_prompt(args.system_prompt_standardize_file)
        cleaned_effects = result_df[cleaned_col].dropna().unique().tolist()
        std_cache = asyncio.run(process_standardize_batch(llm, sys_prompt_std, cleaned_effects, max_concurrency))
        
        std_col = 'TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED'
        result_df[std_col] = result_df[cleaned_col].map(lambda x: std_cache.get(x, {"standard_name": x})["standard_name"])

        # Final Update to Reasoning JSON
        for key in reasoning_output:
            effs = reasoning_output[key].get("effects", [])
            final_std = []
            for e in effs:
                res = std_cache.get(e)
                if res:
                    final_std.append({
                        "original": e,
                        "standardized": res["standard_name"],
                        "reasoning": res.get("reasoning", "")
                    })
            if final_std:
                reasoning_output[key]["standardization"] = final_std

    # --- Final Save ---
    logger.info(f"Saving cleaning reasoning to: {args.reasoning_file}")
    with open(args.reasoning_file, "w", encoding="utf-8") as f:
        json.dump(reasoning_output, f, indent=2, ensure_ascii=False)

    result_df.to_csv(args.output, sep="\t", index=False)
    logger.info(f"Process completed. Results saved to {args.output}")

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print(f"Total time: {time.time() - start_time:.2f}s")
