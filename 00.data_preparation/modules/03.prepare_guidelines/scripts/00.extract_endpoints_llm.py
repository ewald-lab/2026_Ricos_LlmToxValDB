import argparse
import json
import os

import glob
import re
from langchain_community.document_loaders import PyPDFLoader
import sys
from tqdm import tqdm

# Import the model loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.model.llm import get_model

def get_unique_path(directory, base_filename):
    """
    Returns a non-conflicting path by adding a numeric suffix if needed.
    Example: OECD_203_report.md -> OECD_203_report_2.md
    """
    path = os.path.join(directory, base_filename)
    if not os.path.exists(path):
        return path
    
    name, ext = os.path.splitext(base_filename)
    counter = 2
    while True:
        new_filename = f"{name}_{counter}{ext}"
        path = os.path.join(directory, new_filename)
        if not os.path.exists(path):
            return path
        counter += 1


def parse_llm_responses(responses, guideline_number, out_dir, summary_path):
    """
    Parses multiple LLM responses (from chunks) and merges them.
    """
    merged_report = ""
    merged_endpoints_csv_rows = []
    summary_csv_row = ""
    
    for i, response_text in enumerate(responses):
        # Extract markdown report
        md_match = re.search(r'```markdown\n(.*?)\n```', response_text, re.DOTALL)
        if md_match:
            content = md_match.group(1).strip()
            if i == 0:
                merged_report = content
            else:
                # Append observations if they contain actual content
                if "No key observations" not in content and "None found" not in content:
                    obs_match = re.search(r'## (Key Observations|Endpoints).*', content, re.DOTALL)
                    if obs_match:
                        merged_report += "\n\n--- (Continued from next segment) ---\n\n" + obs_match.group(0)

        # Extract CSV blocks
        csv_matches = re.finditer(r'```csv\n(.*?)\n```', response_text, re.DOTALL)
        for m in csv_matches:
            csv_block = m.group(1).strip()
            if "guideline_number,category,endpoint_name" in csv_block:
                lines = csv_block.split('\n')
                header = lines[0]
                rows = lines[1:]
                merged_endpoints_csv_rows.extend(rows)
            elif i == 0 and len(csv_block.split(',')) >= 5:
                # Likely the summary row from the first chunk
                summary_csv_row = csv_block

    # Save merged report
    if merged_report:
        report_path = get_unique_path(out_dir, f"OECD_{guideline_number}_report.md")
        with open(report_path, 'w') as f:
            f.write(merged_report)
            
    # Save merged endpoints CSV
    if merged_endpoints_csv_rows:
        endpoints_path = get_unique_path(out_dir, f"OECD_{guideline_number}_endpoints.csv")
        with open(endpoints_path, 'w') as f:
            f.write("guideline_number,category,endpoint_name,requirement_level,source,footnotes\n")
            # Remove duplicates (simple string comparison)
            seen = set()
            for row in merged_endpoints_csv_rows:
                if row.strip() and row not in seen:
                    f.write(row + "\n")
                    seen.add(row)

    # Save summary row (only from first chunk or recalculated?)
    if summary_csv_row:
        header = "guideline_number,title,adoption_date,species,duration,animals,dose_groups,route,endpoint_source,num_required_endpoints,num_optional_endpoints\n"
        if not os.path.exists(summary_path):
            with open(summary_path, 'w') as f:
                f.write(header)
                
        with open(summary_path, 'a') as f:
            lines = summary_csv_row.split('\n')
            for line in lines:
                if not line.startswith("guideline_number"):
                    f.write(line + "\n")

def main():

    parser = argparse.ArgumentParser(description="Extract endpoints from OECD guidelines using LLM")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--model_provider", default="openai", help="Model provider")
    parser.add_argument("--base_url", help="Base URL for the model")
    parser.add_argument("--skill_file", required=True, help="Path to SKILL.md")
    parser.add_argument("--inputs_dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--outputs_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--model_config", help="Path to model config JSON file")

    
    args = parser.parse_args()
    
    os.makedirs(args.outputs_dir, exist_ok=True)
    
    # Initialize model
    if args.model_config:
        with open(args.model_config, 'r') as f:
            model_kwargs = json.load(f)
        
        # Pop parameters that are not recognized by init_chat_model/OpenAI
        model_kwargs.pop("max_concurrency", 10)
        
        if args.base_url:
            model_kwargs["base_url"] = args.base_url
        # Override temperature if it was fixed before or keep from config
        if "temperature" not in model_kwargs:
            model_kwargs["temperature"] = 0.1
        llm = get_model(**model_kwargs)
    else:
        llm = get_model(
            model=args.model,
            model_provider=args.model_provider,
            base_url=args.base_url,
            temperature=0.1
        )

    
    with open(args.skill_file, 'r') as f:
        skill_prompt = f.read()
        
    # Modify prompt for LLM text output instead of agent
    modified_prompt = skill_prompt + "\n\nIMPORTANT INSTRUCTIONS FOR LOCAL LLM:\n" \
                      "You are a text-based LLM. You do not have tools. Instead of creating files, " \
                      "output the content of the Markdown Report enclosed in ```markdown ... ``` blocks. " \
                      "Output the content of the Endpoints CSV enclosed in ```csv ... ``` blocks. " \
                      "Output the content of the Summary CSV row enclosed in another ```csv ... ``` block."
    
    summary_path = get_unique_path(args.outputs_dir, "OECD_guideline_summary.csv")
    
    pdf_files = glob.glob(os.path.join(args.inputs_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {args.inputs_dir}. Please check the path and ensure PDFs are present.", flush=True)
        return

    for pdf_path in tqdm(pdf_files, desc="Processing OECD Guidelines"):
        filename = os.path.basename(pdf_path)

        guideline_number = filename.replace('.pdf', '')
        print(f"Processing Guideline {guideline_number}...", flush=True)
        
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        pdf_text = "\n".join([p.page_content for p in pages])
        
        # Windowing logic
        chunk_size = 30000
        overlap = 2000
        chunks = []
        if len(pdf_text) > chunk_size:
            for i in range(0, len(pdf_text), chunk_size - overlap):
                chunks.append(pdf_text[i:i + chunk_size])
        else:
            chunks = [pdf_text]
            
        responses = []
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}/{len(chunks)}...", flush=True)
            prompt = modified_prompt.replace("$ARGUMENTS", guideline_number)
            if i > 0:
                prompt += "\n\nIMPORTANT: You already provided the summary fields in the previous chunk. " \
                          "In this chunk, focus ONLY on extracting NEW endpoints and key observations found in this specific text. " \
                          "Output the CSV block with any NEW endpoints. " \
                          "In the Markdown report, you can omit the header/summary table and just provide the 'Key Observations' and 'Endpoints' sections."
            
            prompt += f"\n\nHere is the extracted text from this part of the PDF:\n\n{chunk}"
            
            try:
                response = llm.invoke(prompt)
                responses.append(response.content)
            except Exception as e:
                print(f"  Error in chunk {i+1}: {e}", flush=True)
        
        if responses:
            try:
                print(f"Merging results for {guideline_number}...", flush=True)
                parse_llm_responses(responses, guideline_number, args.outputs_dir, summary_path)
                print(f"Guideline {guideline_number} done.", flush=True)
            except Exception as e:
                print(f"Error merging results for {guideline_number}: {e}", flush=True)

if __name__ == "__main__":
    main()
