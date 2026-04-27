/*
 * Mapping: Core logic processing a chunk of descriptions against one vLLM instance
 */
process TEXT_TO_ONTO_MAPPING {
    tag "${chunk_id} → ${base_url}"
    publishDir "${params.output_dir}/all", mode: 'copy', pattern: "ToxValDB_*"

    input:
    tuple val(chunk_id), path(chunk_file), val(base_url), val(model_config_path)
    path ontology_db

    output:
    path "ToxValDB_Classified_${chunk_id}.csv"
    path "ToxValDB_Reasoning_${chunk_id}.json"

    script:
    // Derive ontology name and suffix
    def ont_name = file(params.ontology_file).baseName
    def suffix = "_${ont_name}_${params.mode}"
    def out_csv = "ToxValDB_Classified_${chunk_id}.csv"
    def out_json = "ToxValDB_Reasoning_${chunk_id}.json"
    """
    python3 -u \'${projectDir.parent}/00.main.py\' \\
        --input_file ${chunk_file} \\
        --output_file ${out_csv} \\
        --reasoning_file ${out_json} \\
        --ontology_embedding_db ${ontology_db} \\
        --model_config ${model_config_path} \\
        --base_url ${base_url} \\
        --target_column "${params.target_column}" \\
        --mode ${params.mode} \\
        --consensus_runs ${params.consensus_runs} \\
        ${params.rag_config ? "--rag_config ${params.rag_config}" : ""} \\
        --embeddings_model_id ${params.embeddings_model_id} \\
        --seed ${params.seed} \\
        ${params.target_embedding_db ? "--target_embedding_db ${params.target_embedding_db}" : ""} \\
        ${params.debug_rows ? "--debug_rows ${params.debug_rows instanceof List ? params.debug_rows.join(' ') : params.debug_rows}" : ""} \\
        ${params.random_debug_rows ? "--random_debug_rows ${params.random_debug_rows}" : ""} \\
        ${params.system_prompt_file ? "--system_prompt_file ${params.system_prompt_file}" : ""} \\
        ${params.ontology_definitions ? "--ontology_definitions" : ""} \\
        ${params.include_superiors ? "--include_superiors" : ""} \\
        ${params.global_definitions_db ? "--global_definitions_db ${params.global_definitions_db}" : ""} \\
        ${params.num_backups ? "--num_backups ${params.num_backups}" : ""} \\
        ${params.context_ontology ? "--context_ontology \"${params.context_ontology}\"" : ""} \\
        --hf_home ${params.hf_home} \\
        --hf_transfer ${params.hf_transfer}
    """
}

/*
 * Aggregate: Merges all chunk results into one file
 */
process AGGREGATE_RESULTS {
    publishDir "${params.output_dir}/all", mode: 'copy'

    input:
    path "chunk_csvs/*"
    path "chunk_jsons/*"

    output:
    path "ToxValDB_Classified_all.csv"
    path "ToxValDB_Reasoning_all.json"

    script:
    """
    python3 -u -c "
import pandas as pd
import json
import glob

# Merge CSVs
csv_files = sorted(glob.glob('chunk_csvs/ToxValDB_Classified_*.csv'))
print(f'Aggregating {len(csv_files)} CSV files...')
dfs = []
for f in csv_files:
    try:
        df = pd.read_csv(f, sep='\t')
        dfs.append(df)
        print(f'  Loaded {f}: {len(df)} rows')
    except Exception as e:
        print(f'  Warning: Could not read {f}: {e}')

if dfs:
    result = pd.concat(dfs, ignore_index=True)
    result.to_csv('ToxValDB_Classified_all.csv', sep='\t', index=False)
    print(f'Total: {len(result)} rows saved')
else:
    print('ERROR: No valid CSV files found!')
    exit(1)

# Merge JSONs
json_files = sorted(glob.glob('chunk_jsons/ToxValDB_Reasoning_*.json'))
print(f'Aggregating {len(json_files)} JSON files...')
merged = {}
for f in json_files:
    try:
        with open(f) as fh:
            merged.update(json.load(fh))
        print(f'  Loaded {f}')
    except Exception as e:
        print(f'  Warning: Could not read {f}: {e}')

json.dump(merged, open('ToxValDB_Reasoning_all.json', 'w'), indent=4)
print(f'Total: {len(merged)} entries saved')
"
    """
}
