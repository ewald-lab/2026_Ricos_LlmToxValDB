#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Import Modules
include { PROCESS_ONTOLOGY } from './modules/ontology.nf'
include { TEXT_TO_ONTO_MAPPING; AGGREGATE_RESULTS } from './modules/mapping.nf'

/*
 * Split input TSV into N chunks (one per vLLM instance)
 */
process SPLIT_INPUT {
    input:
    path input_csv
    val  n_chunks

    output:
    path "chunk_*.tsv"

    script:
    """
    python3 -c "
import pandas as pd

df = pd.read_csv('${input_csv}', sep='\t')
n = ${n_chunks}
chunk_size = max(1, len(df) // n)

for i in range(n):
    start = i * chunk_size
    end = len(df) if i == n - 1 else start + chunk_size
    chunk = df.iloc[start:end]
    chunk_id = str(i).zfill(4)
    chunk.to_csv(f'chunk_{chunk_id}.tsv', sep='\t', index=False)
"
    """
}

workflow {
    if (!params.input_csv || !params.model_config || !params.rag_config) {
        error "Missing required parameters: --input_csv, --model_config, --rag_config"
    }

    // Parse base_url and model_config: each supports single value or list
    def base_urls    = params.base_url instanceof List    ? params.base_url    : [params.base_url]
    def model_cfgs   = params.model_config instanceof List ? params.model_config : [params.model_config]
    def n_instances  = base_urls.size()

    // If model_config is a single value, broadcast it to all instances
    if (model_cfgs.size() == 1 && n_instances > 1) {
        model_cfgs = model_cfgs * n_instances
    } else if (model_cfgs.size() != n_instances) {
        error "model_config list length (${model_cfgs.size()}) must match base_url list length (${n_instances}), or be a single value"
    }

    log.info "vLLM instances: ${n_instances}"
    [base_urls, model_cfgs].transpose().each { url, cfg -> log.info "  ${url}  →  ${cfg}" }

    ch_input_csv = Channel.fromPath(params.input_csv)

    // --- Stage 1: Prep Ontology (Conditional Auto-detection) ---
    if (!params.ontology_file) {
        error "Missing required parameter: --ontology_file"
    }

    def ont_file = file(params.ontology_file)
    def derived_db = "${projectDir}/../../inputs/ontology_dbs/${params.embeddings_model_id}/${ont_file.baseName}.db"

    if (file(derived_db).exists()) {
        log.info "Found existing ontology database: ${derived_db}. Skipping processing."
        ch_ontology_db = Channel.fromPath(derived_db)
    } else {
        log.info "Existing database not found. Processing ontology: ${params.ontology_file}"
        ch_ontology = Channel.fromPath(params.ontology_file)
        PROCESS_ONTOLOGY(ch_ontology)
        ch_ontology_db = PROCESS_ONTOLOGY.out
    }

    // --- Stage 2: Split by number of vLLM instances (or run whole file if only 1) ---
    if (n_instances > 1) {
        log.info "Splitting input into ${n_instances} chunks for parallel processing."
        SPLIT_INPUT(ch_input_csv, n_instances)

        // Pair each chunk with its corresponding vLLM base_url and model_config
        ch_chunks = SPLIT_INPUT.out
            .flatten()
            .toSortedList { it.name }
            .flatMap { files ->
                [files, base_urls, model_cfgs].transpose().collect { file, url, cfg ->
                    tuple(file.baseName, file, url, cfg)
                }
            }
    } else {
        log.info "Single vLLM instance — processing entire file at once."
        ch_chunks = ch_input_csv.map { f -> tuple("full", f, base_urls[0], model_cfgs[0]) }
    }

    // --- Stage 3: Map ---
    TEXT_TO_ONTO_MAPPING(ch_chunks, ch_ontology_db.first())

    // --- Stage 4: Aggregate (only when multiple chunks) ---
    if (n_instances > 1) {
        AGGREGATE_RESULTS(
            TEXT_TO_ONTO_MAPPING.out[0].collect(),
            TEXT_TO_ONTO_MAPPING.out[1].collect()
        )
    }
}
