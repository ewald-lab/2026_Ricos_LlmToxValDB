/*
 * Process Ontology: Generates embeddings database
 */
process PROCESS_ONTOLOGY {
    tag "${ontology_file.baseName}"
    publishDir "${params.output_dir}/ontology", mode: 'copy'

    input:
    path ontology_file

    output:
    path "ontology_embeddings.db"

    script:
    """
    python3 -m process_ontology \\
        --ontology_file ${ontology_file} \\
        --ontology_db ontology_embeddings.db \\
        --model_config ${params.model_config} \\
        ${params.base_url ? "--base_url ${params.base_url}" : ""} \\
        ${params.root_uri ? "--root_uri ${params.root_uri}" : ""} \\
        ${params.prompt_template ? "--prompt_template ${params.prompt_template}" : ""} \\
        ${params.skip_ontology_embeddings ? "--skip_embeddings" : ""} \\
        ${params.rag_config ? "--rag_config ${params.rag_config}" : ""} \\
        ${params.global_definitions_db ? "--global_definitions_db ${params.global_definitions_db}" : ""} \\
        --embeddings_model_id ${params.embeddings_model_id} \\
        --hf_home ${params.hf_home} \\
        --hf_transfer ${params.hf_transfer}
    """
}
