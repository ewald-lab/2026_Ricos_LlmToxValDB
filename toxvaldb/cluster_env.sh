#!/bin/bash
# Shared cluster/environment helpers for ToxValDB SLURM and local runs.
#
# Optional environment overrides:
#   CLUSTER_DOMAIN : Domain suffix appended to bare node names (e.g., "ebi.ac.uk").
#                    Leave unset when passing full URLs, FQDNs, or local hostnames.
#   GCC_LIB_PATH   : Directory containing a newer libstdc++ (e.g., Spack GCC lib64)
#                    to prepend to LD_LIBRARY_PATH if it exists on the host.

resolve_base_url() {
    local endpoint="$1"
    local port="${2:-8000}"

    if [[ "$endpoint" =~ ^https?:// ]]; then
        printf '%s\n' "$endpoint"
    elif [[ "$endpoint" == "localhost" || "$endpoint" == *.* || -z "${CLUSTER_DOMAIN:-}" ]]; then
        printf 'http://%s:%s/v1\n' "$endpoint" "$port"
    else
        printf 'http://%s.%s:%s/v1\n' "$endpoint" "${CLUSTER_DOMAIN#.}" "$port"
    fi
}

setup_cluster_env() {
    if command -v module >/dev/null 2>&1; then
        module purge 2>/dev/null || true
        if [ "$#" -gt 0 ]; then
            module load "$@" 2>/dev/null || true
        fi
    fi

    if [ -n "${GCC_LIB_PATH:-}" ] && [ -d "${GCC_LIB_PATH}" ]; then
        export LD_LIBRARY_PATH="${GCC_LIB_PATH}:${LD_LIBRARY_PATH:-}"
    fi
}

