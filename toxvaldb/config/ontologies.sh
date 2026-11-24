#!/bin/bash
# Bash accessor for toxvaldb/config/ontologies.json, the single source of truth
# for the ontologies used across every module. Jobs that source this file must
# not hardcode a URL, an rdflib format, a root term, a per-ontology threshold or
# a DB location -- every per-ontology fact lives in the JSON and nowhere else.
#
# Jobs move onto this file one stage at a time. The expert/LLM evaluation jobs
# under 03.evaluation/modules/{01.eval_experts,02.eval_llms} read
# inputs/owl_files/<id>.ttl directly and are out of scope while that
# evaluation is not being run; migrate them here if it restarts.
#
#   source "${TOXVALDB_ROOT}/config/ontologies.sh"
#   ontology_config mp                  # or: ontology_by_index "${SLURM_ARRAY_TASK_ID}"
#   python foo.py --owl_file "${ONTOLOGY_FILE}" ${ONTOLOGY_FORMAT_FLAG}
#
# Each ontology declares its own source policy via `pin` in the JSON: the frozen
# ones resolve to their local snapshot (that is what produced the published
# results), and ones added after the freeze resolve to their canonical URL.
# ONTOLOGY_SOURCE=local|url overrides the policy for every ontology at once --
# that is the deliberate-refresh path, not something a results run should do.
#
# Optional environment overrides:
#   ONTOLOGY_SOURCE : "local" or "url"; see above.
#   JQ_MODULE       : Environment module that provides jq on hosts where it is
#                     not already on PATH (default: "jq").

# Every relative path in the JSON is resolved against toxvaldb/, the parent of
# this directory. It is taken from this file's own location rather than from
# the caller's TOXVALDB_ROOT, so the helper and the JSON it reads cannot come
# from two different checkouts.
_ONTOLOGY_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONTOLOGY_TOXVALDB_ROOT="$(dirname "${_ONTOLOGY_CONFIG_DIR}")"
ONTOLOGY_CONFIG_JSON="${_ONTOLOGY_CONFIG_DIR}/ontologies.json"

# _ontology_require_jq -- make sure jq is on PATH, loading it from an
# environment module when the host does not ship it.
#
# Every function that reads the JSON calls this first, rather than the check
# running once at source time: job scripts call setup_cluster_env
# (cluster_env.sh) after sourcing this file, and its `module purge` unloads a
# jq that came from a module. Lmod's own error, if any, is left on stderr so a
# failed load says why.
_ontology_require_jq() {
    command -v jq >/dev/null 2>&1 && return 0

    local module_name="${JQ_MODULE:-jq}"
    local tried="no 'module' command to load ${module_name} with"
    if type module >/dev/null 2>&1; then
        module load "${module_name}" >/dev/null
        tried="tried 'module load ${module_name}'"
    fi
    if ! command -v jq >/dev/null 2>&1; then
        echo "ontologies.sh: jq is required but not on PATH (${tried})" >&2
        return 1
    fi
}
_ontology_require_jq || { return 1 2>/dev/null || exit 1; }

# Canonical order. Slurm array indices map into this list, so appending is safe
# but reordering changes what `--array=N` selects.
mapfile -t ONTOLOGY_IDS < <(jq -r '.order[]' "${ONTOLOGY_CONFIG_JSON}")
if [ "${#ONTOLOGY_IDS[@]}" -eq 0 ]; then
    echo "ontologies.sh: could not read .order from ${ONTOLOGY_CONFIG_JSON}" >&2
    return 1 2>/dev/null || exit 1
fi

# ontology_verify_checksum <file> <expected_sha256>
#
# Snapshots are gitignored, so the checksum in the JSON is the only in-repo
# record of which bytes produced a given result. Verifying is cheap next to the
# GPU work that follows.
ontology_verify_checksum() {
    local file="${1:-}" expected="${2:-}"

    [ -z "${expected}" ] && return 0

    local actual
    actual="$(sha256sum "${file}" | cut -d' ' -f1)" || return 1
    if [ "${actual}" != "${expected}" ]; then
        echo "ontology_config: checksum mismatch for ${file}" >&2
        echo "  expected ${expected}" >&2
        echo "  actual   ${actual}" >&2
        echo "  The snapshot no longer matches the one recorded in ontologies.json." >&2
        echo "  Restore it, or update local_sha256 if the change was deliberate." >&2
        return 1
    fi
}

# ontology_config <id>
#
# Populates, for the given ontology:
#   ONTOLOGY_ID, ONTOLOGY_FILE, ONTOLOGY_FORMAT, ONTOLOGY_SOURCE_KIND,
#   ONTOLOGY_ROOT_URI, ONTOLOGY_ROOT_TERM, ONTOLOGY_TREE_MIN_CASES,
#   ONTOLOGY_RELEASE_SEEN, ONTOLOGY_DB_DIR, ONTOLOGY_EXCLUDE_URIS
# plus ONTOLOGY_ROOT_URI_FLAG / ONTOLOGY_FORMAT_FLAG / ONTOLOGY_EXCLUDE_URIS_FLAG,
# ready to splat unquoted into a python call (each empty when the field is unset).
#
# The globals are written only once everything has resolved. On failure they
# keep whatever the last successful call left, so they never describe a mix of
# two ontologies.
ontology_config() {
    local id="${1:-}"

    _ontology_require_jq || return 1

    if [ -z "${id}" ]; then
        echo "ontology_config: no ontology id given" >&2
        return 1
    fi
    if ! jq -e --arg id "${id}" '.ontologies[$id]' "${ONTOLOGY_CONFIG_JSON}" >/dev/null 2>&1; then
        echo "ontology_config: unknown ontology '${id}' (known: ${ONTOLOGY_IDS[*]})" >&2
        return 1
    fi

    # Emit shell assignments rather than a delimited record: tab is an IFS
    # whitespace character, so `read` would collapse the empty url/url_format
    # fields of a local-only ontology and shift every later value left. @sh
    # also quotes the root terms, which contain spaces.
    #
    # Capture before eval: `eval "$(cmd)" || ...` reports eval's status, not the
    # command's, so a jq failure would silently leave every variable unset.
    local _assignments
    _assignments="$(jq -r --arg id "${id}" '
        .ontologies[$id] as $o |
        "_pin=\($o.pin | @sh)",
        "_db_dir=\($o.db_dir | @sh)",
        "_url=\($o.url // "" | @sh)",
        "_url_format=\($o.url_format // "" | @sh)",
        "_local_file=\($o.local_file // "" | @sh)",
        "_local_format=\($o.local_format // "" | @sh)",
        "_local_sha256=\($o.local_sha256 // "" | @sh)",
        "_local_dir=\(.local_dir | @sh)",
        "_root_uri=\($o.root_uri // "" | @sh)",
        "_root_term=\($o.root_term // "" | @sh)",
        "_exclude_uris=\($o.exclude_uris // [] | join(" ") | @sh)",
        "_tree_min_cases=\($o.tree_min_cases | tostring | @sh)",
        "_release_seen=\($o.release_seen // "" | @sh)"
    ' "${ONTOLOGY_CONFIG_JSON}")" || return 1

    local _pin _url _url_format _local_file _local_format _local_sha256 _local_dir
    local _db_dir _root_uri _root_term _exclude_uris _tree_min_cases _release_seen
    local _file _format _kind
    eval "${_assignments}"

    # ONTOLOGY_SOURCE overrides the per-ontology pin when set. Validate it:
    # an unrecognised value used to fall through to the snapshot silently,
    # which meant a typo quietly chose a source nobody asked for.
    # Default every optional environment variable and positional parameter read
    # in this file: a caller may run under `set -u`, where a bare
    # ${ONTOLOGY_SOURCE} aborts the job instead of meaning "unset".
    if [ -n "${ONTOLOGY_SOURCE:-}" ] \
       && [ "${ONTOLOGY_SOURCE}" != "local" ] && [ "${ONTOLOGY_SOURCE}" != "url" ]; then
        echo "ontology_config: invalid ONTOLOGY_SOURCE='${ONTOLOGY_SOURCE}' (expected 'local' or 'url')" >&2
        return 1
    fi
    local want="${ONTOLOGY_SOURCE:-${_pin}}"
    local local_path="${ONTOLOGY_TOXVALDB_ROOT}/${_local_dir}/${_local_file}"

    if [ "${want}" == "url" ] && [ -n "${_url}" ]; then
        _file="${_url}"
        _format="${_url_format}"
        _kind="url"
    else
        if [ "${want}" == "url" ]; then
            echo "ontology_config: '${id}' has no canonical URL; using the local snapshot." >&2
        fi
        # A url-only entry (mpath) declares no local_file; say so rather than
        # reporting a missing file named after the empty field.
        if [ -z "${_local_file}" ]; then
            echo "ontology_config: '${id}' keeps no local snapshot; it resolves only from ${_url}." >&2
            return 1
        fi
        if [ ! -f "${local_path}" ]; then
            echo "ontology_config: local snapshot missing: ${local_path}" >&2
            if [ -z "${_url}" ]; then
                echo "  '${id}' has no canonical URL, so this file is required." >&2
            else
                echo "  Set ONTOLOGY_SOURCE=url to load '${id}' from ${_url} instead," >&2
                echo "  but note that changes the graph the results were built from." >&2
            fi
            return 1
        fi
        ontology_verify_checksum "${local_path}" "${_local_sha256}" || return 1
        _file="${local_path}"
        _format="${_local_format}"
        _kind="local"
    fi

    ONTOLOGY_ID="${id}"
    ONTOLOGY_FILE="${_file}"
    ONTOLOGY_FORMAT="${_format}"
    ONTOLOGY_SOURCE_KIND="${_kind}"
    ONTOLOGY_DB_DIR="${_db_dir}"
    ONTOLOGY_ROOT_URI="${_root_uri}"
    ONTOLOGY_ROOT_TERM="${_root_term}"
    ONTOLOGY_EXCLUDE_URIS="${_exclude_uris}"
    ONTOLOGY_TREE_MIN_CASES="${_tree_min_cases}"
    ONTOLOGY_RELEASE_SEEN="${_release_seen}"
    ONTOLOGY_ROOT_URI_FLAG=""
    [ -n "${ONTOLOGY_ROOT_URI}" ] && ONTOLOGY_ROOT_URI_FLAG="--root_uri ${ONTOLOGY_ROOT_URI}"
    ONTOLOGY_FORMAT_FLAG=""
    [ -n "${ONTOLOGY_FORMAT}" ] && ONTOLOGY_FORMAT_FLAG="--ontology_format ${ONTOLOGY_FORMAT}"
    ONTOLOGY_EXCLUDE_URIS_FLAG=""
    [ -n "${ONTOLOGY_EXCLUDE_URIS}" ] && ONTOLOGY_EXCLUDE_URIS_FLAG="--exclude_uris ${ONTOLOGY_EXCLUDE_URIS}"
    # The test above fails for every ontology without exclusions; without this
    # the function would return that status and `ontology_config x || exit 1`
    # would abort the job.
    return 0
}

# ontology_group_ids <group> -- print one ontology id per line.
#
# "all" is every entry in .order; any other name is looked up in .groups. Jobs
# should ask for a group rather than carrying their own list -- that is what
# stops the subsets drifting apart across scripts.
ontology_group_ids() {
    local group="${1:-all}"

    _ontology_require_jq || return 1

    if [ "${group}" == "all" ]; then
        printf '%s\n' "${ONTOLOGY_IDS[@]}"
        return 0
    fi

    if ! jq -e --arg g "${group}" '.groups[$g]' "${ONTOLOGY_CONFIG_JSON}" >/dev/null 2>&1; then
        local known
        known="$(jq -r '.groups | keys | join(", ")' "${ONTOLOGY_CONFIG_JSON}")"
        echo "ontology_group_ids: unknown group '${group}' (known: all, ${known})" >&2
        return 1
    fi
    jq -r --arg g "${group}" '.groups[$g][]' "${ONTOLOGY_CONFIG_JSON}"
}

# _ontology_group_array <group> <array_name> -- read a group's ids into the
# caller's array.
#
# `mapfile < <(ontology_group_ids ...) || return 1` cannot do this: the status
# it tests is mapfile's, so an unknown group would yield an empty array and
# carry on.
_ontology_group_array() {
    local _listing
    _listing="$(ontology_group_ids "${1:-}")" || return 1
    if [ -z "${_listing}" ]; then
        echo "ontologies.sh: group '${1:-}' has no ontologies" >&2
        return 1
    fi
    mapfile -t "$2" <<< "${_listing}"
}

# ontology_by_index <n> [group] -- map a Slurm array task id onto a group,
# defaulting to every configured ontology.
ontology_by_index() {
    local idx="${1:-}"
    local group="${2:-all}"

    # Guard non-negative integers explicitly: bash counts a negative index from
    # the end of the array, so `-1` would silently resolve to the last ontology.
    if ! [[ "${idx}" =~ ^[0-9]+$ ]]; then
        echo "ontology_by_index: invalid index '${idx}' (expected a non-negative integer)" >&2
        ontology_echo_index_map "${group}" >&2 || true
        return 1
    fi

    local ids
    _ontology_group_array "${group}" ids || return 1

    # An #SBATCH --array range cannot be computed from this file (Slurm parses
    # the directive before the script runs), so it is the one place an ontology
    # count is duplicated. Catch the drift rather than silently skipping one.
    if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] \
       && [ "${SLURM_ARRAY_TASK_COUNT}" -ne "${#ids[@]}" ]; then
        echo "ontology_by_index: WARNING --array covers ${SLURM_ARRAY_TASK_COUNT} task(s)" \
             "but group '${group}' has ${#ids[@]} ontologies; update the #SBATCH --array range." >&2
    fi

    local id="${ids[$idx]:-}"
    if [ -z "${id}" ]; then
        echo "ontology_by_index: invalid index '${idx}' for group '${group}'" \
             "(valid: 0-$(( ${#ids[@]} - 1 )))" >&2
        ontology_echo_index_map "${group}" >&2
        return 1
    fi
    ontology_config "${id}"
}

# ontology_echo_index_map [group] -- the array index -> ontology mapping, so a
# job log always records which ontology a given --array index actually selected.
ontology_echo_index_map() {
    local group="${1:-all}" i
    local ids
    _ontology_group_array "${group}" ids || return 1
    echo "  Array index map (group '${group}'):"
    for i in "${!ids[@]}"; do
        echo "    [${i}] ${ids[$i]}"
    done
}

# ontology_echo_config -- log the resolved config, for job output.
ontology_echo_config() {
    ontology_require_resolved "ontology_echo_config" || return 1
    echo "  Ontology   : ${ONTOLOGY_ID}"
    echo "  Source     : ${ONTOLOGY_FILE}"
    echo "  Source kind: ${ONTOLOGY_SOURCE_KIND}${ONTOLOGY_SOURCE:+ (forced by ONTOLOGY_SOURCE=${ONTOLOGY_SOURCE})}"
    echo "  Format     : ${ONTOLOGY_FORMAT}"
    echo "  Release    : ${ONTOLOGY_RELEASE_SEEN} (recorded; check the log for the versionIRI actually loaded)"
    echo "  Root URI   : ${ONTOLOGY_ROOT_URI:-<none>}"
    echo "  Root term  : ${ONTOLOGY_ROOT_TERM}"
    echo "  Excluded   : ${ONTOLOGY_EXCLUDE_URIS:-<none>}"
}

# ontology_require_resolved <caller> -- fail loudly instead of emitting empty
# strings when a helper is called before ontology_config populated the globals.
ontology_require_resolved() {
    if [ -z "${ONTOLOGY_ID:-}" ]; then
        echo "${1:-ontologies.sh}: no ontology resolved -- call ontology_config <id> or ontology_by_index <n> first" >&2
        return 1
    fi
}

# ontology_db_root <prepare_module_root> -- where this ontology's generated
# embedding/definition DBs live, echoed to stdout.
#
# The per-ontology part (db_dir) comes from ontologies.json, so this function
# holds no ontology-specific knowledge. Which ontologies build into outputs
# and which into outputs_local, and why, is recorded there.
ontology_db_root() {
    local module_root="${1:-}"

    if [ -z "${module_root}" ]; then
        echo "ontology_db_root: no module root given" >&2
        return 1
    fi
    ontology_require_resolved "ontology_db_root" || return 1
    echo "${module_root}/${ONTOLOGY_DB_DIR}"
}

# ontology_db_root_for <id> <prepare_module_root> -- the same answer for a named
# ontology, without resolving its source graph.
#
# Callers that only want DB locations -- the coverage jobs walk a whole group to
# collect them -- would otherwise call ontology_config per ontology, paying a
# checksum over every snapshot (and failing outright when one is absent) to
# learn a path that does not depend on the source at all.
ontology_db_root_for() {
    local id="${1:-}" module_root="${2:-}" db_dir

    _ontology_require_jq || return 1

    # Same validation contract as ontology_config: an id that is not in the
    # config and an entry that is there but declares no db_dir are different
    # mistakes, and reporting both as "unknown ontology" sends the reader
    # looking in the wrong place.
    if [ -z "${id}" ]; then
        echo "ontology_db_root_for: no ontology id given" >&2
        return 1
    fi
    if [ -z "${module_root}" ]; then
        echo "ontology_db_root_for: no module root given" >&2
        return 1
    fi
    if ! jq -e --arg id "${id}" '.ontologies[$id]' "${ONTOLOGY_CONFIG_JSON}" >/dev/null 2>&1; then
        echo "ontology_db_root_for: unknown ontology '${id}' (known: ${ONTOLOGY_IDS[*]})" >&2
        return 1
    fi

    db_dir="$(jq -r --arg id "${id}" '.ontologies[$id].db_dir // empty' \
        "${ONTOLOGY_CONFIG_JSON}")" || return 1
    if [ -z "${db_dir}" ]; then
        echo "ontology_db_root_for: '${id}' declares no db_dir in ${ONTOLOGY_CONFIG_JSON}" >&2
        return 1
    fi
    echo "${module_root}/${db_dir}"
}

# ontology_db_roots_for_group <group> <prepare_module_root> -- the distinct DB
# roots the group spans, one per line, in canonical order.
#
# The ontologies in a group do not all share a root (db_dir differs between
# entries), so a job that reads DBs for a whole group has to
# search every root the group touches. Which ids share a root is a fact about
# the config, so the walk and the de-duplication live here rather than being
# hand-rolled in each such job.
ontology_db_roots_for_group() {
    local group="${1:-}" module_root="${2:-}" id root
    local ids roots=()

    if [ -z "${module_root}" ]; then
        echo "ontology_db_roots_for_group: no module root given" >&2
        return 1
    fi
    _ontology_group_array "${group}" ids || return 1

    for id in "${ids[@]}"; do
        root="$(ontology_db_root_for "${id}" "${module_root}")" || return 1
        roots+=("${root}")
    done
    printf '%s\n' "${roots[@]}" | awk '!seen[$0]++'
}
