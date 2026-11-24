#!/bin/bash
# Restore the local ontology snapshots listed in toxvaldb/config/ontologies.json.
#
# The snapshots are gitignored but they are NOT disposable: the ontologies
# pinned to one (pin=local) are the sources the published results were built
# from. Each is verified against the local_sha256 recorded in the config, which
# is the only in-repo record of which bytes produced a result. Ontologies that
# declare no local_file are loaded from their URL at run time and are skipped.
#
#   ./download_ontologies.sh            # restore whatever is missing or corrupt
#   ./download_ontologies.sh --force    # re-fetch everything
#
# Optional environment overrides:
#   ONTOLOGY_SNAPSHOT_SRC : Directory that already holds the snapshots (for
#                           example the owl_files directory of another
#                           checkout). A file found there is copied from it.
#   JQ_MODULE             : See toxvaldb/config/ontologies.sh.
#
# A pin=local snapshot is restored only from ONTOLOGY_SNAPSHOT_SRC, never from
# the ontology's URL: the URL is not guaranteed to serve the same bytes, or
# even the same serialization. A pin=url snapshot not found in
# ONTOLOGY_SNAPSHOT_SRC is fetched from its URL.
#
# A file is put in place only after its checksum matches. A copy or download
# that does not verify is discarded and whatever was there before is kept.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOXVALDB_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# Provides ONTOLOGY_CONFIG_JSON and ONTOLOGY_TOXVALDB_ROOT, and makes sure jq
# is available.
source "${TOXVALDB_ROOT}/config/ontologies.sh" || exit 1

CONFIG_JSON="${ONTOLOGY_CONFIG_JSON}"
[ -f "${CONFIG_JSON}" ] || { echo "error: not found: ${CONFIG_JSON}" >&2; exit 1; }

DEST_DIR="${ONTOLOGY_TOXVALDB_ROOT}/$(jq -r '.local_dir' "${CONFIG_JSON}")"
SNAPSHOT_SRC="${ONTOLOGY_SNAPSHOT_SRC:-}"

FORCE=0
[ "${1:-}" == "--force" ] && FORCE=1

if [ -n "${SNAPSHOT_SRC}" ] && [ ! -d "${SNAPSHOT_SRC}" ]; then
    echo "error: ONTOLOGY_SNAPSHOT_SRC is not a directory: ${SNAPSHOT_SRC}" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"

# One field per line rather than @tsv: tab is an IFS whitespace character, so
# `read` collapses runs of them and an entry with no local_file (one pinned to
# its url) would shift its url left into the file variable. The same trap is
# documented in toxvaldb/config/ontologies.sh.
failed=0
while read -r id && read -r pin && read -r file && read -r expected && read -r url; do
    # An ontology pinned to its url needs no snapshot at all, and declares no
    # local_file. Without this guard the destination would resolve to DEST_DIR
    # itself, the fetch would fail, and one url-only entry would stop the other
    # snapshots being restored.
    #
    # Check the pin before calling it a skip: an entry pinned to its local
    # snapshot that declares no local_file is a broken config, and announcing
    # it as "tracks its url" would exit 0 on a missing required snapshot.
    if [ -z "${file}" ]; then
        if [ "${pin}" == "url" ]; then
            echo "skip     ${id}: tracks its url, no local snapshot"
            continue
        fi
        echo "         MISSING ${id}: pinned to its local snapshot but declares no" >&2
        echo "         local_file in ${CONFIG_JSON}." >&2
        failed=1
        continue
    fi

    dest="${DEST_DIR}/${file}"

    if [ "${FORCE}" -eq 0 ] && [ -f "${dest}" ]; then
        if ontology_verify_checksum "${dest}" "${expected}" 2>/dev/null; then
            echo "ok       ${id} (${file})"
            continue
        fi
        echo "stale    ${id}: checksum differs, re-fetching"
    fi

    # Fetch next to the destination, so the final mv is a rename on the same
    # filesystem and never follows a symlinked destination into its target.
    tmp="${dest}.download.$$"

    if [ -n "${SNAPSHOT_SRC}" ] && [ -f "${SNAPSHOT_SRC}/${file}" ]; then
        echo "copying  ${id} from ${SNAPSHOT_SRC}"
        cp "${SNAPSHOT_SRC}/${file}" "${tmp}" || {
            echo "         FAILED to copy ${id}" >&2; rm -f "${tmp}"; failed=1; continue; }
    elif [ "${pin}" == "local" ]; then
        echo "         MISSING ${id}: pinned to its local snapshot, which is not fetched" >&2
        echo "         from a URL. Set ONTOLOGY_SNAPSHOT_SRC to a directory that holds" >&2
        echo "         ${file}, or ask a colleague for the file." >&2
        failed=1
        continue
    elif [ -n "${url}" ]; then
        echo "fetching ${id} from ${url}"
        curl -sSL --retry 3 --retry-delay 2 --max-time 900 -o "${tmp}" "${url}" || {
            echo "         FAILED to fetch ${id}" >&2; rm -f "${tmp}"; failed=1; continue; }
    else
        echo "         MISSING ${id}: it has no URL. Set ONTOLOGY_SNAPSHOT_SRC to a" >&2
        echo "         directory that holds ${file}, or ask a colleague for the file." >&2
        failed=1
        continue
    fi

    if ontology_verify_checksum "${tmp}" "${expected}" && mv -f "${tmp}" "${dest}"; then
        echo "         verified ${id}"
    else
        rm -f "${tmp}"
        echo "         NOT RESTORED ${id}: ${dest} was left as it was." >&2
        # An ontology that tracks a moving URL will legitimately differ from
        # its recorded snapshot once a new release is out.
        echo "         If this ontology tracks a URL, a new version may have been" >&2
        echo "         released since the snapshot was taken. Update local_sha256 and" >&2
        echo "         release_seen in ${CONFIG_JSON} only if that is intended." >&2
        failed=1
    fi
done < <(jq -r '.ontologies | to_entries[]
                | (.key,
                   (.value.pin          // ""),
                   (.value.local_file   // ""),
                   (.value.local_sha256 // ""),
                   (.value.url          // ""))' "${CONFIG_JSON}")

if [ "${failed}" -ne 0 ]; then
    echo "
One or more ontologies could not be restored or did not verify." >&2
    exit 1
fi
echo "
All ontology snapshots present and verified."
