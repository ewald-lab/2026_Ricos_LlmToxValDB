#!/bin/bash
# Restore the local ontology snapshots listed in toxvaldb/config/ontologies.json.
#
# The snapshots are gitignored (~0.5GB) but they are NOT disposable: eight of
# the fourteen ontologies keep one, and five of those (aop, hp, hpath, mp, txpo)
# are pinned to it, so they are the sources the published results were built
# from. Each is verified against the local_sha256 recorded in the config, which
# is the only in-repo record of which bytes produced a result. The other six
# ontologies are loaded from their URL at run time and are skipped here.
#
#   ./download_ontologies.sh            # restore whatever is missing or corrupt
#   ./download_ontologies.sh --force    # re-fetch everything
#
# Optional environment overrides:
#   ONTOLOGY_SNAPSHOT_SRC : Directory that already holds the snapshots (for
#                           example the owl_files directory of another
#                           checkout). A file found there is copied from it;
#                           anything else is fetched from the ontology's URL.
#   JQ_MODULE             : See toxvaldb/config/ontologies.sh.
#
# aop and hpath have no URL, and the hp, mp and txpo snapshots are Turtle
# files that their URLs do not serve, so the five pinned snapshots can only be
# restored from ONTOLOGY_SNAPSHOT_SRC. A URL download that does not match the
# recorded checksum is reported as a failure.

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
        actual="$(sha256sum "${dest}" | cut -d' ' -f1)"
        if [ "${actual}" == "${expected}" ]; then
            echo "ok       ${id} (${file})"
            continue
        fi
        echo "stale    ${id}: checksum differs, re-fetching"
    fi

    if [ -n "${SNAPSHOT_SRC}" ] && [ -f "${SNAPSHOT_SRC}/${file}" ]; then
        echo "copying  ${id} from ${SNAPSHOT_SRC}"
        cp "${SNAPSHOT_SRC}/${file}" "${dest}" || {
            echo "         FAILED to copy ${id}" >&2; failed=1; continue; }
    elif [ -n "${url}" ]; then
        echo "fetching ${id} from ${url}"
        curl -sSL --retry 3 --retry-delay 2 --max-time 900 -o "${dest}" "${url}" || {
            echo "         FAILED to fetch ${id}" >&2; failed=1; continue; }
    else
        echo "         MISSING ${id}: it has no URL, so it can only be copied from a" >&2
        echo "         directory that already holds ${file}. Set ONTOLOGY_SNAPSHOT_SRC" >&2
        echo "         to that directory, or ask a colleague for the file." >&2
        failed=1
        continue
    fi

    actual="$(sha256sum "${dest}" | cut -d' ' -f1)"
    if [ "${actual}" == "${expected}" ]; then
        echo "         verified ${id}"
    else
        echo "         CHECKSUM MISMATCH for ${id}" >&2
        echo "           expected ${expected}" >&2
        echo "           actual   ${actual}" >&2
        # Ontologies that track a moving URL will legitimately differ from the
        # pinned snapshot; anything pinned to local must match exactly.
        echo "         If this ontology tracks a URL, upstream has released a new" >&2
        echo "         version since the snapshot was taken. Update local_sha256 and" >&2
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
