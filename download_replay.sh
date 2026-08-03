#!/usr/bin/env bash

set -euo pipefail

################################################################################
# Configuration
################################################################################

COMPETITION="pokemon-tcg-ai-battle"

################################################################################
# Usage
################################################################################

usage() {
cat << EOF
Download all replay JSONs from a Kaggle simulation submission.

Usage:
    ./download_replay.sh --submission_id <submission_id> [--output-dir <directory>]

Arguments:
    --submission_id    Kaggle submission ID (required)

    --output-dir       Root output directory
                       Default: ./replays

Examples:
    ./download_replay.sh --submission_id 54773249

    ./download_replay.sh \
        --submission_id 54773249 \
        --output-dir ./my_replays
EOF
exit 1
}

################################################################################
# Parse arguments
################################################################################

SUBMISSION_ID=""
OUTPUT_ROOT="./replays"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submission_id)
            [[ $# -lt 2 ]] && usage
            SUBMISSION_ID="$2"
            shift 2
            ;;
        --output-dir)
            [[ $# -lt 2 ]] && usage
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

if [[ -z "$SUBMISSION_ID" ]]; then
    usage
fi

################################################################################
# Check Kaggle CLI
################################################################################

if ! command -v kaggle >/dev/null 2>&1; then
    echo "ERROR: Kaggle CLI is not installed."
    echo
    echo "Install it with:"
    echo
    echo "    pip install kaggle"
    exit 1
fi

################################################################################
# Check authentication
################################################################################

echo "Checking Kaggle authentication..."

if ! kaggle competitions list >/dev/null 2>&1; then
    echo
    echo "Kaggle authentication not found."
    echo
    echo "Please authenticate first."
    echo
    echo "1. Go to https://www.kaggle.com/settings"
    echo "2. Create a new API Token"
    echo "3. Save kaggle.json to:"
    echo
    echo "   ~/.kaggle/kaggle.json"
    echo
    echo "4. Run:"
    echo
    echo "   chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo "Authentication OK."
echo

################################################################################
# Prepare output directory
################################################################################

OUTPUT_DIR="${OUTPUT_ROOT}/${SUBMISSION_ID}"

mkdir -p "$OUTPUT_DIR"

################################################################################
# Fetch episode list
################################################################################

echo "Fetching episodes for submission ${SUBMISSION_ID}..."

TMPFILE=$(mktemp)

if ! kaggle competitions episodes "$SUBMISSION_ID" > "$TMPFILE"; then
    echo
    echo "Failed to fetch episodes."
    echo
    echo "Possible reasons:"
    echo "  - Invalid submission ID"
    echo "  - Submission is private"
    echo "  - Kaggle API limitation"
    rm -f "$TMPFILE"
    exit 1
fi

EPISODES=$(awk '
NR > 1 {
    if ($1 ~ /^[0-9]+$/)
        print $1
}
' "$TMPFILE")

rm -f "$TMPFILE"

COUNT=$(echo "$EPISODES" | grep -c '^[0-9]' || true)

if [[ "$COUNT" -eq 0 ]]; then
    echo
    echo "No episodes found."
    exit 1
fi

echo "Found ${COUNT} episodes."
echo

################################################################################
# Download replays
################################################################################

cd "$OUTPUT_DIR"

# Is this episode already on disk?
#
# `kaggle competitions replay` names its output "episode-<id>-replay.json",
# not "<id>.json" — so a check against the bare id never matches and every
# run re-downloads the whole submission (~900 episodes for the larger ones).
# The legacy bare-id name is still accepted so anything fetched under an
# older layout keeps counting as present.
#
# -s, not -f: a download killed partway leaves a zero-byte file behind, and
# treating that as "already downloaded" would cache the failure permanently.
# An empty file is retried instead.
have_replay() {
    local episode_id="$1"
    [[ -s "episode-${episode_id}-replay.json" ]] || [[ -s "${episode_id}.json" ]]
}

CURRENT=1
SKIPPED=0
DOWNLOADED=0
FAILED=0

while read -r EPISODE_ID; do

    [[ -z "$EPISODE_ID" ]] && continue

    echo "[$CURRENT/$COUNT] Episode ${EPISODE_ID}"

    if have_replay "$EPISODE_ID"; then
        echo "  Already downloaded — skipping."
        ((SKIPPED++))
    elif kaggle competitions replay "$EPISODE_ID"; then
        echo "  Downloaded."
        ((DOWNLOADED++))
    else
        echo "  Failed."
        ((FAILED++))
    fi

    ((CURRENT++))

done <<< "$EPISODES"

################################################################################
# Done
################################################################################

echo
echo "========================================"
echo "Finished!"
echo "Submission : ${SUBMISSION_ID}"
echo "Saved to   : ${OUTPUT_DIR}"
echo "Skipped    : ${SKIPPED} (already present)"
echo "Downloaded : ${DOWNLOADED}"
echo "Failed     : ${FAILED}"
echo "========================================"