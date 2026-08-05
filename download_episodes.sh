#!/usr/bin/env bash

set -euo pipefail

################################################################################
# Configuration
################################################################################

INDEX_DATASET="kaggle/pokemon-tcg-ai-battle-episodes-index"

# Every daily dataset is named <DAILY_PREFIX><YYYY-MM-DD>. The manifest carries
# the slug explicitly, so this is only used as a fallback when --date is passed
# without a manifest on disk.
DAILY_PREFIX="kaggle/pokemon-tcg-ai-battle-episodes-"

################################################################################
# Usage
################################################################################

usage() {
cat << EOF
Download Pokemon TCG AI Battle episode data from Kaggle.

The published data is two-tiered:

  1. An *index* dataset (a single manifest.csv) listing one daily dataset per
     day, with episode counts, sizes and score stats.
  2. One *daily* dataset per day, holding the actual episode JSONs. These are
     large — 3 to 21 GB each — so they are never downloaded implicitly.

Usage:
    ./download_episodes.sh [--index] [--list]
    ./download_episodes.sh --date <YYYY-MM-DD> [--date ...]
    ./download_episodes.sh --latest <N>
    ./download_episodes.sh --all

Arguments:
    --index            Refresh the index dataset only (default action when no
                       other action is given).

    --list             Print the manifest as a table and exit. Implies --index
                       if no manifest is present yet.

    --date <date>      Download the daily dataset for this date. Repeatable.

    --latest <N>       Download the N most recent daily datasets.

    --all              Download every daily dataset in the manifest.
                       This is hundreds of GB. You will be asked to confirm.

    --output-dir       Root output directory.
                       Default: ./data

    --dry-run          Print what would be downloaded, download nothing.

    --yes              Skip the size confirmation prompt.

Layout:
    <output-dir>/episodes_index/manifest.csv
    <output-dir>/episodes/<YYYY-MM-DD>/...

Examples:
    ./download_episodes.sh --list

    ./download_episodes.sh --date 2026-08-01

    ./download_episodes.sh --latest 3 --dry-run
EOF
exit 1
}

################################################################################
# Parse arguments
################################################################################

OUTPUT_ROOT="./data"
DO_INDEX=0
DO_LIST=0
DO_ALL=0
LATEST=0
DRY_RUN=0
ASSUME_YES=0
DATES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --index)
            DO_INDEX=1
            shift
            ;;
        --list)
            DO_LIST=1
            shift
            ;;
        --date)
            [[ $# -lt 2 ]] && usage
            DATES+=("$2")
            shift 2
            ;;
        --latest)
            [[ $# -lt 2 ]] && usage
            LATEST="$2"
            shift 2
            ;;
        --all)
            DO_ALL=1
            shift
            ;;
        --output-dir)
            [[ $# -lt 2 ]] && usage
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --yes|-y)
            ASSUME_YES=1
            shift
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

if [[ ! "$LATEST" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --latest expects a number, got: ${LATEST}"
    exit 1
fi

for DATE in "${DATES[@]:-}"; do
    [[ -z "$DATE" ]] && continue
    if [[ ! "$DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "ERROR: --date expects YYYY-MM-DD, got: ${DATE}"
        exit 1
    fi
done

# With no action at all, the useful default is "get me the index".
WANTS_DAILY=0
if [[ ${#DATES[@]} -gt 0 || "$LATEST" -gt 0 || "$DO_ALL" -eq 1 ]]; then
    WANTS_DAILY=1
fi

if [[ "$DO_INDEX" -eq 0 && "$DO_LIST" -eq 0 && "$WANTS_DAILY" -eq 0 ]]; then
    DO_INDEX=1
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

if ! kaggle datasets files "$INDEX_DATASET" >/dev/null 2>&1; then
    echo
    echo "Kaggle authentication not found, or the index dataset is unreachable."
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
# Index
################################################################################

INDEX_DIR="${OUTPUT_ROOT}/episodes_index"
MANIFEST="${INDEX_DIR}/manifest.csv"

fetch_index() {
    echo "Fetching index dataset ${INDEX_DATASET}..."
    mkdir -p "$INDEX_DIR"
    kaggle datasets download -d "$INDEX_DATASET" -p "$INDEX_DIR" --unzip --force
    echo "Manifest: ${MANIFEST}"
    echo
}

if [[ "$DO_INDEX" -eq 1 ]]; then
    fetch_index
fi

# --list and any daily selection both need a manifest to read.
if [[ ! -f "$MANIFEST" ]] && [[ "$DO_LIST" -eq 1 || "$WANTS_DAILY" -eq 1 ]]; then
    fetch_index
fi

################################################################################
# Manifest helpers
################################################################################

# manifest.csv columns:
#   date,daily_dataset_slug,daily_dataset_url,episode_count,total_bytes,
#   top_avg_score,median_avg_score

human_bytes() {
    awk -v b="$1" 'BEGIN {
        split("B KB MB GB TB", u, " ")
        i = 1
        while (b >= 1024 && i < 5) { b /= 1024; i++ }
        printf "%.1f %s", b, u[i]
    }'
}

if [[ "$DO_LIST" -eq 1 ]]; then
    awk -F, '
    NR == 1 { next }
    {
        bytes = $5
        split("B KB MB GB TB", u, " ")
        i = 1
        while (bytes >= 1024 && i < 5) { bytes /= 1024; i++ }
        printf "%-12s %8d episodes  %10.1f %-2s  top %9.2f  median %9.2f\n",
            $1, $4, bytes, u[i], $6, $7
    }
    ' "$MANIFEST"
    echo
    [[ "$WANTS_DAILY" -eq 0 ]] && exit 0
fi

################################################################################
# Select daily datasets
################################################################################

SELECTED=()

if [[ "$WANTS_DAILY" -eq 1 ]]; then

    ALL_DATES=$(awk -F, 'NR > 1 && $1 != "" { print $1 }' "$MANIFEST" | sort)

    if [[ "$DO_ALL" -eq 1 ]]; then
        while read -r D; do
            [[ -n "$D" ]] && SELECTED+=("$D")
        done <<< "$ALL_DATES"
    fi

    if [[ "$LATEST" -gt 0 ]]; then
        while read -r D; do
            [[ -n "$D" ]] && SELECTED+=("$D")
        done <<< "$(echo "$ALL_DATES" | tail -n "$LATEST")"
    fi

    for DATE in "${DATES[@]:-}"; do
        [[ -z "$DATE" ]] && continue
        if ! grep -q "^${DATE}," "$MANIFEST"; then
            echo "ERROR: ${DATE} is not in the manifest. Run --list to see available dates."
            exit 1
        fi
        SELECTED+=("$DATE")
    done

    # --all, --latest and repeated --date can overlap; keep first occurrence.
    SELECTED=($(printf '%s\n' "${SELECTED[@]}" | awk '!seen[$0]++'))
fi

if [[ ${#SELECTED[@]} -eq 0 ]]; then
    exit 0
fi

################################################################################
# Confirm size
################################################################################

TOTAL_BYTES=0
TOTAL_EPISODES=0

echo "Selected ${#SELECTED[@]} daily dataset(s):"
echo

for DATE in "${SELECTED[@]}"; do
    ROW=$(grep -m1 "^${DATE}," "$MANIFEST")
    BYTES=$(echo "$ROW" | cut -d, -f5)
    EPISODES=$(echo "$ROW" | cut -d, -f4)
    TOTAL_BYTES=$((TOTAL_BYTES + BYTES))
    TOTAL_EPISODES=$((TOTAL_EPISODES + EPISODES))
    printf "  %-12s %8d episodes  %s\n" "$DATE" "$EPISODES" "$(human_bytes "$BYTES")"
done

echo
echo "Total: ${TOTAL_EPISODES} episodes, $(human_bytes "$TOTAL_BYTES") (compressed download may be smaller)"
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run — nothing downloaded."
    exit 0
fi

if [[ "$ASSUME_YES" -eq 0 ]]; then
    read -r -p "Proceed? [y/N] " REPLY
    case "$REPLY" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
    echo
fi

################################################################################
# Download
################################################################################

EPISODES_ROOT="${OUTPUT_ROOT}/episodes"

CURRENT=1
SKIPPED=0
DOWNLOADED=0
FAILED=0

# Counters use `X=$((X + 1))`, never `((X++))`: post-increment evaluates to the
# value *before* the increment, so an arithmetic expansion returning 0 is exit
# status 1 and `set -e` would abort the script the first time a counter goes
# 0 -> 1. See the same note in download_replay.sh.
for DATE in "${SELECTED[@]}"; do

    SLUG=$(grep -m1 "^${DATE}," "$MANIFEST" | cut -d, -f2)
    [[ -z "$SLUG" ]] && SLUG="${DAILY_PREFIX}${DATE}"

    # The manifest stores the bare slug; the API wants owner/slug.
    if [[ "$SLUG" != */* ]]; then
        SLUG="kaggle/${SLUG}"
    fi

    DEST="${EPISODES_ROOT}/${DATE}"

    echo "[$CURRENT/${#SELECTED[@]}] ${DATE} (${SLUG})"

    # A completed unzip leaves files behind; a killed one can leave an empty
    # directory. Only a non-empty destination counts as already downloaded.
    if [[ -d "$DEST" ]] && [[ -n "$(ls -A "$DEST" 2>/dev/null)" ]]; then
        echo "  Already downloaded — skipping."
        SKIPPED=$((SKIPPED + 1))
    else
        mkdir -p "$DEST"
        if kaggle datasets download -d "$SLUG" -p "$DEST" --unzip; then
            echo "  Downloaded."
            DOWNLOADED=$((DOWNLOADED + 1))
        else
            echo "  Failed."
            FAILED=$((FAILED + 1))
            # Leave nothing behind that a later run would mistake for success.
            rmdir "$DEST" 2>/dev/null || true
        fi
    fi

    CURRENT=$((CURRENT + 1))

done

################################################################################
# Done
################################################################################

echo
echo "========================================"
echo "Finished!"
echo "Saved to   : ${EPISODES_ROOT}"
echo "Skipped    : ${SKIPPED} (already present)"
echo "Downloaded : ${DOWNLOADED}"
echo "Failed     : ${FAILED}"
echo "========================================"
