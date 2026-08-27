#!/bin/bash
#
# Downloads the model weights the pipeline needs into models/.
#
# The weights are not in the repository (they are gitignored); they come from
# the upstream project's release assets.
#
# Usage:
#   ./scripts/fetch_models.sh          # the two required models (~258 MB)
#   ./scripts/fetch_models.sh --force  # re-download files that already exist
#
# Override the source with MODELS_BASE_URL, or the destination with MODELS_DIR.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
MODELS_RELEASE="${MODELS_RELEASE:-2024.01.31}"
MODELS_BASE_URL="${MODELS_BASE_URL:-https://github.com/TareHimself/manga-translator/releases/download/$MODELS_RELEASE}"

# Required by every run: bubble detection and text segmentation. The LaMa
# cleaner fetches its own weights on first use.
REQUIRED_MODELS=(detection.pt segmentation.pt)

FORCE=0

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        -h|--help)
            sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Try '$0 --help'." >&2
            exit 2
            ;;
    esac
done

models=("${REQUIRED_MODELS[@]}")

mkdir -p "$MODELS_DIR"

for model in "${models[@]}"; do
    dest="$MODELS_DIR/$model"

    if [ -f "$dest" ] && [ "$FORCE" -eq 0 ]; then
        echo "$model already present, skipping (use --force to re-download)"
        continue
    fi

    echo "Downloading $model ..."

    # Download to a temporary name so an interrupted transfer never looks like
    # a complete model file to the next run.
    tmp="$dest.part"
    if ! curl -fL --progress-bar -o "$tmp" "$MODELS_BASE_URL/$model"; then
        rm -f "$tmp"
        echo "Failed to download $model from $MODELS_BASE_URL/$model" >&2
        exit 1
    fi

    # A truncated transfer or an HTML error page would be far smaller than any
    # of these checkpoints, all of which run to tens of megabytes.
    size=$(wc -c < "$tmp")
    if [ "$size" -lt 1000000 ]; then
        rm -f "$tmp"
        echo "Downloaded $model is only $size bytes, which is not a model file." >&2
        exit 1
    fi

    mv "$tmp" "$dest"
done

echo
echo "Models are in $MODELS_DIR"
