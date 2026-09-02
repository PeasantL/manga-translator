#!/bin/bash
#
# Fetches the model weights the pipeline runs on into the Hugging Face cache.
#
# Nothing here has to be run: every stage downloads what it needs the first time
# it is used. Run it to get the download over with before a chapter is waiting
# on it, or to find out that a machine cannot reach huggingface.co before it is
# halfway through one.
#
# Usage:
#   ./fetch_models.sh          # about 5 GB, into HF_HOME
#   ./fetch_models.sh --force  # fetch again even if the cache has them
#
# Set HF_HOME to say where the cache is. Inside the container that is /cache,
# which is a volume, so a rebuild does not re-download any of this.

set -euo pipefail

FORCE=0

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        -h|--help)
            sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Try '$0 --help'." >&2
            exit 2
            ;;
    esac
done

FORCE=$FORCE python3 - <<'PY'
import os
import sys

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    sys.exit(
        "huggingface_hub is not installed. This runs in the environment the "
        "pipeline runs in:\n"
        "    pip install -r requirements.txt\n"
        "or, for the container:\n"
        "    docker compose run --rm mangatranslate ./fetch_models.sh"
    )

force = os.environ.get("FORCE") == "1"

# Detection and OCR are whole model repositories, since transformers wants the
# config and the tokenizer beside the weights; the other two are one file each.
# The patterns keep the onnx copies of the detector, and the sample images in
# the OCR repository, out of a cache that has no use for either.
repositories = [
    ("ogkalu/comic-text-and-bubble-detector", ["*.json", "*.safetensors"]),
    ("jzhang533/PaddleOCR-VL-For-Manga", ["*.json", "*.jinja", "*.py", "*.model", "*.safetensors"]),
]

files = [
    ("ogkalu/comic-text-segmenter-yolov8m", "comic-text-segmenter.pt"),
    ("TareHimself/AnimeMangaInpainting-torchscript", "anime_manga_lama.pt"),
]

for repo, patterns in repositories:
    print(f"Fetching {repo} ...", flush=True)
    snapshot_download(repo, allow_patterns=patterns, force_download=force)

for repo, name in files:
    print(f"Fetching {name} from {repo} ...", flush=True)
    hf_hub_download(repo, name, force_download=force)

print()
print("Weights are in", os.environ.get("HF_HOME", "the default Hugging Face cache"))
PY
