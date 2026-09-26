#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
INPUT_VIDEO="$REPO_ROOT/data/video-source/nimbus.mp4"
REFERENCE_DIR="$REPO_ROOT/data/reference-images"
OUTPUT_VIDEO="$REPO_ROOT/output/nimbus_labelled.mp4"
OUTPUT_CSV="$REPO_ROOT/output/matches.csv"
CACHE_DIR="$REPO_ROOT/cache"
VIDEO_FILE_ID="1CM1IWUN59ZWml9MwgrvSHXz_9AirIiuU"
RUN_DIR=""

cleanup() {
  if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
    rm -rf -- "$RUN_DIR"
  fi
}
trap cleanup EXIT

CHARACTERS=(
  "Harry Potter"
  "Ron Weasley"
  "Hermione Granger"
  "Prof. McGonagall"
  "Prof. Severus Snape"
)

missing_references=()
for character in "${CHARACTERS[@]}"; do
  character_dir="$REFERENCE_DIR/$character"
  image_count=0
  if [[ -d "$character_dir" ]]; then
    image_count="$(find "$character_dir" -type f \( \
      -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
    \) -print | awk 'END { print NR + 0 }')"
  fi
  if (( image_count < 2 )); then
    missing_references+=("$character")
  fi
done

if (( ${#missing_references[@]} > 0 )); then
  echo "Each character needs at least two curated reference images in:" >&2
  echo "  $REFERENCE_DIR/<Character>/" >&2
  echo "Missing or incomplete: ${missing_references[*]}" >&2
  echo "Reference photos must be supplied by the owner; this script never downloads them." >&2
  exit 1
fi

for media_tool in ffmpeg ffprobe; do
  if ! command -v "$media_tool" >/dev/null 2>&1; then
    echo "$media_tool is required to preserve and verify the source audio." >&2
    echo "Install ffmpeg (which includes ffprobe) and rerun this script." >&2
    exit 1
  fi
done

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if ! PYTHON_BIN="$(command -v python3.11)"; then
    echo "Python 3.11 is required. Install it and rerun this script." >&2
    exit 1
  fi
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'; then
  echo "PYTHON_BIN must point to Python 3.11; got: $PYTHON_BIN" >&2
  exit 1
fi

cd "$REPO_ROOT"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Creating Python 3.11 environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "Installing pinned dependencies"
"$VENV_PYTHON" -m pip install --disable-pip-version-check -r "$REPO_ROOT/requirements.txt"

mkdir -p "$(dirname "$INPUT_VIDEO")" "$REFERENCE_DIR" "$CACHE_DIR" \
  "$(dirname "$OUTPUT_VIDEO")"

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Downloading Nimbus source video"
  "$VENV_PYTHON" -m gdown "$VIDEO_FILE_ID" -O "$INPUT_VIDEO"
fi

RUN_DIR="$(mktemp -d "$(dirname "$OUTPUT_VIDEO")/.full-run.XXXXXX")"
VIDEO_ONLY_OUTPUT="$RUN_DIR/video-only.mp4"
MUXED_OUTPUT="$RUN_DIR/nimbus_labelled.mp4"
STAGED_CSV="$RUN_DIR/matches.csv"

echo "Running the full CPU-only stride-1 pipeline"
"$VENV_PYTHON" label_video.py \
  --input "$INPUT_VIDEO" \
  --output "$VIDEO_ONLY_OUTPUT" \
  --ref-dir "$REFERENCE_DIR" \
  --stride 1 \
  --batch-size 8 \
  --threshold 0.30 \
  --pin-strategy all \
  --cache-dir "$CACHE_DIR" \
  --csv "$STAGED_CSV"

echo "Restoring the source AAC audio"
ffmpeg -hide_banner -loglevel error -y \
  -i "$VIDEO_ONLY_OUTPUT" \
  -i "$INPUT_VIDEO" \
  -map 0:v:0 -map 1:a:0 \
  -c:v copy -c:a copy -shortest \
  "$MUXED_OUTPUT"

if [[ "$(ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_type -of csv=p=0 "$MUXED_OUTPUT")" != "video" ]]; then
  echo "Audio restoration failed verification: no video stream." >&2
  exit 1
fi
if [[ "$(ffprobe -v error -select_streams a:0 \
  -show_entries stream=codec_type -of csv=p=0 "$MUXED_OUTPUT")" != "audio" ]]; then
  echo "Audio restoration failed verification: no audio stream." >&2
  exit 1
fi

mv "$MUXED_OUTPUT" "$OUTPUT_VIDEO"
mv "$STAGED_CSV" "$OUTPUT_CSV"

echo "Complete"
echo "Labelled video (source audio preserved): $OUTPUT_VIDEO"
echo "Detection evidence: $OUTPUT_CSV"
echo "Run report: $REPO_ROOT/docs/results/m6-temporal-smoothing-and-audio.md"
