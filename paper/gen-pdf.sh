#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<EOF
Usage:
  $0 [paper.md] [output.pdf] [bibliography.bib]

Examples:
  $0
  $0 paper/paper.md
  $0 paper/paper.md paper/paper.pdf
  $0 paper/paper.md paper/paper.pdf paper/references.bib
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

command -v pandoc >/dev/null 2>&1 || {
    echo "Error: pandoc is not installed." >&2
    exit 1
}

command -v xelatex >/dev/null 2>&1 || {
    echo "Error: xelatex is not installed." >&2
    exit 1
}

# Convert a path to an absolute path.
absolute_path() {
    local path="$1"
    local directory
    local filename

    directory="$(dirname "$path")"
    filename="$(basename "$path")"

    directory="$(cd "$directory" && pwd)"
    printf '%s/%s\n' "$directory" "$filename"
}

PAPER_PATH="${1:-paper/paper.md}"

if [[ ! -f "$PAPER_PATH" ]]; then
    echo "Error: input file does not exist: $PAPER_PATH" >&2
    exit 1
fi

PAPER_PATH="$(absolute_path "$PAPER_PATH")"
PAPER_DIR="$(dirname "$PAPER_PATH")"
PAPER_FILE="$(basename "$PAPER_PATH")"

if [[ $# -ge 2 ]]; then
    OUTPUT_PATH="$2"
else
    OUTPUT_PATH="${PAPER_PATH%.*}.pdf"
fi

# Create the output directory if necessary, then make the path absolute.
mkdir -p "$(dirname "$OUTPUT_PATH")"
OUTPUT_PATH="$(absolute_path "$OUTPUT_PATH")"

# Optional bibliography supplied as the third argument.
BIBLIOGRAPHY_ARGS=()

if [[ $# -ge 3 ]]; then
    BIBLIOGRAPHY_PATH="$3"

    if [[ ! -f "$BIBLIOGRAPHY_PATH" ]]; then
        echo "Error: bibliography file does not exist: $BIBLIOGRAPHY_PATH" >&2
        exit 1
    fi

    BIBLIOGRAPHY_PATH="$(absolute_path "$BIBLIOGRAPHY_PATH")"
    BIBLIOGRAPHY_ARGS+=(--bibliography="$BIBLIOGRAPHY_PATH")
fi

# Temporary LaTeX header used to enable line numbering.
HEADER_FILE="$(mktemp "${TMPDIR:-/tmp}/plain-pdf-header.XXXXXX.tex")"

cleanup() {
    rm -f "$HEADER_FILE"
}

trap cleanup EXIT

echo "Building PDF:"
echo "  Input:  $PAPER_PATH"
echo "  Output: $OUTPUT_PATH"

# Run from the paper directory so relative image, CSL, and bibliography
# paths in the Markdown metadata continue to work.
cd "$PAPER_DIR"

pandoc "$PAPER_FILE" \
    --standalone \
    --from=markdown \
    --to=latex \
    --pdf-engine=xelatex \
    --citeproc \
    --include-in-header="$HEADER_FILE" \
    --metadata=reference-section-title:References \
    --variable=geometry:margin=1in \
    --resource-path="$PAPER_DIR" \
    "${BIBLIOGRAPHY_ARGS[@]}" \
    --output="$OUTPUT_PATH"

echo "PDF created successfully: $OUTPUT_PATH"