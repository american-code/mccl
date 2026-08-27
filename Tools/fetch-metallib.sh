#!/usr/bin/env bash
#
# Install MLX's prebuilt Metal shader library next to a binary that links MCCLMLX,
# so the GPU backend works under plain SwiftPM — on a CI box, on a lab node, or on
# any machine without a working Xcode.
#
# Adapted from Tools/fetch-metallib.sh in SwiftSci (github.com/…/SwiftSci), which
# solved this first for its Frame/Neural targets. The mechanism and the pip-wheel
# trick are theirs; the version auto-detection and the multi-destination handling
# are the additions mccl needs.
#
# Why this exists
# ---------------
# mlx-swift compiles the MLX C++ core, but it does NOT build `mlx.metallib`: that
# shader library is produced by Xcode's build system, and `swift build` never runs
# it. A binary without the metallib links and starts, then throws
# `[metal::Device] Not found: mlx.metallib` the first time it touches the GPU.
#
# The prebuilt, version-matched metallib ships inside the `mlx-metal` pip wheel, so
# we take it from there. MLX's loader looks in the directory of the running binary
# first, which is why DEST_DIR is a build directory and not a system path.
#
# The version MUST match the MLX core bundled by mlx-swift. That is not the
# mlx-swift package version: mlx-swift 0.31.4 bundles MLX core 0.31.1. It is read
# from <checkout>/Source/Cmlx/mlx/mlx/version.h, which this script does for you.
#
# Usage
# -----
#   Tools/fetch-metallib.sh                       # auto-detect version, install into
#                                                 # .build/debug and .build/release
#   Tools/fetch-metallib.sh 0.31.1 .build/release # explicit version and destination
#
# For `swift test`, the xctest bundle runs out of .build/<config>/, so the default
# destinations cover it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

detect_version() {
    local header
    for header in "$ROOT"/.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/version.h; do
        [ -f "$header" ] || continue
        # Anchored on `#define`: the MLX_VERSION_NUMERIC expression mentions all
        # three macros by name and would otherwise overwrite every field with a
        # macro name.
        awk '
            /^#define MLX_VERSION_MAJOR / { major = $3 }
            /^#define MLX_VERSION_MINOR / { minor = $3 }
            /^#define MLX_VERSION_PATCH / { patch = $3 }
            END { if (major != "") print major "." minor "." patch }
        ' "$header"
        return
    done
}

VERSION="${1:-}"
if [ -z "$VERSION" ]; then
    VERSION="$(detect_version || true)"
fi
if [ -z "$VERSION" ]; then
    echo "could not read MLX core version from .build/checkouts/mlx-swift; run" >&2
    echo "  swift package resolve" >&2
    echo "first, or pass the version explicitly: Tools/fetch-metallib.sh 0.31.1" >&2
    exit 1
fi

if [ "$#" -ge 2 ]; then
    DESTS=("${@:2}")
else
    # Two kinds of destination, because MLX resolves the library relative to the
    # binary that *contains* the MLX code (dladdr on one of its own symbols), and
    # with SwiftPM's static linking that is whatever binary is running:
    #
    #   .build/<config>/                              — mccltrain, mcclbench
    #   .build/<config>/*.xctest/Contents/MacOS/      — `swift test`
    #
    # Both are covered so that a single invocation makes `swift test` and the
    # executables work. Configurations that have not been built yet are skipped.
    DESTS=()
    for CONFIG in debug release; do
        [ -d "$ROOT/.build/$CONFIG" ] || continue
        # `.build/debug` is a symlink into `.build/<triple>/debug`, and `find`
        # does not follow symlinked start points; resolve it first.
        REAL="$(cd "$ROOT/.build/$CONFIG" && pwd -P)"
        DESTS+=("$REAL")
        while IFS= read -r BUNDLE; do
            DESTS+=("$BUNDLE")
        done < <(find "$REAL" -maxdepth 3 -type d -path "*.xctest/Contents/MacOS" 2>/dev/null)
    done
    if [ "${#DESTS[@]}" -eq 0 ]; then
        echo "nothing built yet under .build — run 'swift build' or 'swift build --build-tests' first" >&2
        exit 1
    fi
fi

command -v python3 >/dev/null || { echo "python3 required (to read PyPI's JSON index)" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# The wheel is fetched with `curl` off PyPI's JSON index rather than with
# `pip download`, because `pip download` resolves wheel *tags*, and a system pip
# old enough not to understand a current tag will report the version as simply
# not existing. The lab nodes ship pip 21.2.4 on Python 3.9, which sees
# mlx-metal only up to 0.29.3 and fails on anything newer with a misleading
# "no matching distribution" — while curl fetches the same file happily.
#
# The metallib inside is a compiled Apple GPU shader library, so any arm64 wheel
# for the version carries the same one; the Python tag is irrelevant to it.
echo "Downloading mlx-metal==${VERSION} ..."
WHEEL_URL="$(
    curl -fsSL "https://pypi.org/pypi/mlx-metal/${VERSION}/json" 2>/dev/null | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for entry in data.get("urls", []):
    name = entry.get("filename", "")
    if name.endswith(".whl") and "arm64" in name:
        print(entry["url"])
        break
'
)"

if [ -n "$WHEEL_URL" ] && curl -fsSL -o "$TMP/mlx_metal.whl" "$WHEEL_URL"; then
    WHL="$TMP/mlx_metal.whl"
else
    echo "PyPI JSON fetch failed, falling back to pip ..." >&2
    python3 -m pip download "mlx-metal==${VERSION}" --no-deps -d "$TMP" >/dev/null
    WHL="$(ls "$TMP"/mlx_metal-*.whl)"
fi

unzip -o -j "$WHL" "mlx/lib/mlx.metallib" -d "$TMP" >/dev/null

for DEST in "${DESTS[@]}"; do
    mkdir -p "$DEST"
    cp "$TMP/mlx.metallib" "$DEST/mlx.metallib"
    echo "Installed mlx.metallib (${VERSION}) -> ${DEST}/mlx.metallib"
done
