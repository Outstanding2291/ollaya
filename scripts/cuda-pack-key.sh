#!/bin/sh
# Print the key of a CUDA pack: the sha256 of everything that decides what is in it.
#
#   scripts/cuda-pack-key.sh <platform> cuda|cuda12
#
# A CUDA pack (lib/ollaya/cuda_v13 or cuda_v12) holds third-party libraries only: Microsoft's ONNX
# Runtime GPU build, NVIDIA's CUDA and cuDNN libraries from their wheels, and llama.cpp's CUDA
# backend. Their versions and checksums are pinned in scripts/package.sh, scripts/llama-cpp.sh and
# the pack's requirements file, so the key hashes those three files with the platform and the
# pack. The release workflow uploads the key with every pack and reuses the previous release's
# archives when the key is unchanged, instead of building and compressing about 3 GB again. Any
# change to one of the three files, a version bump or not, builds the packs again.
set -eu

[ $# -eq 2 ] || { sed -n '2,/^$/s/^# \{0,1\}//p' "$0" >&2; exit 2; }
platform=$1 pack=$2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
case $pack in
    cuda) requirements=packaging/cuda-requirements.txt ;;
    cuda12) requirements=packaging/cuda12-requirements.txt ;;
    *) echo "cuda-pack-key.sh: unknown pack: $pack (cuda or cuda12)" >&2; exit 2 ;;
esac

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
{
    printf 'ollaya-cuda-pack-key 1 %s %s\n' "$platform" "$pack"
    for f in scripts/package.sh scripts/llama-cpp.sh "$requirements"; do
        printf '%s %s\n' "$(sha256_of "$ROOT/$f")" "$f"
    done
} >"$tmp"
sha256_of "$tmp"
