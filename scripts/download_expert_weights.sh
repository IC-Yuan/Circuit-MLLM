#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HAWP_DIR="${PROJECT_ROOT}/hawp/checkpoints"
DEEPLSD_DIR="${PROJECT_ROOT}/DeepLSD/weights"
mkdir -p "${HAWP_DIR}" "${DEEPLSD_DIR}"

curl -fL \
  https://github.com/cherubicXN/hawp-torchhub/releases/download/HAWPv3/hawpv3-imagenet-03a84.pth \
  -o "${HAWP_DIR}/hawpv3-imagenet-03a84.pth"

curl -fL \
  https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar \
  -o "${DEEPLSD_DIR}/deeplsd_md.tar"

echo "Expert weights downloaded:"
echo "  ${HAWP_DIR}/hawpv3-imagenet-03a84.pth"
echo "  ${DEEPLSD_DIR}/deeplsd_md.tar"
