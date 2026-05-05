#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT_DIR"
python -m optimal_samples_system solve --m 45 --n 9 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9 \
  --seed 42 --restarts 5 --save
