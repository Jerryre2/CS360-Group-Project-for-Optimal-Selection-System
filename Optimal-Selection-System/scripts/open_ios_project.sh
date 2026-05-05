#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
open "$ROOT_DIR/OptimalSelectionMobile/OptimalSelectionMobile.xcodeproj"
