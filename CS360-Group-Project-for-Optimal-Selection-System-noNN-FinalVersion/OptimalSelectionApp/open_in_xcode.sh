#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

open -a Xcode "$SCRIPT_DIR/OptimalSelectionApp.xcodeproj"
