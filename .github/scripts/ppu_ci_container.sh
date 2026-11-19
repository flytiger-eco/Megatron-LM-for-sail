#!/usr/bin/env bash
set -euo pipefail
control_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -I "$control_dir/ppu_ci_container.py" "$@"
