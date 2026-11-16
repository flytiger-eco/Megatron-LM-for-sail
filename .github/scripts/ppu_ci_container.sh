#!/usr/bin/env bash
set -euo pipefail
control_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:?Missing container operation}" in
    build|build-image|fingerprint|publish-image|restore-control|restore-runtime|offline-probe)
        # Bootstrap and historical compatibility operations run on the CPU host.
        exec python3 -I -B "$control_dir/ppu_ci_container.py" "$@"
        ;;
    *)
        # New task consumers never fall back to the training or host interpreter.
        exec /opt/ppu-ci-control-venv/bin/python -I -B "$control_dir/ppu_ci_container.py" "$@"
        ;;
esac
