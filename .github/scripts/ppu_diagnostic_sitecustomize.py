"""Python stack sampling used only by standalone diagnostic subprocesses; never loaded into
the formal training environment.
"""

import faulthandler
import json
import os
import signal
import sys
from pathlib import Path

_stack_stream = None


def install() -> None:
    """Open a dedicated stack file for the current interpreter or fork worker."""
    global _stack_stream
    destination = os.environ.get("PPU_DIAGNOSTIC_STACK_DIR")
    if not destination:
        return
    if _stack_stream is not None:
        _stack_stream.close()
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    _stack_stream = (directory / f"python-{os.getpid()}.log").open("a", buffering=1)
    _stack_stream.write(
        json.dumps({"pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv}) + "\n"
    )
    faulthandler.enable(file=_stack_stream, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=_stack_stream, all_threads=True)
    # The control process sends the sampling signal; no extra watchdog thread starts
    # in the interpreter under test.
    (directory / f"python-{os.getpid()}.ready").touch()


if os.environ.get("PPU_DIAGNOSTIC_STACK_DIR"):
    install()
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=install)
