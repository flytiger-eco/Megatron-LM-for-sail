"""仅独立诊断子进程使用的 Python 栈采样，不载入正式训练环境。"""

import faulthandler
import json
import os
import signal
import sys
from pathlib import Path

_stack_stream = None


def install() -> None:
    """为当前解释器或 fork worker 打开独立栈文件。"""
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
    # 由控制进程发送采样信号，不在被测解释器启动额外 watchdog 线程。
    (directory / f"python-{os.getpid()}.ready").touch()


if os.environ.get("PPU_DIAGNOSTIC_STACK_DIR"):
    install()
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=install)
