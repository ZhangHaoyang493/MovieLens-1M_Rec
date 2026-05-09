"""这个文件提供简单计时器，便于统计数据处理和训练阶段耗时。"""

from __future__ import annotations

import time
from contextlib import ContextDecorator


class Timer(ContextDecorator):
    """用于统计代码块执行时长。"""

    def __init__(self) -> None:
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.perf_counter() - self.start_time
