"""这个文件提供统一日志初始化工具，便于脚本和训练过程复用。"""

from __future__ import annotations

import logging
from pathlib import Path

from recsys.utils.io import ensure_dir


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """创建或获取统一配置的日志对象。"""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        ensure_dir(log_path.parent)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
