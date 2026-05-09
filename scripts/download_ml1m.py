"""这个脚本用于下载并校验 MovieLens-1M 原始数据集。"""

from __future__ import annotations

import argparse
from pathlib import Path

from recsys.data.downloader import download_movielens_1m, validate_ml1m_files
from recsys.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="下载 MovieLens-1M 数据集")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="原始数据根目录")
    return parser.parse_args()


def main() -> None:
    """执行下载逻辑。"""

    logger = get_logger("scripts.download_ml1m")
    args = parse_args()
    logger.info("开始执行 MovieLens-1M 数据下载脚本。")
    logger.info("原始数据目录: %s", args.raw_dir)

    logger.info("准备下载或复用已存在的数据集。")
    extracted_dir = download_movielens_1m(Path(args.raw_dir))

    logger.info("下载阶段完成，开始校验原始文件完整性。")
    validate_ml1m_files(extracted_dir)
    logger.info("原始文件校验通过。")
    logger.info("数据集已准备完成，目录位置: %s", extracted_dir)


if __name__ == "__main__":
    main()
