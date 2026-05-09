"""这个脚本用于将 artifacts 中的特征表拼接成模型训练使用的宽表样本。"""

from __future__ import annotations

import argparse
from pathlib import Path

from recsys.data.sample_builder import build_and_save_model_inputs
from recsys.utils.io import load_dataframe, load_experiment_config
from recsys.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="构建模型输入宽表")
    parser.add_argument("--config", type=str, required=True, help="实验配置文件路径")
    return parser.parse_args()


def main() -> None:
    """执行模型输入宽表构建流程。"""

    logger = get_logger("scripts.build_model_inputs")
    args = parse_args()
    logger.info("开始执行模型输入宽表构建脚本。")
    logger.info("使用配置文件: %s", args.config)

    config = load_experiment_config(args.config)
    processed_dir = Path(config["dataset"]["processed_dir"])
    interim_dir = Path(config["dataset"]["interim_dir"])
    artifacts_dir = Path(config["dataset"]["artifacts_dir"])
    model_config = config["model"]
    logger.info("处理后样本目录: %s", processed_dir)
    logger.info("中间交互目录: %s", interim_dir)
    logger.info("特征表目录: %s", artifacts_dir)
    interaction_path = interim_dir / "interactions.parquet"
    if not interaction_path.exists():
        interaction_path = interim_dir / "interactions.csv"
    if not interaction_path.exists():
        raise FileNotFoundError(f"未找到可用交互表: {interim_dir / 'interactions.parquet'} 或 {interim_dir / 'interactions.csv'}")

    # 这里提前读取一次只是为了日志里能看到序列构建依赖的正样本规模。
    interactions = load_dataframe(interaction_path)
    logger.info("历史序列来源交互表: %s，样本数: %s", interaction_path, len(interactions))
    logger.info("序列最大长度: %s", model_config.get("sequence_max_len", 20))

    output_paths = build_and_save_model_inputs(
        processed_dir=processed_dir,
        artifacts_dir=artifacts_dir,
        interactions_path=interaction_path,
        max_sequence_length=int(model_config.get("sequence_max_len", 20)),
    )
    logger.info("模型输入宽表构建完成，输出路径: %s", output_paths)


if __name__ == "__main__":
    main()
