"""这个脚本用于执行 MovieLens-1M 数据预处理与训练集、测试集切分。"""

from __future__ import annotations

import argparse

from recsys.data.preprocess import preprocess_movielens
from recsys.data.splitter import build_binary_samples, save_splits, split_interactions
from recsys.utils.io import load_dataframe, load_experiment_config
from recsys.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="预处理 MovieLens-1M 数据")
    # 这里要求显式传入实验配置，避免把数据处理规则写死在脚本里。
    parser.add_argument("--config", type=str, required=True, help="实验配置文件路径")
    return parser.parse_args()


def main() -> None:
    """执行预处理和切分逻辑。"""

    logger = get_logger("scripts.preprocess_ml1m")
    args = parse_args()
    logger.info("开始执行 MovieLens-1M 数据预处理脚本。")
    logger.info("使用配置文件: %s", args.config)

    # 先读取实验配置，后续的数据处理、切分和负样本数量都从这里取。
    config = load_experiment_config(args.config)
    logger.info("配置文件加载完成。")

    # 执行原始数据解析、过滤、ID 重映射，并返回中间产物路径。
    logger.info("开始解析原始数据并执行基础预处理。")
    preprocess_paths = preprocess_movielens(config)
    logger.info("基础预处理完成，中间产物: %s", preprocess_paths)

    # 读取已经完成预处理的交互表，后续切分和负样本构造都基于它进行。
    interactions = load_dataframe(preprocess_paths["interactions_path"])
    logger.info("交互表加载完成，样本数: %s", len(interactions))

    # 先按配置把正样本交互切分成训练集和测试集两部分。
    train_df, test_df = split_interactions(interactions, config["split"])
    logger.info(
        "正样本切分完成，训练集: %s，测试集: %s",
        len(train_df),
        len(test_df),
    )

    # 物品总数用于从未交互物品集合中随机采负样本。
    num_items = int(interactions["item_id"].max()) + 1
    logger.info("检测到物品总数: %s", num_items)

    # 负样本采样参数单独放在配置里，便于后续调整训练难度。
    negative_config = config.get("negative_sampling", {})
    logger.info("负样本配置: %s", negative_config)

    # 为训练集补充更多负样本，形成可用于二分类训练的数据。
    train_df = build_binary_samples(
        positive_df=train_df,
        full_interactions=interactions,
        num_items=num_items,
        num_neg=int(negative_config.get("train_num_neg", 4)),
        seed=int(config["split"]["seed"]),
    )
    logger.info("训练集负样本补充完成，当前样本数: %s", len(train_df))

    # 测试集同样补负样本，并使用不同随机种子避免和训练集重复。
    test_df = build_binary_samples(
        positive_df=test_df,
        full_interactions=interactions,
        num_items=num_items,
        num_neg=int(negative_config.get("test_num_neg", 1)),
        seed=int(config["split"]["seed"]) + 1,
    )
    logger.info("测试集负样本补充完成，当前样本数: %s", len(test_df))

    # 将最终得到的训练集和测试集文件保存到 processed 目录下。
    split_paths = save_splits(preprocess_paths["processed_dir"], train_df, test_df)
    logger.info("处理后数据保存完成，输出路径: %s", split_paths)
    logger.info("数据预处理脚本执行结束。")


if __name__ == "__main__":
    main()
