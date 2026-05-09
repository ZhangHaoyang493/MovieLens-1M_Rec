"""这个脚本用于构建并保存用户、物品和交互特征。"""

from __future__ import annotations

import argparse
from pathlib import Path

from recsys.features.interaction_features import InteractionFeatureBuilder
from recsys.features.item_features import ItemFeatureBuilder
from recsys.features.user_features import UserFeatureBuilder
from recsys.utils.io import load_dataframe, load_experiment_config, save_dataframe
from recsys.utils.logger import get_logger


def _resolve_existing_path(candidates: list[Path]) -> Path:
    """从候选路径中找到第一个存在的文件。"""

    # 预处理阶段可能保存成 parquet，也可能因为依赖缺失回退成 csv，
    # 所以这里按候选顺序查找第一个真实存在的文件。
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到可用输入文件，候选路径: {candidates}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="构建特征")
    # 特征构建依赖实验配置里的数据目录定义，因此这里要求显式传配置文件。
    parser.add_argument("--config", type=str, required=True, help="实验配置文件路径")
    return parser.parse_args()


def main() -> None:
    """执行特征构建逻辑。"""

    logger = get_logger("scripts.build_features")
    args = parse_args()
    logger.info("开始执行特征构建脚本。")
    logger.info("使用配置文件: %s", args.config)

    config = load_experiment_config(args.config)

    # 中间数据目录里放预处理后的结构化表，artifacts 目录里放特征产物。
    interim_dir = Path(config["dataset"]["interim_dir"])
    artifacts_dir = Path(config["dataset"]["artifacts_dir"])
    logger.info("中间数据目录: %s", interim_dir)
    logger.info("特征输出目录: %s", artifacts_dir)

    # 分别定位用户表、物品表和交互表的真实文件路径。
    user_path = _resolve_existing_path([interim_dir / "users.parquet", interim_dir / "users.csv"])
    item_path = _resolve_existing_path([interim_dir / "items.parquet", interim_dir / "items.csv"])
    interaction_path = _resolve_existing_path([interim_dir / "interactions.parquet", interim_dir / "interactions.csv"])
    logger.info("已定位用户表: %s", user_path)
    logger.info("已定位物品表: %s", item_path)
    logger.info("已定位交互表: %s", interaction_path)

    users = load_dataframe(user_path)
    items = load_dataframe(item_path)
    interactions = load_dataframe(interaction_path)
    logger.info("输入数据加载完成，用户数: %s，物品数: %s，交互数: %s", len(users), len(items), len(interactions))

    # 用户特征主要来自人口属性字段，例如性别、年龄、职业。
    logger.info("开始构建用户特征。")
    user_features = UserFeatureBuilder().build(users)
    logger.info("用户特征构建完成，样本数: %s", len(user_features))

    # 物品特征主要来自电影标题和类型字段，例如年份、类型数量。
    logger.info("开始构建物品特征。")
    item_features = ItemFeatureBuilder().build(items)
    logger.info("物品特征构建完成，样本数: %s", len(item_features))

    # 交互特征主要是用户和物品层面的统计信息，例如交互频次。
    logger.info("开始构建交互统计特征。")
    interaction_features = InteractionFeatureBuilder().build(interactions)
    logger.info("交互特征构建完成，样本数: %s", len(interaction_features))

    # 最终统一保存到 artifacts 目录，供后续扩展模型或做数据检查使用。
    user_feature_path = save_dataframe(artifacts_dir / "user_features.csv", user_features)
    item_feature_path = save_dataframe(artifacts_dir / "item_features.csv", item_features)
    interaction_feature_path = save_dataframe(artifacts_dir / "interaction_features.csv", interaction_features)
    logger.info("用户特征已保存: %s", user_feature_path)
    logger.info("物品特征已保存: %s", item_feature_path)
    logger.info("交互特征已保存: %s", interaction_feature_path)
    logger.info("特征构建脚本执行结束。")


if __name__ == "__main__":
    main()
