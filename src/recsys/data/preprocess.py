"""这个文件负责将原始评分数据转换为基础 Deep 模型可用的处理后数据。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from recsys.data.parser import parse_movies, parse_ratings, parse_users
from recsys.utils.io import ensure_dir, save_dataframe, save_json


def build_training_interactions(
    ratings: pd.DataFrame,
    positive_strategy: str = "binary_all",
    positive_threshold: float | None = None,
) -> pd.DataFrame:
    """将显式评分转换为二分类训练交互。"""

    # 先复制一份评分表，避免直接修改外部传入的数据。
    interactions = ratings.copy()

    # 如果采用阈值策略，只保留高于阈值的评分记录作为正样本。
    if positive_strategy == "binary_threshold":
        if positive_threshold is None:
            raise ValueError("使用 binary_threshold 时必须提供 positive_threshold")
        interactions = interactions[interactions["rating"] >= positive_threshold].copy()
    elif positive_strategy != "binary_all":
        raise ValueError(f"不支持的正样本策略: {positive_strategy}")

    # 这里先把所有保留下来的交互统一标为正样本，负样本后续在切分阶段补。
    interactions["label"] = 1
    return interactions


def remap_ids(
    users: pd.DataFrame,
    movies: pd.DataFrame,
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    """将原始用户和物品 ID 映射为连续整数。"""

    # 只基于当前交互表中真实出现过的用户和物品构建映射。
    user_ids = sorted(interactions["raw_user_id"].unique().tolist())
    item_ids = sorted(interactions["raw_item_id"].unique().tolist())

    # 连续整数 ID 便于后续直接喂给 embedding 层。
    user_mapping = {str(raw_id): idx for idx, raw_id in enumerate(user_ids)}
    item_mapping = {str(raw_id): idx for idx, raw_id in enumerate(item_ids)}

    # 用户表和物品表中未出现在交互里的记录，这里直接裁掉。
    users = users[users["raw_user_id"].isin(user_ids)].copy()
    movies = movies[movies["raw_item_id"].isin(item_ids)].copy()
    interactions = interactions.copy()

    # 把原始 ID 列映射成模型训练实际使用的连续索引列。
    users["user_id"] = users["raw_user_id"].map(lambda value: user_mapping[str(value)])
    movies["item_id"] = movies["raw_item_id"].map(lambda value: item_mapping[str(value)])
    interactions["user_id"] = interactions["raw_user_id"].map(lambda value: user_mapping[str(value)])
    interactions["item_id"] = interactions["raw_item_id"].map(lambda value: item_mapping[str(value)])

    # 映射表单独保存，方便后面做结果回查或排查问题。
    mapping = {"user_id_map": user_mapping, "item_id_map": item_mapping}
    return users, movies, interactions, mapping


def filter_interactions(
    interactions: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pd.DataFrame:
    """按用户和物品交互数过滤交互数据。"""

    # 这里先复制一份，避免在原始评分表上做原地过滤。
    filtered = interactions.copy()

    # 如果配置要求最少用户交互数，就先筛掉交互太少的用户。
    if min_user_interactions > 1:
        user_counts = filtered["raw_user_id"].value_counts()
        keep_users = user_counts[user_counts >= min_user_interactions].index
        filtered = filtered[filtered["raw_user_id"].isin(keep_users)].copy()

    # 再按物品最小交互数筛一遍，去掉极少出现的物品。
    if min_item_interactions > 1:
        item_counts = filtered["raw_item_id"].value_counts()
        keep_items = item_counts[item_counts >= min_item_interactions].index
        filtered = filtered[filtered["raw_item_id"].isin(keep_items)].copy()

    return filtered


def preprocess_movielens(config: dict[str, Any]) -> dict[str, Path]:
    """执行 MovieLens-1M 的完整预处理流程。"""

    # 所有预处理参数都从 dataset 配置中读取，避免脚本里写死路径和规则。
    dataset_config = config["dataset"]
    raw_dir = Path(dataset_config["raw_dir"]) / "ml-1m"

    # 预处理阶段会同时产出中间表、处理后数据目录和辅助映射文件。
    interim_dir = ensure_dir(dataset_config["interim_dir"])
    processed_dir = ensure_dir(dataset_config["processed_dir"])
    artifacts_dir = ensure_dir(dataset_config["artifacts_dir"])

    # 先把三个原始文件解析成结构化 DataFrame。
    users = parse_users(raw_dir / "users.dat")
    movies = parse_movies(raw_dir / "movies.dat")
    ratings = parse_ratings(raw_dir / "ratings.dat")

    # 过滤掉低频用户和低频物品，保证训练样本更稳定。
    ratings = filter_interactions(
        interactions=ratings,
        min_user_interactions=dataset_config.get("min_user_interactions", 1),
        min_item_interactions=dataset_config.get("min_item_interactions", 1),
    )

    # 将原始评分转换成训练阶段使用的正样本交互表。
    interactions = build_training_interactions(
        ratings=ratings,
        positive_strategy=dataset_config.get("positive_strategy", "binary_all"),
        positive_threshold=dataset_config.get("positive_threshold"),
    )

    # 对用户和物品做连续 ID 重映射，便于后续模型训练。
    users, movies, interactions, mapping = remap_ids(users, movies, interactions)

    # 统一排序后再落盘，便于后续复查和保证结果稳定。
    users = users.sort_values("user_id").reset_index(drop=True)
    movies = movies.sort_values("item_id").reset_index(drop=True)
    interactions = interactions.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # 中间表优先写成 parquet，不可用时会自动回退成 csv。
    users_path = interim_dir / "users.parquet"
    items_path = interim_dir / "items.parquet"
    interactions_path = interim_dir / "interactions.parquet"
    users_path = save_dataframe(users_path, users)
    items_path = save_dataframe(items_path, movies)
    interactions_path = save_dataframe(interactions_path, interactions)

    # ID 映射单独保存为 json，方便调试和结果反查。
    save_json(artifacts_dir / "id_mappings.json", mapping)

    return {
        "users_path": users_path,
        "items_path": items_path,
        "interactions_path": interactions_path,
        "mapping_path": artifacts_dir / "id_mappings.json",
        "processed_dir": processed_dir,
    }
