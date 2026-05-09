"""这个文件负责将交互数据切分为训练、测试集合，并补充二分类负样本。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recsys.utils.io import ensure_dir, save_dataframe


def random_split(
    interactions: pd.DataFrame,
    train_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按照随机方式切分交互数据。"""

    # 切分比例必须能严格组成完整数据集，否则切分结果会不一致。
    if round(train_ratio + test_ratio, 6) != 1.0:
        raise ValueError("训练、测试比例之和必须为 1.0")

    # 先随机打乱数据，再按比例顺序切分，避免原始顺序带来偏差。
    shuffled = interactions.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    total = len(shuffled)

    # 根据总样本数计算训练集边界位置。
    train_end = int(total * train_ratio)

    # 按边界切出两份数据，并复制为独立 DataFrame。
    train_df = shuffled.iloc[:train_end].copy()
    test_df = shuffled.iloc[train_end:].copy()
    return train_df, test_df


def leave_one_out_split(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按用户时间顺序切分，保留每个用户最后一次交互作为测试集。"""

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    # 逐个用户处理，保证测试集中的样本是该用户时间上最后一次交互。
    for _, group in interactions.groupby("user_id", sort=False):
        ordered = group.sort_values("timestamp").reset_index(drop=True)

        # 如果用户只有一条交互，只能放进训练集，否则测试集会失去训练上下文。
        if len(ordered) <= 1:
            train_parts.append(ordered.copy())
            continue

        train_parts.append(ordered.iloc[:-1].copy())
        test_parts.append(ordered.iloc[-1:].copy())

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else interactions.iloc[0:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else interactions.iloc[0:0].copy()
    return train_df, test_df


def leave_two_out_split(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按用户时间顺序切分，优先保留每个用户最后两次交互作为测试集。"""

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    # 逐个用户处理，尽量把最后两次交互放进测试集，同时避免训练集被掏空。
    for _, group in interactions.groupby("user_id", sort=False):
        ordered = group.sort_values("timestamp").reset_index(drop=True)

        # 用户只有一条交互时，只能放进训练集。
        if len(ordered) <= 1:
            train_parts.append(ordered.copy())
            continue

        # 用户只有两条交互时，保守起见保留第一条做训练，最后一条做测试。
        if len(ordered) == 2:
            train_parts.append(ordered.iloc[:1].copy())
            test_parts.append(ordered.iloc[-1:].copy())
            continue

        # 正常情况下保留最后两次交互作为测试集，其余交互作为训练集。
        train_parts.append(ordered.iloc[:-2].copy())
        test_parts.append(ordered.iloc[-2:].copy())

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else interactions.iloc[0:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else interactions.iloc[0:0].copy()
    return train_df, test_df


def split_interactions(
    interactions: pd.DataFrame,
    split_config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """根据配置选择切分策略。"""

    # 这里统一通过配置决定切分方式，方便后续扩展其他策略。
    split_type = str(split_config["split_type"])
    if split_type == "random":
        return random_split(
            interactions=interactions,
            train_ratio=float(split_config["train_ratio"]),
            test_ratio=float(split_config["test_ratio"]),
            seed=int(split_config["seed"]),
        )
    if split_type == "leave_one_out":
        return leave_one_out_split(interactions=interactions)
    if split_type == "leave_two_out":
        return leave_two_out_split(interactions=interactions)
    raise ValueError(f"不支持的数据切分方式: {split_type}")


def build_binary_samples(
    positive_df: pd.DataFrame,
    full_interactions: pd.DataFrame,
    num_items: int,
    num_neg: int,
    seed: int,
) -> pd.DataFrame:
    """为正样本表补充随机负样本，生成二分类训练数据。"""

    # 如果不需要补负样本，直接返回正样本表副本。
    if num_neg <= 0:
        return positive_df.copy()

    # 先为每个用户建立已交互物品集合，后面采负样本时要避开这些物品。
    user_histories = (
        full_interactions.groupby("user_id")["item_id"].apply(list).map(set).to_dict()
    )

    # 使用独立随机数生成器，保证采样结果可复现。
    rng = np.random.default_rng(seed)
    negative_rows: list[dict[str, int | float]] = []

    # 对每一条正样本都补固定数量的负样本，形成二分类训练格式。
    for _, row in positive_df.iterrows():
        user_id = int(row["user_id"])
        seen_items = user_histories.get(user_id, set())
        sampled = 0

        # 只从该用户未交互的物品里采样负样本。
        while sampled < num_neg:
            candidate = int(rng.integers(low=0, high=num_items))
            if candidate in seen_items:
                continue

            # 负样本沿用用户信息和时间戳，只把物品改成未交互物品，标签置为 0。
            negative_rows.append(
                {
                    "raw_user_id": int(row["raw_user_id"]),
                    "raw_item_id": -1,
                    "rating": 0.0,
                    "timestamp": int(row["timestamp"]),
                    "label": 0,
                    "user_id": user_id,
                    "item_id": candidate,
                }
            )
            sampled += 1

    # 把正负样本合并后再打乱，避免训练时正负样本严格成块出现。
    negative_df = pd.DataFrame(negative_rows)
    combined = pd.concat([positive_df, negative_df], ignore_index=True)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def save_splits(
    processed_dir: str | Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Path]:
    """保存切分结果。"""

    # 先确保 processed 目录存在。
    base_dir = ensure_dir(processed_dir)
    train_path = base_dir / "train.csv"
    test_path = base_dir / "test.csv"

    # 两份数据分别落盘，后续训练和测试会直接从这些文件读取。
    train_path = save_dataframe(train_path, train_df)
    test_path = save_dataframe(test_path, test_df)
    return {"train_path": train_path, "test_path": test_path}
