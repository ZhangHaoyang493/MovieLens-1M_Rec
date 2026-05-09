"""这个文件测试随机切分、leave-one-out、leave-two-out 和二分类样本构造逻辑。"""

from __future__ import annotations

import pandas as pd

from recsys.data.preprocess import build_training_interactions
from recsys.data.splitter import build_binary_samples, leave_one_out_split, leave_two_out_split, random_split


def test_random_split_keeps_total_size(interaction_frame: pd.DataFrame) -> None:
    """随机切分后总样本数应保持不变。"""

    interactions = build_training_interactions(interaction_frame)
    train_df, test_df = random_split(interactions, 0.8, 0.2, seed=2026)
    assert len(train_df) + len(test_df) == len(interactions)


def test_leave_one_out_uses_last_interaction_as_test(interaction_frame: pd.DataFrame) -> None:
    """leave-one-out 应将每个用户最后一次交互放入测试集。"""

    interactions = build_training_interactions(interaction_frame)
    interactions["user_id"] = [0, 0, 0, 1, 1, 2]
    interactions["item_id"] = [10, 11, 12, 20, 21, 30]
    train_df, test_df = leave_one_out_split(interactions)

    # 用户 0 的最后一次交互 item_id=12，用户 1 的最后一次交互 item_id=21。
    assert 12 in set(test_df["item_id"].tolist())
    assert 21 in set(test_df["item_id"].tolist())
    assert 12 not in set(train_df[train_df["user_id"] == 0]["item_id"].tolist())
    assert 21 not in set(train_df[train_df["user_id"] == 1]["item_id"].tolist())


def test_leave_two_out_uses_last_two_interactions_as_test(interaction_frame: pd.DataFrame) -> None:
    """leave-two-out 应将每个用户最后两次交互尽量放入测试集。"""

    interactions = build_training_interactions(interaction_frame)
    interactions["user_id"] = [0, 0, 0, 1, 1, 2]
    interactions["item_id"] = [10, 11, 12, 20, 21, 30]
    train_df, test_df = leave_two_out_split(interactions)

    # 用户 0 有三条交互，因此最后两条应进入测试集。
    assert {11, 12}.issubset(set(test_df[test_df["user_id"] == 0]["item_id"].tolist()))
    assert 10 in set(train_df[train_df["user_id"] == 0]["item_id"].tolist())

    # 用户 1 只有两条交互，保守规则下保留一条训练、一条测试。
    assert 21 in set(test_df[test_df["user_id"] == 1]["item_id"].tolist())
    assert 20 in set(train_df[train_df["user_id"] == 1]["item_id"].tolist())


def test_build_binary_samples_adds_negative_labels(interaction_frame: pd.DataFrame) -> None:
    """二分类样本构造后应包含负样本标签。"""

    interactions = build_training_interactions(interaction_frame)
    interactions["user_id"] = [0, 0, 0, 1, 1, 2]
    interactions["item_id"] = [0, 1, 2, 0, 3, 4]
    sampled = build_binary_samples(
        positive_df=interactions.iloc[:2].copy(),
        full_interactions=interactions,
        num_items=6,
        num_neg=2,
        seed=2026,
    )
    assert 0 in set(sampled["label"].tolist())
    assert 1 in set(sampled["label"].tolist())
