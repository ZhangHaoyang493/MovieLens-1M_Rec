"""这个文件测试训练交互构建和 ID 重映射逻辑。"""

from __future__ import annotations

import pandas as pd

from recsys.data.preprocess import build_training_interactions, remap_ids


def test_build_training_interactions_binary_threshold(interaction_frame: pd.DataFrame) -> None:
    """阈值模式下只保留高分交互。"""

    interactions = build_training_interactions(interaction_frame, positive_strategy="binary_threshold", positive_threshold=4)
    assert interactions["rating"].min() >= 4
    assert set(interactions["label"].unique()) == {1}


def test_remap_ids_returns_dense_ids(interaction_frame: pd.DataFrame) -> None:
    """重映射后的用户和物品 ID 应该从 0 开始连续。"""

    users = pd.DataFrame(
        {
            "raw_user_id": [1, 2, 3],
            "gender": ["M", "F", "M"],
            "age": [18, 25, 35],
            "occupation": [1, 2, 3],
            "zip_code": ["1", "2", "3"],
        }
    )
    items = pd.DataFrame({"raw_item_id": [10, 11, 12, 13, 14], "title": ["a", "b", "c", "d", "e"], "genres": ["A"] * 5})
    interactions = build_training_interactions(interaction_frame)
    remapped_users, remapped_items, remapped_interactions, _ = remap_ids(users, items, interactions)
    assert remapped_users["user_id"].tolist() == [0, 1, 2]
    assert remapped_items["item_id"].tolist() == [0, 1, 2, 3, 4]
    assert remapped_interactions["user_id"].min() == 0
