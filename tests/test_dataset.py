"""这个文件测试基础交互数据集的基本行为。"""

from __future__ import annotations

import pandas as pd

from recsys.data.dataset import InteractionDataset


def test_interaction_dataset_returns_required_fields() -> None:
    """交互数据集应返回训练所需字段。"""

    interactions = pd.DataFrame({"user_id": [0, 0], "item_id": [0, 1], "label": [1, 1]})
    dataset = InteractionDataset(interactions=interactions)
    sample = dataset[0]
    assert set(sample.keys()) == {"user_id", "item_id", "label"}


def test_interaction_dataset_returns_extra_numeric_features() -> None:
    """数据集应能按配置返回额外数值特征列。"""

    interactions = pd.DataFrame(
        {
            "user_id": [0, 0],
            "item_id": [0, 1],
            "label": [1, 0],
            "gender_code": [1, 1],
            "age": [25.0, 25.0],
        }
    )
    dataset = InteractionDataset(interactions=interactions, feature_columns=["gender_code", "age"])
    sample = dataset[0]
    assert set(sample.keys()) == {"user_id", "item_id", "label", "gender_code", "age"}


def test_interaction_dataset_parses_sequence_columns() -> None:
    """数据集应能把序列字符串列解析成定长 LongTensor。"""

    interactions = pd.DataFrame(
        {
            "user_id": [0],
            "item_id": [1],
            "label": [1],
            "hist_item_ids": ["0 2 3 4"],
            "hist_len": [3],
        }
    )
    dataset = InteractionDataset(interactions=interactions, feature_columns=["hist_len"], sequence_columns=["hist_item_ids"])
    sample = dataset[0]
    assert sample["hist_item_ids"].tolist() == [0, 2, 3, 4]
