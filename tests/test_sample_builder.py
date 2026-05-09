"""这个文件测试模型输入宽表的特征拼接逻辑。"""

from __future__ import annotations

import pandas as pd

from recsys.data.sample_builder import build_model_input_frame


def test_build_model_input_frame_merges_numeric_features() -> None:
    """宽表样本应能拼接用户、物品和交互统计特征。"""

    samples = pd.DataFrame(
        {
            "user_id": [0, 0],
            "item_id": [1, 2],
            "label": [1, 0],
            "timestamp": [3, 4],
        }
    )
    user_features = pd.DataFrame(
        {
            "user_id": [0],
            "gender": ["M"],
            "gender_code": [1],
            "age": [25],
            "occupation": [3],
        }
    )
    item_features = pd.DataFrame(
        {
            "item_id": [1, 2],
            "title": ["a", "b"],
            "genres": ["Action", "Comedy"],
            "release_year": [1995, 1996],
            "genre_count": [1, 1],
        }
    )
    interaction_features = pd.DataFrame(
        {
            "user_id": [0, 0],
            "item_id": [1, 2],
            "timestamp": [1, 2],
            "user_interaction_count": [2, 2],
            "item_interaction_count": [10, 20],
        }
    )
    full_positive_interactions = pd.DataFrame(
        {
            "user_id": [0, 0, 0],
            "item_id": [0, 1, 2],
            "timestamp": [1, 2, 3],
        }
    )

    merged = build_model_input_frame(
        samples,
        user_features,
        item_features,
        interaction_features,
        full_positive_interactions=full_positive_interactions,
        max_sequence_length=3,
    )

    assert "gender" not in merged.columns
    assert "title" not in merged.columns
    assert "genres" not in merged.columns
    assert merged["gender_code"].tolist() == [1, 1]
    assert merged["age"].tolist() == [25, 25]
    assert merged["release_year"].tolist() == [1995, 1996]
    assert merged["user_interaction_count"].tolist() == [2, 2]
    assert merged["item_interaction_count"].tolist() == [10, 20]
    assert merged["hist_item_ids"].tolist() == ["0 1 2", "1 2 3"]
    assert merged["hist_len"].tolist() == [2, 3]
