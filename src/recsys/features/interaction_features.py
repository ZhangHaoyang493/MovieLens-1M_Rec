"""这个文件负责构建交互级基础统计特征。"""

from __future__ import annotations

import pandas as pd


class InteractionFeatureBuilder:
    """构建用户和物品的交互频次特征。"""

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        # 先保留交互主键和时间戳，后续统计特征都围绕这些字段展开。
        features = frame[["user_id", "item_id", "timestamp"]].copy()

        # 分别统计每个用户、每个物品在交互表中出现的次数。
        user_counts = frame.groupby("user_id").size().rename("user_interaction_count")
        item_counts = frame.groupby("item_id").size().rename("item_interaction_count")

        # 将统计结果回填到交互表中，形成逐条交互可直接使用的统计特征。
        features = features.merge(user_counts, on="user_id", how="left")
        features = features.merge(item_counts, on="item_id", how="left")
        return features
