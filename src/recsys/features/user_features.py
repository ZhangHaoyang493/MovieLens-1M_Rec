"""这个文件负责构建用户侧基础特征。"""

from __future__ import annotations

import pandas as pd


class UserFeatureBuilder:
    """构建用户人口属性特征。"""

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        # 当前用户侧只保留最基础的人口属性字段，作为后续扩展特征的起点。
        features = frame[["user_id", "gender", "age", "occupation"]].copy()

        # 性别字段先转成离散编码，便于后续直接送入模型或进一步做特征处理。
        features["gender_code"] = features["gender"].astype("category").cat.codes
        return features
