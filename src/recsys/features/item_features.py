"""这个文件负责构建物品侧基础特征。"""

from __future__ import annotations

import re

import pandas as pd


class ItemFeatureBuilder:
    """构建电影类型和年份特征。"""

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        # 物品侧当前主要依赖电影标题和类型字段构造基础特征。
        features = frame[["item_id", "title", "genres"]].copy()

        # 从标题末尾的年份信息中提取上映年份。
        features["release_year"] = features["title"].map(self._extract_year)

        # 统计每部电影包含多少个类型，作为一个简单的内容复杂度特征。
        features["genre_count"] = features["genres"].fillna("").map(
            lambda value: len([genre for genre in value.split("|") if genre])
        )
        return features

    @staticmethod
    def _extract_year(title: str) -> int | None:
        """从标题中提取年份。"""

        # MovieLens 标题通常以 "(1995)" 这种形式结尾，这里直接按该格式提取。
        matched = re.search(r"\((\d{4})\)$", title)
        return int(matched.group(1)) if matched else None
