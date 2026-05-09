"""这个文件提供基础 Deep 模型训练所需的 Dataset 定义。"""

from __future__ import annotations

import pandas as pd
import torch
from pandas.api.types import is_float_dtype, is_integer_dtype
from torch.utils.data import Dataset


class InteractionDataset(Dataset[dict[str, torch.Tensor]]):
    """将交互表转换为 Deep 模型训练或验证使用的样本。"""

    def __init__(
        self,
        interactions: pd.DataFrame,
        feature_columns: list[str] | None = None,
        sequence_columns: list[str] | None = None,
    ) -> None:
        self.interactions = interactions.reset_index(drop=True)
        self.feature_columns = list(feature_columns or [])
        self.sequence_columns = list(sequence_columns or [])
        self.feature_dtypes = self._infer_feature_dtypes()

    def __len__(self) -> int:
        return len(self.interactions)

    def _infer_feature_dtypes(self) -> dict[str, torch.dtype]:
        """根据列类型推断额外特征应使用的张量类型。"""

        feature_dtypes: dict[str, torch.dtype] = {}
        for column in self.feature_columns:
            if is_integer_dtype(self.interactions[column]):
                feature_dtypes[column] = torch.long
                continue
            if is_float_dtype(self.interactions[column]):
                feature_dtypes[column] = torch.float32
                continue
            raise TypeError(f"不支持的特征列类型: {column} -> {self.interactions[column].dtype}")
        return feature_dtypes

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.interactions.iloc[index]
        sample = {
            "user_id": torch.tensor(int(row["user_id"]), dtype=torch.long),
            "item_id": torch.tensor(int(row["item_id"]), dtype=torch.long),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
        }
        for column in self.feature_columns:
            dtype = self.feature_dtypes[column]
            if dtype == torch.long:
                sample[column] = torch.tensor(int(row[column]), dtype=dtype)
            else:
                sample[column] = torch.tensor(float(row[column]), dtype=dtype)
        for column in self.sequence_columns:
            values = [int(value) for value in str(row[column]).split() if value]
            sample[column] = torch.tensor(values, dtype=torch.long)
        return sample
