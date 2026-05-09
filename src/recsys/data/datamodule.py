"""这个文件使用 PyTorch Lightning 封装训练和测试阶段的数据加载逻辑。"""

from __future__ import annotations

from pathlib import Path

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader

from recsys.data.dataset import InteractionDataset
from recsys.utils.io import load_dataframe


class MovieLensDataModule(L.LightningDataModule):
    """为 Deep 模型提供训练和测试的 DataLoader。"""

    def __init__(
        self,
        processed_dir: str | Path,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        # processed_dir 指向预处理后样本所在目录，里面至少应包含 train.csv 和 test.csv。
        self.processed_dir = Path(processed_dir)
        # batch_size 和 num_workers 直接交给 DataLoader 使用。
        self.batch_size = batch_size
        self.num_workers = num_workers
        # setup 之前先置空，避免外部误用未初始化的数据集对象。
        self.train_dataset: InteractionDataset | None = None
        self.test_dataset: InteractionDataset | None = None
        # 额外特征列由 setup 时根据实际输入宽表自动推断。
        self.feature_columns: list[str] = []
        self.sequence_columns: list[str] = []
        # 用户数和物品数用于后续初始化 embedding 大小。
        self.num_users = 0
        self.num_items = 0

    def _resolve_split_path(self, preferred_name: str, fallback_name: str) -> Path:
        """优先返回模型输入宽表路径，不存在时回退到基础样本路径。"""

        preferred = self.processed_dir / preferred_name
        if preferred.exists():
            return preferred
        return self.processed_dir / fallback_name

    def setup(self, stage: str | None = None) -> None:
        """加载切分后的数据并构造数据集。"""

        del stage
        # 如果已经构建了模型输入宽表，则优先使用它；否则退回基础 train/test 样本。
        train_path = self._resolve_split_path("train_model_input.csv", "train.csv")
        test_path = self._resolve_split_path("test_model_input.csv", "test.csv")
        train_df = load_dataframe(train_path)
        test_df = load_dataframe(test_path)
        # 为了拿到完整的用户和物品 ID 范围，这里把 train/test 合并后再统计最大 ID。
        merged = pd.concat([train_df, test_df], ignore_index=True)
        # 项目里的 user_id 和 item_id 都是从 0 开始的连续整数，所以总数等于 max + 1。
        self.num_users = int(merged["user_id"].max()) + 1
        self.num_items = int(merged["item_id"].max()) + 1
        # 如果样本里已经包含特征宽表，则把除基础三列外的其他数值列全部透传给 Dataset。
        ignored_columns = {"user_id", "item_id", "label"}
        self.sequence_columns = [column for column in train_df.columns if column.startswith("hist_") and column.endswith("_ids")]
        self.feature_columns = [
            column
            for column in train_df.columns
            if column not in ignored_columns
            and column not in self.sequence_columns
            and pd.api.types.is_numeric_dtype(train_df[column])
        ]
        self.train_dataset = InteractionDataset(
            train_df,
            feature_columns=self.feature_columns,
            sequence_columns=self.sequence_columns,
        )
        self.test_dataset = InteractionDataset(
            test_df,
            feature_columns=self.feature_columns,
            sequence_columns=self.sequence_columns,
        )

    def train_dataloader(self) -> DataLoader:
        """返回训练集 DataLoader。"""

        if self.train_dataset is None:
            raise RuntimeError("DataModule 尚未 setup。")
        # 训练阶段启用 shuffle，避免样本顺序固定带来的训练偏差。
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """返回测试集 DataLoader。"""

        if self.test_dataset is None:
            raise RuntimeError("DataModule 尚未 setup。")
        # 测试阶段关闭 shuffle，保证评估顺序稳定，便于复现和排查问题。
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
