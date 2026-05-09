"""这个文件测试基础 Deep 模型与 Lightning DataModule 的基本行为。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from recsys.data.datamodule import MovieLensDataModule
from recsys.models.base import BasePointwiseRecModel
from recsys.models.deep import DeepModel


def test_datamodule_setup_and_model_forward(tmp_path: Path) -> None:
    """DataModule 应能加载数据，模型应能完成一次前向计算。"""

    frame = pd.DataFrame(
        {
            "user_id": [0, 1, 0, 1],
            "item_id": [0, 1, 1, 0],
            "label": [1, 1, 0, 0],
        }
    )
    frame.iloc[:2].to_csv(tmp_path / "train.csv", index=False)
    frame.iloc[2:].to_csv(tmp_path / "test.csv", index=False)

    data_module = MovieLensDataModule(processed_dir=tmp_path, batch_size=2, num_workers=0)
    data_module.setup("fit")
    model = DeepModel(
        num_users=data_module.num_users,
        num_items=data_module.num_items,
        model_config={"embedding_dim": 8, "mlp_hidden_dims": [16, 8], "dropout": 0.0},
        trainer_config={"learning_rate": 0.001, "weight_decay": 0.0},
    )
    batch = next(iter(data_module.train_dataloader()))
    logits = model(batch["user_id"], batch["item_id"])
    assert logits.shape[0] == 2
    assert isinstance(model, BasePointwiseRecModel)


def test_datamodule_prefers_model_input_files(tmp_path: Path) -> None:
    """如果存在模型输入宽表，DataModule 应优先读取宽表并暴露额外特征列。"""

    base_frame = pd.DataFrame(
        {
            "user_id": [0, 1, 0, 1],
            "item_id": [0, 1, 1, 0],
            "label": [1, 1, 0, 0],
        }
    )
    wide_frame = base_frame.assign(
        gender_code=[1, 0, 1, 0],
        age=[25.0, 30.0, 25.0, 30.0],
        hist_item_ids=["0 1 2", "0 0 2", "1 2 3", "0 1 0"],
        hist_len=[2, 1, 3, 1],
    )
    base_frame.iloc[:2].to_csv(tmp_path / "train.csv", index=False)
    base_frame.iloc[2:].to_csv(tmp_path / "test.csv", index=False)
    wide_frame.iloc[:2].to_csv(tmp_path / "train_model_input.csv", index=False)
    wide_frame.iloc[2:].to_csv(tmp_path / "test_model_input.csv", index=False)

    data_module = MovieLensDataModule(processed_dir=tmp_path, batch_size=2, num_workers=0)
    data_module.setup("fit")
    batch = next(iter(data_module.train_dataloader()))

    assert "gender_code" in batch
    assert "age" in batch
    assert "hist_item_ids" in batch
    assert "hist_len" in batch


def test_deep_model_compute_logits_with_sequence_features() -> None:
    """DeepModel 应能消费历史物品序列并输出一维 logit。"""

    model = DeepModel(
        num_users=3,
        num_items=5,
        model_config={"embedding_dim": 8, "mlp_hidden_dims": [16, 8], "dropout": 0.0},
        trainer_config={"learning_rate": 0.001, "weight_decay": 0.0},
    )
    batch = {
        "user_id": torch.tensor([0, 1], dtype=torch.long),
        "item_id": torch.tensor([1, 2], dtype=torch.long),
        "hist_item_ids": torch.tensor([[0, 2, 3], [0, 0, 4]], dtype=torch.long),
        "hist_len": torch.tensor([2, 1], dtype=torch.long),
    }
    logits = model.compute_logits(batch)
    assert logits.shape == (2,)
