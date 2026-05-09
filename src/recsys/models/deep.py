"""这个文件定义基于 PyTorch Lightning 的基础 Deep 训练模型。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from recsys.models.base import BasePointwiseRecModel


def build_mlp(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Sequential:
    """构建基础多层感知机。"""

    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    return nn.Sequential(*layers)


class DeepModel(BasePointwiseRecModel):
    """使用用户和物品嵌入做二分类训练的基础 Deep 模型。"""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        model_config: dict[str, Any],
        trainer_config: dict[str, Any],
    ) -> None:
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            model_config=model_config,
            trainer_config=trainer_config,
        )
        embedding_dim = int(model_config["embedding_dim"])
        hidden_dims = list(model_config.get("mlp_hidden_dims", [128, 64, 32]))
        dropout = float(model_config.get("dropout", 0.1))

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # 序列特征需要 padding token，因此物品 embedding 额外预留 0 号位置。
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.mlp = build_mlp(embedding_dim * 3, hidden_dims, dropout)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """计算用户和物品对的匹配分数。"""

        user_vector = self.user_embedding(user_ids)
        # 目标物品与序列物品共用一套 embedding，0 号位预留给 padding。
        item_vector = self.item_embedding(item_ids + 1)
        history_vector = torch.zeros_like(user_vector)
        hidden = self.mlp(torch.cat([user_vector, item_vector, history_vector], dim=-1))
        return self.output_layer(hidden).squeeze(-1)

    def compute_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """在基础 user-item 表示之外，额外接入历史物品序列的平均池化表示。"""

        user_vector = self.user_embedding(batch["user_id"])
        item_vector = self.item_embedding(batch["item_id"] + 1)

        if "hist_item_ids" in batch:
            history_item_ids = batch["hist_item_ids"]
            history_embeddings = self.item_embedding(history_item_ids)
            history_mask = (history_item_ids != 0).unsqueeze(-1).float()
            masked_history = history_embeddings * history_mask
            history_lengths = history_mask.sum(dim=1).clamp_min(1.0)
            history_vector = masked_history.sum(dim=1) / history_lengths
        else:
            history_vector = torch.zeros_like(user_vector)

        hidden = self.mlp(torch.cat([user_vector, item_vector, history_vector], dim=-1))
        return self.output_layer(hidden).squeeze(-1)
