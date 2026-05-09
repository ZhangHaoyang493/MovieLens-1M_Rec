"""这个文件定义点式推荐模型共享的训练与评估基类。"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import lightning as L
import torch
from torch.nn import functional as F

from recsys.metrics import grouped_ranking_metrics


class BasePointwiseRecModel(L.LightningModule):
    """封装点式二分类推荐模型通用训练与测试协议。"""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        model_config: dict[str, Any],
        trainer_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["num_users", "num_items"])
        self.num_users = num_users
        self.num_items = num_items
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.learning_rate = float(trainer_config["learning_rate"])
        self.weight_decay = float(trainer_config.get("weight_decay", 0.0))
        self._test_user_ids: list[torch.Tensor] = []
        self._test_labels: list[torch.Tensor] = []
        self._test_logits: list[torch.Tensor] = []

    @abstractmethod
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """返回用户与物品样本对的打分 logit。"""

    def compute_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """从标准 batch 中取出输入并调用模型前向。"""

        return self.forward(batch["user_id"], batch["item_id"])

    def compute_loss(self, batch: dict[str, torch.Tensor], logits: torch.Tensor) -> torch.Tensor:
        """计算点式二分类训练损失。"""

        return F.binary_cross_entropy_with_logits(logits, batch["label"])

    def compute_accuracy(self, batch: dict[str, torch.Tensor], logits: torch.Tensor) -> torch.Tensor:
        """基于 0.5 阈值计算批级准确率。"""

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()
        return (predictions == batch["label"]).float().mean()

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """执行训练或测试共用的前向、损失与准确率计算。"""

        logits = self.compute_logits(batch)
        loss = self.compute_loss(batch, logits)
        accuracy = self.compute_accuracy(batch, logits)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=(stage == "train"))
        self.log(f"{stage}_acc", accuracy, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """执行单个训练批次。"""

        del batch_idx
        return self._shared_step(batch, "train")

    def on_test_epoch_start(self) -> None:
        """在测试轮开始前清空缓存。"""

        self._test_user_ids.clear()
        self._test_labels.clear()
        self._test_logits.clear()

    def collect_test_outputs(self, batch: dict[str, torch.Tensor], logits: torch.Tensor) -> None:
        """缓存整轮测试指标汇总所需的用户、标签和分数。"""

        self._test_user_ids.append(batch["user_id"].detach().cpu())
        self._test_labels.append(batch["label"].detach().cpu())
        self._test_logits.append(logits.detach().cpu())

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """执行单个测试批次，并缓存整轮测试指标所需输出。"""

        del batch_idx
        logits = self.compute_logits(batch)
        loss = self.compute_loss(batch, logits)
        accuracy = self.compute_accuracy(batch, logits)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("test_acc", accuracy, prog_bar=True, on_epoch=True, on_step=False)
        self.collect_test_outputs(batch, logits)
        return loss

    def compute_test_metrics(self) -> dict[str, float]:
        """根据整轮测试缓存统一计算排序与分类指标。"""

        if not self._test_user_ids:
            return {}

        user_ids = torch.cat(self._test_user_ids).numpy()
        labels = torch.cat(self._test_labels).numpy()
        logits = torch.cat(self._test_logits).numpy()
        return grouped_ranking_metrics(user_ids=user_ids, labels=labels, scores=logits)

    def on_test_epoch_end(self) -> None:
        """在整轮测试结束后汇总并记录测试指标。"""

        metrics = self.compute_test_metrics()
        if not metrics:
            return

        self.log("test_auc", metrics["auc"], prog_bar=True, on_epoch=True, on_step=False)
        self.log("test_logloss", metrics["logloss"], prog_bar=True, on_epoch=True, on_step=False)
        self.log("test_gauc", metrics["gauc"], prog_bar=True, on_epoch=True, on_step=False)
        self.log("test_ndcg", metrics["ndcg"], prog_bar=True, on_epoch=True, on_step=False)
        self.log("test_mrr", metrics["mrr"], prog_bar=True, on_epoch=True, on_step=False)
        self._test_user_ids.clear()
        self._test_labels.clear()
        self._test_logits.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """创建优化器。"""

        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
