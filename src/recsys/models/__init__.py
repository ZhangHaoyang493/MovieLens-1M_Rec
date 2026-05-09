"""这个包包含模型注册表以及当前已接入的模型实现。"""

from __future__ import annotations

from typing import Any

from recsys.models.base import BasePointwiseRecModel
from recsys.models.deep import DeepModel

MODEL_REGISTRY = {
    "deep": DeepModel,
}


def build_model(
    model_config: dict[str, Any],
    trainer_config: dict[str, Any],
    num_users: int,
    num_items: int,
):
    """根据配置中的模型名构建模型实例。"""

    model_name = str(model_config["model_name"]).lower()
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"未注册的模型类型: {model_name}，当前可用模型: {available}")

    model_class = MODEL_REGISTRY[model_name]
    return model_class(
        num_users=num_users,
        num_items=num_items,
        model_config=model_config,
        trainer_config=trainer_config,
    )


__all__ = ["BasePointwiseRecModel", "DeepModel", "build_model", "MODEL_REGISTRY"]
