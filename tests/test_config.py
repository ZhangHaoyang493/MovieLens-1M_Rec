"""这个文件测试实验配置的组合加载逻辑。"""

from __future__ import annotations

from pathlib import Path

from recsys.utils.io import load_experiment_config


def test_load_experiment_config_composes_sections() -> None:
    """实验配置应能正确组合 data、model、trainer 基础配置。"""

    config = load_experiment_config(Path("configs/experiment/deep_ml1m.yaml"))
    assert config["experiment_name"] == "deep_ml1m"
    assert config["dataset"]["dataset_name"] == "movielens1m"
    assert config["model"]["model_name"] == "deep"
    assert "max_epochs" in config["trainer"]
    assert config["split"]["split_type"] == "leave_one_out"
