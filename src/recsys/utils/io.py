"""这个文件提供配置文件和常见结构化文件的读写工具。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回 Path 对象。"""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置文件。"""

    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def merge_dicts(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个字典。"""

    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """按 data/model/trainer 基础配置与实验覆盖项组合加载实验配置。"""

    experiment_path = Path(path)
    experiment_config = load_yaml(experiment_path)

    # 这里约定实验配置只负责声明基础配置文件路径和少量覆盖项。
    config_refs = experiment_config.get("config_refs", {})
    if not config_refs:
        return experiment_config

    project_root = experiment_path.parent.parent.parent

    # 依次加载 data、model、trainer 基础配置，再叠加实验级覆盖。
    dataset_config = load_yaml(project_root / config_refs["data"])
    model_config = load_yaml(project_root / config_refs["model"])
    trainer_config = load_yaml(project_root / config_refs["trainer"])

    composed = {
        "experiment_name": experiment_config["experiment_name"],
        "dataset": dataset_config,
        "model": model_config,
        "trainer": trainer_config,
    }

    # split、negative_sampling、features 通常只在实验层声明。
    for section in ["split", "negative_sampling", "features"]:
        if section in experiment_config:
            composed[section] = experiment_config[section]

    # dataset/model/trainer 如果在实验层有覆盖项，则递归合并。
    for section in ["dataset", "model", "trainer"]:
        if section in experiment_config:
            composed[section] = merge_dicts(composed[section], experiment_config[section])

    return composed


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """保存 YAML 配置文件。"""

    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def load_json(path: str | Path) -> dict[str, Any]:
    """读取 JSON 文件。"""

    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    """保存 JSON 文件。"""

    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_dataframe(path: str | Path, frame: pd.DataFrame) -> Path:
    """优先保存为 parquet，失败时回退为 csv，并返回实际保存路径。"""

    target = Path(path)
    ensure_dir(target.parent)
    if target.suffix == ".parquet":
        try:
            frame.to_parquet(target, index=False)
            return target
        except (ImportError, ValueError):
            fallback = target.with_suffix(".csv")
            frame.to_csv(fallback, index=False)
            return fallback
    frame.to_csv(target, index=False)
    return target


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """根据后缀加载 DataFrame。"""

    source = Path(path)
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    return pd.read_csv(source)
