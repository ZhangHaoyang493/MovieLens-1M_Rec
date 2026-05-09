"""这个文件负责将特征表拼接到训练和测试样本上，生成模型输入宽表。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from recsys.utils.io import ensure_dir, load_dataframe, save_dataframe


def _resolve_existing_path(candidates: list[Path]) -> Path:
    """从候选路径中找到第一个存在的文件。"""

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到可用输入文件，候选路径: {candidates}")


def _select_numeric_feature_columns(frame: pd.DataFrame, key_columns: set[str]) -> list[str]:
    """从特征表中选出可直接进入模型输入宽表的数值列。"""

    columns: list[str] = []
    for column in frame.columns:
        if column in key_columns:
            continue
        if is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def _build_interaction_stat_tables(interaction_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """将交互级特征表拆成可按用户和物品回填的统计表。"""

    # 交互特征表按样本粒度存储，这里提取用户侧统计和物品侧统计两张小表，
    # 以便后续对训练/测试样本中的正负物品都能稳定 merge。
    user_columns = ["user_id"]
    item_columns = ["item_id"]

    if "user_interaction_count" in interaction_features.columns:
        user_columns.append("user_interaction_count")
    if "item_interaction_count" in interaction_features.columns:
        item_columns.append("item_interaction_count")

    user_stats = interaction_features[user_columns].drop_duplicates(subset=["user_id"]).copy()
    item_stats = interaction_features[item_columns].drop_duplicates(subset=["item_id"]).copy()
    return user_stats, item_stats


def _serialize_history_item_ids(history_item_ids: list[int]) -> str:
    """将定长历史物品序列序列化为 CSV 可保存的字符串。"""

    return " ".join(str(item_id) for item_id in history_item_ids)


def _build_sequence_features(
    samples: pd.DataFrame,
    full_positive_interactions: pd.DataFrame,
    max_sequence_length: int,
) -> pd.DataFrame:
    """为每条样本构建基于用户历史正反馈的定长物品序列特征。"""

    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length 必须大于 0")

    ordered_interactions = full_positive_interactions.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    user_histories: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in ordered_interactions.itertuples(index=False):
        # 序列里把 item_id 整体加 1，预留 0 作为 padding token。
        user_histories[int(row.user_id)].append((int(row.timestamp), int(row.item_id) + 1))

    history_rows: list[dict[str, int | str]] = []
    for row in samples.itertuples(index=False):
        user_id = int(row.user_id)
        timestamp = int(row.timestamp)
        history = [
            item_id
            for past_timestamp, item_id in user_histories.get(user_id, [])
            if past_timestamp < timestamp
        ]
        truncated = history[-max_sequence_length:]
        padded = [0] * (max_sequence_length - len(truncated)) + truncated
        history_rows.append(
            {
                "hist_item_ids": _serialize_history_item_ids(padded),
                "hist_len": len(truncated),
            }
        )

    return pd.DataFrame(history_rows)


def build_model_input_frame(
    samples: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    interaction_features: pd.DataFrame,
    full_positive_interactions: pd.DataFrame,
    max_sequence_length: int,
) -> pd.DataFrame:
    """将用户、物品、交互统计特征拼接到样本表中。"""

    merged = samples.copy()

    # 用户和物品侧只拼接数值特征，避免把原始字符串列直接带进训练宽表。
    user_columns = ["user_id", *_select_numeric_feature_columns(user_features, {"user_id"})]
    item_columns = ["item_id", *_select_numeric_feature_columns(item_features, {"item_id"})]
    merged = merged.merge(user_features[user_columns], on="user_id", how="left")
    merged = merged.merge(item_features[item_columns], on="item_id", how="left")

    # 交互特征当前主要作为用户/物品统计特征使用，不按样本主键逐条 merge。
    user_stats, item_stats = _build_interaction_stat_tables(interaction_features)
    merged = merged.merge(user_stats, on="user_id", how="left")
    merged = merged.merge(item_stats, on="item_id", how="left")
    # 序列特征按用户历史正反馈构造，当前只生成历史物品 ID 序列和实际长度。
    sequence_features = _build_sequence_features(
        samples=merged,
        full_positive_interactions=full_positive_interactions,
        max_sequence_length=max_sequence_length,
    )
    merged = pd.concat([merged.reset_index(drop=True), sequence_features], axis=1)
    return merged


def build_and_save_model_inputs(
    processed_dir: str | Path,
    artifacts_dir: str | Path,
    interactions_path: str | Path,
    max_sequence_length: int,
) -> dict[str, Path]:
    """读取样本表与特征表，生成模型输入宽表并保存。"""

    processed_path = Path(processed_dir)
    artifacts_path = Path(artifacts_dir)

    train_samples = load_dataframe(processed_path / "train.csv")
    test_samples = load_dataframe(processed_path / "test.csv")
    user_features = load_dataframe(_resolve_existing_path([artifacts_path / "user_features.csv"]))
    item_features = load_dataframe(_resolve_existing_path([artifacts_path / "item_features.csv"]))
    interaction_features = load_dataframe(_resolve_existing_path([artifacts_path / "interaction_features.csv"]))
    full_positive_interactions = load_dataframe(interactions_path)

    train_model_input = build_model_input_frame(
        train_samples,
        user_features,
        item_features,
        interaction_features,
        full_positive_interactions=full_positive_interactions,
        max_sequence_length=max_sequence_length,
    )
    test_model_input = build_model_input_frame(
        test_samples,
        user_features,
        item_features,
        interaction_features,
        full_positive_interactions=full_positive_interactions,
        max_sequence_length=max_sequence_length,
    )

    ensure_dir(processed_path)
    train_path = save_dataframe(processed_path / "train_model_input.csv", train_model_input)
    test_path = save_dataframe(processed_path / "test_model_input.csv", test_model_input)
    return {"train_model_input_path": train_path, "test_model_input_path": test_path}
