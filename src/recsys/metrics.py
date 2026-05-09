"""测试阶段使用的分类与排序指标计算工具。"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def average_rank(values: np.ndarray) -> np.ndarray:
    """为一维数组计算 average rank，兼容并列分数。"""

    # 先按升序做稳定排序，后面要基于排序结果为相同分数分配平均名次。
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)

    # 用双指针扫描排序后的数组，把连续相等的一段视为一个 tie group。
    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1

        # average rank 的定义是这一段名次区间的平均值，名次从 1 开始计数。
        average = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average
        start = end

    return ranks


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """计算二分类 AUC。"""

    # 空输入无法定义 AUC，直接返回 nan。
    if labels.size == 0:
        return float("nan")

    # AUC 至少要同时有正负样本，否则 ROC 曲线退化。
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    # 用 rank 公式计算 AUC：
    # AUC = (正样本名次和 - 正样本理论最小名次和) / (正负样本两两配对数)
    ranks = average_rank(scores)
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_logloss(labels: np.ndarray, probabilities: np.ndarray, eps: float = 1e-12) -> float:
    """计算二分类 Logloss。"""

    # 空输入时没有可定义的平均损失。
    if labels.size == 0:
        return float("nan")

    # 先裁剪概率，避免 log(0) 导致数值溢出。
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    # 标准二分类交叉熵，对每个样本分别计算后取平均。
    losses = -(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    return float(losses.mean())


def user_ndcg(labels: np.ndarray, scores: np.ndarray) -> float:
    """计算单个用户的 NDCG。"""

    # 单个用户没有候选样本时无法计算排序指标。
    if labels.size == 0:
        return float("nan")

    # 先按模型分数从高到低排序，得到模型实际给出的推荐顺序。
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    # rank 从 1 开始时，discount 定义为 1 / log2(rank + 1)。
    discounts = 1.0 / np.log2(np.arange(2, sorted_labels.size + 2))
    # 当前排序下的增益之和就是 DCG。
    dcg = float((sorted_labels * discounts).sum())

    # 理想排序把所有正样本放前面，对应 IDCG。
    ideal_labels = np.sort(labels)[::-1]
    ideal_dcg = float((ideal_labels * discounts).sum())
    # 如果该用户没有任何正样本，则 NDCG 无定义。
    if ideal_dcg <= 0.0:
        return float("nan")
    return dcg / ideal_dcg


def user_mrr(labels: np.ndarray, scores: np.ndarray) -> float:
    """计算单个用户的 MRR。"""

    # 没有候选样本时无法计算倒数排名。
    if labels.size == 0:
        return float("nan")

    # 仍然先按分数降序排列，MRR 只关心第一个正样本出现的位置。
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    positive_positions = np.flatnonzero(sorted_labels > 0)
    # 没有正样本时，该用户的 MRR 无定义。
    if positive_positions.size == 0:
        return float("nan")
    # 第一个正样本排第 k 位时，该用户 MRR = 1 / k。
    return 1.0 / float(positive_positions[0] + 1)


def grouped_ranking_metrics(
    user_ids: Iterable[int],
    labels: Iterable[float],
    scores: Iterable[float],
) -> dict[str, float]:
    """按用户聚合计算 AUC、Logloss、GAUC、NDCG、MRR。"""

    # 统一转成 numpy 数组，方便后面做分组和向量化计算。
    user_array = np.asarray(list(user_ids), dtype=np.int64)
    label_array = np.asarray(list(labels), dtype=np.float64)
    score_array = np.asarray(list(scores), dtype=np.float64)
    # 模型传入的是 logit，Logloss 需要概率，因此先做 sigmoid。
    probability_array = 1.0 / (1.0 + np.exp(-score_array))

    # 全局 AUC 和全局 Logloss 不分用户，直接在整个测试集上计算。
    metrics = {
        "auc": binary_auc(label_array, score_array),
        "logloss": binary_logloss(label_array, probability_array),
    }

    # 没有任何测试样本时，分组指标统一返回 nan。
    if user_array.size == 0:
        metrics["gauc"] = float("nan")
        metrics["ndcg"] = float("nan")
        metrics["mrr"] = float("nan")
        return metrics

    # 下面三个指标都需要先按用户分组后再聚合。
    unique_users = np.unique(user_array)
    gauc_numerator = 0.0
    gauc_denominator = 0.0
    ndcg_values: list[float] = []
    mrr_values: list[float] = []

    for user_id in unique_users:
        # 针对当前用户取出该用户所有候选样本的标签和分数。
        mask = user_array == user_id
        group_labels = label_array[mask]
        group_scores = score_array[mask]
        group_size = int(mask.sum())

        # GAUC 先算用户级 AUC，再按组大小做加权平均。
        group_auc = binary_auc(group_labels, group_scores)
        if not np.isnan(group_auc):
            gauc_numerator += group_auc * group_size
            gauc_denominator += group_size

        # NDCG 和 MRR 都是先算用户级指标，再对用户做简单平均。
        group_ndcg = user_ndcg(group_labels, group_scores)
        if not np.isnan(group_ndcg):
            ndcg_values.append(group_ndcg)

        group_mrr = user_mrr(group_labels, group_scores)
        if not np.isnan(group_mrr):
            mrr_values.append(group_mrr)

    # 如果所有用户的 AUC 都无定义，则 GAUC 也返回 nan。
    metrics["gauc"] = float("nan") if gauc_denominator == 0.0 else gauc_numerator / gauc_denominator
    # 用户级 NDCG / MRR 采用宏平均，不再额外按组大小加权。
    metrics["ndcg"] = float("nan") if not ndcg_values else float(np.mean(ndcg_values))
    metrics["mrr"] = float("nan") if not mrr_values else float(np.mean(mrr_values))
    return metrics
