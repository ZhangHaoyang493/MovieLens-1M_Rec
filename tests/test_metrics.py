"""这个文件测试测试集评估指标的计算逻辑。"""

from __future__ import annotations

import pytest

from recsys.metrics import grouped_ranking_metrics


def test_grouped_ranking_metrics_match_expected_values() -> None:
    """分组排序指标应与手工可验证结果一致。"""

    metrics = grouped_ranking_metrics(
        user_ids=[0, 0, 0, 1, 1, 1],
        labels=[1, 0, 0, 0, 1, 0],
        scores=[3.0, 2.0, 1.0, 1.0, 3.0, 2.0],
    )

    assert metrics["auc"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["logloss"] == pytest.approx(1.1629256834, rel=1e-6)
    assert metrics["gauc"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["ndcg"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["mrr"] == pytest.approx(1.0, rel=1e-6)
