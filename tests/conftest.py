"""这个文件提供测试阶段复用的样例数据和 fixture。"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def interaction_frame() -> pd.DataFrame:
    """构造一个小型交互表供测试使用。"""

    return pd.DataFrame(
        {
            "raw_user_id": [1, 1, 1, 2, 2, 3],
            "raw_item_id": [10, 11, 12, 10, 13, 14],
            "rating": [5, 4, 3, 5, 4, 5],
            "timestamp": [1, 2, 3, 1, 2, 1],
        }
    )
