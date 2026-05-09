"""这个文件负责解析 MovieLens-1M 的 users、movies、ratings 原始文件。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _parse_dat_file(path: str | Path, columns: list[str]) -> pd.DataFrame:
    """解析使用双冒号分隔的 dat 文件。"""

    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=columns,
        encoding="latin-1",
    )


def parse_users(path: str | Path) -> pd.DataFrame:
    """解析用户文件。"""

    return _parse_dat_file(path, ["raw_user_id", "gender", "age", "occupation", "zip_code"])


def parse_movies(path: str | Path) -> pd.DataFrame:
    """解析电影文件。"""

    return _parse_dat_file(path, ["raw_item_id", "title", "genres"])


def parse_ratings(path: str | Path) -> pd.DataFrame:
    """解析评分文件。"""

    return _parse_dat_file(path, ["raw_user_id", "raw_item_id", "rating", "timestamp"])
