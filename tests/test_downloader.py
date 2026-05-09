"""这个文件测试数据下载模块中的路径校验逻辑。"""

from __future__ import annotations

from pathlib import Path

import pytest

from recsys.data.downloader import validate_ml1m_files


def test_validate_ml1m_files_raise_when_missing(tmp_path: Path) -> None:
    """缺少文件时应该抛出异常。"""

    with pytest.raises(FileNotFoundError):
        validate_ml1m_files(tmp_path)
