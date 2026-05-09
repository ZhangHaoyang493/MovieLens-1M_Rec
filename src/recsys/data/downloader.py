"""这个文件负责下载并解压 MovieLens-1M 原始数据集。"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from recsys.utils.io import ensure_dir
from tqdm import tqdm

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


class DownloadProgressBar(tqdm):
    """为 urlretrieve 提供下载进度条回调。"""

    def update_to(
        self,
        block_num: int = 1,
        block_size: int = 1,
        total_size: int | None = None,
    ) -> None:
        """根据 urllib 的回调参数更新进度条。"""

        # 当服务端返回文件总大小时，同步更新进度条总量。
        if total_size is not None:
            self.total = total_size

        # urllib 回调给出的是累计块数，因此这里换算出当前新增字节数。
        downloaded = block_num * block_size
        self.update(downloaded - self.n)


def download_movielens_1m(raw_dir: str | Path) -> Path:
    """下载并解压 MovieLens-1M 数据集。"""

    # 先确保原始数据根目录存在。
    raw_path = ensure_dir(raw_dir)
    target_dir = raw_path / "ml-1m"
    ratings_file = target_dir / "ratings.dat"

    # 如果核心文件已经存在，说明数据集之前已经准备好，直接复用。
    if ratings_file.exists():
        return target_dir

    # 先把官方压缩包下载到原始数据目录下，并显示下载进度条。
    zip_path = raw_path / "ml-1m.zip"
    with DownloadProgressBar(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc="下载 ml-1m.zip",
    ) as progress_bar:
        urllib.request.urlretrieve(MOVIELENS_1M_URL, zip_path, reporthook=progress_bar.update_to)

    # 下载完成后解压到 raw 目录。
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(raw_path)

    extracted_dir = raw_path / "ml-1m"

    # 解压后如果目标目录不存在，说明下载或解压过程异常。
    if not extracted_dir.exists():
        raise FileNotFoundError("解压后未找到 ml-1m 目录")

    # 数据已经解压完成，删除压缩包，避免重复占用空间。
    zip_path.unlink(missing_ok=True)
    return extracted_dir


def validate_ml1m_files(raw_dir: str | Path) -> None:
    """校验 MovieLens-1M 所需原始文件是否存在。"""

    base_dir = Path(raw_dir)

    # 这三个文件是后续解析流程必须依赖的最小文件集合。
    expected_files = ["users.dat", "movies.dat", "ratings.dat"]
    missing = [file_name for file_name in expected_files if not (base_dir / file_name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"缺少 MovieLens-1M 原始文件: {missing_text}")
