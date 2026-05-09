"""这个文件负责解析运行设备，统一 CPU 和 GPU 的选择逻辑。"""

from __future__ import annotations

import torch


def resolve_device(device_name: str) -> torch.device:
    """根据配置返回可用设备。"""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)
