"""这个脚本用于读取实验配置并执行基于 PyTorch Lightning 的模型训练。"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import CSVLogger

from recsys.data.datamodule import MovieLensDataModule
from recsys.models import build_model
from recsys.utils.io import load_experiment_config
from recsys.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="训练排序推荐模型")
    # 训练入口完全依赖实验配置文件驱动，这里要求显式传入配置路径。
    parser.add_argument("--config", type=str, required=True, help="实验配置文件路径")
    return parser.parse_args()


def main() -> None:
    """执行训练流程。"""

    # 先初始化脚本级日志器，方便观察训练主流程每一步执行到哪里。
    script_logger = get_logger("scripts.train")
    args = parse_args()
    script_logger.info("开始执行模型训练脚本。")
    script_logger.info("使用配置文件: %s", args.config)

    # 从配置文件中读取训练参数、模型参数和数据目录定义。
    config = load_experiment_config(args.config)
    trainer_config = config["trainer"]
    processed_dir = Path(config["dataset"]["processed_dir"])
    script_logger.info("训练配置加载完成。")
    script_logger.info("处理后数据目录: %s", processed_dir)
    script_logger.info("实验名称: %s", config["experiment_name"])

    # 固定随机种子，尽量保证实验结果可复现。
    script_logger.info("开始设置随机种子: %s", trainer_config["seed"])
    L.seed_everything(int(trainer_config["seed"]), workers=True)

    # 先构建 Lightning DataModule，它负责统一提供训练集和测试集 DataLoader。
    script_logger.info("开始构建 Lightning DataModule。")
    data_module = MovieLensDataModule(
        processed_dir=processed_dir,
        batch_size=int(trainer_config["batch_size"]),
        num_workers=int(trainer_config["num_workers"]),
    )

    # 这里先显式 setup，拿到用户数和物品数，供模型初始化 embedding 大小。
    data_module.setup("fit")
    script_logger.info(
        "DataModule 构建完成，用户数: %s，物品数: %s，训练 batch_size: %s",
        data_module.num_users,
        data_module.num_items,
        trainer_config["batch_size"],
    )

    # 根据配置中的 model_name 动态构建模型，避免训练入口写死具体模型类。
    script_logger.info("开始根据配置构建模型，模型类型: %s", config["model"]["model_name"])
    model = build_model(
        model_config=config["model"],
        trainer_config=trainer_config,
        num_users=data_module.num_users,
        num_items=data_module.num_items,
    )
    script_logger.info("模型构建完成，模型配置: %s", config["model"])

    # 使用 Lightning 的 CSVLogger，把训练过程日志写到 outputs/logs 下。
    script_logger.info("开始准备 Lightning 日志器。")
    trainer_logger = CSVLogger(save_dir="outputs/logs", name=config["experiment_name"])
    script_logger.info("Lightning 日志目录: outputs/logs/%s", config["experiment_name"])

    # 当前脚本显式关闭 checkpoint 保存，所以这里只保留日志输出。
    script_logger.info("当前训练脚本未启用 checkpoint 自动保存。")

    # 根据配置构建 Lightning Trainer，训练轮数、精度、设备等都由配置控制。
    script_logger.info("开始构建 Lightning Trainer。")
    trainer = L.Trainer(
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"],
        precision=trainer_config["precision"],
        max_epochs=trainer_config["max_epochs"],
        gradient_clip_val=trainer_config["gradient_clip_val"],
        log_every_n_steps=trainer_config["log_every_n_steps"],
        logger=trainer_logger,
        enable_checkpointing=False,
    )
    script_logger.info("Trainer 构建完成，最大训练轮数: %s", trainer_config["max_epochs"])

    # 正式启动训练流程，训练数据由 DataModule 自动提供。
    script_logger.info("开始执行模型训练。")
    trainer.fit(model=model, datamodule=data_module)

    # 训练完成后直接在测试集上跑一次结果，当前项目不再单独划分验证集。
    script_logger.info("训练阶段完成，开始执行测试集评估。")
    trainer.test(model=model, datamodule=data_module)
    script_logger.info("训练脚本执行结束。")


if __name__ == "__main__":
    main()
