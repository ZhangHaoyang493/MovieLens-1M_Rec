# 这个 Makefile 用于简化 MovieLens-1M 项目的常用操作入口。

UV ?= uv
PYTHON ?= python
CONFIG ?= configs/experiment/deep_ml1m.yaml

.PHONY: help sync download preprocess features build-model-inputs train pipeline test

help:
	@echo "可用命令："
	@echo "  make sync        安装或同步项目依赖"
	@echo "  make download    下载 MovieLens-1M 数据集"
	@echo "  make preprocess  执行数据预处理与训练集/测试集切分"
	@echo "  make features    构建用户、物品、交互特征"
	@echo "  make build-model-inputs  将 artifacts 特征表拼接成训练宽表"
	@echo "  make train       执行模型训练"
	@echo "  make pipeline    按顺序执行 download -> preprocess -> features -> build-model-inputs -> train"
	@echo "  make test        运行测试"
	@echo ""
	@echo "可选参数："
	@echo "  CONFIG=<path>    指定实验配置文件，默认 $(CONFIG)"

sync:
	$(UV) sync

download:
	$(UV) run $(PYTHON) scripts/download_ml1m.py

preprocess:
	$(UV) run $(PYTHON) scripts/preprocess_ml1m.py --config $(CONFIG)

features:
	$(UV) run $(PYTHON) scripts/build_features.py --config $(CONFIG)

build-model-inputs:
	$(UV) run $(PYTHON) scripts/build_model_inputs.py --config $(CONFIG)

train:
	$(UV) run $(PYTHON) scripts/train.py --config $(CONFIG)

pipeline: download preprocess features build-model-inputs train

test:
	$(UV) run pytest
