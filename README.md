# MovieLens-1M 精排论文复现实验项目

这个项目的最终目标是面向 `MovieLens-1M` 的推荐算法**精排方向论文复现实验**。

当前代码处于一个精简过渡状态，暂时只保留四条主链能力：

- 数据下载
- 数据预处理
- 特征提取
- 基于 PyTorch Lightning 的模型训练

当前实现说明：

- 当前只保留一个基础 `Deep` 模型
- 当前还没有恢复到完整的“多模型精排论文复现实验框架”
- 后续项目会继续向“多模型、统一评估、独立实验配置”的方向扩展
- 当前训练入口已经支持通过配置文件中的 `model.model_name` 动态选择模型

后续做论文复现时，需要特别区分两类测试协议：

- 传统打分式精排模型
  - 测试时通常显式构造正样本和负样本候选
- 生成式推荐模型
  - 测试集可以不显式保存负样本
  - 但评估时仍然需要定义候选背景或全物品空间

项目当前真实状态、约束和未来方向，请优先参考 [AGENT.md](./AGENT.md)。

## 快速开始

```bash
uv sync
uv run python scripts/download_ml1m.py
uv run python scripts/preprocess_ml1m.py --config configs/experiment/deep_ml1m.yaml
uv run python scripts/build_features.py --config configs/experiment/deep_ml1m.yaml
uv run python scripts/train.py --config configs/experiment/deep_ml1m.yaml
```

## 使用 `uv`

推荐统一使用 `uv` 管理环境和依赖，不使用 `venv` 手动创建环境。

常用命令：

```bash
uv sync
uv run python scripts/download_ml1m.py
uv run python scripts/preprocess_ml1m.py --config configs/experiment/deep_ml1m.yaml
uv run python scripts/build_features.py --config configs/experiment/deep_ml1m.yaml
uv run python scripts/train.py --config configs/experiment/deep_ml1m.yaml
uv run pytest
```

## 使用 `make`

为了简化常用操作，项目根目录提供了 `Makefile`。

常用命令：

```bash
make sync
make download
make preprocess
make features
make train
make pipeline
make test
```

默认使用的配置文件是：

```bash
configs/experiment/deep_ml1m.yaml
```

如果你要切换配置，可以这样传参：

```bash
make preprocess CONFIG=configs/experiment/deep_ml1m.yaml
make train CONFIG=configs/experiment/deep_ml1m.yaml
```

## 目录说明

- `configs/`：配置文件
- `data/`：原始数据、中间数据、处理后数据
- `scripts/`：命令行入口
- `src/recsys/`：核心代码
- `tests/`：测试代码
- `outputs/`：日志和模型输出

## 配置组织方式

项目当前采用“基础配置 + 实验配置覆盖”的组合加载方式。

- `configs/data/`
  - 放数据基础配置
- `configs/model/`
  - 放模型基础配置
- `configs/trainer/`
  - 放训练基础配置
- `configs/experiment/`
  - 只负责引用基础配置，并声明切分方式、负采样和少量覆盖项

当前默认实验配置 [deep_ml1m.yaml](./configs/experiment/deep_ml1m.yaml) 会组合加载：

- `configs/data/movielens1m.yaml`
- `configs/model/deep.yaml`
- `configs/trainer/lightning.yaml`

## 执行步骤说明

如果你按标准流程从零开始跑一个实验，建议按下面顺序执行。

### 1. 数据下载

作用：

- 下载 `MovieLens-1M`
- 校验 `users.dat`、`movies.dat`、`ratings.dat` 是否存在

命令：

```bash
uv run python scripts/download_ml1m.py
```

输出位置：

- `data/raw/ml-1m/`

### 2. 数据预处理

作用：

- 解析原始数据文件
- 将显式评分转换为训练样本
- 过滤低频用户和低频物品
- 将用户 ID 和物品 ID 重映射为连续整数
- 为训练集和测试集补充负样本
- 生成训练集和测试集

命令：

```bash
uv run python scripts/preprocess_ml1m.py --config configs/experiment/deep_ml1m.yaml
```

主要输出：

- `data/interim/users.parquet` 或 `users.csv`
- `data/interim/items.parquet` 或 `items.csv`
- `data/interim/interactions.parquet` 或 `interactions.csv`
- `data/artifacts/id_mappings.json`
- `data/processed/train.csv`
- `data/processed/test.csv`

切分方式说明：

- `leave_one_out`：当前默认方式，每个用户最后一次交互进入测试集，其余交互进入训练集
- `random`：按配置比例随机切分训练集和测试集
- `leave_two_out`：每个用户最后两次交互优先进入测试集；若用户交互不足 3 次，则至少保留 1 条训练样本

### 3. 特征提取

作用：

- 构建用户特征
- 构建物品特征
- 构建交互统计特征

命令：

```bash
uv run python scripts/build_features.py --config configs/experiment/deep_ml1m.yaml
```

主要输出：

- `data/artifacts/user_features.csv`
- `data/artifacts/item_features.csv`
- `data/artifacts/interaction_features.csv`

### 4. 模型训练

作用：

- 读取实验配置
- 构建 PyTorch Lightning `DataModule`
- 构建基础 `Deep` 模型
- 使用 PyTorch Lightning Trainer 在训练集上训练
- 在测试集上输出测试结果
- 保存训练日志

命令：

```bash
uv run python scripts/train.py --config configs/experiment/deep_ml1m.yaml
```

主要输出：

- `outputs/logs/`

## 文档导航

- 代理上下文与项目真实状态：[`AGENT.md`](./AGENT.md)
- 整体结构说明：[`docs/design/project_overview.md`](./docs/design/project_overview.md)
- 数据流程说明：[`docs/design/data_pipeline.md`](./docs/design/data_pipeline.md)
- 训练与特征流程说明：[`docs/design/workflow_guide.md`](./docs/design/workflow_guide.md)
