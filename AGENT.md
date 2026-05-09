# AGENT.md

## 1. 文件目的

这个文件用于给后续进入仓库的 Codex / AI 编码代理提供统一上下文。

目标：

- 让代理快速理解项目的真实代码状态
- 让代理理解项目的最终发展方向
- 约束代理在修改项目时同步更新文档

硬性要求：

- 以后凡是对项目结构、主流程、配置方式、训练逻辑、数据切分逻辑、模型范围做修改，必须同步更新本文件
- 如果 `README.md`、`docs/design/*` 与本文件冲突，以代码真实实现为准，并立即修正文档

## 2. 项目最终目标

这个项目的最终目标不是单模型训练脚手架，而是：

- 面向 `MovieLens-1M` 的推荐算法**精排方向论文复现实验框架**
- 重点用于复现和测试较新的推荐算法论文效果
- 重点关注精排模型，而不是召回阶段

因此后续项目演进方向应当是：

- 支持多模型
- 支持统一训练接口
- 支持统一测试评估
- 支持不同论文独立配置
- 支持复现实验记录与结果对比

额外说明：

- 后续恢复评估框架时，不能假设所有模型都使用同一种测试集组织方式
- 传统打分式精排模型和生成式推荐模型，测试协议大概率需要分开设计

## 3. 当前代码状态

当前代码还没有恢复到“完整论文复现实验框架”，而是一个**精简版过渡状态**。

当前只保留四条主链能力：

- 数据下载
- 数据预处理
- 特征提取
- 基于 PyTorch Lightning 的基础 Deep 模型训练

当前代码**暂时不包含**以下能力：

- 多模型管理机制
- 独立评估脚本
- 多论文实验组织
- 复现实验结果管理

这意味着：

- 当前仓库的实现状态 < 最终项目目标
- 以后新增功能时，应优先朝“精排论文复现实验框架”方向恢复，而不是继续收缩

## 4. 代理修改方向约束

后续代理修改本项目时，应遵守以下方向约束：

1. 不要把当前精简版误认为最终形态
2. 后续扩展优先考虑“论文复现框架能力”，而不是继续做成单模型 Demo
3. 扩展时优先保留当前已有的数据下载、预处理、特征提取链路
4. 后续如果恢复多模型，应优先围绕“精排模型”恢复，不要默认引入召回模块
5. 后续如果恢复评估模块，应优先围绕精排测试效果恢复

## 5. 当前技术栈

- Python
- PyTorch
- PyTorch Lightning
- pandas
- numpy
- YAML 配置
- `uv` 作为环境和依赖管理工具

环境约定：

- 不使用手动 `venv`
- 推荐使用：

```bash
uv sync
```

## 6. 当前目录结构

### 根目录关键文件

- `README.md`
  - 面向使用者的入口说明
- `PLAN.md`
  - 历史规划文档，范围比当前代码更大
- `AGENT.md`
  - 面向代理的当前状态与最终方向说明，修改项目时必须同步更新
- `pyproject.toml`
  - 项目依赖与打包配置
- `requirements.txt`
  - 兼容型依赖文件

### `configs/`

当前只保留与精简版主链直接相关的配置。

- `configs/data/movielens1m.yaml`
- `configs/data/feature_basic.yaml`
- `configs/model/deep.yaml`
- `configs/trainer/lightning.yaml`
- `configs/experiment/deep_ml1m.yaml`

注意：

- 当前配置已经切换为“基础配置 + 实验配置覆盖”的组合加载模式
- `configs/experiment/*.yaml` 不应重复拷贝整份 data/model/trainer 配置
- 实验配置应优先通过 `config_refs` 引用基础配置，再只写实验级差异项
- 后续做论文复现时，应扩展为“每篇论文 / 每个模型独立实验配置”

### `scripts/`

当前只保留四个脚本入口：

- `download_ml1m.py`
- `preprocess_ml1m.py`
- `build_features.py`
- `train.py`

### `src/recsys/`

#### `src/recsys/data/`

- `downloader.py`
  - 下载和校验 `MovieLens-1M`
- `parser.py`
  - 解析 `users.dat`、`movies.dat`、`ratings.dat`
- `preprocess.py`
  - 显式评分转训练正样本、过滤、ID 重映射
- `splitter.py`
  - `random` / `leave_one_out` / `leave_two_out` 切分
  - 补二分类负样本
- `dataset.py`
  - 训练 / 测试交互数据集
- `datamodule.py`
  - Lightning DataModule

#### `src/recsys/features/`

- `user_features.py`
- `item_features.py`
- `interaction_features.py`

#### `src/recsys/models/`

- `__init__.py`
  - 模型注册表与统一构建入口
- `deep.py`
  - 当前唯一保留的基础 Deep 模型
  - 这是当前过渡状态，不代表最终只保留这一个模型

#### `src/recsys/utils/`

- `io.py`
- `logger.py`
- `seed.py`
- `timer.py`
- `env.py`

### `docs/design/`

当前保留：

- `project_overview.md`
- `data_pipeline.md`
- `workflow_guide.md`

这些文档应与本文件保持一致。

## 7. 当前主流程

当前默认执行顺序如下：

1. 下载数据

```bash
uv run python scripts/download_ml1m.py
```

2. 数据预处理

```bash
uv run python scripts/preprocess_ml1m.py --config configs/experiment/deep_ml1m.yaml
```

3. 特征提取

```bash
uv run python scripts/build_features.py --config configs/experiment/deep_ml1m.yaml
```

4. 模型训练

```bash
uv run python scripts/train.py --config configs/experiment/deep_ml1m.yaml
```

## 8. 当前数据处理与切分逻辑

### 原始数据

原始数据位于：

- `data/raw/ml-1m/users.dat`
- `data/raw/ml-1m/movies.dat`
- `data/raw/ml-1m/ratings.dat`

### 预处理阶段

`preprocess.py` 负责：

- 解析原始评分数据
- 过滤低频用户和低频物品
- 将显式评分转换为训练正样本
- 做用户 / 物品连续 ID 重映射

### 切分与负样本

`scripts/preprocess_ml1m.py` 会继续调用 `splitter.py`：

1. 先切分正样本交互
2. 再为训练集和测试集补负样本

当前支持三种切分：

#### `random`

```yaml
split:
  split_type: random
  train_ratio: 0.8
  test_ratio: 0.2
  seed: 2026
```

说明：

- 当前项目默认不使用 `random`
- 它保留为可选切分方式

#### `leave_one_out`

```yaml
split:
  split_type: leave_one_out
  seed: 2026
```

逻辑：

- 每个用户最后一次交互进入测试集
- 其余交互进入训练集

说明：

- 这是当前项目默认切分方式

#### `leave_two_out`

```yaml
split:
  split_type: leave_two_out
  seed: 2026
```

逻辑：

- 正常情况下，每个用户最后两次交互进入测试集
- 其余交互进入训练集
- 如果用户交互不足 3 次，则保守处理，至少保留 1 条训练样本

## 9. 当前训练逻辑

训练逻辑在：

- `scripts/train.py`
- `src/recsys/data/datamodule.py`
- `src/recsys/models/deep.py`

当前行为：

- 只在训练集上训练
- 训练完成后只在测试集上跑结果
- 不使用验证集

注意：

- 当前还不是“统一论文评估框架”
- 这是过渡实现

## 10. 当前模型约束

当前只保留一个基础 `Deep` 模型：

- 输入：`user_id`、`item_id`
- 结构：用户嵌入 + 物品嵌入 + MLP
- 任务：二分类
- 损失：`binary_cross_entropy_with_logits`

注意：

- 当前训练主路径并没有把 `build_features.py` 生成的特征显式送进模型
- 这些特征文件目前更多用于后续扩展和数据检查
- 后续做论文复现时，模型输入和特征接入方式大概率会扩展

## 11. 未来推荐扩展顺序

如果后续要把项目恢复到“精排论文复现实验框架”，建议优先按这个顺序扩展：

1. 恢复多模型管理机制
2. 恢复统一测试评估模块
3. 把实验配置改成模型 / 论文独立配置
4. 引入实验结果记录与对比
5. 再逐步加入新的精排论文模型

## 12. 关于测试协议的项目约束

后续恢复论文复现能力时，应明确区分两类测试协议。

### 12.1 传统打分式精排模型

适用对象：

- CTR / 精排类打分模型
- 给定 `user-item` 对输出匹配分数的模型

测试集组织：

- 可以显式构造正样本和负样本
- 常见做法是“正样本 + 负样本候选集”
- 负样本可来自随机采样、固定候选集或全物品集合背景

评估重点：

- 目标物品是否排在前面
- AUC / LogLoss / NDCG / HitRate / MRR 一类指标

### 12.2 生成式推荐模型

适用对象：

- 给定用户历史，直接生成下一个物品或一个 Top-K 物品序列的模型

测试集组织：

- 测试集可以不显式保存负样本文件
- 但评估时仍然需要一个比较背景
- 这个背景可以是全物品集合、过滤已见物品后的全物品集合，或者固定候选集

评估重点：

- 真实下一个物品是否出现在生成结果中
- Hit@K / NDCG@K / MRR / Recall@K 一类指标

### 12.3 代理实现要求

后续如果新增生成式推荐论文复现：

- 不要直接套用当前二分类负采样测试集
- 必须先在文档中声明测试协议
- 必须说明测试集是否显式包含负样本
- 必须说明评估背景是“负样本候选集”还是“全物品空间”

## 13. 修改项目时的同步更新要求

只要发生以下任一变化，必须同步更新 `AGENT.md`：

- 增删脚本入口
- 增删配置文件
- 改变主流程顺序
- 改变数据切分方式
- 改变正负样本定义
- 改变训练框架
- 改变模型输入
- 恢复多模型、评估模块、论文复现模块

建议同步更新的文件包括：

- `AGENT.md`
- `README.md`
- `docs/design/project_overview.md`
- `docs/design/data_pipeline.md`
- `docs/design/workflow_guide.md`

## 14. 代理修改原则

后续代理在修改本项目时应遵守：

1. 区分“当前代码状态”和“最终项目目标”
2. 不要把当前单模型实现误判为最终设计
3. 新增能力时优先向“精排论文复现实验框架”靠拢
4. 如果文档和代码冲突，以代码真实实现为准，但必须立刻补文档
