# MovieLens-1M 推荐算法复现项目规划

## 1. 项目目标

基于 MovieLens-1M 数据集构建一个使用 PyTorch 的推荐算法研究型项目，目标如下：

- 支持经典和近期推荐论文的可复现实验。
- 将数据处理、特征工程、模型定义、训练、评估、实验管理拆分为清晰模块。
- 在新增模型、负采样策略、评估协议时保持低耦合、易扩展。
- 采用标准工程化目录，既方便研究迭代，也便于后续整理为规范开源项目。

项目初期建议优先面向 **隐式反馈排序模型复现**，同时预留兼容能力，支持：

- 评分预测任务
- 序列推荐任务
- 使用侧信息的特征型推荐任务

## 2. 复现范围建议

不要一开始就做成过度泛化的大框架，建议分阶段推进。

### 第一阶段：公共底座

优先实现大多数论文都会用到的通用流程：

- 数据集下载与原始文件校验
- 用户、物品、评分、时间戳的统一预处理
- 训练集 / 验证集 / 测试集切分
- 负采样
- 统一评估指标计算
- 基于配置文件的训练入口

### 第二阶段：基线模型

建议先实现几类代表性模型：

- `MF` / `BPR`
- `NCF`
- `LightGCN`
- `SASRec` 或其他序列模型基线

这几类模型分别覆盖矩阵分解、深度交互、图推荐、序列推荐。完成这些之后，后续复现新论文时，大多数工作会变成“加一个模型模块”或“补一个特定数据处理逻辑”。

### 第三阶段：论文复现工作流

对于每一篇要复现的论文，都单独维护：

- 独立实验配置
- 论文预处理假设和实现说明
- 超参数与评估协议记录
- 检查点、日志、最终指标输出目录

## 3. 推荐目录结构

```text
MovieLens-1M_Rec/
├── PLAN.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/
│   ├── data/
│   │   ├── movielens1m.yaml
│   │   ├── split_random.yaml
│   │   ├── split_leave_one_out.yaml
│   │   └── feature_basic.yaml
│   ├── model/
│   │   ├── mf.yaml
│   │   ├── bpr.yaml
│   │   ├── ncf.yaml
│   │   ├── lightgcn.yaml
│   │   └── sasrec.yaml
│   ├── trainer/
│   │   ├── default.yaml
│   │   ├── gpu.yaml
│   │   └── early_stop.yaml
│   └── experiment/
│       ├── mf_ml1m.yaml
│       ├── ncf_ml1m.yaml
│       ├── lightgcn_ml1m.yaml
│       └── sasrec_ml1m.yaml
├── data/
│   ├── raw/
│   │   └── ml-1m/
│   ├── interim/
│   ├── processed/
│   └── artifacts/
├── docs/
│   ├── papers/
│   │   ├── baseline_list.md
│   │   └── reproduction_notes_template.md
│   ├── design/
│   │   ├── data_pipeline.md
│   │   ├── model_registry.md
│   │   └── evaluation_protocol.md
│   └── experiments/
├── notebooks/
│   ├── 01_eda_ml1m.ipynb
│   ├── 02_feature_checks.ipynb
│   └── 03_result_analysis.ipynb
├── scripts/
│   ├── download_ml1m.py
│   ├── preprocess_ml1m.py
│   ├── build_features.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_experiment.py
├── src/
│   └── recsys/
│       ├── __init__.py
│       ├── constants.py
│       ├── registry.py
│       ├── utils/
│       │   ├── io.py
│       │   ├── logger.py
│       │   ├── seed.py
│       │   ├── timer.py
│       │   └── env.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── schema.py
│       │   ├── downloader.py
│       │   ├── parser.py
│       │   ├── preprocess.py
│       │   ├── splitter.py
│       │   ├── sampler.py
│       │   ├── dataset.py
│       │   ├── datamodule.py
│       │   └── feature_store.py
│       ├── features/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── user_features.py
│       │   ├── item_features.py
│       │   ├── interaction_features.py
│       │   └── sequence_features.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── layers/
│       │   │   ├── mlp.py
│       │   │   ├── embedding.py
│       │   │   ├── attention.py
│       │   │   └── graph.py
│       │   ├── mf.py
│       │   ├── bpr.py
│       │   ├── ncf.py
│       │   ├── lightgcn.py
│       │   └── sasrec.py
│       ├── losses/
│       │   ├── __init__.py
│       │   ├── bpr_loss.py
│       │   ├── ce_loss.py
│       │   └── reg_loss.py
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── ranking.py
│       │   ├── rating.py
│       │   └── coverage.py
│       ├── trainer/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   ├── callbacks.py
│       │   ├── checkpoint.py
│       │   └── evaluator.py
│       └── experiments/
│           ├── __init__.py
│           ├── runner.py
│           ├── tracker.py
│           └── report.py
├── tests/
│   ├── conftest.py
│   ├── test_downloader.py
│   ├── test_preprocess.py
│   ├── test_splitter.py
│   ├── test_sampler.py
│   ├── test_dataset.py
│   ├── test_metrics.py
│   └── test_models.py
├── outputs/
│   ├── logs/
│   ├── checkpoints/
│   ├── metrics/
│   └── predictions/
└── references/
    ├── papers/
    └── configs_from_papers/
```

## 4. 各模块职责划分

### `configs/`

所有实验参数都放在配置文件中，不要把关键实验设置硬编码在 Python 文件里。

- `data/`：数据路径、切分策略、序列长度、负采样参数
- `model/`：embedding 维度、隐藏层维度、层数、dropout、图传播层数
- `trainer/`：优化器、学习率调度、batch size、epoch、设备、AMP、早停
- `experiment/`：组合一次完整实验所需的 data + model + trainer 配置

建议：

- 使用 YAML，便于阅读和维护
- 每类配置只负责一类问题
- 运行时通过统一配置加载器做合并

### `data/`

按数据生命周期划分存储：

- `raw/`：原始下载文件，不做修改
- `interim/`：解析后、清洗后的中间表
- `processed/`：编码后的 ID、切分结果、张量、邻接矩阵、序列等
- `artifacts/`：映射表、词表、统计量、缓存特征等辅助产物

这一步必须严格区分。很多复现失败的问题，本质上都是原始数据和加工数据混在一起导致的。

### `scripts/`

这里放命令行入口，但只做薄封装，核心逻辑必须在 `src/recsys/` 中。

建议脚本职责：

- `download_ml1m.py`：下载并校验 MovieLens-1M
- `preprocess_ml1m.py`：解析 `ratings/users/movies`，输出规范化表
- `build_features.py`：生成通用特征和模型特定特征
- `train.py`：读取单个实验配置，执行一次训练
- `evaluate.py`：对指定 checkpoint 执行评估
- `run_experiment.py`：按多个随机种子批量运行实验

### `src/recsys/data/`

这是复现项目的核心模块，负责：

- 下载原始数据
- 解析 `users.dat`、`movies.dat`、`ratings.dat`
- 清洗字段、类型转换
- 用户 ID / 物品 ID 重映射为连续整数
- 必要时按最小交互数过滤
- 生成数据切分：
  - 随机切分
  - 时间切分
  - Leave-one-out 切分
- 负采样：
  - 离线负样本
  - 在线负采样
- 封装成 PyTorch `Dataset` 和 `DataLoader`

重要原则：

- 模型无关的预处理与模型相关的预处理必须分开。
- 例如：LightGCN 的图邻接矩阵可以作为 processed artifact，SASRec 的序列截断逻辑则应放在对应的数据构建逻辑中。

### `src/recsys/features/`

即使初期模型对侧信息依赖不强，也建议明确拆出特征工程模块。

MovieLens-1M 可提取的特征包括：

- 用户特征：
  - 性别
  - 年龄桶
  - 职业
  - 邮编区域信息（如果后续实验需要）
- 物品特征：
  - 电影标题中的年份
  - 多热编码的电影类型
  - 类型数量
- 交互特征：
  - 评分值
  - 时间戳
  - 新近性特征
  - 用户/物品交互频次
- 序列特征：
  - 用户历史交互序列
  - 截断或 padding 后的序列
  - 时间间隔特征

建议设计：

- `base.py` 定义统一特征构建接口
- 每个文件只负责一类特征
- 最终通过 `feature_store.py` 持久化输出

### `src/recsys/models/`

每个模型独立一个文件，共用层放在 `layers/` 下。

建议统一抽象基类：

- `base.py` 中定义：
  - `forward`
  - `calculate_loss`
  - `predict`
  - `full_sort_predict`

这样训练和评估代码可以复用，不需要为每个模型单独写一套流程。

### `src/recsys/losses/`

不同论文往往采用不同训练目标，损失函数不要直接埋在模型文件中。

建议拆分出：

- BPR loss
- Binary cross entropy / cross entropy
- L2 正则项
- 后续如需复现新论文，再加入对比学习损失或辅助损失

### `src/recsys/metrics/`

评估指标必须独立于训练逻辑。

排序指标：

- `NDCG@K`
- `HitRate@K`
- `MAP@K`
- `MRR@K`
- `Precision@K`

评分预测指标：

- `RMSE`
- `MAE`

其他分析指标：

- 覆盖率
- 流行度偏置
- 新颖性

### `src/recsys/trainer/`

封装完整训练循环：

- 前向传播
- 反向传播
- 优化器更新
- 学习率调度
- 验证循环
- checkpoint 保存 / 加载
- early stopping
- 混合精度支持
- 指标记录

Trainer 要尽量通用，避免每加一个模型就改一遍训练流程。

### `src/recsys/experiments/`

这一层的目标是把“论文复现”从临时脚本变成规范流程。

职责包括：

- 实验命名
- 随机种子管理
- 输出目录创建
- 多次运行结果汇总
- 最终实验报告生成
- 导出与论文表格对比所需结果

## 5. 数据处理流程规划

建议的数据流水线如下：

1. 下载 MovieLens-1M 到 `data/raw/ml-1m/`
2. 校验是否存在以下原始文件：
   - `users.dat`
   - `movies.dat`
   - `ratings.dat`
3. 将原始文件解析为结构化表
4. 统一字段名与数据类型
5. 将用户 ID / 物品 ID 重映射为连续整数
6. 按目标论文的实验协议生成数据切分
7. 生成模型无关的通用特征
8. 生成模型相关的处理结果：
   - MF / BPR 用的交互矩阵
   - LightGCN 用的图结构
   - SASRec 用的用户行为序列
9. 将所有产物保存到 `data/processed/`
10. 通过 PyTorch dataset / dataloader 加载处理后数据

建议产出文件示例：

- `users.parquet`
- `items.parquet`
- `interactions.parquet`
- `id_mappings.json`
- `train.csv`
- `valid.csv`
- `test.csv`
- `user_history.pkl`
- `lightgcn_graph.npz`
- `feature_stats.json`

## 6. 排序任务中的正负样本定义

MovieLens-1M 原始数据是显式评分数据。在本项目中，如果目标是做排序模型评估，而不是评分预测，那么需要先把显式反馈转换为适合排序学习的隐式反馈表示。

### 6.1 正样本定义

默认建议：

- 用户与物品发生过一次交互，就视为一个正样本。
- 更具体地说，`ratings.dat` 中出现的 `(user, item)` 记录可以先视为正交互候选。

但为了让定义更稳定，建议在配置中支持两种策略：

- `binary_all`：
  - 所有出现过的评分记录都视为正样本
  - 适合大多数经典隐式推荐复现
- `binary_threshold`：
  - 只有评分大于等于某个阈值的交互才视为正样本
  - 例如 `rating >= 4`
  - 适合希望弱化低分行为噪声的实验

项目默认建议优先采用：

- `binary_all`，因为 MovieLens-1M 在很多排序论文复现里通常会被先二值化处理，只保留“是否交互”这一事实。

如果后续复现的论文明确要求：

- 仅保留高分交互作为正样本
- 或者剔除低分样本

则必须在对应的实验配置和论文记录中明确写出。

### 6.2 负样本定义

在排序任务里，MovieLens-1M 没有天然标注的负反馈，所以负样本通常不能直接从原始数据中读取，而是通过“未交互物品”构造。

默认定义：

- 对于某个用户 `u`，所有没有出现在其历史交互集合中的物品，都可视为负样本候选集。

也就是说：

- 正样本：用户真实交互过的物品
- 负样本：用户从未交互过的物品

需要注意：

- 这不代表用户真的“不喜欢”这些物品，只表示“未观察到交互”。
- 因此这类负样本本质上是未观测样本，而不是显式负反馈。

### 6.3 训练阶段的负采样

训练时通常不会把所有未交互物品都拿来参与损失计算，而是对每个正样本动态或离线采样若干负样本。

推荐两种训练方式：

- Pairwise 训练：
  - 样本形式为 `(user, pos_item, neg_item)`
  - 典型模型：`BPR`, `LightGCN`
- Pointwise 训练：
  - 样本形式为 `(user, item, label)`
  - 正样本标签为 `1`
  - 负样本标签为 `0`
  - 典型模型：`NCF`

训练阶段建议默认规则：

- 每个正样本随机采样 `1~n` 个负样本
- 负样本必须从该用户未交互物品中采样
- 支持以下采样策略：
  - 均匀随机采样
  - 按物品流行度采样
  - hard negative 采样（后续高级实验再加）

初版建议默认实现：

- 均匀随机负采样

因为这是最常见、最稳定、最容易和论文设定对齐的方案。

### 6.4 验证与测试阶段的正负样本

排序评估时，正负样本的定义要与训练阶段区分开。

推荐约定如下：

- 验证 / 测试正样本：
  - 来自验证集或测试集中的真实目标物品
- 验证 / 测试负样本：
  - 来自该用户未交互过的其他物品

如果采用 leave-one-out：

- 每个用户在验证集 / 测试集中通常只有 1 个目标正样本
- 模型需要在“目标正样本 + 一组负样本”中进行排序

### 6.5 评估时的两种常见做法

项目中建议同时支持两种评估方式，并在配置里显式声明。

- `full-sort evaluation`
  - 对每个用户，将目标正样本与其所有未交互物品一起排序
  - 更严格，也更接近真实排序任务
  - 更适合 MovieLens-1M 这类规模可控的数据集
- `sampled evaluation`
  - 对每个用户，将目标正样本与固定数量负样本一起排序
  - 计算更快
  - 但结果更依赖采样方式，不同论文之间可比性更差

默认建议：

- 优先使用 `full-sort evaluation`

如果论文原文使用 sampled evaluation，则必须记录：

- 每个正样本配多少个负样本
- 负样本是否固定
- 负样本如何采样

### 6.6 配置层面需要显式声明的内容

为了避免复现时产生歧义，以下内容必须进入配置文件：

- 是否将评分二值化
- 正样本阈值，例如 `rating >= 4`
- 训练负采样数
- 训练负采样策略
- 评估方式是 `full-sort` 还是 `sampled`
- 如果是 sampled evaluation，负样本数量是多少
- 是否固定验证 / 测试负样本集合

建议在 `configs/data/` 中体现类似字段：

```yaml
feedback_type: implicit
positive_strategy: binary_all
positive_threshold: null
train_negative_sampling:
  strategy: uniform
  num_neg: 1
eval_protocol:
  mode: full_sort
  sampled_neg_num: null
```

### 6.7 本项目的默认定义

如果没有特殊论文要求，本项目默认采用以下规则：

- 将 MovieLens-1M 转换为隐式反馈排序任务
- 所有历史交互均视为正样本
- 所有未交互物品均视为负样本候选
- 训练时使用均匀随机负采样
- 评估时使用 leave-one-out + full-sort

这样定义的优点是：

- 简单
- 与主流排序模型训练方式一致
- 便于后续统一复现 `BPR`、`NCF`、`LightGCN`、`SASRec`

## 7. 特征提取规划

MovieLens-1M 数据量不大，特征提取建议做得显式、可缓存、可检查。

### 基础通用特征

- 编码后的 `user_id`
- 编码后的 `item_id`
- 评分值
- 时间戳
- 用户人口属性
- 电影类型信息

### 可选衍生特征

- 用户交互次数
- 物品流行度
- 用户平均评分
- 物品平均评分
- 用户类型偏好分布
- 最近交互窗口统计
- 序列位置编码
- 时间间隔分桶

### 设计原则

特征提取要分成两层：

- `common features`：多个模型都能复用
- `model-specific features`：只在特定模型需要时构建

不要把全部特征逻辑都堆进一个超长的预处理脚本，否则后面按论文协议调整时会非常难维护。

## 8. 模型训练规划

### 训练入口设计

建议统一一个训练入口：

```bash
python scripts/train.py --config configs/experiment/lightgcn_ml1m.yaml
```

这个脚本只负责调度：

- 加载配置
- 设置随机种子
- 初始化 dataset / datamodule
- 通过 registry 创建模型
- 初始化 trainer
- 执行训练与验证
- 保存最佳 checkpoint 和指标

### 训练系统建议一开始就支持的能力

- 固定随机种子，支持可复现
- CPU / 单卡 GPU 运行
- gradient clipping
- early stopping
- 按最佳验证指标保存 checkpoint
- 可配置评估频率
- 多随机种子重复实验

### 输出规范

每个实验建议生成独立目录，例如：

```text
outputs/checkpoints/lightgcn_ml1m/seed_2026_01/
outputs/logs/lightgcn_ml1m/seed_2026_01/
outputs/metrics/lightgcn_ml1m/seed_2026_01.json
```

这样不会把不同论文、不同随机种子的结果混在一起。

## 9. 评估协议规划

这一部分对论文复现非常关键。

不同论文的差异通常不只在模型本身，还可能体现在：

- 数据切分方式
- 评估物品集合构造方式
- 评估时是否使用负采样
- full-sort 还是 sampled evaluation
- 指标截断值，比如 `K = [5, 10, 20]`

因此项目应强制做到：

- 评估协议通过配置文件声明
- 指标计算模块独立于模型模块
- 实验报告记录本次运行使用的精确评估方式

建议默认排序指标：

- `NDCG@10`
- `NDCG@20`
- `HitRate@10`
- `MRR@10`
- `Precision@10`

建议默认做法：

- 对排序模型复现优先使用 leave-one-out + full-sort evaluation
- 如果实现和论文原文存在偏差，必须在 `docs/papers/` 中明确记录

## 10. 论文复现工作流

每复现一篇论文，至少产出以下三类内容：

- `configs/experiment/` 中的一个实验配置
- `docs/papers/` 中的一份论文复现说明
- `docs/experiments/` 中的一份实验结果总结

建议论文说明模板包含：

- 论文标题
- 发表会议 / 年份
- 任务类型
- 原始数据集设置
- 预处理假设
- 模型结构摘要
- 损失函数
- 优化器和超参数
- 评估协议
- 复现结果
- 与论文结果的差距
- 差距可能原因分析

这部分会决定你的项目是“真正可研究、可回溯”的复现工程，还是只是“能跑一次代码”的实验仓库。

## 11. 工程规范建议

### 代码规范

- 使用 `src/` 布局
- 使用类型标注
- 脚本文件保持轻量
- 数据逻辑与模型逻辑彻底分离
- 不要让 notebook 成为唯一真实实现

### 依赖建议

核心依赖：

- `torch`
- `numpy`
- `pandas`
- `scipy`
- `pyyaml`
- `tqdm`

常用扩展：

- `pyarrow`：保存 parquet
- `scikit-learn`：辅助切分或预处理
- `tensorboard` 或 `wandb`：实验跟踪
- `pytest`：测试

### 测试优先级

建议优先为最脆弱的部分写测试：

- 原始文件解析
- ID 重映射
- 数据切分正确性
- 负采样正确性
- 指标计算正确性
- 模型输出 shape 基本检查

## 12. 建议的实现顺序

建议按下面顺序推进，返工最少：

1. 搭项目骨架和依赖文件
2. 实现 MovieLens-1M 下载与解析
3. 实现统一预处理和切分逻辑
4. 实现 PyTorch dataset / dataloader
5. 实现指标和评估工具
6. 实现 `MF` / `BPR` 基线
7. 实现 `NCF`
8. 实现 `LightGCN`
9. 实现 `SASRec`
10. 实现实验批量运行和复现报告

原因：

- 只要数据接口先稳定，后续新增模型的成本就会明显下降
- 如果评估协议不对，后面所有模型结果都会失真，所以指标与评估应优先于复杂模型

## 13. 最小可用版本建议

如果你想尽快有一个可以跑通的第一版，建议将 `v0` 定义为：

- 自动下载 MovieLens-1M
- 预处理成隐式反馈交互数据
- 支持随机切分和 leave-one-out 切分
- 支持训练 `BPR` 和 `NCF`
- 支持 `NDCG@10` 和 `HitRate@10` 评估
- 按配置名保存日志、checkpoint 和指标结果

做到这里，项目就已经具备了比较扎实的论文复现底座。

## 14. 这份规划之后的下一步

基于这份规划，建议下一步直接开始落地：

1. 创建完整项目骨架
2. 添加 `pyproject.toml` 和基础依赖
3. 实现 MovieLens-1M 下载与解析脚手架
4. 先完成第一个可运行基线 `BPR`

这是从“规划阶段”最快进入“可运行复现阶段”的路径。
