# 项目整体框架说明

## 1. 项目定位

这个项目已经收缩为一个最小可用训练工程，只保留四条能力：

- 数据下载
- 数据预处理
- 特征提取
- 基于 PyTorch Lightning 的 Deep 模型训练

项目不再包含：

- 多模型注册机制
- 排序评估脚本
- 论文复现管理
- 实验报告汇总

## 2. 推荐目录结构说明

### `configs/`

负责维护所有实验配置。

- `configs/data/`
  - 数据路径、正样本策略、切分方式、特征开关
- `configs/model/`
  - 各模型结构参数
- `configs/trainer/`
  - 训练批大小、epoch、学习率、设备、早停
- `configs/experiment/`
  - 一次完整实验的组合配置
  - 通过引用 data/model/trainer 基础配置并追加实验差异项来组织

### `scripts/`

负责命令行入口，文件尽量轻量。

- `download_ml1m.py`
  - 下载并校验 MovieLens-1M 数据
- `preprocess_ml1m.py`
  - 执行预处理和训练集、测试集切分
- `build_features.py`
  - 构建用户、物品、交互特征
- `train.py`
  - 启动单次训练

### `src/recsys/`

这是项目核心代码所在目录。

- `constants.py`
  - 默认路径与常量

#### `src/recsys/utils/`

通用工具层。

- `io.py`
  - YAML、JSON、DataFrame 的读写与目录创建
- `logger.py`
  - 日志初始化
- `seed.py`
  - 随机种子设置
- `timer.py`
  - 简单计时器
- `env.py`
  - 设备解析

#### `src/recsys/data/`

数据处理层。

- `downloader.py`
  - 数据下载与原始文件校验
- `parser.py`
  - 解析 `users.dat`、`movies.dat`、`ratings.dat`
- `preprocess.py`
  - 显式评分转换、过滤、ID 重映射
- `splitter.py`
  - 随机切分、leave-one-out 和 leave-two-out 切分
- `dataset.py`
  - 训练集和测试集的 PyTorch Dataset
- `datamodule.py`
  - Lightning DataModule

#### `src/recsys/features/`

特征工程层。

- `user_features.py`
  - 用户属性特征
- `item_features.py`
  - 电影类型、年份等特征
- `interaction_features.py`
  - 用户/物品交互频次等统计特征
#### `src/recsys/models/`

模型层。

- `__init__.py`
  - 模型注册表与统一构建入口
- `deep.py`
  - 基于 LightningModule 的基础 Deep 模型

### `tests/`

负责基础行为测试。

- `test_preprocess.py`
  - 预处理和 ID 重映射测试
- `test_splitter.py`
  - 数据切分测试
- `test_dataset.py`
  - Dataset 输出字段测试
- `test_training.py`
  - Deep 模型训练基础测试

### `docs/`

负责项目设计和使用文档。

- `docs/design/`
  - 工程结构、数据流、训练流程说明

### `outputs/`

负责训练产出。

- `outputs/logs/`
  - 日志
- `outputs/checkpoints/`
  - 模型参数

## 3. 核心执行链路

当前项目默认执行链路如下：

1. `scripts/download_ml1m.py`
   - 下载原始数据
2. `scripts/preprocess_ml1m.py`
   - 解析原始数据
   - 转换交互标签
   - 重映射用户和物品 ID
   - 切分训练集和测试集
3. `scripts/build_features.py`
   - 构建并保存特征
4. `scripts/train.py`
   - 读取配置
   - 构建 Lightning DataModule
   - 构建 LightningModule
   - 在训练集上训练
   - 在测试集上输出结果

## 4. 当前默认建模假设

项目默认采用以下假设：

- 任务类型：用户-物品二分类训练
- 标签定义：交互记录默认为正样本标签 `1`
- 切分方式：默认 `leave_one_out`，同时支持 `random` 和 `leave_two_out`
- 模型类型：基础 Deep MLP
- 训练框架：PyTorch Lightning

## 5. 面向论文复现时的测试协议提醒

当项目后续恢复到“精排论文复现实验框架”时，测试协议不能只保留一种。

至少应区分：

- 传统打分式精排模型
  - 测试集通常显式包含正样本和负样本候选
- 生成式推荐模型
  - 测试集可以只保留用户历史和真实下一个物品
  - 但评估时仍然要声明比较背景是全物品空间还是候选集

## 6. 建议阅读顺序

如果是第一次看这个项目，建议按下面顺序理解：

1. 本文件 `project_overview.md`
2. `data_pipeline.md`
3. `workflow_guide.md`
4. `PLAN.md`
