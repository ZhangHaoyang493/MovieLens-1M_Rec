# 数据流水线说明

## 1. 流水线目标

这个项目的数据流水线目标是把 MovieLens-1M 的原始显式评分数据转换成适合基础 Deep 模型训练和测试的结构化输入。

## 2. 原始输入文件

MovieLens-1M 原始数据位于：

- `data/raw/ml-1m/users.dat`
- `data/raw/ml-1m/movies.dat`
- `data/raw/ml-1m/ratings.dat`

这些文件由 [scripts/download_ml1m.py](/Users/zhy/Desktop/MovieLens-1M_Rec/scripts/download_ml1m.py:1) 调用 [src/recsys/data/downloader.py](/Users/zhy/Desktop/MovieLens-1M_Rec/src/recsys/data/downloader.py:1) 下载与校验。

## 3. 数据处理阶段

### 第一步：原始文件解析

由 [src/recsys/data/parser.py](/Users/zhy/Desktop/MovieLens-1M_Rec/src/recsys/data/parser.py:1) 完成。

职责：

- 解析 `users.dat`
- 解析 `movies.dat`
- 解析 `ratings.dat`
- 将双冒号分隔格式转为 DataFrame

### 第二步：显式评分转训练标签

由 [src/recsys/data/preprocess.py](/Users/zhy/Desktop/MovieLens-1M_Rec/src/recsys/data/preprocess.py:1) 中的 `build_training_interactions` 完成。

默认规则：

- 所有已发生交互的 `(user, item)` 都视为正样本
- 默认使用 `binary_all`
- 如果实验有要求，也支持按评分阈值做正样本筛选

### 第三步：交互过滤

仍由 `preprocess.py` 负责。

职责：

- 过滤交互次数过少的用户
- 过滤交互次数过少的物品

这个步骤用于保证训练数据基本可用，也方便和部分论文设定对齐。

### 第四步：ID 重映射

仍由 `preprocess.py` 中的 `remap_ids` 完成。

职责：

- 将原始 `raw_user_id` 映射为从 `0` 开始的连续 `user_id`
- 将原始 `raw_item_id` 映射为从 `0` 开始的连续 `item_id`

这个步骤是推荐模型训练的基础，因为嵌入层通常要求连续整数索引。

### 第五步：保存中间结果

预处理结果默认保存到：

- `data/interim/users.parquet` 或 `users.csv`
- `data/interim/items.parquet` 或 `items.csv`
- `data/interim/interactions.parquet` 或 `interactions.csv`
- `data/artifacts/id_mappings.json`

## 4. 数据切分阶段

由 [src/recsys/data/splitter.py](/Users/zhy/Desktop/MovieLens-1M_Rec/src/recsys/data/splitter.py:1) 完成。

当前支持：

- `random`
- `leave_one_out`
- `leave_two_out`

当前默认推荐：

- `leave_one_out`

两种模式说明：

- `random`
  - 先全局随机打乱交互，再按比例切成训练集和测试集
- `leave_one_out`
  - 对每个用户按时间排序，保留最后一次交互作为测试集，其余交互作为训练集
- `leave_two_out`
  - 对每个用户按时间排序，优先保留最后两次交互作为测试集，其余交互作为训练集
  - 如果用户交互不足 3 次，则至少保留 1 条训练样本

当使用 `leave_one_out` 时：

- 不需要提供 `train_ratio`
- 不需要提供 `test_ratio`
- 只需要提供 `split_type: leave_one_out`

当使用 `leave_two_out` 时：

- 不需要提供 `train_ratio`
- 不需要提供 `test_ratio`
- 只需要提供 `split_type: leave_two_out`

切分之后还会补一层二分类负样本：

- 训练集：默认每个正样本补 `4` 个负样本
- 测试集：默认每个正样本补 `1` 个负样本

这里的负样本来源于：

- 用户没有交互过的物品集合

切分结果默认保存到：

- `data/processed/train.csv`
- `data/processed/test.csv`

## 5. Dataset 与 DataLoader 组织

由以下文件负责：

- [src/recsys/data/dataset.py](/Users/zhy/Desktop/MovieLens-1M_Rec/src/recsys/data/dataset.py:1)
- [src/recsys/data/datamodule.py](/Users/zhy/Desktop/MovieLens-1M_Rec/src/recsys/data/datamodule.py:1)

当前只保留一种训练数据形式：

- `(user_id, item_id, label)`

`datamodule.py` 负责把切分后的 `train/test` 文件组织成 Lightning DataModule。

## 6. 特征构建入口

由 [scripts/build_features.py](/Users/zhy/Desktop/MovieLens-1M_Rec/scripts/build_features.py:1) 调用以下模块完成：

- [src/recsys/features/user_features.py](/Users/zhy/Desktop/MovieLens-1M_Rec/src/recsys/features/user_features.py:1)
- [src/recsys/features/item_features.py](/Users/zhy/Desktop/MovieLens-1M_Rec/src/recsys/features/item_features.py:1)
- [src/recsys/features/interaction_features.py](/Users/zhy/Desktop/MovieLens-1M_Rec/src/recsys/features/interaction_features.py:1)
## 7. 当前默认数据流

当前默认完整数据流为：

1. 下载原始数据
2. 解析原始文件
3. 转换为训练标签
4. 过滤低频用户 / 物品
5. 重映射用户和物品 ID
6. 保存中间结果
7. 生成训练集 / 测试集
8. 生成 Dataset 与 Lightning DataModule
9. 提供给训练脚本使用

## 8. 面向论文复现时的测试集说明

当前代码默认仍然是“二分类训练 / 测试集补负样本”的组织方式，这更接近传统打分式模型。

但如果后续要支持生成式推荐论文，需要注意：

- 生成式模型的测试集不一定需要显式保存负样本文件
- 测试集可以只包含：
  - 用户历史
  - 真实下一个物品
- 不过评估时仍然需要定义比较背景：
  - 全物品集合
  - 过滤已见物品后的全物品集合
  - 或固定候选集

所以未来恢复论文复现能力时，数据流水线很可能需要按模型类型分出两套测试集组织方式。
