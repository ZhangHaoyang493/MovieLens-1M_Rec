# 训练与特征流程说明

## 1. 文档目的

这份文档专门说明这个项目里“数据预处理、特征提取、模型训练”三个过程应该怎么执行，以及每一步对应哪些输入、输出和代码入口。

如果你想真正跑通一个实验，优先看这份文档。

## 2. 完整执行顺序

推荐执行顺序如下：

1. 下载数据
2. 预处理数据
3. 提取特征
4. 训练模型
5. 查看 Lightning 训练输出

## 3. 数据下载

脚本：

- [download_ml1m.py](/Users/zhy/Desktop/MovieLens-1M_Rec/scripts/download_ml1m.py:1)

命令：

```bash
uv run python scripts/download_ml1m.py
```

做的事情：

- 下载 `MovieLens-1M`
- 解压数据文件
- 校验原始文件是否齐全

输入：

- 无

输出：

- `data/raw/ml-1m/users.dat`
- `data/raw/ml-1m/movies.dat`
- `data/raw/ml-1m/ratings.dat`

## 4. 数据预处理

脚本：

- [preprocess_ml1m.py](/Users/zhy/Desktop/MovieLens-1M_Rec/scripts/preprocess_ml1m.py:1)

核心实现：

- [preprocess.py](/Users/zhy/Desktop/MovieLens-1M_Rec/src/recsys/data/preprocess.py:1)
- [splitter.py](/Users/zhy/Desktop/MovieLens-1M_Rec/src/recsys/data/splitter.py:1)

命令：

```bash
uv run python scripts/preprocess_ml1m.py --config configs/experiment/deep_ml1m.yaml
```

说明：

- 这里传入的是实验配置
- 脚本内部会自动组合加载 data/model/trainer 基础配置

做的事情：

- 解析原始用户、电影、评分文件
- 将显式评分转为训练标签
- 过滤低频用户和低频物品
- 把原始 ID 转成连续整数 ID
- 为每个正样本补充随机负样本
- 按配置切分训练集和测试集

输入：

- `data/raw/ml-1m/users.dat`
- `data/raw/ml-1m/movies.dat`
- `data/raw/ml-1m/ratings.dat`
- `configs/experiment/*.yaml`

输出：

- `data/interim/users.parquet` 或 `users.csv`
- `data/interim/items.parquet` 或 `items.csv`
- `data/interim/interactions.parquet` 或 `interactions.csv`
- `data/artifacts/id_mappings.json`
- `data/processed/train.csv`
- `data/processed/test.csv`

默认处理规则：

- 所有已交互物品视为正样本
- 从用户未交互物品中随机采样负样本
- 默认切分方式为 `leave_one_out`
- 如果使用 `leave_one_out`，则每个用户最后一次交互进入测试集
- 如果使用 `leave_two_out`，则每个用户最后两次交互优先进入测试集

`split` 配置示例：

默认 Leave-one-out 切分：

```yaml
split:
  split_type: leave_one_out
  seed: 2026
```

随机切分：

```yaml
split:
  split_type: random
  train_ratio: 0.8
  test_ratio: 0.2
  seed: 2026
```

Leave-two-out 切分：

```yaml
split:
  split_type: leave_two_out
  seed: 2026
```

## 5. 特征提取

脚本：

- [build_features.py](/Users/zhy/Desktop/MovieLens-1M_Rec/scripts/build_features.py:1)

核心实现：

- [user_features.py](/Users/zhy/Desktop/MovieLens-1M_Rec/src/recsys/features/user_features.py:1)
- [item_features.py](/Users/zhy/Desktop/MovieLens-1M_Rec/src/recsys/features/item_features.py:1)
- [interaction_features.py](/Users/zhy/Desktop/MovieLens-1M_Rec/src/recsys/features/interaction_features.py:1)

命令：

```bash
uv run python scripts/build_features.py --config configs/experiment/deep_ml1m.yaml
```

说明：

- 特征脚本同样通过实验配置间接拿到数据目录定义

做的事情：

- 生成用户人口属性特征
- 生成电影类型和年份特征
- 生成用户和物品交互统计特征

输入：

- `data/interim/users.*`
- `data/interim/items.*`
- `data/interim/interactions.*`

输出：

- `data/artifacts/user_features.csv`
- `data/artifacts/item_features.csv`
- `data/artifacts/interaction_features.csv`

说明：

- 当前基础 Deep 模型主输入仍然是 `user_id` 和 `item_id`
- 这些特征文件主要用于后续扩展和数据检查

## 6. 模型训练

脚本：

- [train.py](/Users/zhy/Desktop/MovieLens-1M_Rec/scripts/train.py:1)

核心实现：

- [datamodule.py](/Users/zhy/Desktop/MovieLens-1M_Rec/src/recsys/data/datamodule.py:1)
- [deep.py](/Users/zhy/Desktop/MovieLens-1M_Rec/src/recsys/models/deep.py:1)

命令：

```bash
uv run python scripts/train.py --config configs/experiment/deep_ml1m.yaml
```

说明：

- 训练脚本会先组合加载基础 data/model/trainer 配置
- 再根据实验配置中的覆盖项生成最终运行配置

做的事情：

- 读取实验配置
- 设置随机种子
- 构建 Lightning DataModule
- 构建基础 Deep 模型
- 使用 Lightning Trainer 在训练集上执行训练
- 在测试集上执行测试结果计算

输入：

- `configs/experiment/*.yaml`
- `data/processed/train.csv`
- `data/processed/test.csv`

输出：

- `outputs/logs/<experiment_name>/`

## 7. 常见配置调整方式

### 调整正样本定义

修改实验配置中的数据部分，例如：

```yaml
dataset:
  positive_strategy: binary_threshold
  positive_threshold: 4
```

## 8. 建议实践

如果你要稳定迭代实验，建议每次都按下面这套流程执行：

1. 复制一份实验配置
2. 修改数据处理规则与模型参数
3. 先执行数据预处理
4. 再执行特征提取
5. 再执行训练
6. 最后查看 Lightning 日志和 checkpoint

这样实验过程会比较清晰，也方便回溯。

## 9. 后续支持生成式推荐时的注意事项

当前文档默认描述的是传统打分式模型流程，因此测试集会显式补负样本。

如果后续新增生成式推荐论文复现，需要单独说明：

- 测试集是否显式包含负样本
- 模型输出的是打分还是直接生成 Top-K
- 评估时使用的是候选集比较还是全物品空间比较

不要直接把当前二分类测试集组织方式套到生成式模型上。
