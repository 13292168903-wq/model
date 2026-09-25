# Fashion-MNIST 置信度分流实验

这是我在学习 PyTorch 训练流程时完成的第一个完整小项目。项目从 LeNet 的搭建、训练和测试开始，之后增加了一个更轻量的 MLP，并尝试让 MLP 先处理样本，把低置信度样本交给 LeNet。

这个实验受到大小模型协同和样本分流思路的启发，当前目标不是复现完整论文，而是先跑通一个可以观察“准确率与强模型调用比例”关系的最小实验。

## 分流思路

```text
Fashion-MNIST 图片
        |
        v
      MLP 预测
        |
        +-- 置信度 >= 阈值 --> 使用 MLP 结果
        |
        +-- 置信度 <  阈值 --> 交给 LeNet，使用 LeNet 结果
```

MLP 输出 10 个类别的 logits，经过 softmax 后取最大概率作为置信度。当前脚本按 batch 处理数据，并通过布尔索引只把低置信度样本送入 LeNet。

## 模型

| 模型 | 结构概览 | 参数量 | 项目中的作用 |
|---|---|---:|---|
| MLP | `Flatten -> Linear(784, 32) -> ReLU -> Linear(32, 10)` | 25,450 | 先处理全部样本的小模型 |
| LeNet | 两层卷积和池化 + 三层全连接 | 61,706 | 处理 MLP 低置信度样本的较强模型 |

两个模型都使用 Fashion-MNIST，输入形状为 `(batch, 1, 28, 28)`，输出形状为 `(batch, 10)`。

## 当前结果

以下结果来自当前保存的模型参数和 Fashion-MNIST 官方测试集，共 10,000 张图片：

| 方法 | 测试准确率 | LeNet 处理比例 |
|---|---:|---:|
| MLP 单独预测 | 86.16% | 0% |
| LeNet 单独预测 | 89.77% | 100% |
| 置信度分流，阈值 `0.95` | 89.77% | 46.14% |

这个结果说明，在当前模型和阈值下，只把 46.14% 的测试样本交给 LeNet，分流系统取得了与全量 LeNet 相同的测试准确率。

需要说明的是，这仍是初步实验：阈值 `0.95` 还没有在两个模型共享的独立验证集上规范选择；“LeNet 处理比例”也只是计算开销的粗略代理，不能直接等同于实际推理时间或算力节省。

## 项目结构

```text
LeNet_5/
|-- model.py                 # LeNet 结构
|-- model_train.py           # LeNet 训练与验证
|-- model_test.py            # LeNet 测试和测试集加载
|-- best_model.pth           # LeNet 参数
|-- data_shunt.py            # MLP/LeNet 置信度分流
`-- mlp/
    |-- model.py             # MLP 结构
    |-- model_train.py       # MLP 训练与验证
    |-- model_test.py        # MLP 测试
    `-- best_MLP_model.pth   # MLP 参数
```

## 运行分流实验

环境中需要安装：

```bash
pip install torch torchvision pandas matplotlib torchinfo
```

在项目根目录运行：

```bash
python data_shunt.py
```

Fashion-MNIST 在本地不存在时会由 torchvision 自动下载。脚本会优先使用 Mac 的 MPS，MPS 不可用时使用 CPU，并输出测试集样本数、转交数量、转交比例和分流准确率。

本次实验环境为 Python 3.14.7、PyTorch 2.14.0 和 torchvision 0.29.0。

## 学习过程

- 2026-09-20：搭建 LeNet，完成 Fashion-MNIST 数据加载和模型前向结构检查。
- 2026-09-22：跟写训练循环，并开始拆解 `zero_grad -> forward -> loss -> backward -> step` 五个步骤。
- 2026-09-23：完成训练、验证和测试流程，梳理 batch、logits、loss、梯度以及 epoch 统计量之间的关系。
- 2026-09-24：完成 MLP，加载两个模型的参数，并从单张样本扩展到完整测试集的置信度分流。

这个过程中最主要的收获不是记住训练模板，而是逐渐理解了数据在一个 batch 中的形状变化、交叉熵如何使用全部 logits、梯度存放在参数的 `.grad` 中，以及如何累计每轮的 loss 和 accuracy。

## 仍需完善

- 固定随机种子，让两个模型使用相同的训练集和验证集划分。
- 在共同验证集上选择阈值，测试集只用于最终评估。
- 比较不同阈值下的准确率和转交比例，画出二者的关系。
- 记录真实推理耗时或计算量，不能只用转交比例代表效率。
