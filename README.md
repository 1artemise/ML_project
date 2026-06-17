# ML Project

一个基于 `RandomForest` 的信用卡欺诈检测实验项目，重点研究极端类别不平衡条件下的采样策略、阈值调整和模型参数对结果的影响。

## 项目内容

- `credit_fraud_experiment.py`
  主实验脚本，包含数据预处理、五组对照实验、阈值扫描、`SMOTE` 比例比较和 `RandomForest` 参数实验。

- `tests/`
  轻量测试，用于验证主要函数行为是否正常。

- `dataset/creditcard.csv`
  实验使用的数据集。

## 环境要求

- Python 3.10+

## 安装

在项目根目录执行：

```powershell
pip install -r requirements.txt
```

## 使用方法

### 1. 运行完整实验

```powershell
python credit_fraud_experiment.py
```

运行后会在项目目录下生成实验结果和图像文件。

### 2. 运行测试

```powershell
python -m unittest tests.test_credit_fraud_experiment -v
```

## 数据说明

本项目使用 Kaggle 信用卡欺诈数据集 `creditcard.csv`。  
如果你不上传数据集到 GitHub，需要自己将数据文件放到：

```text
dataset/creditcard.csv
```

否则脚本运行时会报找不到数据集。

## 当前实验主线

当前报告中保留的最终方案为：

- 模型：`RandomForest`
- 采样：`SMOTE`
- `sampling_ratio = 0.3`
- 阈值：`0.5`
- 参数：
  - `n_estimators = 300`
  - `max_depth = None`
  - `min_samples_leaf = 1`

## 说明

- `机器学习大作业.docx` 不属于最小可运行单元，不建议上传。
- `.venv/`、`__pycache__/`、`.idea/` 这类本地环境文件也不应上传。
