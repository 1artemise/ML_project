from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
DATA_PATH = "dataset/creditcard.csv"
OUTPUT_DIR = Path(".")

np.random.seed(RANDOM_STATE)


# 检查绘图和采样相关依赖是否已安装。
def check_deps() -> bool:
    missing_pkgs = []
    required_modules = {
        "imblearn": "imbalanced-learn",
        "seaborn": "seaborn",
    }

    for module_name, pkg_name in required_modules.items():
        if importlib.util.find_spec(module_name) is None:
            missing_pkgs.append(pkg_name)

    if missing_pkgs:
        print("缺少依赖包：" + ", ".join(missing_pkgs))
        return False

    return True


# 读取数据集，并确认目标列 Class 存在。
def load_df(file: str | Path | Any) -> pd.DataFrame:
    df = pd.read_csv(file)

    if "Class" not in df.columns:
        raise ValueError("数据无效：缺少 Class 列。")

    return df


# 标准化 Amount，并按分层方式切分训练集和测试集。
def prep_data(df: pd.DataFrame):
    if "Amount" not in df.columns:
        raise ValueError('输入数据缺少 "Amount" 列。')

    proc_df = df.copy()
    scaler = StandardScaler()
    proc_df["Amount_scaled"] = scaler.fit_transform(proc_df[["Amount"]])
    proc_df = proc_df.drop(columns=["Amount"])

    X = proc_df.drop(columns=["Class"])
    y = proc_df["Class"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )


# 对训练集做采样，测试集保持原始分布。
def apply_sampling(X_train, y_train, method, sampling_ratio=1.0):
    if method == "none":
        return X_train, y_train

    if method == "smote":
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(
            random_state=RANDOM_STATE,
            sampling_strategy=sampling_ratio,
        )
        return sampler.fit_resample(X_train, y_train)

    if method == "smote_tomek":
        from imblearn.combine import SMOTETomek

        sampler = SMOTETomek(
            random_state=RANDOM_STATE,
            sampling_strategy=sampling_ratio,
        )
        return sampler.fit_resample(X_train, y_train)

    raise ValueError(f"不支持的采样方式：{method}")


# 训练随机森林；需要时给少数类加 class_weight。
def train_rf(
    X_train,
    y_train,
    use_class_weight,
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=1,
):
    model_cfg = {
        "n_estimators": n_estimators,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
    }

    if use_class_weight:
        model_cfg["class_weight"] = "balanced"

    model = RandomForestClassifier(**model_cfg)
    model.fit(X_train, y_train)
    return model


# 输出评估结果
def eval_model(model, X_test, y_test, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print(f"\n分类报告（threshold={threshold:.2f}）")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    roc_auc = roc_auc_score(y_test, y_proba)
    prec_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_curve, prec_curve)

    prec, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=[1],
        average=None,
        zero_division=0,
    )

    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"PR-AUC:  {pr_auc:.6f}")

    return {
        "precision_fraud": float(prec[0]),
        "recall_fraud": float(recall[0]),
        "f1_fraud": float(f1[0]),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


# 固定模型扫描阈值
def scan_thresholds(model, X_test, y_test, thresholds):
    rows = []

    for threshold in thresholds:
        metrics = eval_model(model, X_test, y_test, threshold=threshold).copy()
        metrics["threshold"] = threshold
        rows.append(metrics)

    return pd.DataFrame(rows)[
        [
            "threshold",
            "precision_fraud",
            "recall_fraud",
            "f1_fraud",
            "roc_auc",
            "pr_auc",
        ]
    ]


# 固定阈值后，比较不同 SMOTE 比例对模型结果的影响。
def scan_smote_ratios(
    X_train,
    X_test,
    y_train,
    y_test,
    ratios,
    threshold=0.3,
):
    rows = []

    for ratio in ratios:
        print("\n" + "-" * 60)
        print(f"当前 SMOTE 比例：{ratio}")

        X_train_sampled, y_train_sampled = apply_sampling(
            X_train,
            y_train,
            "smote",
            sampling_ratio=ratio,
        )
        model = train_rf(X_train_sampled, y_train_sampled, use_class_weight=False)
        metrics = eval_model(model, X_test, y_test, threshold=threshold).copy()
        metrics["sampling_ratio"] = ratio
        metrics["threshold"] = threshold
        metrics["train_size_after_sampling"] = int(len(y_train_sampled))
        rows.append(metrics)

    return pd.DataFrame(rows)[
        [
            "sampling_ratio",
            "threshold",
            "precision_fraud",
            "recall_fraud",
            "f1_fraud",
            "roc_auc",
            "pr_auc",
            "train_size_after_sampling",
        ]
    ]


# 固定阈值和 SMOTE 比例后，比较不同随机森林参数组合。
def scan_rf_params(
    X_train,
    X_test,
    y_train,
    y_test,
    rf_cfgs,
    threshold=0.3,
    sampling_ratio=1.0,
):
    rows = []

    X_train_sampled, y_train_sampled = apply_sampling(
        X_train,
        y_train,
        "smote",
        sampling_ratio=sampling_ratio,
    )

    for rf_cfg in rf_cfgs:
        print("\n" + "-" * 60)
        print(f"当前 RF 参数：{rf_cfg['name']}")

        model = train_rf(
            X_train_sampled,
            y_train_sampled,
            use_class_weight=False,
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            min_samples_leaf=rf_cfg["min_samples_leaf"],
        )
        metrics = eval_model(model, X_test, y_test, threshold=threshold).copy()
        metrics["rf_name"] = rf_cfg["name"]
        metrics["n_estimators"] = rf_cfg["n_estimators"]
        metrics["max_depth"] = rf_cfg["max_depth"]
        metrics["min_samples_leaf"] = rf_cfg["min_samples_leaf"]
        metrics["threshold"] = threshold
        metrics["sampling_ratio"] = sampling_ratio
        rows.append(metrics)

    return pd.DataFrame(rows)[
        [
            "rf_name",
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "sampling_ratio",
            "threshold",
            "precision_fraud",
            "recall_fraud",
            "f1_fraud",
            "roc_auc",
            "pr_auc",
        ]
    ]


# 绘制类别分布、ROC、PR、指标对比和混淆矩阵。
def plot_res(
    orig_df: pd.DataFrame,
    curves: dict[str, dict[str, np.ndarray | float]],
    res_df: pd.DataFrame,
    y_test_final: pd.Series,
    y_pred_final: np.ndarray,
    out_dir: str | Path = OUTPUT_DIR,
) -> None:
    import seaborn as sns

    out_path = Path(out_dir)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    class_counts = orig_df["Class"].value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        x=class_counts.index.astype(str),
        y=class_counts.values,
        palette=["#4C78A8", "#F58518"],
        hue=class_counts.index.astype(str),
        legend=False,
    )
    ax.set_title("Class Distribution of Credit Card Transactions")
    ax.set_xlabel("Class (0 = Normal, 1 = Fraud)")
    ax.set_ylabel("Number of Samples")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    plt.tight_layout()
    plt.savefig(out_path / "class_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for exp_name, vals in curves.items():
        plt.plot(
            vals["fpr"],
            vals["tpr"],
            linewidth=1.8,
            label=f"{exp_name} (AUC={vals['roc_auc']:.4f})",
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.title("ROC Curves for Fraud Detection Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path / "roc_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for exp_name, vals in curves.items():
        plt.plot(
            vals["recall_curve"],
            vals["precision_curve"],
            linewidth=1.8,
            label=f"{exp_name} (PR-AUC={vals['pr_auc']:.4f})",
        )
    baseline = orig_df["Class"].mean()
    plt.axhline(
        baseline,
        linestyle="--",
        color="gray",
        linewidth=1,
        label=f"Fraud Rate Baseline ({baseline:.4f})",
    )
    plt.title("Precision-Recall Curves for Fraud Detection Models")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path / "pr_curve.png", dpi=300)
    plt.close()

    metrics_long = res_df.melt(
        id_vars="experiment",
        value_vars=["recall_fraud", "f1_fraud", "roc_auc", "pr_auc"],
        var_name="metric",
        value_name="score",
    )
    metric_labels = {
        "recall_fraud": "Recall",
        "f1_fraud": "F1-score",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }
    metrics_long["metric"] = metrics_long["metric"].map(metric_labels)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=metrics_long,
        x="experiment",
        y="score",
        hue="metric",
        palette="Set2",
    )
    ax.set_title("Comparison of Fraud-Class Metrics Across Experiments")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(out_path / "metrics_bar.png", dpi=300)
    plt.close()

    cm = confusion_matrix(y_test_final, y_pred_final, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Normal", "Predicted Fraud"],
        yticklabels=["Actual Normal", "Actual Fraud"],
    )
    ax.set_title("Confusion Matrix of Final Model")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(out_path / "confusion_matrix.png", dpi=300)
    plt.close()


# 保存特征重要性，并打印前 10 个最重要特征。
def save_feat_imp(model, feat_names: list[str], out_dir: str | Path = OUTPUT_DIR):
    feat_imp_df = pd.DataFrame(
        {
            "feature": feat_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    out_path = Path(out_dir) / "feature_importance.csv"
    feat_imp_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n最重要的 10 个特征")
    print(feat_imp_df.head(10).to_string(index=False))
    print(f"\n特征重要性已保存到：{out_path}")

    return feat_imp_df


def run_exp() -> None:
    # 记录整个实验流程的总耗时
    start_time = time.time()
    print("=" * 80)
    print("信用卡欺诈检测实验")
    print("核心指标：fraud-class Recall、PR-AUC、ROC-AUC")
    print("=" * 80)

    # 检查依赖是否齐全
    if not check_deps():
        sys.exit(1)

    # 确认数据集存在
    data_path = Path(DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(f"没有找到数据集：{data_path.resolve()}")

    print(f"\n读取数据：{data_path}")
    df = load_df(data_path)
    X_train, X_test, y_train, y_test = prep_data(df)

    # 五组对比实验：基线、仅类别权重、仅 SMOTE、SMOTETomek、SMOTETomek+类别权重。
    experiments = [
        {
            "name": "Exp1_Baseline_RF",
            "sampling": "none",
            "use_class_weight": False,
        },
        {
            "name": "Exp2_ClassWeight_RF",
            "sampling": "none",
            "use_class_weight": True,
        },
        {
            "name": "Exp3_SMOTE_RF",
            "sampling": "smote",
            "use_class_weight": False,
        },
        {
            "name": "Exp4_SMOTETomek_RF",
            "sampling": "smote_tomek",
            "use_class_weight": False,
        },
        {
            "name": "Exp5_SMOTETomek_ClassWeight_RF",
            "sampling": "smote_tomek",
            "use_class_weight": True,
        },
    ]

    res = []
    curves = {}
    final_model = None
    y_pred_final = None

    # 执行实验，收集结果表和画图所需曲线。
    for config in experiments:
        exp_start = time.time()
        name = config["name"]
        print("\n" + "=" * 80)
        print(f"开始实验：{name}")
        print(
            f"采样方式：{config['sampling']} | "
            f"class_weight balanced：{config['use_class_weight']}"
        )

        X_train_sampled, y_train_sampled = apply_sampling(
            X_train,
            y_train,
            config["sampling"],
        )

        model = train_rf(
            X_train_sampled,
            y_train_sampled,
            config["use_class_weight"],
        )
        metrics = eval_model(model, X_test, y_test)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        prec_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)

        metrics["experiment"] = name
        metrics["train_size_after_sampling"] = int(len(y_train_sampled))
        res.append(metrics)

        curves[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "precision_curve": prec_curve,
            "recall_curve": recall_curve,
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
        }

        if name == "Exp5_SMOTETomek_ClassWeight_RF":
            final_model = model
            y_pred_final = y_pred

        elapsed = time.time() - exp_start
        print(f"实验完成，用时 {elapsed:.2f} 秒")

    results_df = pd.DataFrame(res)[
        [
            "experiment",
            "precision_fraud",
            "recall_fraud",
            "f1_fraud",
            "roc_auc",
            "pr_auc",
            "train_size_after_sampling",
        ]
    ]

    results_path = OUTPUT_DIR / "results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("全部实验结果")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print(f"\n结果已保存到：{results_path}")

    if final_model is None or y_pred_final is None:
        raise RuntimeError("最终模型训练失败。")

    # 生成图片和特征重要性文件，便于后续分析或写报告。
    print("\n正在生成图像...")
    plot_res(df, curves, results_df, y_test, y_pred_final, OUTPUT_DIR)
    print("已保存图像：")
    print("- class_distribution.png")
    print("- roc_curve.png")
    print("- pr_curve.png")
    print("- metrics_bar.png")
    print("- confusion_matrix.png")

    save_feat_imp(final_model, list(X_train.columns), OUTPUT_DIR)

    # 固定最终模型，扫描一组阈值，观察 Recall 和 Precision 的变化。
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("\n开始扫描阈值...")
    threshold_df = scan_thresholds(final_model, X_test, y_test, thresholds)
    threshold_path = OUTPUT_DIR / "threshold_results.csv"
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    print("\n阈值扫描结果")
    print(threshold_df.to_string(index=False))
    print(f"\n阈值扫描结果已保存到：{threshold_path}")

    # 固定阈值 0.3，比较不同 SMOTE 比例的影响。
    smote_ratios = [0.2, 0.3, 0.5, 0.8, 1.0]
    print("\n开始比较不同 SMOTE 比例...")
    smote_ratio_df = scan_smote_ratios(
        X_train,
        X_test,
        y_train,
        y_test,
        smote_ratios,
        threshold=0.3,
    )
    smote_ratio_path = OUTPUT_DIR / "smote_ratio_results.csv"
    smote_ratio_df.to_csv(smote_ratio_path, index=False, encoding="utf-8-sig")
    print("\nSMOTE 比例比较结果")
    print(smote_ratio_df.to_string(index=False))
    print(f"\nSMOTE 比例结果已保存到：{smote_ratio_path}")

    # 固定 threshold=0.3 和 SMOTE=1.0 后，再比较一组 RF 参数。
    rf_cfgs = [
        {"name": "rf_200_none_1", "n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
        {"name": "rf_300_none_1", "n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        {"name": "rf_200_15_1", "n_estimators": 200, "max_depth": 15, "min_samples_leaf": 1},
        {"name": "rf_300_15_2", "n_estimators": 300, "max_depth": 15, "min_samples_leaf": 2},
        {"name": "rf_300_20_5", "n_estimators": 300, "max_depth": 20, "min_samples_leaf": 5},
    ]
    print("\n开始比较不同 RF 参数...")
    rf_df = scan_rf_params(
        X_train,
        X_test,
        y_train,
        y_test,
        rf_cfgs,
        threshold=0.3,
        sampling_ratio=1.0,
    )
    rf_path = OUTPUT_DIR / "rf_param_results.csv"
    rf_df.to_csv(rf_path, index=False, encoding="utf-8-sig")
    print("\nRF 参数比较结果")
    print(rf_df.to_string(index=False))
    print(f"\nRF 参数结果已保存到：{rf_path}")

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"实验完成，总用时 {total_elapsed:.2f} 秒。")
    print("=" * 80)


if __name__ == "__main__":
    run_exp()
