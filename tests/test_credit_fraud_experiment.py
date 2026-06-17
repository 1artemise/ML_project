import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import credit_fraud_experiment as exp


class DummyProbModel:
    def __init__(self, probs):
        self.probs = np.asarray(probs)

    def predict(self, _X):
        return (self.probs >= 0.5).astype(int)

    def predict_proba(self, _X):
        return np.column_stack([1 - self.probs, self.probs])


class DummyTrainModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        probs = np.full(len(X), 0.2, dtype=float)
        return np.column_stack([1 - probs, probs])


class DummyFeatureModel:
    feature_importances_ = np.array([0.7, 0.3])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        probs = np.full(len(X), 0.3, dtype=float)
        return np.column_stack([1 - probs, probs])


class DummyRFModel:
    def __init__(self, probs):
        self.probs = np.asarray(probs)

    def predict(self, X):
        return (self.probs[: len(X)] >= 0.5).astype(int)

    def predict_proba(self, X):
        probs = self.probs[: len(X)]
        return np.column_stack([1 - probs, probs])


class CreditFraudExperimentTests(unittest.TestCase):
    def make_df(self):
        rows = []
        for i in range(40):
            rows.append(
                {
                    "Time": i,
                    "V1": float(i % 5),
                    "V2": float(i % 7),
                    "Amount": float(i + 1),
                    "Class": 1 if i % 10 == 0 else 0,
                }
            )
        return pd.DataFrame(rows)

    def test_load_df_requires_class_column(self):
        csv_data = io.StringIO("Time,V1,Amount\n0,1.0,10.0\n")

        with self.assertRaises(ValueError):
            exp.load_df(csv_data)

    def test_prep_data_scales_amount_and_preserves_stratification(self):
        df = self.make_df()

        X_train, X_test, y_train, y_test = exp.prep_data(df)

        self.assertIn("Amount_scaled", X_train.columns)
        self.assertNotIn("Amount", X_train.columns)
        self.assertIn("Time", X_train.columns)
        self.assertEqual(len(X_test), 8)
        self.assertEqual(int(y_train.sum()), 3)
        self.assertEqual(int(y_test.sum()), 1)

    def test_apply_sampling_none_returns_original_objects(self):
        df = self.make_df()
        X = df.drop(columns=["Class"])
        y = df["Class"]

        X_sampled, y_sampled = exp.apply_sampling(X, y, "none")

        self.assertIs(X_sampled, X)
        self.assertIs(y_sampled, y)

    def test_eval_model_returns_fraud_metrics_and_auc_values(self):
        X_test = pd.DataFrame({"feature": range(6)})
        y_test = pd.Series([0, 0, 1, 0, 1, 1])
        model = DummyProbModel([0.01, 0.2, 0.8, 0.4, 0.7, 0.9])

        with redirect_stdout(io.StringIO()):
            metrics = exp.eval_model(model, X_test, y_test)

        self.assertEqual(metrics["precision_fraud"], 1.0)
        self.assertEqual(metrics["recall_fraud"], 1.0)
        self.assertEqual(metrics["f1_fraud"], 1.0)
        self.assertGreaterEqual(metrics["roc_auc"], 0.0)
        self.assertLessEqual(metrics["roc_auc"], 1.0)
        self.assertGreaterEqual(metrics["pr_auc"], 0.0)
        self.assertLessEqual(metrics["pr_auc"], 1.0)

    def test_eval_model_threshold_changes_recall(self):
        X_test = pd.DataFrame({"feature": range(4)})
        y_test = pd.Series([0, 1, 1, 0])
        model = DummyProbModel([0.1, 0.45, 0.8, 0.3])

        with redirect_stdout(io.StringIO()):
            metrics_high = exp.eval_model(model, X_test, y_test, threshold=0.5)
            metrics_low = exp.eval_model(model, X_test, y_test, threshold=0.4)

        self.assertLess(metrics_high["recall_fraud"], metrics_low["recall_fraud"])

    def test_scan_thresholds_returns_one_row_per_threshold(self):
        X_test = pd.DataFrame({"feature": range(4)})
        y_test = pd.Series([0, 1, 1, 0])
        model = DummyProbModel([0.1, 0.45, 0.8, 0.3])
        thresholds = [0.4, 0.5, 0.6]

        with redirect_stdout(io.StringIO()):
            threshold_df = exp.scan_thresholds(model, X_test, y_test, thresholds)

        self.assertEqual(list(threshold_df["threshold"]), thresholds)
        self.assertEqual(len(threshold_df), 3)
        self.assertIn("recall_fraud", threshold_df.columns)

    def test_apply_sampling_passes_sampling_ratio_to_smote(self):
        df = self.make_df()
        X = df.drop(columns=["Class"])
        y = df["Class"]

        class FakeSMOTE:
            def __init__(self, random_state, sampling_strategy):
                self.random_state = random_state
                self.sampling_strategy = sampling_strategy

            def fit_resample(self, X_train, y_train):
                return X_train, y_train

        with patch("imblearn.over_sampling.SMOTE", FakeSMOTE):
            X_sampled, y_sampled = exp.apply_sampling(
                X,
                y,
                "smote",
                sampling_ratio=0.4,
            )

        self.assertIs(X_sampled, X)
        self.assertIs(y_sampled, y)

    def test_scan_smote_ratios_returns_one_row_per_ratio(self):
        df = self.make_df()
        X_train = df.drop(columns=["Class"]).iloc[:32].reset_index(drop=True)
        X_test = df.drop(columns=["Class"]).iloc[32:].reset_index(drop=True)
        y_train = df["Class"].iloc[:32].reset_index(drop=True)
        y_test = df["Class"].iloc[32:].reset_index(drop=True)
        ratios = [0.2, 0.5, 1.0]
        seen_ratios = []

        def fake_apply_sampling(X, y, _m, sampling_ratio=1.0):
            seen_ratios.append(sampling_ratio)
            return X, y

        with (
            patch.object(exp, "apply_sampling", side_effect=fake_apply_sampling),
            patch.object(exp, "train_rf", return_value=DummyFeatureModel()),
            patch.object(exp, "eval_model", return_value={
                "precision_fraud": 0.6,
                "recall_fraud": 0.8,
                "f1_fraud": 0.685,
                "roc_auc": 0.9,
                "pr_auc": 0.7,
            }),
            patch.object(exp, "save_feat_imp"),
            redirect_stdout(io.StringIO()),
        ):
            ratio_df = exp.scan_smote_ratios(
                X_train,
                X_test,
                y_train,
                y_test,
                ratios,
                threshold=0.3,
            )

        self.assertEqual(seen_ratios, ratios)
        self.assertEqual(list(ratio_df["sampling_ratio"]), ratios)
        self.assertEqual(len(ratio_df), 3)
        self.assertIn("threshold", ratio_df.columns)

    def test_scan_rf_params_returns_one_row_per_config(self):
        df = self.make_df()
        X_train = df.drop(columns=["Class"]).iloc[:32].reset_index(drop=True)
        X_test = df.drop(columns=["Class"]).iloc[32:].reset_index(drop=True)
        y_train = df["Class"].iloc[:32].reset_index(drop=True)
        y_test = df["Class"].iloc[32:].reset_index(drop=True)
        rf_cfgs = [
            {"name": "rf_a", "n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
            {"name": "rf_b", "n_estimators": 300, "max_depth": 15, "min_samples_leaf": 2},
        ]

        with (
            patch.object(exp, "apply_sampling", return_value=(X_train, y_train)),
            patch.object(exp, "train_rf") as train_rf_mock,
            redirect_stdout(io.StringIO()),
        ):
            train_rf_mock.side_effect = [
                DummyRFModel([0.2] * len(X_test)),
                DummyRFModel([0.3] * len(X_test)),
            ]
            rf_df = exp.scan_rf_params(
                X_train,
                X_test,
                y_train,
                y_test,
                rf_cfgs,
                threshold=0.3,
                sampling_ratio=1.0,
            )

        self.assertEqual(list(rf_df["rf_name"]), ["rf_a", "rf_b"])
        self.assertEqual(list(rf_df["threshold"]), [0.3, 0.3])
        self.assertEqual(list(rf_df["sampling_ratio"]), [1.0, 1.0])
        self.assertIn("f1_fraud", rf_df.columns)

    def test_run_exp_prints_chinese_summary_and_skips_low_value_shape_logs(self):
        df = self.make_df()
        X_train = df.drop(columns=["Class"]).iloc[:32].reset_index(drop=True)
        X_test = df.drop(columns=["Class"]).iloc[32:].reset_index(drop=True)
        y_train = df["Class"].iloc[:32].reset_index(drop=True)
        y_test = df["Class"].iloc[32:].reset_index(drop=True)
        dummy_model = DummyTrainModel()
        metrics = {
            "precision_fraud": 0.5,
            "recall_fraud": 0.5,
            "f1_fraud": 0.5,
            "roc_auc": 0.7,
            "pr_auc": 0.3,
        }

        with (
            patch.object(exp, "check_deps", return_value=True),
            patch.object(exp.Path, "exists", return_value=True),
            patch.object(exp, "load_df", return_value=df),
            patch.object(exp, "prep_data", return_value=(X_train, X_test, y_train, y_test)),
            patch.object(
                exp,
                "apply_sampling",
                side_effect=lambda X, y, _m, sampling_ratio=1.0: (X, y),
            ),
            patch.object(exp, "train_rf", return_value=dummy_model),
            patch.object(exp, "eval_model", return_value=metrics.copy()),
            patch.object(exp, "plot_res"),
            patch.object(exp, "save_feat_imp"),
            patch.object(exp, "OUTPUT_DIR", Path(".")),
            redirect_stdout(io.StringIO()) as out,
        ):
            exp.run_exp()

        output = out.getvalue()
        self.assertIn("信用卡欺诈检测实验", output)
        self.assertIn("全部实验结果", output)
        self.assertIn("开始实验：Exp1_Baseline_RF", output)
        self.assertIn("实验完成，用时", output)
        self.assertNotIn("Dataset shape", output)
        self.assertNotIn("Train shape", output)
        self.assertNotIn("Training size after sampling", output)


if __name__ == "__main__":
    unittest.main()
