"""
train_classifier.py — обучение LightGBM классификатора на исторических сделках бота.

Целевая переменная: realized_r > 0 (бинарная классификация: прибыльная сделка или нет).
Используем realized_r (реальный исход), а не плановый rr.

Модель работает в SHADOW MODE: только считает вероятность и логирует её,
не влияет на реальные торговые решения.

Модель сохраняется на SD-карту (/sdcard/bybit_bot_model/), не в папку бота.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

CSV_PATH = "trades_audited_clean.csv"
MODEL_DIR = "/sdcard/bybit_bot_model"
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.txt")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.json")
MIN_TRADES_RECOMMENDED = 500


def load_data(path):
    if not os.path.exists(path):
        print(f"Файл не найден: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Загружено сделок: {len(df)}")
    return df


def build_features(df):
    df = df.copy()
    df["target"] = (df["realized_r"] > 0).astype(int)
    df["symbol_code"] = df["symbol"].astype("category").cat.codes
    df["side_code"] = (df["side"] == "Buy").astype(int)
    df["session_code"] = df["session"].astype("category").cat.codes
    df["sl_dist_pct"] = ((df["entry"] - df["sl"]).abs() / df["entry"]) * 100
    df["tp_dist_pct"] = ((df["tp"] - df["entry"]).abs() / df["entry"]) * 100

    feature_cols = ["symbol_code", "side_code", "session_code", "rr", "sl_dist_pct", "tp_dist_pct"]
    df_clean = df.dropna(subset=feature_cols + ["target"])
    dropped = len(df) - len(df_clean)
    if dropped:
        print(f"Отброшено строк с пропусками: {dropped}")

    X = df_clean[feature_cols]
    y = df_clean["target"]
    return X, y, feature_cols


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "num_leaves": 15, "learning_rate": 0.05, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1, "min_data_in_leaf": 10,
    }

    model = lgb.train(params, train_data, num_boost_round=200, valid_sets=[val_data],
                       callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])

    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = float("nan")

    print("\n=== Результаты на тестовой выборке ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC-AUC:  {auc:.3f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Loss", "Win"]))

    print("\n=== Важность признаков ===")
    importance = dict(zip(X.columns, model.feature_importance(importance_type="gain")))
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.1f}")

    return model, acc, auc


def save_model(model, feature_cols):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump({"features": feature_cols}, f, ensure_ascii=False, indent=2)
    print(f"\nМодель сохранена: {MODEL_PATH}")
    print(f"Список признаков сохранён: {FEATURES_PATH}")


def main():
    df = load_data(CSV_PATH)

    if len(df) < MIN_TRADES_RECOMMENDED:
        print(f"\n⚠️  ВНИМАНИЕ: в датасете {len(df)} сделок, рекомендуемый минимум — {MIN_TRADES_RECOMMENDED}.")
        print("Модель будет обучена, но должна оставаться в SHADOW MODE до накопления большего числа сделок.\n")

    X, y, feature_cols = build_features(df)
    print(f"\nРаспределение классов: Win={y.sum()} ({y.mean()*100:.1f}%), Loss={(1-y).sum()}")

    if len(X) < 50:
        print("Слишком мало данных для обучения (< 50 сделок). Прерываю.")
        sys.exit(1)

    model, acc, auc = train_model(X, y)
    save_model(model, feature_cols)

    print("\n=== ИТОГ ===")
    print(f"Обучено на {len(X)} сделках. Accuracy={acc:.3f}, AUC={auc:.3f}")
    print("Модель в SHADOW MODE — не влияет на реальную торговлю.")


if __name__ == "__main__":
    main()
