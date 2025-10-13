#!/usr/bin/env python3
"""
train_ml_multiclass_xgb.py

Обучает XGBoost multiclass модель (0=stop,1=TP1,...,3=TP_HIGH)
по данным из SQLite таблицы ml_trading_data.

Сохраняет:
 - model -> models/xgb_v1.pkl
 - feature order -> models/feature_order.json
 - training report -> models/training_report.json
"""

import os
import argparse
import json
import sqlite3
from collections import Counter
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier

DEFAULT_DB = "ml_trading.db"
DEFAULT_TABLE = "ml_trading_data"
DEFAULT_OUT_MODEL = "models/xgb_v1.pkl"
DEFAULT_FEATURES_FILE = "models/feature_order.json"
DEFAULT_REPORT_FILE = "models/training_report.json"
DEFAULT_MIN_SAMPLES = 100
RANDOM_STATE = 42

ML_FEATURE_ORDER = [
    "cdv_ltf_value", "adx_htf_value", "trix_htf", "macd_momentum", "rsi_htf",
    "stoch_ltf", "volume_normalized_body", "final_scores", "macd_htf_diff",
    "volume_delta", "cdv_ltf_volume", "macd_htf_momentum", "volume_value",
    "adx_ltf_value", "trix_ltf", "macd_current_diff", "potential_value",
    "rsi_prev_htf", "atr_ltf_value", "rsi_ltf", "macd_prev_diff", "stoch_htf",
    "stoch_prev_htf", "volume_mean", "rsi_delta_ltf", "rsi_delta_htf", "stoch_delta_ltf"
]

RANDOM_STATE = 42

# -----------------------------
# Функции для загрузки данных и построения target
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--table", default=DEFAULT_TABLE)
    p.add_argument("--out", default=DEFAULT_OUT_MODEL)
    p.add_argument("--features-file", default=DEFAULT_FEATURES_FILE)
    p.add_argument("--report-file", default=DEFAULT_REPORT_FILE)
    p.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()

def read_table(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def build_target(df):
    def get_target(row):
        if int(row.get("stop_loss_hit", 0)) == 1: return 0
        if int(row.get("take3_hit", 0)) == 1: return 2
        if int(row.get("take4_hit", 0)) == 1: return 3
        if int(row.get("take5_hit", 0)) == 1: return 3
        if int(row.get("take2_hit", 0)) == 1: return 2
        if int(row.get("take1_hit", 0)) == 1: return 1
        return -1
    df["target"] = df.apply(get_target, axis=1)
    return df[df["target"] >= 0]

def summarize_top_symbols(df, min_take_level=3, min_hits=2):
    df_tp = df[df["target"] >= min_take_level]
    counter = Counter(df_tp["symbol"].dropna())
    filtered = [(s, c) for s, c in counter.items() if c >= min_hits]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered

def ensure_model_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# -----------------------------
# Основной код
# -----------------------------
def main():
    args = parse_args()
    print("Loading data from:", args.db, args.table)
    df = read_table(args.db, args.table)
    print("Total rows:", len(df))

    df = df[
        (df.get("take1_hit", 0)==1) | (df.get("take2_hit",0)==1) |
        (df.get("take3_hit",0)==1) | (df.get("take4_hit",0)==1) |
        (df.get("take5_hit",0)==1) | (df.get("stop_loss_hit",0)==1)
    ]
    print("Completed signals:", len(df))
    if len(df) < args.min_samples:
        print("Not enough completed signals. Exiting.")
        return

    df = build_target(df)
    print("After target apply:", len(df))
    print("Target distribution:\n", df["target"].value_counts())

    # фичи
    available_feats = [f for f in ML_FEATURE_ORDER if f in df.columns]
    available_feats = list(dict.fromkeys(available_feats))  # уникальные
    print("Using features:", available_feats)

    # категориальные признаки
    categorical_cols = [
        "symbol", "signal_ltf", "signal_htf", "trend_global",
        "adx_mode_config","supertrend_mode","structural_mode_config",
        "rsi_1h_long_type","rsi_1h_short_type","rsi_4h_long_type","rsi_4h_short_type",
        "macd_1h_long_type","macd_1h_short_type","macd_4h_long_type","macd_4h_short_type",
        "sd_zone_type","div_tf_config"
    ]
    for col in categorical_cols:
        if col in available_feats:
            df[col] = df[col].fillna("NA")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"{col} mapping:", mapping)

    X = df[available_feats].fillna(0).astype(float)
    y = df["target"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

    # выбор модели
    num_samples = len(X_train)
    num_classes = len(set(y_train))

    if num_samples < 1000:
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            n_estimators=50,
            max_depth=3,
            learning_rate=0.2,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="mlogloss"
        )
        model_name = "model_learn"
    elif num_samples < 2000:
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="mlogloss"
        )
        model_name = "model_balanse"
    else:
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            n_estimators=1500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="mlogloss"
        )
        model_name = "model_base"

    eval_set = [(X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mlogloss",  # multiclass logloss
        early_stopping_rounds=40,
        verbose=True
    )


    # метрики
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro")
    cls_report = classification_report(y_val, y_pred, digits=4)
    cm = confusion_matrix(y_val, y_pred)
    print("Validation Accuracy:", acc)
    print("Balanced Accuracy:", bal_acc)
    print("F1 macro:", f1_macro)
    print("Classification report:\n", cls_report)
    print("Confusion matrix:\n", cm)

    # feature importance
    try:
        fi = dict(zip(available_feats, model.feature_importances_))
        fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        fi = {}

    top_symbols = summarize_top_symbols(df)
    top_symbols_stable = summarize_top_symbols(df, min_hits=3)

    ensure_model_dir(args.out)
    dump(model, args.out)
    with open(args.features_file, "w", encoding="utf-8") as f:
        json.dump(available_feats, f, indent=2, ensure_ascii=False)

    report = {
        "db": args.db,
        "table": args.table,
        "rows_total": int(len(df)),
        "train_shape": [int(x) for x in X_train.shape],
        "val_shape": [int(x) for x in X_val.shape],
        "num_classes": int(num_classes),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_macro),
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist(),
        "feature_importances": fi,
        "top_symbols_tp3_plus": top_symbols,
        "top_symbols_tp3_stable": top_symbols_stable,
        "model_path": args.out
    }

    ensure_model_dir(args.report_file)
    with open(args.report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Selected model:", model_name)
    print("Model saved to:", args.out)
    print("Feature order saved to:", args.features_file)
    print("Training report saved to:", args.report_file)
    print("Top symbols (TP3+):", top_symbols)
    print("Top symbols stable:", top_symbols_stable)
    print("Done.")

if __name__ == "__main__":
    main()
