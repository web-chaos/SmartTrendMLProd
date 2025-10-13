#!/usr/bin/env python3
"""
train_ml_binary.py

Обучает LightGBM binary классификатор:
 target = 1 если достигнут любой take (take1..take5)
 target = 0 если достигнут stop_loss

Сохраняет:
 - модель -> models/lgbm_binary_v1.pkl
 - feature order -> models/feature_order_binary.json
 - отчет -> models/training_report_binary.json
"""
import os
import argparse
import sqlite3
import json
from collections import Counter

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, f1_score, precision_score, recall_score
)
import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

RANDOM_STATE = 42

# ---- тут можно положить тот же расширенный список фичей, что обсуждали ----
ML_FEATURE_ORDER = [
    "atr_ltf_value","atr_ltf_score",
    "atr_htf_value",
    "adx_ltf_value","adx_htf_value","adx_ltf_score","adx_htf_score",
    "trix_ltf","trix_htf","trix_long_score","trix_short_score",
    "macd_current_diff","macd_htf_diff","macd_momentum","macd_htf_momentum",
    "stoch_ltf","stoch_htf","stoch_prev_ltf","stoch_prev_htf","stoch_delta_ltf","stoch_delta_htf",
    "rsi_ltf","rsi_htf","rsi_long_score","rsi_short_score","rsi_prev_ltf","rsi_prev_htf","rsi_delta_ltf","rsi_delta_htf",
    "ema_diff_ltf","ema_diff_htf","ema_fast","ema_slow",
    "cdv_ltf_value","cdv_htf_value","cdv_ltf_volume","cdv_htf_volume",
    "volume_value","volume_delta","volume_mean","volume_normalized_body","volume_score",
    "candle_pattern_score","candle_pattern_passed","candle_pattern_sd_bonus",
    "structural_score","potential_value","final_scores","scores_config",
    "div_long_total_score","div_short_total_score",
    "macd_score_long","macd_score_short",
    # бинарные passed-флаги (если есть)
    "adx_ltf_passed","adx_htf_passed","rsi_long_passed","rsi_short_passed",
    "macd_long","macd_short","volume_passed","ema_long_passed","ema_short_passed",
    "trix_long_passed","trix_short_passed","div_long_passed","div_short_passed"
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="ml_trading.db", help="SQLite DB path")
    p.add_argument("--table", default="ml_trading_data", help="Table name")
    p.add_argument("--out", default="models/lgbm_binary_v1.pkl", help="Output model file (.pkl)")
    p.add_argument("--features-file", default="models/feature_order_binary.json", help="Save feature order")
    p.add_argument("--report-file", default="models/training_report_binary.json", help="Save training report")
    p.add_argument("--min-samples", type=int, default=80, help="Min completed signals required")
    p.add_argument("--min-pos", type=int, default=20, help="Min positive class (TP1+) required")
    p.add_argument("--test-size", type=float, default=0.2, help="Validation split")
    return p.parse_args()

def read_table(db_path, table):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df

def build_binary_target(df):
    # 1 если любой take_hit == 1, 0 если stop_loss_hit == 1, -1 иначе
    def f(row):
        try:
            if int(row.get("take1_hit", 0)) == 1 or int(row.get("take2_hit", 0)) == 1 \
               or int(row.get("take3_hit", 0)) == 1 or int(row.get("take4_hit", 0)) == 1 \
               or int(row.get("take5_hit", 0)) == 1:
                return 1
            if int(row.get("stop_loss_hit", 0)) == 1:
                return 0
            return -1
        except Exception:
            return -1
    df["target"] = df.apply(f, axis=1)
    return df

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def main():
    args = parse_args()
    print("DB:", args.db, "table:", args.table)
    df = read_table(args.db, args.table)
    print("Rows in table:", len(df))

    # выбрать только завершённые сигналы (имеют take или stop)
    df = df[
        (df.get("take1_hit", 0) == 1) | (df.get("take2_hit", 0) == 1) |
        (df.get("take3_hit", 0) == 1) | (df.get("take4_hit", 0) == 1) |
        (df.get("take5_hit", 0) == 1) | (df.get("stop_loss_hit", 0) == 1)
    ].copy()
    print("Completed signals:", len(df))
    if len(df) < args.min_samples:
        print(f"Not enough completed signals (<{args.min_samples}). Exiting.")
        return

    df = build_binary_target(df)
    df = df[df["target"] >= 0].copy()
    print("After building binary target:", len(df))
    pos = int(df["target"].sum())
    neg = int(len(df) - pos)
    print("Positive (TP1+):", pos, "Negative (Stop):", neg)

    if pos < args.min_pos or neg < args.min_pos:
        print(f"Not enough examples per class (pos<{args.min_pos} or neg<{args.min_pos}). Exiting.")
        return

    # реальный список фичей из ML_FEATURE_ORDER, которые есть в таблице
    available_feats = [f for f in ML_FEATURE_ORDER if f in df.columns]
    # добавим пару ключевых, если не попали
    for e in ["final_scores","entry_price","volume_value","rsi_ltf","adx_ltf_value"]:
        if e in df.columns and e not in available_feats:
            available_feats.append(e)

    print("Using features:", available_feats)
    X = df[available_feats].fillna(0).astype(float)
    y = df["target"].astype(int)

    # stratify split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size,
                                                      random_state=RANDOM_STATE, stratify=y)

    print("Train:", X_train.shape, "Val:", X_val.shape)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        importance_type="gain"
    )

    # fit with early stopping (LightGBM sklearn wrapper)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(100)  # печатает прогресс каждые 100 итераций
        ]
    )

    feature_importance = dict(zip(ML_FEATURE_ORDER, model.feature_importances_.tolist()))

    # предсказания и метрики
    y_proba = model.predict_proba(X_val)[:,1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred)
    try:
        auc = roc_auc_score(y_val, y_proba)
    except Exception:
        auc = None
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    cm = confusion_matrix(y_val, y_pred).tolist()

    print("Validation metrics:")
    print(" Accuracy:", acc)
    print(" AUC:", auc)
    print(" Precision:", prec, " Recall:", rec, " F1:", f1)
    print(" Confusion matrix:\n", cm)

    # feature importance
    fi = {}
    try:
        imps = model.feature_importances_
        for feat, imp in zip(available_feats, imps):
            fi[feat] = int(imp)
        fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        fi = {}

    # top symbols that reached TP1+ and their counts (helpful)
    top_symbols = Counter(df[df["target"]==1]["symbol"].dropna().values).most_common(50)
    # top symbols for TP3+ (strong signals)
    top_symbols_tp3 = Counter(df[df.get("take3_hit",0)==1]["symbol"].dropna().values).most_common()

    # save model + artifacts
    ensure_dir(args.out)
    dump(model, args.out)
    ensure_dir(args.features_file)
    with open(args.features_file, "w", encoding="utf-8") as f:
        json.dump(available_feats, f, indent=2, ensure_ascii=False)

    report = {
        "db": args.db,
        "table": args.table,
        "rows_used": int(len(df)),
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "val_shape": [int(X_val.shape[0]), int(X_val.shape[1])],
        "pos_count": int(pos),
        "neg_count": int(neg),
        "accuracy": float(acc),
        "auc": float(auc) if auc is not None else None,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "feature_importances": fi,
        "top_symbols_tp1plus": top_symbols,
        "top_symbols_tp3": top_symbols_tp3,
        "model_path": args.out,
        "feature_importance": feature_importance,
    }

    ensure_dir(args.report_file)
    with open(args.report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Model saved to:", args.out)
    print("Features saved to:", args.features_file)
    print("Report saved to:", args.report_file)
    print("Top symbols (TP1+):", top_symbols[:10])
    print("Top symbols (TP3+):", top_symbols_tp3[:10])
    print("Done.")

if __name__ == "__main__":
    main()