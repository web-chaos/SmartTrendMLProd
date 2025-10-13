#!/usr/bin/env python3
"""
train_ml_multiclass.py

Обучает LightGBM multiclass модель (0=stop,1=TP1,...,5=TP5)
по данным из SQLite таблицы ml_trading_data.

Сохраняет:
 - model -> models/lgbm_v1.pkl
 - feature order -> models/feature_order.json
 - training report -> models/training_report.json
"""

import os
import argparse
import json
import sqlite3
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score
import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# ----------------------------
# Параметры по умолчанию
# ----------------------------
DEFAULT_DB = "ml_trading.db"
DEFAULT_TABLE = "ml_trading_data"
DEFAULT_OUT_MODEL = "models/lgbm_v1.pkl"
DEFAULT_FEATURES_FILE = "models/feature_order.json"
DEFAULT_REPORT_FILE = "models/training_report.json"
DEFAULT_MIN_SAMPLES = 100
RANDOM_STATE = 42

# ----------------------------
# Подбор фичей (взят из структуры таблицы)
# можно дополнить/убрать при необходимости
# ----------------------------
# ML_FEATURE_ORDER = [
#     # LTF / HTF базовые
#     "atr_ltf_value", "atr_htf_value",
#     "adx_ltf_value", "adx_htf_value",
#     "trix_ltf", "trix_htf",
#     "macd_current_diff", "macd_htf_diff",
#     "stoch_ltf", "stoch_htf",
#     "rsi_ltf", "rsi_htf",
#     # EMA diffs
#     "ema_diff_ltf", "ema_diff_htf", "ema_fast", "ema_slow",
#     # CDV / volume
#     "cdv_ltf_value", "cdv_htf_value", "cdv_ltf_volume", "cdv_htf_volume",
#     "volume_value", "volume_delta", "volume_mean", "volume_normalized_body",
#     # structural / patterns / scores
#     "candle_pattern_score", "structural_score", "potential_value", "final_scores",
#     # divergence / macd scores
#     "macd_score_long", "macd_score_short", "div_long_total_score", "div_short_total_score",
# ]

# ML_FEATURE_ORDER = [
#     # --- Основные значения индикаторов ---
#     "atr_ltf_value", "atr_ltf_score",
#     "adx_ltf_value", "adx_htf_value", "adx_ltf_score", "adx_htf_score",
#     "rsi_ltf", "rsi_htf", "rsi_long_score", "rsi_short_score",
#     "trix_ltf", "trix_htf", "trix_long_score", "trix_short_score",
#     "macd_current_diff", "macd_htf_diff", "macd_momentum", "macd_htf_momentum",
#     "volume_value", "volume_mean", "volume_score", "volume_delta",

#     # --- CDV ---
#     "cdv_ltf_value", "cdv_htf_value", "cdv_ltf_score", "cdv_htf_score",

#     # --- EMA ---
#     "ema_diff_ltf", "ema_diff_htf", "ema_long_score", "ema_short_score",

#     # --- Потенциал и структура ---
#     "potential_value", "potential_passed",
#     "market_structure_ltf_score", "market_structure_htf_score",
#     "structural_score",

#     # --- Свечные паттерны и бонусы ---
#     "candle_pattern_score", "candle_pattern_passed",
#     "candle_pattern_sd_bonus", "candle_pattern_sd_bonus_score",

#     # --- Дивергенции ---
#     "div_long_total_score", "div_short_total_score",
#     "rsi_1h_long_score", "rsi_1h_short_score",
#     "rsi_4h_long_score", "rsi_4h_short_score",
#     "macd_1h_long_score", "macd_1h_short_score",
#     "macd_4h_long_score", "macd_4h_short_score",

#     # --- Проходы фильтров (бинарные индикаторы) ---
#     "adx_ltf_passed", "adx_htf_passed",
#     "rsi_long_passed", "rsi_short_passed",
#     "macd_long", "macd_short",
#     "volume_passed",
#     "ema_long_passed", "ema_short_passed",
#     "trix_long_passed", "trix_short_passed",
#     "div_long_passed", "div_short_passed",

#     # --- Итоговые оценки ---
#     "final_scores"
# ]

ML_FEATURE_ORDER = [
    # --- Основные значения индикаторов ---
    "atr_ltf_value", "atr_ltf_score",
    "adx_ltf_value", "adx_htf_value", "adx_ltf_score", "adx_htf_score",
    "rsi_ltf", "rsi_htf", "rsi_long_score", "rsi_short_score",
    "trix_ltf", "trix_htf", "trix_long_score", "trix_short_score",
    "macd_current_diff", "macd_htf_diff", "macd_prev_diff", "macd_prev_htf_diff",
    "macd_momentum", "macd_htf_momentum",
    "volume_value", "volume_mean", "volume_score", "volume_delta", "volume_mean_long", "volume_normalized_body",
    "ema_diff_ltf", "ema_diff_htf", "ema_long_score", "ema_short_score",

    # --- CDV ---
    "cdv_ltf_value", "cdv_htf_value", "cdv_ltf_score", "cdv_htf_score",
    "cdv_ltf_volume", "cdv_htf_volume", "cdv_ltf_normalized", "cdv_htf_normalized",

    # --- Потенциал и структура ---
    "potential_value", "potential_passed",
    "market_structure_ltf_score", "market_structure_htf_score",
    "structural_score",

    # --- Свечные паттерны и бонусы ---
    "candle_pattern_score", "candle_pattern_passed",
    "candle_pattern_sd_bonus", "candle_pattern_sd_bonus_score",

    # --- Дивергенции ---
    "div_long_total_score", "div_short_total_score",
    "rsi_1h_long_score", "rsi_1h_short_score",
    "rsi_4h_long_score", "rsi_4h_short_score",
    "macd_1h_long_score", "macd_1h_short_score",
    "macd_4h_long_score", "macd_4h_short_score",

    # --- Проходы фильтров (бинарные индикаторы) ---
    "atr_ltf_passed",
    "adx_ltf_passed", "adx_htf_passed",
    "rsi_long_passed", "rsi_short_passed",
    "macd_long", "macd_short",
    "volume_passed",
    "ema_long_passed", "ema_short_passed",
    "trix_long_passed", "trix_short_passed",
    "div_long_passed", "div_short_passed",
    "cdv_ltf_passed", "cdv_htf_passed",
    "potential_passed",
    "market_structure_ltf_passed", "market_structure_htf_passed",
    "candle_pattern_sd_bonus_passed",

    # --- Таймфреймные категориальные фичи (one-hot/label encoding позже) ---
    "trend_global",
    "adx_mode_config",
    "supertrend_mode",
    "structural_mode_config",
    "rsi_1h_long_type", "rsi_1h_short_type", "rsi_4h_long_type", "rsi_4h_short_type",
    "macd_1h_long_type", "macd_1h_short_type", "macd_4h_long_type", "macd_4h_short_type",
    "sd_zone_type",

    # --- Итоговые оценки ---
    "final_scores"
]



# Оставляем только те фичи, что реально есть в таблице при чтении — код ниже проверит


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DEFAULT_DB, help="Path to sqlite DB")
    p.add_argument("--table", default=DEFAULT_TABLE, help="Table name with signals")
    p.add_argument("--out", default=DEFAULT_OUT_MODEL, help="Output model path (.pkl)")
    p.add_argument("--features-file", default=DEFAULT_FEATURES_FILE, help="Save feature order json")
    p.add_argument("--report-file", default=DEFAULT_REPORT_FILE, help="Training report JSON")
    p.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES, help="Min completed signals to train")
    p.add_argument("--test-size", type=float, default=0.2, help="Validation size")
    return p.parse_args()


def read_table(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def build_target(df):
    # priority: TP5 -> TP4 -> TP3 -> TP2 -> TP1 -> STOP
    def get_target(row):
        if int(row.get("stop_loss_hit", 0)) == 1:
            return 0
        if int(row.get("take5_hit", 0)) == 1:
            return 5
        if int(row.get("take4_hit", 0)) == 1:
            return 4
        if int(row.get("take3_hit", 0)) == 1:
            return 3
        if int(row.get("take2_hit", 0)) == 1:
            return 2
        if int(row.get("take1_hit", 0)) == 1:
            return 1
        return -1

    df["target"] = df.apply(get_target, axis=1)
    return df


def summarize_top_symbols(df, min_take_level=3):
    # + список символов, которые чаще всего достигают TP3 и выше
    df_tp3 = df[df["target"] >= min_take_level]
    counter = Counter(df_tp3["symbol"].dropna().values)
    most_common = counter.most_common(30)
    return most_common


def ensure_model_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main():
    args = parse_args()
    print("Loading data from:", args.db, args.table)
    df = read_table(args.db, args.table)
    print("Total rows in table:", len(df))

    # фильтруем завершённые сигналы
    df = df[
        (df.get("take1_hit", 0) == 1) | (df.get("take2_hit", 0) == 1) |
        (df.get("take3_hit", 0) == 1) | (df.get("take4_hit", 0) == 1) |
        (df.get("take5_hit", 0) == 1) | (df.get("stop_loss_hit", 0) == 1)
    ].copy()
    print("Completed signals (with outcome):", len(df))
    if len(df) < args.min_samples:
        print(f"Not enough completed signals (<{args.min_samples}). Exiting.")
        return

    # целевая переменная multiclass
    df = build_target(df)
    df = df[df["target"] >= 0].copy()
    print("After apply target, rows:", len(df))
    print("Target distribution:\n", df["target"].value_counts().sort_index())

    # уточняем реальный список фичей, т.к. в таблице могут отсутствовать некоторые из ML_FEATURE_ORDER
    available_feats = [f for f in ML_FEATURE_ORDER if f in df.columns]
    # дополнительно возьмём несколько числовых полей часто полезных, если есть
    extras = ["atr_ltf_value", "rsi_ltf", "volume_value", "final_scores"]
    for e in extras:
        if e in df.columns and e not in available_feats:
            available_feats.append(e)

    print("Using features:", available_feats)

    X = df[available_feats].fillna(0).astype(float)
    y = df["target"].astype(int)

    # train/val split stratified по target (чтобы сохранить пропорции классов)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )

    print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

    # подготовка модели
    num_classes = int(y.max()) + 1
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
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


    # обучение с ранней остановкой
    eval_set = [(X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[
            early_stopping(stopping_rounds=40),
            log_evaluation(100)
        ]
    )

    # --- важность признаков ---
    try:
        feature_importance = dict(zip(available_feats, model.feature_importances_.tolist()))
    except Exception:
        feature_importance = {}


    # метрики
    # --- Метрики ---
    # Приводим предсказания и цели к numpy-массивам
    y_pred_raw = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    # Приведение к np.ndarray (исправление типизации)
    y_pred = np.asarray(y_pred_raw).astype(int).ravel()
    y_val = np.asarray(y_val).astype(int).ravel()

    # Расчёт метрик
    acc = accuracy_score(y_val, y_pred)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro")

    # Классический отчёт и матрица ошибок
    cls_report = classification_report(y_val, y_pred, digits=4)
    cm = confusion_matrix(y_val, y_pred)

    print("Validation Accuracy:", acc)
    print("Validation Balanced Accuracy:", bal_acc)
    print("Validation F1 (macro):", f1_macro)
    print("Classification report:\n", cls_report)
    print("Confusion matrix:\n", cm)

    # feature importances
    fi = {}
    try:
        importances = model.feature_importances_
        for feat, imp in zip(available_feats, importances):
            fi[feat] = float(imp)
        # sort
        fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        fi = {}

    # топ символов TP3+
    top_symbols = summarize_top_symbols(df, min_take_level=3)

    # save model + feature order + report
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
        "model_path": args.out,
        "feature_importance": feature_importance,
    }

    # ensure report dir
    ensure_model_dir(args.report_file)
    with open(args.report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Model saved to:", args.out)
    print("Feature order saved to:", args.features_file)
    print("Training report saved to:", args.report_file)
    print("Top symbols (TP3+):", top_symbols)
    print("Done.")


if __name__ == "__main__":
    main()