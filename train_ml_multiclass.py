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
from sklearn.preprocessing import LabelEncoder

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

ML_FEATURE_ORDER1 = [
    # --- Лидеры ---
    "cdv_ltf_value",              # локальный объём CDV — главный
    "adx_htf_value",              # сила тренда по старшему ТФ
    "trix_htf",                   # долгосрочный импульс
    "macd_momentum",              # импульс MACD (основной драйвер)
    "rsi_htf",                    # RSI на старшем ТФ
    "stoch_ltf",                  # стохастик на младшем
    "volume_normalized_body",     # нормализованный объём тела свечи
    "final_scores",               # итоговый скоринг по фильтрам
    "macd_htf_diff",              # расхождение MACD старшего
    "volume_delta",               # изменение объёма

    # --- Средняя группа ---
    "cdv_ltf_volume",             # объём CDV младшего ТФ
    "macd_htf_momentum",          # импульс MACD старшего
    "volume_value",               # чистое значение объёма
    "adx_ltf_value",              # сила тренда младшего ТФ
    "trix_ltf",                   # импульс TRIX младшего ТФ
    "macd_current_diff",          # текущий дифф MACD
    "potential_value",            # потенциал сигнала
    "rsi_prev_htf",               # предыдущее RSI на старшем ТФ
    "atr_ltf_value",              # волатильность младшего ТФ
    "rsi_ltf",                    # RSI младшего ТФ

    # --- Дополнительные важные ---
    "macd_prev_diff",             # предыдущий дифф MACD
    "stoch_htf",                  # стохастик старшего ТФ
    "stoch_prev_htf",             # предыдущее значение стохастика старшего
    "volume_mean",                # средний объём

    # --- Derived / новые признаки ---
    "adx_htf_diff",               # adx_htf_value - adx_ltf_value
    "macd_momentum_diff",         # macd_momentum - macd_htf_momentum
    "rsi_delta_ltf",              # rsi_ltf - rsi_prev_ltf
    "rsi_delta_htf",              # rsi_htf - rsi_prev_htf
    "stoch_delta_ltf",            # stoch_ltf - stoch_prev_ltf
    "trix_delta_htf",             # trix_htf - prev_trix_htf
    "trix_delta_ltf",             # trix_ltf - prev_trix_ltf
    "volume_ratio",               # volume_value / volume_mean
    "candle_struct_score",        # candle_pattern_score + structural_score
    "supertrend_score_total",     # supertrend_ltf_score + supertrend_htf_score
    "cdv_score_total",            # cdv_ltf_score + cdv_htf_score
    "macd_score_total",           # macd_score_long + macd_score_short
    "rsi_score_total",            # rsi_long_score + rsi_short_score
    "trend_up",                   # 1 если adx_htf_value > adx_ltf_value
    "volume_spike",               # 1 если volume_value > volume_mean * 1.2
    "rsi_overbought",             # 1 если rsi_htf > 70
    "rsi_oversold",               # 1 если rsi_htf < 30
]

ML_FEATURE_ORDER = [
    # --- Лидеры ---
    "oi_value",                   # открытый интерес (OI)
    "cdv_ltf_value",              # локальный объём CDV — младший ТФ
    "cdv_htf_value",              # CDV по старшему ТФ
    "adx_ltf_value",              # сила тренда младшего ТФ
    "adx_htf_value",              # сила тренда старшего ТФ
    "trix_ltf",                   # импульс TRIX младшего ТФ
    "trix_htf",                   # импульс TRIX старшего ТФ
    "macd_momentum",              # импульс MACD
    "macd_htf_momentum",          # импульс MACD старшего ТФ
    "macd_current_diff",          # текущий дифф MACD

    # --- Средняя группа ---
    "volume_value",               # объём свечи
    "volume_delta",               # изменение объёма
    "volume_mean",                # средний объём
    "volume_mean_long",           # средний объём по длинному окну
    "volume_normalized_body",     # нормализованный объём тела свечи
    "potential_value",            # потенциал сигнала
    "atr_ltf_value",              # волатильность младшего ТФ
    "rsi_ltf",                    # RSI младшего ТФ
    "rsi_htf",                    # RSI старшего ТФ
    "rsi_prev_ltf",               # предыдущее RSI младшего
    "rsi_prev_htf",               # предыдущее RSI старшего
    "stoch_ltf",                  # стохастик младшего ТФ
    "stoch_htf",                  # стохастик старшего ТФ
    "stoch_prev_ltf",             # предыдущее значение стохастика младшего
    "stoch_prev_htf",             # предыдущее значение стохастика старшего

    # --- Прошлые и derived признаки ---
    "macd_htf_diff",              # расхождение MACD старшего
    "macd_prev_diff",             # предыдущий MACD дифф
    "macd_prev_htf_diff",         # предыдущий MACD дифф старшего ТФ
    "prev_trix_ltf",              # предыдущий TRIX младшего
    "prev_trix_htf",              # предыдущий TRIX старшего
    "trix_momentum_ltf",          # TRIX моментум младшего
    "trix_momentum_htf",          # TRIX моментум старшего
    "stoch_delta_ltf",            # разница стохастика младшего
    "stoch_delta_htf",            # разница стохастика старшего
    "rsi_delta_ltf",              # изменение RSI младшего
    "rsi_delta_htf",              # изменение RSI старшего
    "ema_diff_ltf",               # разница EMA младшего ТФ
    "ema_diff_htf",               # разница EMA старшего ТФ
    "ema_fast",                   # быстрая EMA
    "ema_slow",                   # медленная EMA

    # --- Логические сигналы и флаги ---
    "atr_ltf_passed",
    "adx_ltf_passed",
    "adx_htf_passed",
    "supertrend_ltf_passed",
    "supertrend_htf_passed",
    "cdv_ltf_passed",
    "cdv_htf_passed",
    "potential_passed",
    "market_structure_ltf_passed",
    "market_structure_htf_passed",
    "candle_pattern_passed",
    "candle_pattern_sd_bonus_passed",
    "structural_passed",
    "macd_early_bullish",
    "macd_early_bearish",
    "macd_long",
    "macd_short",
    "volume_passed",
    "early_trix_bullish",
    "early_trix_bearish",
    "trix_classic_long",
    "trix_classic_short",
    "trix_long_passed",
    "trix_short_passed",
    "stoch_long_passed",
    "stoch_short_passed",
    "rsi_long_passed",
    "rsi_short_passed",
    "ema_long_passed",
    "ema_short_passed",
    "div_long_passed",
    "div_short_passed",
    "final_scores",


    # --- Индексы и дивергенции ---
    "div_long_total_score",       # общий счёт дивергенции (без конфигов)
    "div_short_total_score",      # общий счёт дивергенции (без конфигов)
    "rsi_1h_long_score",
    "rsi_1h_short_score",
    "rsi_4h_long_score",
    "rsi_4h_short_score",
    "macd_1h_long_score",
    "macd_1h_short_score",
    "macd_4h_long_score",
    "macd_4h_short_score",
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


def build_target2(df):
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

def build_target(df):
    def get_target(row):
        if int(row.get("stop_loss_hit", 0)) == 1:
            return 0  # STOP
        if int(row.get("take5_hit", 0)) == 1:
            return 3  # TP_HIGH (объединяем TP4+TP5)
        if int(row.get("take4_hit", 0)) == 1:
            return 3  # TP_HIGH
        if int(row.get("take3_hit", 0)) == 1:
            return 2  # TP_MEDIUM
        if int(row.get("take2_hit", 0)) == 1:
            return 2  # TP_MEDIUM
        if int(row.get("take1_hit", 0)) == 1:
            return 1  # TP_LOW
        return -1

    df["target"] = df.apply(get_target, axis=1)
    return df


def summarize_top_symbols(df, min_take_level=3, min_hits=2):
    """
    Возвращает список символов, которые достигали TP3+ минимум min_hits раз.
    """
    df_tp3 = df[df["target"] >= min_take_level]
    counter = Counter(df_tp3["symbol"].dropna().values)
    # фильтруем только тех, кто сделал это >= min_hits раз
    filtered = [(sym, count) for sym, count in counter.items() if count >= min_hits]
    # сортируем по частоте
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered


def ensure_model_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main():
    args = parse_args()
    print("Loading data from:", args.db, args.table)
    df = read_table(args.db, args.table)
    print("Total rows in table:", len(df))

    #TODO: вернуть 5 тейков, раскоментировать
    # фильтруем завершённые сигналы
    df = df[
        (df.get("take1_hit", 0) == 1) | 
        (df.get("take2_hit", 0) == 1) |
        (df.get("take3_hit", 0) == 1) | 
        (df.get("take4_hit", 0) == 1) |
        (df.get("take5_hit", 0) == 1) | 
        (df.get("stop_loss_hit", 0) == 1)
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

    # дополнительно добавляем часто полезные числовые поля
    # extras = ["atr_ltf_value", "rsi_ltf", "volume_value", "final_scores"]
    # available_feats.extend([e for e in extras if e in df.columns])

    # --- удаляем дубликаты, сохраняя порядок ---
    seen = set()
    available_feats_unique = []
    for f in available_feats:
        if f not in seen:
            available_feats_unique.append(f)
            seen.add(f)
    available_feats = available_feats_unique

    print("Using features:", available_feats)

    # --- Категориальные колонки, которые нужно закодировать ---
    # categorical_cols = [
    #     "trend_global",
    #     "adx_mode_config",
    #     "supertrend_mode",
    #     "structural_mode_config",
    #     "rsi_1h_long_type", "rsi_1h_short_type", "rsi_4h_long_type", "rsi_4h_short_type",
    #     "macd_1h_long_type", "macd_1h_short_type", "macd_4h_long_type", "macd_4h_short_type",
    #     "sd_zone_type"
    # ]

    categorical_cols = [
        "symbol",                   # актив/валюта
        "signal_ltf",               # сигнал младшего ТФ
        "signal_htf",               # сигнал старшего ТФ
        # "trend_global",             
        "adx_mode_config",
        "supertrend_mode",
        "structural_mode_config",
        "rsi_1h_long_type",
        "rsi_1h_short_type",
        "rsi_4h_long_type",
        "rsi_4h_short_type",
        "macd_1h_long_type",
        "macd_1h_short_type",
        "macd_4h_long_type",
        "macd_4h_short_type",
        "sd_zone_type",
        "div_tf_config"
    ]

    # !!!
    if "trend_global" in available_feats:
        df["trend_global"] = df["trend_global"].map({"long": 1, "short": -1}).fillna(0)


    for col in categorical_cols:
        if col in available_feats:
            df[col] = df[col].fillna("NA")  # заменяем пустые на "NA"
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            
            # безопасный вывод для Pylance
            mapping = {str(cls): idx for idx, cls in enumerate(le.classes_)}
            print(f"{col} mapping:", mapping)

        # Очистка только НЕкатегориальных колонок от нечисловых значений
        non_categorical_feats = [col for col in available_feats if col not in categorical_cols]

        for col in non_categorical_feats:
            # Проверяем, есть ли в колонке строковые или байтовые значения
            if df[col].dtype == object:
                # Заменяем нечисловые значения на 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Если есть байтовые значения, конвертируем их в числовые
            if df[col].dtype == object and any(isinstance(x, bytes) for x in df[col].dropna()):
                df[col] = df[col].apply(lambda x: float(int.from_bytes(x, 'little')) if isinstance(x, bytes) else x)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    X = df[available_feats].fillna(0).astype(float)
    y = df["target"].astype(int)

    # train/val split stratified по target (чтобы сохранить пропорции классов)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )

    print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

    # подготовка модели
    # num_classes = int(y.max()) + 1

    def get_auto_model(X, y, RANDOM_STATE=42):
        num_samples = len(X)
        num_classes = len(set(y))

        print(f"[INFO] Обнаружено {num_samples} обучающих примеров. Определяем оптимальную модель...")

        # === до 500 сигналов ===
        if num_samples < 1000:
            print("[MODEL] Выбрана лёгкая модель: model_learn (малый объём данных)")
            model_name = "model_learn"
            model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=num_classes,
                n_estimators=50,
                learning_rate=0.2,
                max_depth=3,
                num_leaves=6,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )

            return model, num_classes, model_name

        # === от 500 до 2000 ===
        elif num_samples < 2000:
            print("[MODEL] Выбрана сбалансированная модель: model_balanse (оптимум по данным)")
            model_name = "model_balanse"
            model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=num_classes,
                n_estimators=300,
                learning_rate=0.1,
                max_depth=5,
                num_leaves=20,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=0.5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            return model, num_classes, model_name

        # === свыше 2000 ===
        else:
            print("[MODEL] Выбрана продвинутая модель: model_base (много данных, высокая точность)")
            model_name = "model_base"
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
        
            return model, num_classes, model_name


    model, num_classes, model_name = get_auto_model(X_train, y_train)
    #model.fit(X_train, y_train)

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
    top_symbols_stable = summarize_top_symbols(df, min_take_level=3, min_hits=3)


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
        "top_symbols_tp3_stable": top_symbols_stable,
        "model_path": args.out,
        "feature_importance": feature_importance,
    }

    # ensure report dir
    ensure_model_dir(args.report_file)
    with open(args.report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Selected model:", model_name)
    print("Model saved to:", args.out)
    print("Feature order saved to:", args.features_file)
    print("Training report saved to:", args.report_file)
    print("Top symbols (TP3+):", top_symbols)
    print("Top symbols (TP3+) stable:", top_symbols_stable)
    print("Done.")


if __name__ == "__main__":
    main()