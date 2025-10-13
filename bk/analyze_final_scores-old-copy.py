#!/usr/bin/env python3
import os
import sqlite3
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

# === Настройки ===
DEFAULT_DB = "ml_trading.db"
DEFAULT_TABLE = "ml_trading_data"
DEFAULT_OUT_MODEL = "models/"
MODEL_FILE = os.path.join(DEFAULT_OUT_MODEL, "final_scores_lgbm.txt")
REPORT_FILE = os.path.join(DEFAULT_OUT_MODEL, "final_scores_report.json")
PLOT_FILE = os.path.join(DEFAULT_OUT_MODEL, "final_scores_success_rate.png")
HIST_FILE = os.path.join(DEFAULT_OUT_MODEL, "final_scores_histogram.png")

os.makedirs(DEFAULT_OUT_MODEL, exist_ok=True)

# === Подключение к БД ===
conn = sqlite3.connect(DEFAULT_DB)

query = f"""
SELECT
    final_scores,
    scores_config,
    target,
    take1_profit,
    take2_profit,
    take3_profit,
    take4_profit,
    take5_profit,
    signal_passed
FROM {DEFAULT_TABLE}
WHERE signal_passed = 1 AND final_scores IS NOT NULL
"""
df = pd.read_sql_query(query, conn)
conn.close()

# === Флаг успешной сделки ===
df["success"] = (
    (df["target"] == 1)
    | (df["take1_profit"] > 0)
    | (df["take2_profit"] > 0)
    | (df["take3_profit"] > 0)
    | (df["take4_profit"] > 0)
    | (df["take5_profit"] > 0)
).astype(int)

df = df.dropna(subset=["final_scores", "scores_config"])

if df.empty:
    raise ValueError("Нет данных после фильтрации по signal_passed = 1 и final_scores/scores_config")
if df["success"].sum() == 0:
    raise ValueError("Нет успешных сделок. Проверь критерии успеха.")

# === Базовая статистика ===
success_df = df[df["success"] == 1]
min_score = success_df["final_scores"].min()
max_score = success_df["final_scores"].max()
mean_score = success_df["final_scores"].mean()

print("📊 Базовая статистика по успешным сделкам:")
print(f"  Минимальный final_scores: {min_score:.2f}")
print(f"  Максимальный final_scores: {max_score:.2f}")
print(f"  Средний final_scores: {mean_score:.2f}")

# === Обучение LightGBM ===
X = df[["final_scores", "scores_config"]]
y = df["success"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    random_state=42,
)
model.fit(X_train, y_train)
model.booster_.save_model(MODEL_FILE)

# === Предсказание вероятности успеха ===
y_pred_proba = model.predict_proba(X_test)[:, 1]
results = X_test.copy()
results["prob_success"] = y_pred_proba
recommended_threshold = results.loc[results["prob_success"] > 0.5, "final_scores"].min()
if pd.isna(recommended_threshold):
    recommended_threshold = results["final_scores"].median()
print(f"\n🔮 Рекомендуемый порог final_scores: {recommended_threshold:.2f}")

# === Гистограмма успешных и неуспешных final_scores ===
plt.figure(figsize=(10,6))
plt.hist(df[df['success']==0]['final_scores'], bins=20, alpha=0.5, label='Неуспешные', color='red')
plt.hist(df[df['success']==1]['final_scores'], bins=20, alpha=0.7, label='Успешные', color='green')
plt.title("Распределение final_scores по успешности сделок")
plt.xlabel("final_scores")
plt.ylabel("Количество сигналов")
plt.legend()
plt.tight_layout()
plt.savefig(HIST_FILE, dpi=150)
plt.show()
plt.close()
print(f"📈 Гистограмма сохранена: {HIST_FILE}")

# === Успешность по бакетам ===
df['bucket'] = pd.cut(df['final_scores'], bins=15)
success_rate = df.groupby('bucket')['success'].mean().fillna(0)

names = success_rate.index.astype(str)
values = success_rate.values

plt.figure(figsize=(10,6))
bars = plt.barh(names[::-1], values[::-1])
plt.title("📊 Успешность по диапазонам final_scores")
plt.xlabel("Доля успешных сделок")
plt.ylabel("Диапазон final_scores")
plt.axvline(x=0.5, color='red', linestyle='--', label="Порог вероятности успеха 0.5")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.show()
plt.close()
print(f"📈 Бар диаграмма успешности сохранена: {PLOT_FILE}")

# === Важность признаков ===
feature_importance = {feature: float(imp) for feature, imp in zip(X.columns, model.feature_importances_)}

# === Отчёт ===
report = {
    "min_final_score": round(float(min_score), 4),
    "max_final_score": round(float(max_score), 4),
    "mean_final_score": round(float(mean_score), 4),
    "recommended_threshold": round(float(recommended_threshold), 4),
    "total_samples": len(df),
    "success_samples": int(df["success"].sum()),
    "feature_importance": feature_importance,
    "model_file": MODEL_FILE,
    "plot_file": PLOT_FILE,
    "hist_file": HIST_FILE
}

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=4)

print(f"\n✅ Отчёт сохранён: {REPORT_FILE}")
print(f"✅ Модель сохранена: {MODEL_FILE}")