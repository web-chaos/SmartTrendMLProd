import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import sqlite3

# --- Загружаем данные ---
conn = sqlite3.connect("ml_trading.db")
df = pd.read_sql("SELECT * FROM ml_trading_data", conn)
conn.close()

# --- Загружаем модель и порядок фичей ---
model = joblib.load("models/lgbm_v1.pkl")
with open("models/feature_order.json", "r", encoding="utf-8") as f:
    features = json.load(f)

X = df[features]
y = df['target']

# --- 1. Вариация признаков ---
print("=== Вариация признаков ===")
print(X.nunique().sort_values(ascending=True))

# --- 2. Корреляция ---
corr = X.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Корреляция признаков")
plt.show()

# --- 3. Важность признаков по LightGBM ---
importance = model.feature_importances_
feat_imp = pd.DataFrame({"feature": features, "importance": importance})
feat_imp = feat_imp.sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x="importance", y="feature", data=feat_imp)
plt.title("Feature Importance")
plt.show()

print("=== Важность признаков ===")
print(feat_imp)
