import json
import matplotlib.pyplot as plt

# Путь к сохранённым данным
FEATURE_FILE = "models/feature_order_binary.json"
REPORT_FILE = "models/training_report_binary.json"

# Загружаем важности признаков
try:
    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        report = json.load(f)
    importance = report.get("feature_importance", {})
except Exception as e:
    print(f"❌ Не удалось загрузить feature importance: {e}")
    importance = {}

if not importance:
    print("⚠️ В файле отчёта нет данных по важности признаков.")
    exit()

# Преобразуем в список и сортируем
features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# Берём топ-20 признаков
top_n = 20
top_features = features[:top_n]
names = [f[0] for f in top_features]
values = [f[1] for f in top_features]

# Рисуем график
plt.figure(figsize=(10, 6))
plt.barh(names[::-1], values[::-1])
plt.title(f"📊 Топ-{top_n} признаков по важности (LightGBM binary)")
plt.xlabel("Важность признака")
plt.ylabel("Название признака")
plt.tight_layout()
plt.show()
