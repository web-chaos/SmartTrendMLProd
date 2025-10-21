"""
analyze_ml_predictions.py
─────────────────────────
Анализ ML-модели с расширенной визуализацией TP1–TP5 и распределением уверенности.

⚙️ Требования:
    pip install pandas matplotlib tabulate

Автор: ChatGPT (GPT-5)
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import os

DB_PATH = "ml_trading.db"
TABLE_NAME = "ml_trading_data"
REPORT_PATH = "ml_report.txt"

def analyze_ml_predictions():
    if not os.path.exists(DB_PATH):
        print(f"❌ Не найден файл базы данных: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT id, symbol, timestamp,
               take1_hit, take2_hit, take3_hit, take4_hit, take5_hit,
               stop_loss_hit,
               ml_predicted_label, ml_confidence, ml_strength,
               ml_mode, ml_decision, ml_comment, ml_allowed, signal_passed
        FROM {TABLE_NAME}
        WHERE signal_passed = 1 AND ml_allowed = 1
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("⚠️ Нет данных с signal_passed = 1 и ml_allowed = 1.")
        return

    # Фактический исход сделки
    df["actual"] = df.apply(lambda x:
        "STOP" if x.stop_loss_hit == 1 else
        "TP" if any([x.take1_hit, x.take2_hit, x.take3_hit, x.take4_hit, x.take5_hit]) else
        "NONE", axis=1)
    df["predicted"] = df["ml_predicted_label"].fillna("UNKNOWN").str.upper()
    df["is_correct"] = (
        ((df["predicted"].str.startswith("TP")) & (df["actual"] == "TP")) |
        ((df["predicted"] == "STOP") & (df["actual"] == "STOP"))
    )

    # Частичные FN по TP1–TP5
    for i in range(1,6):
        df[f"partial_FN_TP{i}"] = ((df["predicted"]=="STOP") & (df[f"take{i}_hit"]==1))

    df["weighted_score"] = df["is_correct"].astype(float)
    partial_mask = df[[f"partial_FN_TP{i}" for i in range(1,6)]].any(axis=1)
    df.loc[partial_mask, "weighted_score"] = 0.5
    weighted_accuracy = df["weighted_score"].mean() * 100

    tp_preds = df[df["predicted"].str.startswith("TP")]
    stop_preds = df[df["predicted"]=="STOP"]

    stats = {
        "Всего сигналов": len(df),
        "Верных предсказаний": df["is_correct"].sum(),
        "Частичных FN": partial_mask.sum(),
        "Точность модели (%) с учётом частичных FN": round(weighted_accuracy,2),
        "Предсказаний тейков": len(tp_preds),
        "Предсказаний стопов": len(stop_preds),
        "Точность тейков (%)": round((tp_preds["is_correct"].mean() or 0)*100,2),
        "Точность стопов (%)": round((stop_preds["is_correct"].mean() or 0)*100,2),
        "Средняя уверенность (верные)": round(df.loc[df["is_correct"], "ml_confidence"].mean() or 0,3),
        "Средняя уверенность (ошибочные)": round(df.loc[~df["is_correct"], "ml_confidence"].mean() or 0,3)
    }

    errors = df[~df["is_correct"]].head(100)[["symbol","predicted","actual","ml_confidence","ml_comment"]]

    report_lines = ["📊 ОТЧЁТ ПО ML-ПРЕДСКАЗАНИЯМ\n" + "─"*50]
    for k,v in stats.items():
        report_lines.append(f"{k:50}: {v}")
    report_lines.append("\nОшибочные предсказания (первые 10):\n")
    if not errors.empty:
        report_lines.append(tabulate(errors, headers="keys", tablefmt="psql", showindex=False))
    else:
        report_lines.append("Ошибочных предсказаний не обнаружено ✅")

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n📁 Отчёт сохранён в {REPORT_PATH}")

    visualize(df)

def visualize(df):
    plt.figure(figsize=(22,6))

    # 1️⃣ Общая точность
    plt.subplot(1,4,1)
    counts = df["is_correct"].value_counts()
    counts.plot(kind="pie", autopct="%1.1f%%", labels=["Верные","Ошибочные"], ylabel="")
    plt.title("Общая точность модели")

    # 2️⃣ Точность по типам прогнозов
    plt.subplot(1,4,2)
    accuracy_tp = df[df["predicted"].str.startswith("TP")]["is_correct"].mean()*100
    accuracy_stop = df[df["predicted"]=="STOP"]["is_correct"].mean()*100
    plt.bar(["TP","STOP"], [accuracy_tp,accuracy_stop])
    plt.title("Точность по типам прогнозов")
    plt.ylabel("Процент верных")

    # 3️⃣ Частичные FN TP1–TP5
    plt.subplot(1,4,3)
    counts_list = []
    for i in range(1,6):
        correct = ((df["predicted"].str.startswith("TP")) & (df[f"take{i}_hit"]==1)).sum()
        partial = df[f"partial_FN_TP{i}"].sum()
        wrong = len(df) - correct - partial
        counts_list.append([correct, partial, wrong])

    for i, (c,p,w) in enumerate(counts_list):
        plt.bar(f"TP{i+1}", c, color="green")
        plt.bar(f"TP{i+1}", p, bottom=c, color="yellow")
        plt.bar(f"TP{i+1}", w, bottom=c+p, color="red")
    plt.title("Верные / Частичные / Ошибочные TP1–TP5")
    plt.ylabel("Количество сигналов")

    # 4️⃣ Уверенность модели TP1–TP5
    plt.subplot(1,4,4)
    confidence_data = []
    labels = []
    for i in range(1,6):
        # Верные
        correct = df[(df["predicted"].str.startswith("TP")) & (df[f"take{i}_hit"]==1)]["ml_confidence"]
        # Частичные
        partial = df[df[f"partial_FN_TP{i}"]]["ml_confidence"]
        # Ошибочные
        wrong = df[(df["predicted"].str.startswith("TP")) & (df[f"take{i}_hit"]==0)]["ml_confidence"]
        confidence_data.extend([correct, partial, wrong])
        labels.extend([f"TP{i} ✅"]*len(correct) + [f"TP{i} ⚪"]*len(partial) + [f"TP{i} ❌"]*len(wrong))

    if confidence_data:
        conf_df = pd.DataFrame({"TP": labels, "ml_confidence": pd.concat(confidence_data, ignore_index=True)})
        conf_df.boxplot(column="ml_confidence", by="TP", grid=False, rot=45)
        plt.title("Распределение уверенности модели по TP1–TP5")
        plt.suptitle("")
        plt.ylabel("ml_confidence")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_ml_predictions()
