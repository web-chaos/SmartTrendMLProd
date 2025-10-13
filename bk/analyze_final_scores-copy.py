#!/usr/bin/env python3
import os
import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# === Настройки ===
DEFAULT_DB = "ml_trading.db"
DEFAULT_TABLE = "ml_trading_data"
DEFAULT_OUT_MODEL = "models/"
MODEL_FILE = os.path.join(DEFAULT_OUT_MODEL, "final_scores_lgbm.txt")
REPORT_FILE = os.path.join(DEFAULT_OUT_MODEL, "final_scores_report.json")
PLOT_FILE = os.path.join(DEFAULT_OUT_MODEL, "final_scores_success_rate.png")
HIST_FILE = os.path.join(DEFAULT_OUT_MODEL, "final_scores_histogram.png")
ROC_FILE = os.path.join(DEFAULT_OUT_MODEL, "roc_curve.png")
PROB_DIST_FILE = os.path.join(DEFAULT_OUT_MODEL, "probability_distribution.png")

os.makedirs(DEFAULT_OUT_MODEL, exist_ok=True)

class TradingModelAnalyzer:
    def __init__(self, db_path, table_name):
        self.db_path = db_path
        self.table_name = table_name
        self.df = None
        self.model = None
        self.report = {}
        
    def load_and_prepare_data(self):
        """Загрузка и подготовка данных с расширенными фичами"""
        conn = sqlite3.connect(self.db_path)
        
        # Расширенный запрос с дополнительными метриками
        query = f"""
        SELECT
            final_scores,
            scores_config,
            target,
            take1_profit, take2_profit, take3_profit, take4_profit, take5_profit,
            signal_passed
        FROM {self.table_name}
        WHERE signal_passed = 1 AND final_scores IS NOT NULL
        """
        
        self.df = pd.read_sql_query(query, conn)
        conn.close()
        
        if self.df.empty:
            raise ValueError("Нет данных после фильтрации по signal_passed = 1 и final_scores")
            
        # Создание расширенных фич
        self._create_advanced_features()
        
        # Определение целевой переменной
        self._define_success_criteria()
        
        print(f"✅ Загружено {len(self.df)} записей")
        print(f"✅ Создано {len(self.df.columns)} признаков")
        
    def _create_advanced_features(self):
        """Создание расширенных признаков для модели"""
        # Базовые преобразования
        self.df["score_diff"] = self.df["final_scores"] - self.df["scores_config"]
        self.df["score_ratio"] = self.df["final_scores"] / (self.df["scores_config"] + 1e-8)
        self.df["score_abs_diff"] = np.abs(self.df["score_diff"])
        
        # Метрики профита
        profit_columns = [f"take{i}_profit" for i in range(1, 6)]
        self.df["max_profit"] = self.df[profit_columns].max(axis=1)
        self.df["min_profit"] = self.df[profit_columns].min(axis=1)
        self.df["avg_profit"] = self.df[profit_columns].mean(axis=1)
        self.df["profit_std"] = self.df[profit_columns].std(axis=1)
        
        # Бинарные признаки
        self.df["high_score_ratio"] = (self.df["score_ratio"] > 1.5).astype(int)
        self.df["positive_score_diff"] = (self.df["score_diff"] > 0).astype(int)
        
    def _define_success_criteria(self):
        """Улучшенное определение успешности сделок"""
        def calculate_success(row):
            # Критерий 1: целевой сигнал
            if row["target"] == 1:
                return 1
            
            # Критерий 2: любой тейк-профит > 0
            takes = [row[f"take{i}_profit"] for i in range(1, 6)]
            if any(take > 0 for take in takes):
                return 1
                
            return 0
        
        self.df["success"] = self.df.apply(calculate_success, axis=1)
        
        success_rate = self.df["success"].mean()
        print(f"✅ Успешных сделок: {self.df['success'].sum()} ({success_rate:.1%})")
        
        # Проверка на сбалансированность данных
        if success_rate == 1.0:
            print("⚠️  ВНИМАНИЕ: Все сделки успешные! Это может повлиять на качество модели.")
        elif success_rate == 0.0:
            raise ValueError("❌ Нет успешных сделок по заданным критериям")
        elif success_rate > 0.8 or success_rate < 0.2:
            print("⚠️  ВНИМАНИЕ: Сильный дисбаланс классов может повлиять на модель")
    
    def analyze_basic_statistics(self):
        """Расширенный анализ базовой статистики"""
        success_df = self.df[self.df["success"] == 1]
        fail_df = self.df[self.df["success"] == 0]
        
        stats = {
            "success_count": len(success_df),
            "fail_count": len(fail_df),
            "success_rate": self.df["success"].mean(),
            "final_scores_overall": {
                "min": float(self.df["final_scores"].min()),
                "max": float(self.df["final_scores"].max()),
                "mean": float(self.df["final_scores"].mean()),
                "std": float(self.df["final_scores"].std())
            }
        }
        
        # Добавляем статистику по успешным, если они есть
        if len(success_df) > 0:
            stats["final_scores_success"] = {
                "min": float(success_df["final_scores"].min()),
                "max": float(success_df["final_scores"].max()),
                "mean": float(success_df["final_scores"].mean()),
                "std": float(success_df["final_scores"].std())
            }
        
        # Добавляем статистику по неуспешным, если они есть
        if len(fail_df) > 0:
            stats["final_scores_fail"] = {
                "min": float(fail_df["final_scores"].min()),
                "max": float(fail_df["final_scores"].max()),
                "mean": float(fail_df["final_scores"].mean()),
                "std": float(fail_df["final_scores"].std())
            }
        
        print("\n📊 РАСШИРЕННАЯ СТАТИСТИКА:")
        print(f"   Общее количество сделок: {len(self.df)}")
        print(f"   Успешных: {stats['success_count']} ({stats['success_rate']:.1%})")
        print(f"   Неуспешных: {stats['fail_count']}")
        print(f"   Final_scores общие: {stats['final_scores_overall']['mean']:.2f} ± {stats['final_scores_overall']['std']:.2f}")
        
        if 'final_scores_success' in stats:
            print(f"   Final_scores успешных: {stats['final_scores_success']['mean']:.2f} ± {stats['final_scores_success']['std']:.2f}")
        if 'final_scores_fail' in stats:
            print(f"   Final_scores неуспешных: {stats['final_scores_fail']['mean']:.2f} ± {stats['final_scores_fail']['std']:.2f}")
        
        self.report["basic_statistics"] = stats
        return stats
    
    def train_model(self):
        """Обучение модели с расширенной валидацией"""
        # Выбор признаков
        feature_columns = ["final_scores", "scores_config", "score_diff", "score_ratio", 
                          "max_profit", "avg_profit", "high_score_ratio", "positive_score_diff"]
        
        # Убедимся, что все признаки существуют
        available_features = [col for col in feature_columns if col in self.df.columns]
        X = self.df[available_features]
        y = self.df["success"]
        
        # Проверяем, есть ли оба класса
        if len(np.unique(y)) < 2:
            print("⚠️  Только один класс в данных. Используем упрощенный анализ.")
            return self._handle_single_class_case(X, y)
        
        # Стратифицированное разделение
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Настройка модели LightGBM
        self.model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=100,  # Уменьшили для скорости
            learning_rate=0.05,
            max_depth=3,  # Уменьшили для предотвращения переобучения
            num_leaves=10,
            random_state=42,
            verbosity=-1
        )
        
        print(f"\n🎯 ОБУЧЕНИЕ МОДЕЛИ:")
        print(f"   Признаки: {', '.join(available_features)}")
        print(f"   Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        try:
            # Обучение с обработкой разных версий LightGBM
            self.model.fit(X_train, y_train)
            
            # Кросс-валидация
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Уменьшили для скорости
            cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='roc_auc')
            
            # Предсказания
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Метрики
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            
            # Сохранение модели
            self.model.booster_.save_model(MODEL_FILE)
            
            # Анализ порогов
            optimal_threshold = self._analyze_thresholds(X_test, y_test, y_pred_proba)
            
            model_metrics = {
                "cv_roc_auc_mean": float(cv_scores.mean()),
                "cv_roc_auc_std": float(cv_scores.std()),
                "test_roc_auc": float(roc_auc),
                "optimal_threshold": float(optimal_threshold),
                "feature_importance": dict(zip(available_features, [float(x) for x in self.model.feature_importances_])),
                "classification_report": classification_rep
            }
            
            print(f"   CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"   Test ROC-AUC: {roc_auc:.3f}")
            print(f"   Оптимальный порог: {optimal_threshold:.3f}")
            
            self.report["model_metrics"] = model_metrics
            self.X_test = X_test
            self.y_test = y_test
            self.y_pred_proba = y_pred_proba
            
            return model_metrics
            
        except Exception as e:
            print(f"⚠️  Ошибка при обучении модели: {e}")
            print("🔄 Используем упрощенный анализ...")
            return self._handle_single_class_case(X, y)
    
    def _handle_single_class_case(self, X, y):
        """Обработка случая с одним классом"""
        success_rate = y.mean()
        
        # Простой анализ на основе перцентилей
        if success_rate == 1.0:
            # Все сделки успешные - берем минимальный final_scores
            threshold = X["final_scores"].min()
            recommendation = "Все сделки успешны. Используйте минимальное значение final_scores как порог."
        else:
            # Все сделки неуспешные - берем медиану
            threshold = X["final_scores"].median()
            recommendation = "Все сделки неуспешны. Рекомендуется пересмотреть стратегию."
        
        model_metrics = {
            "cv_roc_auc_mean": 1.0 if success_rate == 1.0 else 0.0,
            "cv_roc_auc_std": 0.0,
            "test_roc_auc": 1.0 if success_rate == 1.0 else 0.0,
            "optimal_threshold": float(threshold),
            "feature_importance": {"final_scores": 1.0},
            "classification_report": {"accuracy": success_rate},
            "special_case": True,
            "recommendation": recommendation
        }
        
        print(f"   📍 Специальный случай: {recommendation}")
        print(f"   📊 Порог: {threshold:.3f}")
        
        self.report["model_metrics"] = model_metrics
        return model_metrics
    
    def _analyze_thresholds(self, X_test, y_test, y_pred_proba):
        """Анализ оптимальных порогов"""
        try:
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            
            # F1-score для каждого порога
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores[:-1])  # Исключаем последний элемент
            optimal_threshold_prob = thresholds[optimal_idx]
            
            # Рекомендуемый порог на основе final_scores
            results = X_test.copy()
            results["prob_success"] = y_pred_proba
            recommended_threshold = results.loc[results["prob_success"] > optimal_threshold_prob, "final_scores"]
            
            if not recommended_threshold.empty:
                final_threshold = recommended_threshold.min()
            else:
                final_threshold = results["final_scores"].median()
                
            return final_threshold
            
        except Exception as e:
            print(f"⚠️  Ошибка в анализе порогов: {e}")
            return X_test["final_scores"].median()
    
    def create_visualizations(self):
        """Создание расширенных визуализаций"""
        plt.style.use('default')
        
        try:
            # 1. Гистограмма распределения
            self._create_histogram()
            
            # 2. Успешность по бакетам (только если есть оба класса)
            if len(np.unique(self.df["success"])) > 1:
                self._create_success_rate_plot()
            else:
                self._create_single_class_plot()
            
            # 3. ROC-кривая (только если есть оба класса)
            if hasattr(self, 'y_test') and len(np.unique(self.y_test)) > 1:
                self._create_roc_curve()
            
            # 4. Распределение final_scores
            self._create_final_scores_distribution()
            
        except Exception as e:
            print(f"⚠️  Ошибка при создании визуализаций: {e}")
    
    def _create_histogram(self):
        """Гистограмма распределения final_scores"""
        plt.figure(figsize=(12, 8))
        
        if len(np.unique(self.df["success"])) > 1:
            success_data = self.df[self.df['success'] == 1]['final_scores']
            fail_data = self.df[self.df['success'] == 0]['final_scores']
            
            plt.hist(fail_data, bins=20, alpha=0.6, label='Неуспешные', color='red', density=True)
            plt.hist(success_data, bins=20, alpha=0.7, label='Успешные', color='green', density=True)
        else:
            # Если только один класс
            plt.hist(self.df['final_scores'], bins=20, alpha=0.7, color='blue', density=True)
            success_rate = self.df["success"].mean()
            label = 'Все успешные' if success_rate == 1.0 else 'Все неуспешные'
            plt.text(0.7, 0.9, label, transform=plt.gca().transAxes, fontsize=12)
        
        plt.title("Распределение final_scores", fontsize=14, fontweight='bold')
        plt.xlabel("final_scores", fontsize=12)
        plt.ylabel("Плотность", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(HIST_FILE, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Гистограмма сохранена: {HIST_FILE}")
    
    def _create_success_rate_plot(self):
        """Успешность по диапазонам final_scores"""
        self.df['bucket'] = pd.cut(self.df['final_scores'], bins=10)  # Уменьшили количество бинов
        success_rate = self.df.groupby('bucket')['success'].agg(['mean', 'count']).fillna(0)
        
        plt.figure(figsize=(12, 8))
        
        # Основной график успешности
        bars = plt.barh(range(len(success_rate)), success_rate['mean'][::-1])
        
        # Цвета в зависимости от успешности
        for i, (idx, row) in enumerate(success_rate[::-1].iterrows()):
            color = 'green' if row['mean'] > 0.5 else 'red'
            bars[i].set_color(color)
            plt.text(row['mean'] + 0.01, i, f"n={int(row['count'])}", 
                    va='center', fontsize=9)
        
        plt.yticks(range(len(success_rate)), [str(x) for x in success_rate.index[::-1]])
        plt.axvline(x=0.5, color='blue', linestyle='--', linewidth=2, label="Порог 50%")
        plt.title("Успешность сделок по диапазонам final_scores", fontsize=14, fontweight='bold')
        plt.xlabel("Доля успешных сделок", fontsize=12)
        plt.ylabel("Диапазон final_scores", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 График успешности сохранен: {PLOT_FILE}")
    
    def _create_single_class_plot(self):
        """График для случая с одним классом"""
        plt.figure(figsize=(12, 8))
        
        success_rate = self.df["success"].mean()
        if success_rate == 1.0:
            color = 'green'
            title = "Все сделки успешные - распределение final_scores"
        else:
            color = 'red'
            title = "Все сделки неуспешные - распределение final_scores"
        
        plt.hist(self.df['final_scores'], bins=20, alpha=0.7, color=color, density=True)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("final_scores", fontsize=12)
        plt.ylabel("Плотность", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Специальный график сохранен: {PLOT_FILE}")
    
    def _create_roc_curve(self):
        """ROC-кривая"""
        if not hasattr(self, 'y_test') or len(np.unique(self.y_test)) < 2:
            return
            
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Классификатор успешности сделок', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(ROC_FILE, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📉 ROC-кривая сохранена: {ROC_FILE}")
    
    def _create_final_scores_distribution(self):
        """Распределение final_scores с порогом"""
        plt.figure(figsize=(12, 8))
        
        threshold = self.report.get('model_metrics', {}).get('optimal_threshold', self.df['final_scores'].median())
        
        plt.hist(self.df['final_scores'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Рекомендуемый порог: {threshold:.2f}')
        
        plt.title("Распределение final_scores с рекомендуемым порогом", fontsize=14, fontweight='bold')
        plt.xlabel("final_scores", fontsize=12)
        plt.ylabel("Количество сделок", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        dist_file = os.path.join(DEFAULT_OUT_MODEL, "final_scores_threshold.png")
        plt.savefig(dist_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Распределение с порогом сохранено: {dist_file}")
    
    def generate_report(self):
        """Генерация расширенного отчета"""
        final_report = {
            "analysis_date": datetime.now().isoformat(),
            "dataset_info": {
                "total_samples": len(self.df),
                "success_samples": int(self.df["success"].sum()),
                "success_rate": float(self.df["success"].mean()),
                "features_used": list(self.df.columns) if hasattr(self, 'df') else []
            },
            **self.report,
            "files_generated": {
                "model": MODEL_FILE,
                "report": REPORT_FILE,
                "success_rate_plot": PLOT_FILE,
                "histogram": HIST_FILE
            },
            "recommendations": self._generate_recommendations()
        }
        
        # Добавляем дополнительные файлы, если они созданы
        if os.path.exists(ROC_FILE):
            final_report["files_generated"]["roc_curve"] = ROC_FILE
        
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=4)
        
        print(f"\n✅ Отчёт сохранён: {REPORT_FILE}")
        return final_report
    
    def _generate_recommendations(self):
        """Генерация рекомендаций на основе анализа"""
        success_rate = self.df["success"].mean()
        threshold = self.report.get('model_metrics', {}).get('optimal_threshold', self.df['final_scores'].median())
        
        recommendations = []
        
        if success_rate == 1.0:
            recommendations.append("🎉 ВСЕ СДЕЛКИ УСПЕШНЫЕ! Текущая стратегия показывает идеальные результаты.")
            recommendations.append(f"📊 Минимальный final_scores: {self.df['final_scores'].min():.2f}")
            recommendations.append(f"📈 Средний final_scores: {self.df['final_scores'].mean():.2f}")
        elif success_rate == 0.0:
            recommendations.append("❌ ВСЕ СДЕЛКИ НЕУСПЕШНЫЕ! Срочно пересмотрите торговую стратегию.")
            recommendations.append("🔍 Проанализируйте критерии входа и параметры индикаторов.")
        else:
            if success_rate > 0.7:
                recommendations.append("📈 Высокий процент успешных сделок. Стратегия эффективна.")
            elif success_rate > 0.5:
                recommendations.append("📊 Умеренный процент успешных сделок. Есть потенциал для улучшения.")
            else:
                recommendations.append("⚠️ Низкий процент успешных сделок. Рекомендуется оптимизация стратегии.")
            
            recommendations.append(f"🎯 Рекомендуемый порог final_scores: {threshold:.2f}")
        
        # Общие рекомендации
        recommendations.append("💡 Рекомендация: тестируйте стратегию на новых данных для проверки устойчивости.")
        
        return recommendations
    
    def run_full_analysis(self):
        """Запуск полного анализа"""
        print("🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА ТОРГОВЫХ СДЕЛОК")
        print("=" * 50)
        
        try:
            self.load_and_prepare_data()
            self.analyze_basic_statistics()
            self.train_model()
            self.create_visualizations()
            report = self.generate_report()
            
            print("\n" + "=" * 50)
            print("✅ АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
            print("\n📋 ОСНОВНЫЕ ВЫВОДЫ:")
            for rec in report["recommendations"]:
                print(f"   • {rec}")
                
            return report
            
        except Exception as e:
            print(f"❌ Ошибка при анализе: {e}")
            raise

# === Запуск анализа ===
if __name__ == "__main__":
    analyzer = TradingModelAnalyzer(DEFAULT_DB, DEFAULT_TABLE)
    report = analyzer.run_full_analysis()