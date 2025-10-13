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
STOP_LOSS_ANALYSIS_FILE = os.path.join(DEFAULT_OUT_MODEL, "stop_loss_analysis.png")
THRESHOLD_OPTIMIZATION_FILE = os.path.join(DEFAULT_OUT_MODEL, "threshold_optimization.png")
PERFORMANCE_ANALYSIS_FILE = os.path.join(DEFAULT_OUT_MODEL, "performance_analysis.png")

os.makedirs(DEFAULT_OUT_MODEL, exist_ok=True)

class TradingModelAnalyzer:
    def __init__(self, db_path, table_name):
        self.db_path = db_path
        self.table_name = table_name
        self.df = None
        self.model = None
        self.report = {}
        
    def get_table_columns(self):
        """Получить список всех колонок в таблице"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({self.table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()
        return columns
    
    def load_and_prepare_data(self):
        """Загрузка и подготовка данных с учетом стоп-лоссов"""
        # Получаем все доступные колонки
        all_columns = self.get_table_columns()
        print(f"📋 Доступные колонки в таблице: {', '.join(all_columns[:10])}...")  # Показываем первые 10
        
        # Базовые обязательные колонки
        required_columns = ['final_scores', 'scores_config', 'target', 'signal_passed', 'stop_loss_hit']
        
        # Проверяем наличие обязательных колонок
        missing_columns = [col for col in required_columns if col not in all_columns]
        if missing_columns:
            raise ValueError(f"❌ Отсутствуют обязательные колонки: {missing_columns}")
        
        # Ищем колонки take_profit
        take_profit_columns = [col for col in all_columns if col.startswith('take') and 'profit' in col]
        print(f"💰 Найдены колонки профитов: {take_profit_columns}")
        
        # Формируем запрос
        selected_columns = required_columns + take_profit_columns
        query = f"""
        SELECT {', '.join(selected_columns)}
        FROM {self.table_name}
        WHERE signal_passed = 1 AND final_scores IS NOT NULL
        """
        
        conn = sqlite3.connect(self.db_path)
        self.df = pd.read_sql_query(query, conn)
        conn.close()
        
        if self.df.empty:
            raise ValueError("Нет данных после фильтрации по signal_passed = 1 и final_scores")
            
        # Создание расширенных фич
        self._create_advanced_features()
        
        # Определение целевой переменной с учетом стоп-лоссов
        self._define_success_criteria()
        
        print(f"✅ Загружено {len(self.df)} записей")
        print(f"✅ Создано {len(self.df.columns)} признаков")
        print(f"✅ Сделок со стоп-лоссом: {self.df['stop_loss_hit'].sum()} ({self.df['stop_loss_hit'].mean():.1%})")
        
    def _create_advanced_features(self):
        """Создание расширенных признаков для модели"""
        # Базовые преобразования
        self.df["score_diff"] = self.df["final_scores"] - self.df["scores_config"]
        self.df["score_ratio"] = self.df["final_scores"] / (self.df["scores_config"] + 1e-8)
        self.df["score_abs_diff"] = np.abs(self.df["score_diff"])
        
        # Метрики профита (используем только существующие колонки)
        profit_columns = [col for col in self.df.columns if col.startswith('take') and 'profit' in col]
        if profit_columns:
            print(f"📊 Используем колонки профитов: {profit_columns}")
            self.df["max_profit"] = self.df[profit_columns].max(axis=1)
            self.df["min_profit"] = self.df[profit_columns].min(axis=1)
            self.df["avg_profit"] = self.df[profit_columns].mean(axis=1)
            self.df["profit_std"] = self.df[profit_columns].std(axis=1)
            self.df["total_profit_potential"] = self.df[profit_columns].sum(axis=1)
        
        # Бинарные признаки
        self.df["high_score_ratio"] = (self.df["score_ratio"] > 1.5).astype(int)
        self.df["positive_score_diff"] = (self.df["score_diff"] > 0).astype(int)
        self.df["very_high_score"] = (self.df["final_scores"] > 9.5).astype(int)
        
    def _define_success_criteria(self):
        """Улучшенное определение успешности сделок с учетом стоп-лоссов"""
        def calculate_success(row):
            # Критерий 1: целевой сигнал И отсутствие стоп-лосса
            if row["target"] == 1 and row["stop_loss_hit"] == 0:
                return 1
            
            # Критерий 2: любой тейк-профит > 0 И отсутствие стоп-лосса
            profit_columns = [col for col in row.index if col.startswith('take') and 'profit' in col]
            takes = [row[col] for col in profit_columns]
            if any(take > 0 for take in takes) and row["stop_loss_hit"] == 0:
                return 1
                
            return 0
        
        self.df["success"] = self.df.apply(calculate_success, axis=1)
        
        success_rate = self.df["success"].mean()
        print(f"✅ Успешных сделок (без стоп-лоссов): {self.df['success'].sum()} ({success_rate:.1%})")
        
        # Проверка на сбалансированность данных
        if success_rate == 1.0:
            print("⚠️  ВНИМАНИЕ: Все сделки успешные! Это может повлиять на качество модели.")
        elif success_rate == 0.0:
            raise ValueError("❌ Нет успешных сделок по заданным критериям")
        elif success_rate > 0.8 or success_rate < 0.2:
            print("⚠️  ВНИМАНИЕ: Сильный дисбаланс классов может повлиять на модель")
    
    def analyze_stop_loss_patterns(self):
        """Анализ паттернов стоп-лоссов"""
        print("\n🔍 АНАЛИЗ СТОП-ЛОССОВ:")
        
        stop_loss_df = self.df[self.df["stop_loss_hit"] == 1]
        no_stop_loss_df = self.df[self.df["stop_loss_hit"] == 0]
        
        if len(stop_loss_df) == 0:
            print("   Нет данных о стоп-лоссах для анализа")
            return self.df["final_scores"].median(), {}
        
        stop_loss_stats = {
            "stop_loss_count": len(stop_loss_df),
            "stop_loss_rate": len(stop_loss_df) / len(self.df),
            "final_scores_stop_loss": {
                "min": float(stop_loss_df["final_scores"].min()),
                "max": float(stop_loss_df["final_scores"].max()),
                "mean": float(stop_loss_df["final_scores"].mean()),
                "std": float(stop_loss_df["final_scores"].std())
            },
            "final_scores_no_stop_loss": {
                "min": float(no_stop_loss_df["final_scores"].min()),
                "max": float(no_stop_loss_df["final_scores"].max()),
                "mean": float(no_stop_loss_df["final_scores"].mean()),
                "std": float(no_stop_loss_df["final_scores"].std())
            }
        }
        
        print(f"   Сделок со стоп-лоссом: {stop_loss_stats['stop_loss_count']} ({stop_loss_stats['stop_loss_rate']:.1%})")
        print(f"   Final_scores при стоп-лоссе: {stop_loss_stats['final_scores_stop_loss']['mean']:.2f} ± {stop_loss_stats['final_scores_stop_loss']['std']:.2f}")
        print(f"   Final_scores без стоп-лосса: {stop_loss_stats['final_scores_no_stop_loss']['mean']:.2f} ± {stop_loss_stats['final_scores_no_stop_loss']['std']:.2f}")
        
        # Анализ оптимального порога для минимизации стоп-лоссов
        optimal_threshold = self._optimize_threshold_for_stop_loss()
        
        self.report["stop_loss_analysis"] = stop_loss_stats
        return optimal_threshold, stop_loss_stats
    
    def _optimize_threshold_for_stop_loss(self):
        """Оптимизация порога для минимизации стоп-лоссов"""
        if self.df["stop_loss_hit"].sum() == 0:
            print("   Нет данных о стоп-лоссах для оптимизации порога")
            return self.df["final_scores"].median()
        
        thresholds = np.linspace(self.df["final_scores"].min(), self.df["final_scores"].max(), 50)
        results = []
        
        for threshold in thresholds:
            filtered_df = self.df[self.df["final_scores"] >= threshold]
            if len(filtered_df) == 0:
                continue
                
            stop_loss_rate = filtered_df["stop_loss_hit"].mean()
            success_rate = filtered_df["success"].mean()
            signals_remaining = len(filtered_df) / len(self.df)
            
            # Композитный score с приоритетом на уменьшение стоп-лоссов
            score = success_rate * (1 - stop_loss_rate) * np.sqrt(signals_remaining)
            
            results.append({
                'threshold': threshold,
                'stop_loss_rate': stop_loss_rate,
                'success_rate': success_rate,
                'signals_remaining': signals_remaining,
                'score': score
            })
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            return self.df["final_scores"].median()
        
        # Находим оптимальные пороги по разным критериям
        optimal_by_score = results_df.loc[results_df['score'].idxmax()]
        optimal_by_stop_loss = results_df.loc[results_df['stop_loss_rate'].idxmin()]
        optimal_by_success = results_df.loc[results_df['success_rate'].idxmax()]
        
        # Ищем порог с минимальным количеством стоп-лоссов и приемлемым количеством сигналов
        acceptable_thresholds = results_df[results_df['signals_remaining'] > 0.3]  # минимум 30% сигналов
        if len(acceptable_thresholds) > 0:
            optimal_balanced = acceptable_thresholds.loc[acceptable_thresholds['stop_loss_rate'].idxmin()]
        else:
            optimal_balanced = optimal_by_stop_loss
        
        print(f"\n🎯 ОПТИМАЛЬНЫЕ ПОРОГИ:")
        print(f"   По композитному score: {optimal_by_score['threshold']:.2f}")
        print(f"     - Стоп-лоссов: {optimal_by_score['stop_loss_rate']:.1%}")
        print(f"     - Успешных: {optimal_by_score['success_rate']:.1%}")
        print(f"     - Сигналов остаётся: {optimal_by_score['signals_remaining']:.1%}")
        
        print(f"   По минимизации стоп-лоссов: {optimal_balanced['threshold']:.2f}")
        print(f"     - Стоп-лоссов: {optimal_balanced['stop_loss_rate']:.1%}")
        print(f"     - Сигналов остаётся: {optimal_balanced['signals_remaining']:.1%}")
        
        print(f"   По максимизации успеха: {optimal_by_success['threshold']:.2f}")
        print(f"     - Успешных: {optimal_by_success['success_rate']:.1%}")
        
        # Визуализация оптимизации порога
        self._plot_threshold_optimization(results_df, optimal_balanced)
        
        self.report["threshold_optimization"] = {
            "optimal_by_score": optimal_by_score.to_dict(),
            "optimal_balanced": optimal_balanced.to_dict(),
            "optimal_by_success": optimal_by_success.to_dict()
        }
        
        return optimal_balanced['threshold']
    
    def _plot_threshold_optimization(self, results_df, optimal_point):
        """Визуализация оптимизации порога"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # График 1: Ставки успеха и стоп-лоссов
        ax1.plot(results_df['threshold'], results_df['success_rate'], 
                label='Доля успешных сделок', color='green', linewidth=2)
        ax1.plot(results_df['threshold'], results_df['stop_loss_rate'], 
                label='Доля стоп-лоссов', color='red', linewidth=2)
        ax1.axvline(x=optimal_point['threshold'], color='blue', linestyle='--', 
                   label=f'Рекомендуемый порог: {optimal_point["threshold"]:.2f}')
        ax1.set_xlabel('Порог final_scores')
        ax1.set_ylabel('Доля')
        ax1.set_title('Влияние порога на успешность и стоп-лоссы')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: Композитный score и оставшиеся сигналы
        ax2.plot(results_df['threshold'], results_df['score'], 
                label='Композитный score', color='purple', linewidth=2)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(results_df['threshold'], results_df['signals_remaining'], 
                     label='Доля оставшихся сигналов', color='orange', linewidth=2, linestyle='--')
        ax2.axvline(x=optimal_point['threshold'], color='blue', linestyle='--')
        ax2.set_xlabel('Порог final_scores')
        ax2.set_ylabel('Композитный score')
        ax2_twin.set_ylabel('Доля сигналов')
        ax2.set_title('Композитный score и количество сигналов')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(THRESHOLD_OPTIMIZATION_FILE, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 График оптимизации порога сохранен: {THRESHOLD_OPTIMIZATION_FILE}")
    
    def analyze_basic_statistics(self):
        """Расширенный анализ базовой статистики"""
        success_df = self.df[self.df["success"] == 1]
        fail_df = self.df[self.df["success"] == 0]
        
        stats = {
            "success_count": len(success_df),
            "fail_count": len(fail_df),
            "success_rate": self.df["success"].mean(),
            "stop_loss_count": int(self.df["stop_loss_hit"].sum()),
            "stop_loss_rate": float(self.df["stop_loss_hit"].mean()),
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
        print(f"   Со стоп-лоссом: {stats['stop_loss_count']} ({stats['stop_loss_rate']:.1%})")
        print(f"   Final_scores общие: {stats['final_scores_overall']['mean']:.2f} ± {stats['final_scores_overall']['std']:.2f}")
        
        if 'final_scores_success' in stats:
            print(f"   Final_scores успешных: {stats['final_scores_success']['mean']:.2f} ± {stats['final_scores_success']['std']:.2f}")
        if 'final_scores_fail' in stats:
            print(f"   Final_scores неуспешных: {stats['final_scores_fail']['mean']:.2f} ± {stats['final_scores_fail']['std']:.2f}")
        
        self.report["basic_statistics"] = stats
        return stats
    
    def train_model(self):
        """Обучение модели с учетом дисбаланса классов"""
        # Выбор признаков
        feature_columns = ["final_scores", "scores_config", "score_diff", "score_ratio"]
        
        # Добавляем дополнительные признаки если они существуют
        additional_features = ["max_profit", "avg_profit", "high_score_ratio", "positive_score_diff"]
        for feature in additional_features:
            if feature in self.df.columns:
                feature_columns.append(feature)
        
        X = self.df[feature_columns]
        y = self.df["success"]
        
        # Проверяем, есть ли оба класса
        if len(np.unique(y)) < 2:
            print("⚠️  Только один класс в данных. Используем упрощенный анализ.")
            return self._handle_single_class_case(X, y)
        
        # Стратифицированное разделение
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Настройка модели LightGBM с учетом дисбаланса
        self.model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=10,
            random_state=42,
            verbosity=-1,
            is_unbalance=True  # Важно для дисбалансированных данных
        )
        
        print(f"\n🎯 ОБУЧЕНИЕ МОДЕЛИ:")
        print(f"   Признаки: {', '.join(feature_columns)}")
        print(f"   Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        try:
            # Обучение
            self.model.fit(X_train, y_train)
            
            # Кросс-валидация
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
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
                "feature_importance": dict(zip(feature_columns, [float(x) for x in self.model.feature_importances_])),
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
        
        # Используем оптимизацию по стоп-лоссам если есть данные
        if 'threshold_optimization' in self.report:
            optimal_threshold = self.report['threshold_optimization']['optimal_balanced']['threshold']
            recommendation = "Используем порог, оптимизированный по стоп-лоссам"
        else:
            # Простой анализ на основе перцентилей
            if success_rate == 1.0:
                # Все сделки успешные - берем минимальный final_scores
                optimal_threshold = X["final_scores"].min()
                recommendation = "Все сделки успешны. Используйте минимальное значение final_scores как порог."
            else:
                # Все сделки неуспешные - берем медиану
                optimal_threshold = X["final_scores"].median()
                recommendation = "Все сделки неуспешны. Рекомендуется пересмотреть стратегию."
        
        model_metrics = {
            "cv_roc_auc_mean": 1.0 if success_rate == 1.0 else 0.0,
            "cv_roc_auc_std": 0.0,
            "test_roc_auc": 1.0 if success_rate == 1.0 else 0.0,
            "optimal_threshold": float(optimal_threshold),
            "feature_importance": {"final_scores": 1.0},
            "classification_report": {"accuracy": success_rate},
            "special_case": True,
            "recommendation": recommendation
        }
        
        print(f"   📍 Специальный случай: {recommendation}")
        print(f"   📊 Порог: {optimal_threshold:.3f}")
        
        self.report["model_metrics"] = model_metrics
        return model_metrics
    
    def _analyze_thresholds(self, X_test, y_test, y_pred_proba):
        """Анализ оптимальных порогов"""
        try:
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            
            # F1-score для каждого порога
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores[:-1])
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
            
            # 2. Анализ стоп-лоссов
            self._create_stop_loss_analysis()
            
            # 3. Успешность по бакетам
            if len(np.unique(self.df["success"])) > 1:
                self._create_success_rate_plot()
            else:
                self._create_single_class_plot()
            
            # 4. ROC-кривая (если есть тестовые данные)
            if hasattr(self, 'y_test') and len(np.unique(self.y_test)) > 1:
                self._create_roc_curve()
            
            # 5. Анализ производительности
            self._create_performance_analysis()
            
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
            plt.hist(self.df['final_scores'], bins=20, alpha=0.7, color='blue', density=True)
            success_rate = self.df["success"].mean()
            label = 'Все успешные' if success_rate == 1.0 else 'Все неуспешные'
            plt.text(0.7, 0.9, label, transform=plt.gca().transAxes, fontsize=12)
        
        # Добавляем рекомендуемый порог
        if 'threshold_optimization' in self.report:
            threshold = self.report['threshold_optimization']['optimal_balanced']['threshold']
            plt.axvline(x=threshold, color='purple', linestyle='--', linewidth=2, 
                       label=f'Рек. порог: {threshold:.2f}')
        
        plt.title("Распределение final_scores", fontsize=14, fontweight='bold')
        plt.xlabel("final_scores", fontsize=12)
        plt.ylabel("Плотность", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(HIST_FILE, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Гистограмма сохранена: {HIST_FILE}")
    
    def _create_stop_loss_analysis(self):
        """Анализ стоп-лоссов"""
        if self.df["stop_loss_hit"].sum() == 0:
            return
            
        plt.figure(figsize=(12, 8))
        
        stop_loss_data = self.df[self.df['stop_loss_hit'] == 1]['final_scores']
        no_stop_loss_data = self.df[self.df['stop_loss_hit'] == 0]['final_scores']
        
        plt.hist(stop_loss_data, bins=15, alpha=0.6, label='Со стоп-лоссом', color='red', density=True)
        plt.hist(no_stop_loss_data, bins=15, alpha=0.7, label='Без стоп-лосса', color='green', density=True)
        
        # Добавляем рекомендуемый порог
        if 'threshold_optimization' in self.report:
            threshold = self.report['threshold_optimization']['optimal_balanced']['threshold']
            plt.axvline(x=threshold, color='purple', linestyle='--', linewidth=2, 
                       label=f'Рек. порог: {threshold:.2f}')
        
        plt.title("Распределение final_scores по наличию стоп-лоссов", fontsize=14, fontweight='bold')
        plt.xlabel("final_scores", fontsize=12)
        plt.ylabel("Плотность", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(STOP_LOSS_ANALYSIS_FILE, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Анализ стоп-лоссов сохранен: {STOP_LOSS_ANALYSIS_FILE}")
    
    def _create_success_rate_plot(self):
        """Успешность по диапазонам final_scores"""
        self.df['bucket'] = pd.cut(self.df['final_scores'], bins=10)
        success_rate = self.df.groupby('bucket')['success'].agg(['mean', 'count']).fillna(0)
        
        plt.figure(figsize=(12, 8))
        
        bars = plt.barh(range(len(success_rate)), success_rate['mean'][::-1])
        
        for i, (idx, row) in enumerate(success_rate[::-1].iterrows()):
            color = 'green' if row['mean'] > 0.5 else 'red'
            bars[i].set_color(color)
            plt.text(row['mean'] + 0.01, i, f"n={int(row['count'])}", va='center', fontsize=9)
        
        plt.yticks(range(len(success_rate)), [str(x) for x in success_rate.index[::-1]])
        plt.axvline(x=0.5, color='blue', linestyle='--', linewidth=2, label="Порог 50%")
        
        # Добавляем рекомендуемый порог успешности
        if 'threshold_optimization' in self.report:
            threshold = self.report['threshold_optimization']['optimal_balanced']['threshold']
            # Находим бакет, в который попадает порог
            for i, bucket in enumerate(success_rate.index):
                if threshold >= bucket.left and threshold <= bucket.right:
                    plt.text(0.02, len(success_rate) - i - 1, f"★ {threshold:.2f}", 
                            va='center', fontsize=10, fontweight='bold', color='purple')
                    break
        
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
    
    def _create_performance_analysis(self):
        """Анализ производительности с рекомендуемым порогом"""
        if 'threshold_optimization' not in self.report:
            return
            
        optimal_threshold = self.report['threshold_optimization']['optimal_balanced']['threshold']
        filtered_df = self.df[self.df['final_scores'] >= optimal_threshold]
        
        if len(filtered_df) == 0:
            return
        
        # Сравнительная статистика
        original_stats = {
            'total_signals': len(self.df),
            'success_rate': self.df['success'].mean(),
            'stop_loss_rate': self.df['stop_loss_hit'].mean()
        }
        
        filtered_stats = {
            'total_signals': len(filtered_df),
            'success_rate': filtered_df['success'].mean(),
            'stop_loss_rate': filtered_df['stop_loss_hit'].mean()
        }
        
        # Визуализация сравнения
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        
        # График 1: Сравнение успешности
        categories = ['Все сигналы', f'Фильтр ≥{optimal_threshold:.2f}']
        success_rates = [original_stats['success_rate'], filtered_stats['success_rate']]
        stop_loss_rates = [original_stats['stop_loss_rate'], filtered_stats['stop_loss_rate']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax[0].bar(x - width/2, success_rates, width, label='Успешные', color='green', alpha=0.7)
        ax[0].bar(x + width/2, stop_loss_rates, width, label='Стоп-лоссы', color='red', alpha=0.7)
        ax[0].set_ylabel('Доля')
        ax[0].set_title('Сравнение успешности и стоп-лоссов')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(categories)
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for i, v in enumerate(success_rates):
            ax[0].text(i - width/2, v + 0.01, f'{v:.1%}', ha='center')
        for i, v in enumerate(stop_loss_rates):
            ax[0].text(i + width/2, v + 0.01, f'{v:.1%}', ha='center')
        
        # График 2: Количество сигналов
        signals_counts = [original_stats['total_signals'], filtered_stats['total_signals']]
        signals_percentage = [1.0, filtered_stats['total_signals'] / original_stats['total_signals']]
        
        ax[1].bar(categories, signals_counts, color=['lightblue', 'blue'], alpha=0.7)
        ax[1].set_ylabel('Количество сигналов')
        ax[1].set_title('Количество сигналов после фильтрации')
        ax[1].grid(True, alpha=0.3)
        
        # Добавляем значения и проценты
        for i, (count, perc) in enumerate(zip(signals_counts, signals_percentage)):
            ax[1].text(i, count + max(signals_counts)*0.01, f'{count}\n({perc:.1%})', 
                      ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(PERFORMANCE_ANALYSIS_FILE, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Анализ производительности сохранен: {PERFORMANCE_ANALYSIS_FILE}")
    
    def generate_report(self):
        """Генерация расширенного отчета"""
        # Собираем информацию о сгенерированных файлах
        files_generated = {
            "model": MODEL_FILE,
            "report": REPORT_FILE,
            "success_rate_plot": PLOT_FILE,
            "histogram": HIST_FILE,
        }
        
        # Добавляем дополнительные файлы, если они созданы
        additional_files = {
            "roc_curve": ROC_FILE,
            "stop_loss_analysis": STOP_LOSS_ANALYSIS_FILE,
            "threshold_optimization": THRESHOLD_OPTIMIZATION_FILE,
            "performance_analysis": PERFORMANCE_ANALYSIS_FILE
        }
        
        for file_name, file_path in additional_files.items():
            if os.path.exists(file_path):
                files_generated[file_name] = file_path
        
        final_report = {
            "analysis_date": datetime.now().isoformat(),
            "dataset_info": {
                "total_samples": len(self.df),
                "success_samples": int(self.df["success"].sum()),
                "success_rate": float(self.df["success"].mean()),
                "stop_loss_samples": int(self.df["stop_loss_hit"].sum()),
                "stop_loss_rate": float(self.df["stop_loss_hit"].mean()),
            },
            **self.report,
            "files_generated": files_generated,
            "recommendations": self._generate_recommendations()
        }
        
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=4)
        
        print(f"\n✅ Отчёт сохранён: {REPORT_FILE}")
        return final_report
    
    def _generate_recommendations(self):
        """Генерация рекомендаций на основе анализа"""
        success_rate = self.df["success"].mean()
        stop_loss_rate = self.df["stop_loss_hit"].mean()
        
        # Определяем оптимальный порог
        if 'threshold_optimization' in self.report:
            optimal_threshold = self.report['threshold_optimization']['optimal_balanced']['threshold']
            threshold_source = "оптимизированный по стоп-лоссам"
            
            # Анализ эффективности порога
            filtered_df = self.df[self.df['final_scores'] >= optimal_threshold]
            if len(filtered_df) > 0:
                new_success_rate = filtered_df['success'].mean()
                new_stop_loss_rate = filtered_df['stop_loss_hit'].mean()
                signals_remaining = len(filtered_df) / len(self.df)
            else:
                new_success_rate = success_rate
                new_stop_loss_rate = stop_loss_rate
                signals_remaining = 0
        elif 'model_metrics' in self.report:
            optimal_threshold = self.report['model_metrics']['optimal_threshold']
            threshold_source = "рассчитанный моделью"
            new_success_rate = success_rate
            new_stop_loss_rate = stop_loss_rate
            signals_remaining = 1.0
        else:
            optimal_threshold = self.df['final_scores'].median()
            threshold_source = "медианный"
            new_success_rate = success_rate
            new_stop_loss_rate = stop_loss_rate
            signals_remaining = 1.0
        
        recommendations = []
        
        # Основные выводы по успешности
        if success_rate >= 0.85:
            recommendations.append("🎉 ОТЛИЧНЫЕ РЕЗУЛЬТАТЫ! Стратегия показывает высокую эффективность.")
        elif success_rate >= 0.7:
            recommendations.append("📈 ХОРОШИЕ РЕЗУЛЬТАТЫ! Стратегия работает стабильно.")
        else:
            recommendations.append("⚠️  ТРЕБУЕТСЯ ОПТИМИЗАЦИЯ! Уровень успешности ниже оптимального.")
        
        # Рекомендации по стоп-лоссам
        if stop_loss_rate > 0.2:
            recommendations.append("🔴 ВЫСОКИЙ УРОВЕНЬ СТОП-ЛОССОВ! Необходимо срочно уменьшить количество убыточных сделок.")
        elif stop_loss_rate > 0.1:
            recommendations.append("🟡 УМЕРЕННЫЙ УРОВЕНЬ СТОП-ЛОССОВ. Есть потенциал для улучшения.")
        else:
            recommendations.append("🟢 НИЗКИЙ УРОВЕНЬ СТОП-ЛОССОВ. Отличный контроль рисков!")
        
        recommendations.append(f"🎯 РЕКОМЕНДУЕМЫЙ ПОРОГ final_scores: {optimal_threshold:.2f} ({threshold_source})")
        
        # Анализ эффективности рекомендуемого порога
        if 'threshold_optimization' in self.report:
            if new_stop_loss_rate < stop_loss_rate:
                recommendations.append(f"✅ С порогом {optimal_threshold:.2f} стоп-лоссы уменьшатся с {stop_loss_rate:.1%} до {new_stop_loss_rate:.1%}")
            if new_success_rate > success_rate:
                recommendations.append(f"📈 С порогом {optimal_threshold:.2f} успешность увеличится с {success_rate:.1%} до {new_success_rate:.1%}")
            recommendations.append(f"📊 Количество сигналов: {signals_remaining:.1%} от исходного")
        
        # Качество модели
        if 'model_metrics' in self.report:
            model_auc = self.report['model_metrics']['test_roc_auc']
            if model_auc < 0.6:
                recommendations.append("🤖 МОДЕЛЬ СЛАБАЯ: Low AUC (ниже 0.6) - модель плохо предсказывает успешность сделок")
            elif model_auc < 0.7:
                recommendations.append("🤖 МОДЕЛЬ УМЕРЕННАЯ: Moderate AUC (0.6-0.7) - есть потенциал для улучшения")
            else:
                recommendations.append("🤖 МОДЕЛЬ СИЛЬНАЯ: Good AUC (выше 0.7) - хорошо предсказывает успешность")
        
        # Финальные рекомендации
        recommendations.append("💡 СОВЕТ: Начните с рекомендуемого порога и постепенно оптимизируйте на основе новых данных")
        recommendations.append("⚖️ БАЛАНС: Ищите оптимальное соотношение между количеством сигналов и их качеством")
        
        return recommendations
    
    def run_full_analysis(self):
        """Запуск полного анализа"""
        print("🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА ТОРГОВЫХ СДЕЛОК")
        print("=" * 50)
        
        try:
            self.load_and_prepare_data()
            self.analyze_basic_statistics()
            optimal_threshold, stop_loss_stats = self.analyze_stop_loss_patterns()
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
            import traceback
            traceback.print_exc()
            raise

# === Запуск анализа ===
if __name__ == "__main__":
    analyzer = TradingModelAnalyzer(DEFAULT_DB, DEFAULT_TABLE)
    report = analyzer.run_full_analysis()