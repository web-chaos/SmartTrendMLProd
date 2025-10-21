# Intraday Trading Bot for Binance Futures - SmartTrend
# Version: 1.0.0
from ml_integration import init_model, ml_handle_signal, generate_signal_text, ML_CONFIG
init_model()

import hashlib
import os
import asyncio
import time
import telebot
import threading
from binance.client import Client
from datetime import datetime, timedelta, timezone
import requests
import yfinance as yf
import uuid
from typing import Optional, Dict, List, Tuple
import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator
import numpy as np
from db_stats import db_stats
# from db_stats import db_stats, export_filter_stats_csv
from session_monitor import session_monitor_loop
import websockets
import json
from websockets.exceptions import ConnectionClosed
import traceback
import uuid

from dotenv import load_dotenv
load_dotenv()

kyiv_tz = timezone(timedelta(hours=3))  # Киев летом (UTC+3)

# === Конфигурация ===
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

CONFIG = {
    # === АВТОМАТИЧЕСКАЯ ТОРГОВЛЯ ===
    "TRADING_ENABLED": False,                      # Включить/выключить автоматическую торговлю
    "MAX_ACTIVE_TRADES": 5,                        # Максимальное количество одновременных сделок

    # === РАЗМЕР ПОЗИЦИИ И РИСК ===
    "POSITION_SIZE_TYPE": "percentage",            # "percentage" или "fixed"
    "POSITION_SIZE_PERCENTAGE": 1.0,               # Размер позиции в % от депозита
    "POSITION_SIZE_FIXED": 5.0,                    # Фиксированный размер позиции в USDT
    "RISK_PER_TRADE": 1.0,                         # Риск на сделку в % от депозита

    # === ПЛЕЧО И МАРЖА ===
    "LEVERAGE": 20,                                # Плечо по умолчанию
    "MAX_LEVERAGE": 20,                            # Максимальное плечо
    "MARGIN_TYPE": "ISOLATED",                     # Тип маржи: "ISOLATED" или "CROSS"

    # === УПРАВЛЕНИЕ ПОЗИЦИЕЙ ===
    "BREAKEVEN_AFTER_TP1": True,                   # Перенос стопа в безубыток после TP1
    "TRAILING_STOP_AFTER_TP2": True,               # Трейлинг-стоп после TP2
    "PARTIAL_TAKE_PROFIT": True,                   # Частичное закрытие позиции
    "PARTIAL_CLOSE_PERCENTS": [20, 20, 20, 20, 20],# Проценты закрытия на TP1–TP5 (в сумме 100)

    # === НАСТРОЙКИ ОРДЕРОВ ===
    "USE_MARKET_ORDERS": True,                     # Использовать рыночные ордера
    "ORDER_TIMEOUT": 30,                           # Таймаут ордера в секундах
    "MAX_SLIPPAGE": 0.001,                         # Максимальное проскальзывание (0.1%)


    "SHOW_FILTERS_DETAILS": True,                   # Показывать детали фильтров в сообщения
    # === ФИЛЬТРЫ ===
    # Включение/отключение фильтров
    "USE_DIVERGENCE_FILTER": True,               # включить фильтр по дивергенциям
    "USE_ATR_FILTER": True,                      # Включить фильтр по ATR (волатильность)
    "USE_VOLUME_FILTER": True,                   # Включить фильтр по объёму
    "USE_ADX_FILTER": True,                      # Включить фильтр по ADX (направление тренда)
    "USE_MACD_FILTER": True,                     # Включить фильтр по MACD + 
    "USE_STOCH_FILTER": True,                    # Включить фильтр по Stochastic + 
    "USE_RSI_FILTER": True,                      # Включить фильтр по RSI
    "USE_EMA_FILTER": True,                      # Включить фильтр по EMA
    "USE_GLOBAL_TREND_FILTER": True,             # Включить фильтр по глобальному тренду
    "USE_MARKET_STRUCTURE_FILTER": True,         # Включить фильтр по рыночной структуре
    "USE_STRUCTURE_HH_HL_FILTER": False,         # Включить фильтр по HH/HL (высокий максимум/низкий минимум)
    "USE_VOLATILITY_FILTER": True,               # Включить фильтр волатильности
    "USE_TRIX_FILTER": True,                     # Включить фильтр по TRIX
    "USE_SUPERTREND_FILTER": False,               # Включить/выключить фильтр SuperTrend
    "USE_SD_ZONE_FILTER": True,                  # Включить фильтр по зонам S/D
    "USE_POTENTIAL_FILTER": True,                # Включить фильтр потенциала движения
    "USE_CDV_FILTER": True,                      # Включить фильтр CDV
    "CDV_CHECK_MULTI_TF": True,                  # Проверка согласованности на нескольких TF
    "USE_STRUCTURAL_FILTER": True,               # Включить фильтр структурных паттернов
    "USE_TREND_METHOD": "TREND",                 # варианты: "ZLEMA", "TRIX", "TREND",  None


    # для теста гет тренд
    "GT_DEBUG_TREND": False,                     # Выводить отладочную информацию по глобальному тренду
    
    "GT_EMA_FAST": 9,                            # Период быстрой EMA для глобального тренда
    "GT_EMA_SLOW": 26,                           # Период медленной EMA для глобального тренда
    "GT_SUPERTREND_ATR_PERIOD": 9,               # Период ATR для SuperTrend
    "GT_SUPERTREND_MULTIPLIER": 2,               # Множитель ATR для SuperTrend

    # Пороговые значения для глобального тренда
    "GT_ADX_THRESHOLD": 18.5,                    # Минимальное значение ADX для сильного тренда
    "GT_TREND_EMA_LENGTH": 50,                   # Период EMA для определения направления тренда
    
    # Параметры для локальных таймфреймов
    "GT_ZLEMA_LENGTH": 20,                       # Период ZLEMA
    "GT_TRIX_WINDOW": 15,                        # Период TRIX
    "GT_MIN_VOLATILITY_RATIO": 0.001,            # Минимальное отношение ATR/Close (фильтр низкой волатильности)
    
    # Критерии подтверждения
    "GT_MIN_LOCAL_CONFIRMATIONS": 2,             # Минимальное количество младших ТФ для подтверждения
    
    # Настройки таймфреймов (при необходимости можно изменить)
    "GT_GLOBAL_TFS": ["1d", "4h"],               # Таймфреймы для анализа глобального тренда
    "GT_LOCAL_TFS": ["1h", "15m", "5m"],         # Таймфреймы для локального подтверждения
    # для теста гет тренд



    # ОБНОВЛЕННЫЕ ВЕСА ФИЛЬТРОВ (по убыванию важности)
    "ADX_SCORE_INCREMENT": 0.5,                  # Снижено: слабый подтверждающий сигнал в логах
    "ATR_SCORE_INCREMENT": 1.7,                  # Без изменений: важный для волатильности
    "VOLUME_SCORE_INCREMENT": 1.8,               # Повышено: ключевой базовый фильтр
    "MACD_SCORE_INCREMENT": 1.8,                 # Повышено: самый частый и сбалансированный подтверждающий фильтр
    "EMA_SCORE_INCREMENT": 1.5,                  # Повышено: надежно определяет тренд
    "TRIX_SCORE_INCREMENT": 1.0,                 # Без изменений: важный, но не ключевой
    "SUPERTREND_SCORE_INCREMENT": 1.0,           # Балл за совпадение тренда на всех необходимых ТФ
    "RSI_SCORE_INCREMENT": 0.8,                  # Без изменений: дополнительный фильтр
    "STOCH_SCORE_INCREMENT": 0.6,                # Без изменений: дополнительный фильтр
    "VOLUME_RELAXED_SCORE_INCREMENT": 0.7,       # Без изменений: ослабленный объем
    "MARKET_SCORE_STRUCTURE_INCREMENT": 0.4,     # Без изменений: второстепенный
    "CANDLE_PATTERN_WITH_SD_BONUS": 1.0,         # Без изменений: сильное подтверждение
    "CANDLE_PATTERN_BASE_SCORE": 0.5,            # Без изменений: базовое подтверждение
    "CDV_SCORE_INCREMENT": 0.5,                  # Баллы за соответствие CDV
    "STRUCTURAL_SCORE_INCREMENT": 1.0,           # Баллы за соответствие структурным паттернам
    "VOLUME_DELTA_BONUS": 0.2,                   # Бонус за положительную дельту объема
    "DIVERGENCE_SCORE_4H": 0.75,                 # вес 4H
    "DIVERGENCE_SCORE_1H": 0.25,                 # вес 1H

    "SIGNAL_SCORE_THRESHOLD": 9.35,               # Снижено: новый порог, основанный на пересмотренных баллах 8.5

    # --- ПАРАМЕТРЫ ФИЛЬТРОВ ---
    # ATR параметры
    "CDV_MIN_THRESHOLD": 0.2,                    # Минимальное отношение CDV/объёма
    "MAX_VOLATILITY_RATIO": 2.2,                 # Максимальное отношение ATR/Price для фильтра
    "MIN_ATR_RATIO": 0.0015,                     # Минимальное отношение ATR/Price для фильтра
    # VOLUME параметры
    "MIN_ABSOLUTE_VOLUME": 15_000,               # Минимальный абсолютный объём для фильтра 18_000
    "MIN_ABSOLUTE_VOLUME_TOLERANCE": 0.75,       # Толерантность для абсолютного объёма (80% от минимума)
    "VOLUME_THRESHOLD_MULTIPLIER": 0.56,         # Мягкий порог объёма 0.45 
    "MIN_VOLUME_RELAXED": 0.52,                  # Мягкий порог объёма относительно среднего 0.35
    "NOISE_VOL_RATIO": 0.50,                     # Более мягкий шум-фильтр
    "NOISE_BODY_RATIO": 0.28,                     # Мягкий порог тела свечи к диапазону
    "VOLUME_ACCUM_WINDOW": 3,                    # Окно накопления объема (N свечей)
    # STOCHASTIC параметры
    "STOCHASTIC_RANGE_LONG": [18, 78],           # Диапазон Stochastic для лонга
    "STOCHASTIC_RANGE_SHORT": [25, 85],          # Диапазон Stochastic для шорта
    # RSI параметры
    "RSI_RANGE_LONG": [18, 73],                  # Диапазон RSI для лонга
    "RSI_RANGE_SHORT": [25, 85],                 # Диапазон RSI для шорта
    # MACD параметры
    "MACD_FAST": 12,                             # Параметр fast для MACD
    "MACD_SLOW": 26,                             # Параметр slow для MACD
    "MACD_SIGNAL": 9,                            # Параметр signal для MACD
    "MACD_THRESHOLD": 0.0025,                    # Порог MACD для фильтра 0,0010 было
    "MACD_MOMENTUM_THRESHOLD": 0.0008,          # Порог момента MACD для фильтра 0,00065 было
    "MACD_DEPTH_THRESHOLD": 0.007,              # Порог глубины MACD для фильтра 0,0035 было
    # TRIX параметры
    "TRIX_WINDOW": 18,                           # Период для TRIX 15 было
    "TRIX_MOMENTUM_THRESHOLD": 0.0005,           # Чувствительность к развороту
    "TRIX_DEPTH_THRESHOLD": 0.006,               # Макс. глубина для раннего сигнала - 0,003 было
    # EMA параметры  
    "EMA_THRESHOLD": 0.0015,                     # Порог EMA для фильтра 0,0020 было
    # ADX параметры  
    "ADX_THRESHOLD": 20,                         # Порог ADX для фильтра тренда
    "ADX_WINDOW": 14,                            # Окно расчёта ADX  
    # Market Structure параметры
    "STRUCTURAL_TOLERANCE": 0.003,               # допуск (0.3%) при сравнении уровней
    # SUPERTREND параметры
    "SUPERTREND_ATR_PERIOD": 9,                  # Период ATR для расчета SuperTrend
    "SUPERTREND_MULTIPLIER": 2.0,                # Множитель для расчета верхней/нижней границы
    # modes для фильтров     
    "SUPERTREND_USE_TWO_TF": True,               # Использовать 2 таймфрейма (HTF + LTF) или только основной
    "STRUCTURAL_MODE": "strict",                 # strict = разделение на усиливающие/фильтрующие, soft = любая формация = +балл
    "RSI_MODE": "both",                          # "single" или "both"
    "ADX_MODE": "both",                          # "both" или "single"

    # === ФИЛЬТР ПО ЗОНАМ ПОДДЕРЖКИ/СОПРОТИВЛЕНИЯ ===
    "SD_MIN_TOUCHES": 2,                         # Минимальное количество касаний уровня
    "SD_LAST_TOUCHED": 30,                       # Максимальное количество свечей с последнего касания
    "SD_SWING_WINDOW": 3,                        # Окно для поиска экстремумов
    "SD_ZONE_TOLERANCE": 0.003,                  # Допуск при объединении уровней
    "SD_TRUE_BREAK_CONFIRM": 2,                  # Свечей для подтверждения пробоя
    "SD_ZONE_DISTANCE_THRESHOLD": 0.015,         # Максимальная дистанция до зоны S/D 0.015
    "SD_MIN_STRENGTH": 1.5,                      # Минимальная сила уровня
    "CONFIRMATION_SD_DISTANCE_THRESHOLD": 0.02,  # Максимальная дистанция до зоны S/D для подтверждения паттерна 0.02
    "CONFIRMATION_SD_MIN_STRENGTH": 1.2,         # Минимальная сила уровня S/D для подтверждения паттерна 1.2

    # POТЕНЦИАЛ ДВИЖЕНИЯ
    "POTENTIAL_TFS": ["15m", "1h"],              # Таймфреймы для оценки потенциала
    "POTENTIAL_TF_WEIGHTS": [0.6, 0.4],          # Веса для таймфреймов потенциала
    "POTENTIAL_THRESHOLD": 0.5,                  # Минимальный балл для пропуска сигнала

#   Скальпинг (M1-M15)	FAST=8, SLOW=16, SIGNAL=5	Максимальная скорость.
#   Внутридневная (M30-H1)	FAST=12, SLOW=20, SIGNAL=7	Баланс скорости и фильтрации.
#   Свинг-трейдинг (H4-D1)	FAST=12, SLOW=26, SIGNAL=9 (как сейчас)	Меньше шума, акцент на тренд.


    # === ГЛОБАЛЬНЫЕ ФИЛЬТРЫ И РЫНОЧНЫЙ АНАЛИЗ ===
    "MARKET_ANALYSIS_TF": ["30m", "2h"],         # Таймфреймы для анализа рынка
    "MARKET_ANALYSIS_SEND": 120,                 # Периодичность отправки анализа (сек)
    "USE_D1_TREND_FILTER": True,                 # Фильтр по тренду D1
    "USE_H4_TREND_FILTER": True,                 # Фильтр по тренду H4
    "TREND_SCORE_MARGIN": 0.2,                   # Минимальная разница в баллах для тренда
    "MIN_TF_TREND_MATCHES": 5,                   # Минимум совпадающих таймфреймов по тренду
    "DEBUG_TREND": False,                        # Выводить отладочную информацию по тренду

    # === ДИВЕРГЕНЦИИ (RSI + MACD) ===
    "USE_DIVERGENCE_TAG": True,                  # Добавлять тег дивергенции в сообщение
    "DIVERGENCE_USE_1H": True,                   # использовать 1ч ТФ
    "DIVERGENCE_USE_4H": True,                   # использовать 4ч ТФ
    "DIVERGENCE_TFS": ["1h", "4h"],              # Таймфреймы для поиска дивергенций
    "DIVERGENCE_LOOKBACK": 5,                    # Сколько свечей назад искать дивергенцию
    "CHECK_RSI_DIVERGENCE": True,                # Включить RSI-дивергенции
    "CHECK_MACD_DIVERGENCE": True,               # Включить MACD-дивергенции

    # === ТАЙМФРЕЙМЫ ДЛЯ ТЕЙКОВ (гибридный режим) ===
    "TP_FVG_TF": "15m",                          # Таймфрейм для поиска FVG (TP1) 15m
    "TP_SWING_TF": "30m",                        # Таймфрейм для поиска swing high/low (TP2) 30m
    "TP_SD_TF": "1h",                            # Таймфрейм для поиска S/D зон (TP3) 1h
    "TP_COUNT": 5,                               # Количество тейк-профитов (рекомендуется 3-5)
    "TP_ATR": [0.65, 1.1, 1.5, 2.0, 3.0],        # Множители ATR для TP1-TP5
    "MIN_TP_DIST_MULT": 1.0,                     # Минимальное расстояние между тейками (в ATR) 1.0      

    # === ЗОНЫ ВХОДА + РАСЧЕТ СТОПОВ ===
    "ENTRY_ZONE_WIDTH_FRACTAL": 0.0035,          # Ширина зоны входа (fractal) 0,005 было
    "STOP_TF": "2h",                             # Таймфрейм для поиска стопа и swing-фракталов
    "SWING_LOOKBACK": 55,                        # Сколько свечей назад искать swing-фрактал
    "SWING_LEFT": 3,                             # Сколько свечей слева для swing-фрактала
    "SWING_RIGHT": 3,                            # Сколько свечей справа для swing-фрактала
    "SWING_VOLUME_MULT": 1.15,                   # Множитель объёма для swing-фильтра
    "SWING_MIN_ATR_RATIO": 0.0015,               # Минимальное ATR/Price для swing
    "SWING_MIN_PRICE_DIST": 0.0025,              # Мин. дистанция swing-уровня от цены
    "STOP_ATR_BUFFER_MULT": 1.2,                 # Доп. буфер от ATR (0.5 = 50% от ATR)
    "STOP_ATR_MULTIPLIER": 1.5,                  # Базовый множитель ATR (1.5x ATR)
    "STOP_MIN_ATR_MULTIPLIER": 1.0,              # Минимальный ATR (1.0x ATR)
    "STOP_MAX_ATR_MULTIPLIER": 3.0,              # Максимальный ATR (3.0x ATR)
    "STOP_MIN_DISTANCE_PCT": 0.035,              # Минимум 5% от цены входа
    "STOP_MAX_DISTANCE_PCT": 0.055,              # Максимум 7% от цены входа

    # === ПРОЧЕЕ ===
    "PER_SYMBOL_DELAY": 20.0,                    # Задержка между анализом монет (сек) - 10 сек было
    "COOLDOWN_AFTER_STOP": 1800,                 # Пауза после стопа (сек)

    # === ТОРГОВЫЕ СЕССИИ ===
    "ENABLE_SESSION_MONITOR": True,              # Включить мониторинг торговых сессий
    "SESSION_MONITOR_INTERVAL": 60,              # Интервал мониторинга сессий (сек)
    "SESSION_PIN_MESSAGE": True,                 # Закреплять сообщение с сессиями в чате
    "SHOW_SESSION_SYMBOL_STATUS": True,          # Показывать статус символов в сессиях

    "ENABLE_WALLETS": False,                     # Включить отображение кошельков в сообщениях
    "WALLETS": {
        "USDT_TRC20": "TFSUWvf4bodQzXn1w9q7td5R4Kc6iwbAo1",
        "USDT_TON": {
            "address": "EQD5mxRgCuRNLxKxeOjG6r14iSroLF5FtomPnet-sgP5xNJb",
            "memo": "168117704"
        }
    },

    # === МОНИТОРИНГ ОТКАТОВ ===
    "RETRACEMENT_ALERTS_ENABLED": True,          # Включить/выключить мониторинг откатов
    "RETRACEMENT_MONITOR_INTERVAL": 1800,        # Проверка каждые 30 минут (в секундах)
    "RETRACEMENT_LEVEL_1": 50,                   # Порог для уровня 1 (желтый) в %
    "RETRACEMENT_LEVEL_2": 70,                   # Порог для уровня 2 (красный) в %
    "RETRACEMENT_ALERT_COOLDOWN": 14400,         # Задержка между уведомлениями одного уровня (4 часа)

    # === ПАРАМЕТРЫ ВХОДА ===
    "CONSOLIDATION_WINDOW": 20,                  # Количество свечей для оценки консолидации
    "CONSOLIDATION_THRESHOLD": 0.003,            # Порог консолидации 0.3%
    "ENTRY_DISTANCE_THRESHOLD": 0.008,           # максимум 1,5% до зоны - 0.01 - 0.015 было
    "ENTRY_CANDIDATE_WEIGHTS": {              
        "fvg": 10,                               # Вес FVG (TP1)
        "sd": 9,                                 # Вес S/D зона (TP3)
        "swing": 8,                              # Вес swing high/low (TP2)
        "accumulation": 7
    },

    # === ПАРАМЕТРЫ БОТА ДЛЯ ДИНАМИЧЕСКОГО СПИСКА ===
    "USE_DYNAMIC_SYMBOLS": False,                    # Использовать динамический список монет
    "TOP_SYMBOLS_LIMIT": 50,                        # Количество монет в топе
    "TOP_SYMBOLS_UPDATE_INTERVAL": 7200,           # Интервал обновления списка монет (в секундах)
                                                    # Доступные варианты:
                                                    # 3600    = 1 час
                                                    # 7200    = 2 часа  
                                                    # 14400   = 4 часа
                                                    # 21600   = 6 часов
                                                    # 43200   = 12 часов (по умолчанию)
                                                    # 86400   = 24 часа (1 день)
                                                    # 172800  = 48 часов (2 дня)
                                                    # 259200  = 72 часа (3 дня)
                                                    # 604800  = 7 дней (1 неделя)
    # === ПАРАМЕТРЫ БОТА ДЛЯ ДИНАМИЧЕСКОГО СПИСКА ===

    # === WEBSOCKET ПАРАМЕТРЫ ===
    "CHUNK_SIZE": 50,

    # === МОНЕТЫ ПОД АНАЛИЗ ===
    "MARKET_ANALYSIS_SYMBOLS": [
        # === Топ активы ===
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOTUSDT", "LTCUSDT",

        # === Layer-1 ===
        "AVAXUSDT", "NEARUSDT", "APTUSDT", "SUIUSDT", "SEIUSDT", "ALGOUSDT",
        "XLMUSDT", "TONUSDT", "INJUSDT", "TIAUSDT",

        # === Layer-2 / Infra ===
        "ARBUSDT", "OPUSDT", "MANTAUSDT", "METISUSDT", "AEVOUSDT", "ZROUSDT",
        "AXLUSDT", "ORDIUSDT",

        # === AI / BigData ===
        "FETUSDT", "WLDUSDT", "RENDERUSDT", "TAOUSDT", "AVAAIUSDT", "AI16ZUSDT", "AIXBTUSDT",

        # === DeFi / DEX ===
        "AAVEUSDT", "UNIUSDT", "LDOUSDT", "COMPUSDT", "KAIAUSDT", "CRVUSDT",
        "GMXUSDT", "SNXUSDT", "1INCHUSDT", "PENDLEUSDT", "CAKEUSDT",
        "JTOUSDT", "RAYSOLUSDT", "HUMAUSDT", "XPLUSDT",

        # === Мемкоины и спекулятивные ===
        "DOGEUSDT", "1000PEPEUSDT", "1000BONKUSDT", "WIFUSDT", "PENGUUSDT",
        "FARTCOINUSDT", "PNUTUSDT", "SUSDT", "USUALUSDT", "MOODENGUSDT",

        # === Новые и хайповые активы ===
        "JUPUSDT", "ENAUSDT", "ONDOUSDT", "MORPHOUSDT", "EIGENUSDT",
        "NEIROUSDT", "GUNUSDT", "APEUSDT", "VANAUSDT",

        # === Privacy ===
        "XMRUSDT", "ETCUSDT", "VIRTUALUSDT", "REZUSDT",

        # === Storage / Web3 ===
        "ICPUSDT", "TRBUSDT", "LPTUSDT", "IOTAUSDT", "GRTUSDT", "SWARMSUSDT", "SUSHIUSDT",

        # === Metaverse / NFT ===
        "MOCAUSDT", "GHSTUSDT", "HYPEUSDT", "ZKUSDT", "XANUSDT", "PAXGUSDT","PLUMEUSDT", "TSTUSDT",

        # === Индексные / прочее ===
        "SPXUSDT", "ETHFIUSDT", "PEOPLEUSDT", "DEGENUSDT",
        "CHILLGUYUSDT", "POPCATUSDT", "BIOUSDT", "PUMPUSDT", "ENSUSDT", "IPUSDT", "POLUSDT", "KASUSDT", "GMTUSDT", "ARKMUSDT", "ALTUSDT",

        # === Прочие (низкая ликвидность/неопределённые) ===
        "KERNELUSDT", "ARCUSDT", "GRIFFAINUSDT", "GASUSDT", "SKLUSDT", "BANDUSDT", "KAVAUSDT", "HAEDALUSDT", "KAITOUSDT", "CFXUSDT", "USELESSUSDT", "STXUSDT", "ZEREBROUSDT", "COAIUSDT", "ASTERUSDT"
    ]

}

# Собираем все уникальные таймфреймы для загрузки данных
CONFIG["REQUIRED_TFS"] = list(set([
    # "5m", "15m", "30m", "1h", "2h", "4h", "1d",
    *CONFIG["GT_GLOBAL_TFS"],
    *CONFIG["GT_LOCAL_TFS"],
    *CONFIG["MARKET_ANALYSIS_TF"],
    CONFIG["STOP_TF"],
    CONFIG["TP_FVG_TF"],
    CONFIG["TP_SWING_TF"],
    CONFIG["TP_SD_TF"],
    *CONFIG["DIVERGENCE_TFS"],
    *CONFIG["POTENTIAL_TFS"],
]))

ORIGINAL_STATIC_SYMBOLS = CONFIG["MARKET_ANALYSIS_SYMBOLS"].copy()

# === Статистика сигналов (обновляется каждый час) ===
daily_stats = {
    "signals_sent": 0,
    "trades_opened": 0,
    "stopped_out": 0,
    "expired": 0,
    "tp1_hit": 0,
    "tp2_hit": 0,
    "tp3_hit": 0,
    "tp4_hit": 0,
    "tp5_hit": 0,
    "closed_breakeven_after_tp1": 0,
    "closed_breakeven_after_tp2": 0,
    "closed_breakeven_after_tp3": 0,
    "closed_breakeven_after_tp4": 0,
    # Новые счетчики для мониторинга откатов
    "retracement_level1_alerts": 0,  # Количество уведомлений уровня 1
    "retracement_level2_alerts": 0,  # Количество уведомлений уровня 2
    "retracement_unique_symbols": set(),  # Уникальные символы с уведомлениями
    "profit_10x": 0.0,
    "profit_20x": 0.0,
    "loss_10x": 0.0,
    "loss_20x": 0.0,
}
pinned_stats_message_id = None  # ID закреплённого сообщения

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
bot = telebot.TeleBot(TG_BOT_TOKEN)


def update_profit_loss_from_trade(symbol, trade, outcome: str, target_idx: Optional[int] = None):
    """
    outcome: 'win' если тейк достигнут, 'loss' если стоп.
    Если outcome == 'win' — берём target{target_idx}.
    Если outcome == 'loss' — берём stop.
    """
    print(f"[DEBUG] update_profit_loss_from_trade called: {symbol}, outcome={outcome}, target_idx={target_idx}")

    try:
        entry = trade.get("entry_real", trade.get("entry"))
        side = trade.get("side", "long")
        if not entry:
            print(f"[WARN] {symbol}: нет entry для расчёта прибыли")
            return

        # Определяем цену выхода
        if outcome == "win":
            if target_idx:
                last_price = trade.get(f"target{target_idx}")
            else:
                # ищем максимальный достигнутый тейк
                last_price = None
                for i in range(5, 0, -1):
                    if trade.get(f"take{i}_hit"):
                        last_price = trade.get(f"target{i}")
                        break
                if not last_price:
                    last_price = trade.get("target1")
        elif outcome == "loss":
            last_price = trade.get("stop")
        else:
            print(f"[WARN] {symbol}: outcome неизвестен — {outcome}")
            return

        if not last_price:
            print(f"[WARN] {symbol}: нет last_price для расчёта ({outcome})")
            return

        entry = float(entry)
        last_price = float(last_price)

        # Расчёт изменения в %
        if side == "long":
            change_pct = ((last_price - entry) / entry) * 100
        else:
            change_pct = ((entry - last_price) / entry) * 100

        if outcome == "win":
            daily_stats["profit_10x"] += change_pct * 10
            daily_stats["profit_20x"] += change_pct * 20
        elif outcome == "loss":
            daily_stats["loss_10x"] += abs(change_pct) * 10
            daily_stats["loss_20x"] += abs(change_pct) * 20

        print(f"[INFO] {symbol}: {outcome.upper()} target{target_idx or '?'} → {change_pct:.2f}%")

    except Exception as e:
        print(f"[ERROR] update_profit_loss_from_trade({symbol}): {e}")


# Глобальный список всех уникальных символов, когда-либо анализировавшихся
def load_df(
    symbol: str,
    timeframes: List[str],
    client: Client,
    limit: int = 100
) -> Dict[str, Optional[pd.DataFrame]]:
    df_dict = {}
    
    for tf in timeframes:
        try:
            klines = client.futures_klines(
                symbol=symbol,
                interval=tf,
                limit=limit
            )
            
            # Используйте правильные названия колонок
            columns = [
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
            ]
            
            df = pd.DataFrame(klines, columns=columns)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df = df.astype({
                "open": float, "high": float, "low": float, 
                "close": float, "volume": float
            })
            
            df_dict[tf] = df
            
        except Exception as e:
            print(f"[ERROR] Failed to load {symbol} {tf}: {str(e)}")
            df_dict[tf] = None
    
    return df_dict

# Удаление тега #new из сообщения
def remove_new_tag(symbol: str, trade: dict):
    try:
        if "original_message_text" not in trade:
            return
        text = trade["original_message_text"]
        if "#new" not in text:
            return
        # Удаляем тег с эмодзи и переносом строки
        new_text = text.replace("🔔 #new\n", "").replace("🔔 #new", "").strip()
        bot.edit_message_text(
            chat_id=TG_CHAT_ID,
            message_id=trade["message_id"],
            text=new_text,
            parse_mode="HTML",
            disable_web_page_preview=True
        )
        trade["original_message_text"] = new_text
        print(f"[INFO] Хештег #new удалён из {symbol}")
    except Exception as e:
        print(f"[ERROR] Не удалось отредактировать сообщение {symbol}: {e}")

# 🟢 Инициализируем БД перед запуском бота
async def init_services():
    await db_stats.init()
    await db_stats.enable_filter_stats() 

asyncio.run(init_services())
# === Обработчик команд Telegram ===
# Команды для бота
@bot.message_handler(commands=['trading_on'])
def handle_trading_on(message):
    CONFIG["TRADING_ENABLED"] = True
    bot.reply_to(message, "✅ Автоматическая торговля включена")

@bot.message_handler(commands=['trading_off'])
def handle_trading_off(message):
    CONFIG["TRADING_ENABLED"] = False
    bot.reply_to(message, "⛔ Автоматическая торговля выключена")

@bot.message_handler(commands=['set_max_trades'])
def handle_set_max_trades(message):
    try:
        max_trades = int(message.text.split()[1])
        CONFIG["MAX_ACTIVE_TRADES"] = max_trades
        bot.reply_to(message, f"✅ Максимальное количество сделок установлено: {max_trades}")
    except:
        bot.reply_to(message, "❌ Использование: /set_max_trades <число>")

@bot.message_handler(commands=['active_trades'])
def handle_active_trades(message):
    active_count = trade_manager.get_active_trades_count()
    bot.reply_to(message, f"Активных сделок: {active_count}/{CONFIG['MAX_ACTIVE_TRADES']}")

@bot.message_handler(commands=['close_trade'])
def handle_close_trade(message):
    try:
        symbol = message.text.split()[1].upper()
        asyncio.create_task(trade_manager.close_trade(symbol, "manual"))
        bot.reply_to(message, f"🔄 Закрытие сделки {symbol}...")
    except:
        bot.reply_to(message, "❌ Использование: /close_trade <SYMBOL>")


@bot.message_handler(commands=['stats'])
def handle_stats_command(message):
    try:
        bot.send_message(
            chat_id=message.chat.id,
            text=get_stats_message(),
            parse_mode='HTML',
            disable_web_page_preview=True
        )
    except Exception as e:
        print(f"[ERROR] /stats: {e}")

# === Обработчик команды сброса статистики ===
@bot.message_handler(commands=['resetstats'])
def handle_reset_stats_command(message):
    try:
        for key in daily_stats:
            if key == "retracement_unique_symbols":
                daily_stats[key] = set()
            elif key in ["profit_10x", "profit_20x", "loss_10x", "loss_20x"]:
                daily_stats[key] = 0  # Сбрасываем новые поля
            else:
                daily_stats[key] = 0

        bot.send_message(
            chat_id=message.chat.id,
            text="📉 Статистика сброшена вручную.",
            parse_mode='HTML'
        )
    except Exception as e:
        print(f"[ERROR] /resetstats: {e}")

@bot.message_handler(commands=['symbols'])
def handle_all_symbols_command(message):
    try:
        if not all_symbols_ever:
            bot.send_message(
                chat_id=message.chat.id,
                text="📭 Нет данных о символах",
                parse_mode='HTML'
            )
            return
            
        # Определяем режим работы
        mode = "динамический" if CONFIG.get("USE_DYNAMIC_SYMBOLS", False) else "статический"
        mode_icon = "🔄" if CONFIG.get("USE_DYNAMIC_SYMBOLS", False) else "📋"
        
        # Сортируем символы для удобства чтения
        sorted_symbols = sorted(list(all_symbols_ever))
        symbols_list = "\n".join([f"• {symbol}" for symbol in sorted_symbols])
        
        # Формируем сообщение с информацией о режиме
        msg = (
            f"📋 <b>Все уникальные символы ({len(sorted_symbols)})</b>\n"
            f"{mode_icon} <b>Режим работы:</b> {mode}\n\n"
            f"{symbols_list}"
        )
        
        bot.send_message(
            chat_id=message.chat.id,
            text=msg,
            parse_mode='HTML',
            disable_web_page_preview=True
        )
    except Exception as e:
        print(f"[ERROR] /symbols: {e}")
        bot.send_message(
            chat_id=message.chat.id,
            text="❌ Ошибка при получении списка символов"
        )

# @bot.message_handler(commands=["export_filter_stats"])
# def handle_export_csv(message):
#     # Запускаем async функцию через глобальный loop в безопасном потоке
#     print("[DEBUG STATS] Получена команда /export_filter_stats")
#     asyncio.run_coroutine_threadsafe(export_filter_stats_csv(bot, message.chat.id), loop)

@bot.message_handler(commands=["clear_filter_stats"])
def handle_clear_filter_stats(message):
    # TODO: Очистить статистику фильтров
    asyncio.run_coroutine_threadsafe(db_stats.clear_filter_stats(), loop)
    bot.send_message(message.chat.id, "Статистика фильтров очищена.")

@bot.message_handler(commands=["filter_stats_off"])
def handle_filter_stats_off(message):
    # TODO: Выключить запись статистики по фильтрам
    asyncio.run_coroutine_threadsafe(db_stats.disable_filter_stats(), loop)
    bot.send_message(message.chat.id, "Запись статистики по фильтрам выключена.")

@bot.message_handler(commands=["filter_stats_on"])
def handle_filter_stats_on(message):
    # TODO: Включить запись статистики по фильтрам
    asyncio.run_coroutine_threadsafe(db_stats.enable_filter_stats(), loop)
    bot.send_message(message.chat.id, "Запись статистики по фильтрам включена.")


@bot.message_handler(commands=["retracement_on"])
def handle_retracement_on(message):
    """Включить мониторинг откатов"""
    CONFIG["RETRACEMENT_ALERTS_ENABLED"] = True
    bot.send_message(
        message.chat.id,
        "✅ Мониторинг откатов включен",
        parse_mode='HTML'
    )

@bot.message_handler(commands=["retracement_off"])
def handle_retracement_off(message):
    """Выключить мониторинг откатов"""
    CONFIG["RETRACEMENT_ALERTS_ENABLED"] = False
    bot.send_message(
        message.chat.id,
        "⛔ Мониторинг откатов выключен",
        parse_mode='HTML'
    )

@bot.message_handler(commands=["retracement_status"])
def handle_retracement_status(message):
    """Показать статус мониторинга откатов"""
    status = "включен" if CONFIG.get("RETRACEMENT_ALERTS_ENABLED", True) else "выключен"
    interval_min = CONFIG.get("RETRACEMENT_MONITOR_INTERVAL", 1800) // 60
    cooldown_hours = CONFIG.get("RETRACEMENT_ALERT_COOLDOWN", 14400) // 3600
    
    bot.send_message(
        message.chat.id,
        f"📊 <b>Статус мониторинга откатов</b>\n\n"
        f"• Состояние: <b>{status}</b>\n"
        f"• Интервал проверки: <b>{interval_min} минут</b>\n"
        f"• Уровень 1: <b>{CONFIG.get('RETRACEMENT_LEVEL_1', 50)}%</b>\n"
        f"• Уровень 2: <b>{CONFIG.get('RETRACEMENT_LEVEL_2', 70)}%</b>\n"
        f"• Задержка между уведомлениями: <b>{cooldown_hours} часов</b>",
        parse_mode='HTML'
    )

@bot.message_handler(commands=['retracement_stats'])
def handle_retracement_stats(message):
    """Показать детальную статистику по откатам"""
    unique_symbols = daily_stats["retracement_unique_symbols"]
    unique_count = len(unique_symbols)
    symbols_list = "\n".join([f"• {symbol}" for symbol in sorted(unique_symbols)]) if unique_symbols else "Нет данных"
    
    bot.send_message(
        message.chat.id,
        f"📊 <b>Детальная статистика мониторинга откатов</b>\n\n"
        f"⚠️  Уведомлений Уровень 1: <b>{daily_stats['retracement_level1_alerts']}</b>\n"
        f"🚨 Уведомлений Уровень 2: <b>{daily_stats['retracement_level2_alerts']}</b>\n"
        f"📈 Уникальных символов: <b>{unique_count}</b>\n\n"
        f"<b>Символы с уведомлениями:</b>\n{symbols_list}",
        parse_mode='HTML',
        disable_web_page_preview=True
    )

# === Обработчики команд для динамического списка монет ===
@bot.message_handler(commands=['dynamic_on'])
def handle_dynamic_on(message):
    """Включение динамического режима"""
    try:
        CONFIG["USE_DYNAMIC_SYMBOLS"] = True
        bot.send_message(
            message.chat.id,
            "✅ Динамический режим включен. Список монет будет обновляться автоматически.",
            parse_mode='HTML'
        )
    except Exception as e:
        print(f"[ERROR] /dynamic_on: {e}")
        bot.send_message(
            message.chat.id,
            "❌ Ошибка при включении динамического режима",
            parse_mode='HTML'
        )

# === Обработчики команд для динамического списка монет ===
@bot.message_handler(commands=['dynamic_off'])
def handle_dynamic_off(message):
    """Выключение динамического режима"""
    try:
        CONFIG["USE_DYNAMIC_SYMBOLS"] = False
        
        # Восстанавливаем исходный статический список из конфига
        # (предполагая, что у вас есть исходный список где-то сохранённый)
        CONFIG["MARKET_ANALYSIS_SYMBOLS"] = ORIGINAL_STATIC_SYMBOLS.copy()
        
        bot.send_message(
            message.chat.id,
            "⛔ <b>Динамический режим выключен. Используется статический список монет.</b>\n\n",
            parse_mode='HTML'
        )
    except Exception as e:
        print(f"[ERROR] /dynamic_off: {e}")


@bot.message_handler(commands=['dynamic_status'])
def handle_dynamic_status(message):
    """Показать статус динамического режима"""
    try:
        mode = "динамический" if CONFIG.get("USE_DYNAMIC_SYMBOLS", False) else "статический"
        mode_icon = "🔄" if CONFIG.get("USE_DYNAMIC_SYMBOLS", False) else "📋"
        update_interval = CONFIG.get("TOP_SYMBOLS_UPDATE_INTERVAL", 43200) / 3600
        
        if CONFIG.get("USE_DYNAMIC_SYMBOLS", False):
            msg = (
                f"{mode_icon} <b>Текущий режим:</b> {mode}\n"
                f"📊 <b>Количество монет:</b> {CONFIG.get('TOP_SYMBOLS_LIMIT', 50)}\n"
                f"⏰ <b>Интервал обновления:</b> каждые {update_interval} часов\n"
                f"🔢 <b>Текущий список:</b> {len(CONFIG['MARKET_ANALYSIS_SYMBOLS'])} монет"
            )
        else:
            msg = (
                f"{mode_icon} <b>Текущий режим:</b> {mode}\n"
                f"📊 <b>Количество монет:</b> {len(CONFIG['MARKET_ANALYSIS_SYMBOLS'])}\n"
                f"📝 <b>Список фиксирован</b>"
            )
            
        bot.send_message(
            message.chat.id,
            msg,
            parse_mode='HTML'
        )
    except Exception as e:
        print(f"[ERROR] /dynamic_status: {e}")
        bot.send_message(
            message.chat.id,
            "❌ Ошибка при получении статуса режима"
        )

@bot.message_handler(commands=['update_interval_1h', 'update_interval_2h', 'update_interval_4h', 'update_interval_6h', 
                              'update_interval_12h', 'update_interval_24h', 'update_interval_3d',
                              'update_interval_7d'])
def handle_update_interval(message):
    """Установка интервала обновления списка монет"""
    try:
        # Определяем интервал из команды
        interval_map = {
            '/update_interval_1h': 3600,
            '/update_interval_2h': 7200,
            '/update_interval_4h': 14400,
            '/update_interval_6h': 21600,
            '/update_interval_12h': 43200,
            '/update_interval_24h': 86400,
            '/update_interval_3d': 259200,
            '/update_interval_7d': 604800
        }
        
        command = message.text.split()[0]  # Берём первую часть текста (команду)
        new_interval = interval_map.get(command)
        
        if new_interval:
            CONFIG["TOP_SYMBOLS_UPDATE_INTERVAL"] = new_interval
            
            # Форматируем интервал для читаемости
            if new_interval < 3600:
                interval_text = f"{new_interval} секунд"
            elif new_interval < 86400:
                hours = new_interval // 3600
                interval_text = f"{hours} часов"
            else:
                days = new_interval // 86400
                interval_text = f"{days} дней"
            
            bot.send_message(
                message.chat.id,
                f"⏰ <b>Интервал обновления изменен</b>\n\n"
                f"Новый интервал: <b>{interval_text}</b>\n"
                f"Следующее обновление через: <b>{interval_text}</b>",
                parse_mode='HTML'
            )
        else:
            bot.send_message(
                message.chat.id,
                "❌ Неизвестная команда интервала"
            )
            
    except Exception as e:
        print(f"[ERROR] Ошибка установки интервала: {e}")
        bot.send_message(
            message.chat.id,
            "❌ Ошибка при изменении интервала обновления"
        )

@bot.message_handler(commands=['current_interval'])
def handle_current_interval(message):
    """Показать текущий интервал обновления"""
    try:
        interval = CONFIG.get("TOP_SYMBOLS_UPDATE_INTERVAL", 43200)
        
        # Форматируем интервал для читаемости
        if interval < 3600:
            interval_text = f"{interval} секунд"
        elif interval < 86400:
            hours = interval // 3600
            interval_text = f"{hours} часов"
        else:
            days = interval // 86400
            interval_text = f"{days} дней"
        
        bot.send_message(
            message.chat.id,
            f"⏰ <b>Текущий интервал обновления</b>\n\n"
            f"Интервал: <b>{interval_text}</b>\n"
            f"Значение в секундах: <code>{interval}</code>",
            parse_mode='HTML'
        )
    except Exception as e:
        print(f"[ERROR] Ошибка получения интервала: {e}")
        bot.send_message(
            message.chat.id,
            "❌ Ошибка при получении интервала обновления"
        )

# === ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ===
active_trades = {}
recently_stopped = {}
all_symbols_ever = set()
retracement_alerts_sent = {}  # Формат: {symbol: {"level": int, "last_alert": timestamp}}

# Глобальный словарь для хранения последних цен
last_prices = {}
cdv_data = {}  # динамические данные CDV по символам

# === WebSocket клиент для получения CDV и объема торгов в реальном времени ===
async def cdv_websocket_client():
    symbols = list(set(CONFIG["MARKET_ANALYSIS_SYMBOLS"] + list(active_trades.keys())))
    streams = [f"{symbol.lower()}@aggTrade" for symbol in symbols]

    # Разделяем на чанки для подключения
    stream_chunks = [streams[i:i + CONFIG["CHUNK_SIZE"]] for i in range(0, len(streams), CONFIG["CHUNK_SIZE"])]

    for chunk in stream_chunks:
        stream_url = f"wss://fstream.binance.com/stream?streams={'/'.join(chunk)}"
        asyncio.create_task(cdv_websocket_connection(stream_url))

# === Функция для подключения к WebSocket CDV и обработки сообщений ===
async def cdv_websocket_connection(stream_url):
    while True:
        try:
            async with websockets.connect(stream_url) as websocket:
                print(f"[CDV WS] Connected to {stream_url}")
                async for message in websocket:
                    data = json.loads(message)
                    if 'data' in data and 's' in data['data']:
                        symbol = data['data']['s']
                        is_buyer_maker = data['data']['m']
                        quantity = float(data['data']['q'])
                        timestamp = data['data']['T'] / 1000

                        # Инициализация под символ
                        if symbol not in cdv_data:
                            cdv_data[symbol] = {}
                            for tf in CONFIG["MARKET_ANALYSIS_TF"]:
                                cdv_data[symbol][tf] = {"delta": 0, "volume": 0}
                            cdv_data[symbol]['last_update'] = time.time()

                        # Обновляем дельту и объем для каждого TF
                        current_time = time.time()
                        for tf in CONFIG["MARKET_ANALYSIS_TF"]:
                            seconds = int(tf[:-1]) * 60 if tf.endswith('m') else int(tf[:-1]) * 3600
                            if timestamp >= current_time - seconds:
                                if is_buyer_maker:
                                    cdv_data[symbol][tf]['delta'] -= quantity
                                else:
                                    cdv_data[symbol][tf]['delta'] += quantity
                                cdv_data[symbol][tf]['volume'] += quantity

                        cdv_data[symbol]['last_update'] = current_time

        except (ConnectionClosed, Exception) as e:
            print(f"[CDV WS] Connection error: {e}, reconnecting in 5 seconds...")
            await asyncio.sleep(5)

# Функция для получения CDV и объема по символу и таймфрейму
def get_cdv_ratio(symbol, timeframe):
    if symbol not in cdv_data or timeframe not in cdv_data[symbol]:
        return None, None

    data = cdv_data[symbol][timeframe]
    if data['volume'] > 0:
        return data['delta'] / data['volume'], data['volume']
    return None, None

# === Функция для получения топовых монет по объему торгов ===
async def fetch_top_symbols():
    """Получение топовых монет по объему торгов"""
    try:
        tickers = client.futures_ticker()
        sorted_tickers = sorted(
            tickers, 
            key=lambda x: float(x['quoteVolume']), 
            reverse=True
        )
        symbols = [
            t['symbol'] for t in sorted_tickers 
            if t['symbol'].endswith('USDT')
        ][:CONFIG['TOP_SYMBOLS_LIMIT']]
        
        return symbols
    except Exception as e:
        print(f"[ERROR] Не удалось получить топовые символы: {e}")
        return None

# === Цикл для периодического обновления списка монет ===
async def update_symbols_loop():
    """Цикл обновления списка монет"""
    while True:
        try:
            if CONFIG["USE_DYNAMIC_SYMBOLS"]:
                new_symbols = await fetch_top_symbols()
                if new_symbols:
                    CONFIG["MARKET_ANALYSIS_SYMBOLS"][:] = new_symbols
                    print(f"[INFO] Обновлен список символов: {new_symbols}")
        except Exception as e:
            print(f"[ERROR] Ошибка в update_symbols_loop: {e}")
        
        await asyncio.sleep(CONFIG["TOP_SYMBOLS_UPDATE_INTERVAL"])

# === WebSocket клиент для получения цен в реальном времени ===
async def binance_websocket_client():
    """WebSocket-клиент для получения цен в реальном времени"""
    # Объединяем символы для анализа и активные сделки
    symbols = list(set(CONFIG["MARKET_ANALYSIS_SYMBOLS"] + list(active_trades.keys())))
    streams = [f"{symbol.lower()}@ticker" for symbol in symbols]
    
    # Разбиваем на группы по 50 символов
    chunk_size = CONFIG["CHUNK_SIZE"]
    stream_chunks = [streams[i:i + chunk_size] for i in range(0, len(streams), chunk_size)]
    
    for chunk in stream_chunks:
        stream_url = f"wss://fstream.binance.com/stream?streams={'/'.join(chunk)}"
        asyncio.create_task(websocket_connection(stream_url))

# === Функция для подключения к WebSocket и обработки сообщений ===
async def websocket_connection(stream_url):
    """Отдельное соединение WebSocket для группы символов"""
    while True:
        try:
            async with websockets.connect(stream_url) as websocket:
                print(f"[WS] Connected to {stream_url}")
                async for message in websocket:
                    data = json.loads(message)
                    if 'data' in data and 's' in data['data']:
                        symbol = data['data']['s']
                        last_price = float(data['data']['c'])
                        last_prices[symbol] = last_price
        except (ConnectionClosed, Exception) as e:
            print(f"[WS] Connection error: {e}, reconnecting in 5 seconds...")
            await asyncio.sleep(5)

# === Функция отправки сообщений в Telegram с обработкой ошибок ===
async def send_message(msg, reply_to_message_id=None):
    try:
        print(f"[SEND] {msg}")
        sent = bot.send_message(
            TG_CHAT_ID, msg,
            parse_mode='HTML',
            disable_web_page_preview=True,
            reply_to_message_id=reply_to_message_id
        )
        return sent.message_id
    except Exception as e:
        print(f"[ERROR] Telegram: {e}")
        return None

# === Расчет разворота у монеты ===
def calculate_retracement_percentage(trade, current_price):
    """
    Расчет процента отката от точки входа к стоп-лоссу.
    Возвращает процент (0-100) или None при ошибке.
    """
    try:
        # Получаем цену входа с проверкой на None
        entry = trade.get("entry_real", trade.get("entry"))
        if entry is None:
            print(f"[WARNING] Отсутствует цена входа для {trade.get('symbol', 'unknown')}")
            return None
        
        # Получаем стоп-лосс с проверкой на None
        stop = trade.get("stop")
        if stop is None:
            print(f"[WARNING] Отсутствует стоп-лосс для {trade.get('symbol', 'unknown')}")
            return None
        
        # Проверяем, что current_price не None
        if current_price is None:
            print(f"[WARNING] Отсутствует текущая цена для {trade.get('symbol', 'unknown')}")
            return None
        
        # Проверяем, что все значения являются числами
        if not all(isinstance(x, (int, float)) for x in [entry, stop, current_price]):
            print(f"[WARNING] Нечисловые значения в данных сделки {trade.get('symbol', 'unknown')}")
            return None
        
        if trade["side"] == "long":
            # Для лонга: откат от entry к stop
            total_risk = entry - stop
            if total_risk <= 0:
                print(f"[WARNING] Некорректный расчет риска для лонга: entry={entry}, stop={stop}")
                return 100  # Если стоп выше входа, считаем откат 100%
                
            current_risk = entry - current_price
            retracement_pct = (current_risk / total_risk) * 100
            
            # Ограничиваем значения от 0 до 100
            return max(0, min(100, retracement_pct))
        else:
            # Для шорта: откат от entry к stop
            total_risk = stop - entry
            if total_risk <= 0:
                print(f"[WARNING] Некорректный расчет риска для шорта: entry={entry}, stop={stop}")
                return 100  # Если стоп ниже входа, считаем откат 100%
                
            current_risk = current_price - entry
            retracement_pct = (current_risk / total_risk) * 100
            
            # Ограничиваем значения от 0 до 100
            return max(0, min(100, retracement_pct))
            
    except Exception as e:
        print(f"[ERROR] calculate_retracement_percentage: {e}")
        print(f"[DEBUG] Trade data: {trade}")
        print(f"[DEBUG] Current price: {current_price}")
        return None

async def send_retracement_alert(symbol, trade, current_price, retracement_pct, alert_level):
    """
    Отправка уведомления об откате в Telegram.
    """
    side_icon = "🟢" if trade["side"] == "long" else "🔴"
    entry = trade.get("entry_real", trade["entry"])
    stop = trade["stop"]
    link = f"https://www.tradingview.com/chart/?symbol=BINANCE%3A{symbol}.P"
    
    # Обновляем статистику
    if alert_level == 1:
        daily_stats["retracement_level1_alerts"] += 1
        daily_stats["retracement_unique_symbols"].add(symbol)
        message = (
            f"⚠️ #ALERT Lvl 1 | {side_icon} <a href='{link}'>{symbol}</a> | ID: <code>{trade['trade_id']}</code>\n"
            f"📉 Откат к стопу: <b>{retracement_pct:.1f}%</b>\n"
            f"📍 ТВХ: <code>{entry}</code> | Цена: <code>{current_price}</code> | Стоп: <code>{stop}</code>\n"
            f"💡 Риск: возможно рынок развернулся против позиции"
        )
    else:
        daily_stats["retracement_level2_alerts"] += 1
        daily_stats["retracement_unique_symbols"].add(symbol)
        message = (
            f"🚨 #ALERT Lvl 2 | {side_icon} <a href='{link}'>{symbol}</a> | ID: <code>{trade['trade_id']}</code>\n"
            f"📉 Откат к стопу: <b>{retracement_pct:.1f}%</b>\n"
            f"📍 ТВХ: <code>{entry}</code> | Цена: <code>{current_price}</code> | Стоп: <code>{stop}</code>\n"
            f"💡 Высокий риск: рассмотрите выход/стоп"
        )
    
    # Отправляем как ответ на оригинальное сообщение о сделке
    await send_message(message, reply_to_message_id=trade["message_id"])

# === Фоновая задача для мониторинга откатов ===
async def retracement_monitor_loop():
    """
    Мониторинг откатов активных сделок.
    Проверяет сделки, которые еще не достигли TP1.
    """
    # Проверяем, включен ли мониторинг откатов
    if not CONFIG.get("RETRACEMENT_ALERTS_ENABLED", True):
        print("[RETRACEMENT] Мониторинг откатов отключен в конфигурации")
        return
    
    while True:
        try:
            # Проверяем, не был ли мониторинг отключен во время работы
            if not CONFIG.get("RETRACEMENT_ALERTS_ENABLED", True):
                print("[RETRACEMENT] Мониторинг откатов отключен, завершаем работу")
                break
                
            print("[RETRACEMENT] Начинаем проверку активных сделок...")
            current_time = time.time()
            
            for symbol, trade in list(active_trades.items()):
                # Пропускаем сделки, которые уже достигли TP1 или закрыты
                if trade.get("partial_taken", False) or trade.get("status") != "open":
                    continue
                
                # Получаем текущую цену
                current_price = last_prices.get(symbol)
                if current_price is None:
                    # Fallback на REST API
                    try:
                        ticker = client.futures_symbol_ticker(symbol=symbol)
                        current_price = float(ticker["price"])
                        last_prices[symbol] = current_price
                    except Exception as e:
                        print(f"[ERROR] Не удалось получить цену для {symbol}: {e}")
                        continue
                
                # Рассчитываем процент отката
                retracement_pct = calculate_retracement_percentage(trade, current_price)
                if retracement_pct is None:
                    continue  # Пропускаем эту сделку, если расчет не удался
                
                # Определяем уровень предупреждения
                alert_level = 0
                if retracement_pct >= CONFIG["RETRACEMENT_LEVEL_2"]:
                    alert_level = 2
                elif retracement_pct >= CONFIG["RETRACEMENT_LEVEL_1"]:
                    alert_level = 1
                
                # Проверяем, нужно ли отправлять уведомление
                last_alert = retracement_alerts_sent.get(symbol, {})
                last_level = last_alert.get("level", 0)
                last_time = last_alert.get("time", 0)
                
                # Получаем задержку из конфига
                cooldown = CONFIG.get("RETRACEMENT_ALERT_COOLDOWN", 14400)
                
                # Отправляем уведомление если:
                # 1. Уровень повысился
                # 2. Или прошло достаточно времени с последнего уведомления того же уровня
                if alert_level > last_level or (alert_level > 0 and current_time - last_time > cooldown):
                    await send_retracement_alert(symbol, trade, current_price, retracement_pct, alert_level)
                    retracement_alerts_sent[symbol] = {"level": alert_level, "time": current_time}
            
            print(f"[RETRACEMENT] Проверка завершена. Следующая через {CONFIG['RETRACEMENT_MONITOR_INTERVAL']} сек.")
            await asyncio.sleep(CONFIG["RETRACEMENT_MONITOR_INTERVAL"])
            
        except Exception as e:
            print(f"[ERROR] retracement_monitor_loop: {e}")
            await asyncio.sleep(300)  # Пауза при ошибке

# === Фоновая задача для анализа рынка и отправки сообщений ===
async def market_analysis_loop(send_message, client, config):

    high_volatility = False
    low_volatility = False
    last_vol_msg_time = 0
    last_vol_state = None

    # === Функция для расчёта индикаторов ===
    def calculate_indicators(df):
        df = df.copy()
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["open"] = df["open"].astype(float)
        df["volume"] = df["volume"].astype(float)

        # EMA и индикаторы (из Версии 2)
        ema_fast = EMAIndicator(df["close"], window=8).ema_indicator()
        ema_slow = EMAIndicator(df["close"], window=21).ema_indicator()

        #MACD
        macd_indicator = MACD(
            df["close"],
            window_slow=config.get("MACD_SLOW", 26),
            window_fast=config.get("MACD_FAST", 12),
            window_sign=config.get("MACD_SIGNAL", 9),
        )
        
        # Рассчитываем гистограмму MACD
        macd_diff = macd_indicator.macd_diff()

        try:
            if len(df) >= 15:
                adx = ADXIndicator(df["high"], df["low"], df["close"], window=config.get("ADX_WINDOW", 14)).adx()
            else:
                adx = None
        except Exception as e:
            print(f"Error calculating ADX: {e}")
            adx = None

        atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
        df["atr"] = atr
        mean_atr = df["atr"].rolling(window=20).mean().iloc[-1]

        #RSI
        rsi = RSIIndicator(df["close"], window=14).rsi()
        prev_rsi = rsi.shift(1).iloc[-1] if len(rsi) >= 2 else rsi.iloc[-1]
        rsi_delta = rsi.iloc[-1] - prev_rsi

        #Stoch
        stoch = StochasticOscillator(
            df["high"], df["low"], df["close"], 
            window=14, smooth_window=3
            ).stoch()
        
        stoch_value = stoch.iloc[-1] if len(stoch) else None
        prev_stoch_value = stoch.iloc[-2] if len(stoch) > 1 else None

        # Stoch - разница между последним и предыдущим значением (импульс)
        stoch_delta = None
        if stoch_value is not None and prev_stoch_value is not None:
            stoch_delta = round(stoch_value - prev_stoch_value, 2)

        # VWAP
        df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()

        # OBV
        obv = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()

        last_candle_body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
        last_candle_range = df["high"].iloc[-1] - df["low"].iloc[-1]

        df["range"] = df["high"] - df["low"]
        mean_range = df["range"].rolling(window=20).mean().iloc[-1]
        
        # Тело свечи и нормализация (из Версии 2)
        candle_body = abs(df["close"].iloc[-2] - df["open"].iloc[-2])
        normalized_body = candle_body / mean_range if mean_range > 0 else candle_body

        # ЗАМЕНА: Расчет объема из Версии 1 вместо Версии 2
        # --- Накопленный объём и дельта ---
        N = config.get("VOLUME_ACCUM_WINDOW", 3)
        volume_accum = df["volume"].iloc[-N:].sum()
        volume_accum_prev = df["volume"].iloc[-2*N:-N].sum() if len(df) >= 2*N else df["volume"].iloc[-N:].sum()
        volume_delta = volume_accum - volume_accum_prev

        vol_mean_accum = df["volume"].ewm(span=10, adjust=False).mean().iloc[-2] * N
        vol_mean_long_accum = df["volume"].rolling(window=50).mean().iloc[-2] * N

        # TRIX (из Версии 2)
        trix_window = config.get("TRIX_WINDOW", 15)
        
        # Рассчитываем тройную EMA
        ema1 = df["close"].ewm(span=trix_window, adjust=False).mean()
        ema2 = ema1.ewm(span=trix_window, adjust=False).mean()
        ema3 = ema2.ewm(span=trix_window, adjust=False).mean()
        
        # TRIX = процентное изменение тройной EMA
        trix = (ema3 - ema3.shift(1)) / ema3.shift(1).replace(0, 1e-10) * 100


        if "open_interest" in df.columns:
            oi = df["open_interest"].astype(float)
            oi_change = oi.pct_change().iloc[-1] * 100  # % изменения
        else:
            oi_change = 0

        return {
            "ema_fast": ema_fast.iloc[-1],
            "ema_slow": ema_slow.iloc[-1],
            "macd": macd_indicator.macd().iloc[-1],
            "macd_signal": macd_indicator.macd_signal().iloc[-1],
            "macd_diff": macd_diff.iloc[-1],
            "prev_macd_diff": macd_diff.iloc[-2] if len(macd_diff) >= 2 else None,
            "adx": adx.iloc[-1] if adx is not None else None,  # Добавлена проверка на None
            "atr": atr.iloc[-1],
            "mean_atr": mean_atr,
            "rsi": rsi.iloc[-1],
            "prev_rsi": prev_rsi,
            "rsi_delta": rsi_delta,
            "stoch": stoch_value,
            "prev_stoch": prev_stoch_value,
            "stoch_delta": stoch_delta,
            # ЗАМЕНА: Используем накопленный объем вместо объема одной свечи
            "volume": volume_accum,
            "volume_delta": volume_delta,
            "vol_mean": vol_mean_accum,
            "vol_mean_long": vol_mean_long_accum,
            "normalized_body": normalized_body,
            "vwap": df["vwap"].iloc[-1],
            "obv": obv.iloc[-1],
            "obv_change": obv.iloc[-1] - obv.iloc[-2],
            "last_close": df["close"].iloc[-1],
            "last_open": df["open"].iloc[-1],
            "last_high": df["high"].iloc[-1],
            "last_low": df["low"].iloc[-1],
            "last_candle_body": last_candle_body,
            "candle_body": candle_body,
            "last_candle_range": last_candle_range,
            "mean_range": mean_range,
            "trix": trix.iloc[-1],
            "prev_trix": trix.iloc[-2] if len(trix) >= 2 else None,
            "oi_change": oi_change
        }

    # === Функция для расчёта ZLEMA ===
    def calculate_zlema(close_series: pd.Series, length: int) -> pd.Series:
        if len(close_series) < length:
            return pd.Series([np.nan] * len(close_series), index=close_series.index)

        lag = (length - 1) // 2
        price_adjusted = close_series + (close_series - close_series.shift(lag))
        return price_adjusted.ewm(span=length, adjust=False).mean()
    
    def is_supertrend_up(df, atr_period: int, multiplier: float) -> bool:
        """
        Определяет направление SuperTrend на последнем баре.
        Возвращает True, если тренд восходящий, False — если нисходящий.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # ATR
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1)
        tr = tr.max(axis=1)
        atr = tr.rolling(atr_period).mean()

        # SuperTrend базовый
        hl2 = (high + low) / 2
        upperband = hl2 + multiplier * atr
        lowerband = hl2 - multiplier * atr

        # Инициализация
        direction = np.ones(len(df))  # 1 = up, -1 = down

        for i in range(1, len(df)):
            if close.iloc[i] > upperband.iloc[i-1]:
                direction[i] = 1
            elif close.iloc[i] < lowerband.iloc[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]

        return direction[-1] == 1

    # === Функция для получения глобального тренда ===
    def get_trend_trix(symbol: str, df_dict, config: dict) -> Optional[str]:
        """
        Определяет направление тренда на основе анализа старших и младших таймфреймов.
        
        Args:
            symbol: Символ актива
            df_dict: Словарь с DataFrame для разных таймфреймов
            config: Словарь с настройками
            
        Returns:
            Optional[str]: 'long', 'short' или None (неопределенный тренд)
        """
        try:
            # Конфигурируемые параметры
            DEBUG = config.get("GT_DEBUG_TREND", False)
            ADX_THRESHOLD = config.get("GT_ADX_THRESHOLD", 20)
            EMA_LENGTH = config.get("GT_TREND_EMA_LENGTH", 50)
            ZLEMA_LENGTH = config.get("GT_ZLEMA_LENGTH", 20)
            TRIX_WINDOW = config.get("GT_TRIX_WINDOW", 15)
            MIN_VOLATILITY_RATIO = config.get("GT_MIN_VOLATILITY_RATIO", 0.001)
            MIN_LOCAL_CONFIRMATIONS = config.get("GT_MIN_LOCAL_CONFIRMATIONS", 2)
            
            # Определяем таймфреймы
            GLOBAL_TFS = ["1d", "4h"]
            LOCAL_TFS = ["1h", "15m", "5m"]
            
            trends = {}
            adx_values = {}
            
            if DEBUG:
                print(f"[DEBUG TREND] Анализ тренда для {symbol}")
            
            # 1. Анализ глобального тренда (D1, 4H)
            for tf in GLOBAL_TFS:
                df = df_dict.get(tf)
                if df is None or df.empty or len(df) < EMA_LENGTH * 2:
                    if DEBUG:
                        print(f"[DEBUG TREND] {tf}: недостаточно данных")
                    trends[tf] = "neutral"
                    adx_values[tf] = 0
                    continue
                    
                # Приводим данные к float
                for col in ["close", "high", "low", "open"]:
                    df[col] = df[col].astype(float)
                    
                # Рассчитываем индикаторы
                ema = EMAIndicator(df["close"], window=EMA_LENGTH).ema_indicator()
                adx = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
                current_adx = adx.iloc[-1]
                adx_values[tf] = current_adx
                
                # Определяем тренд для данного TF
                if current_adx >= ADX_THRESHOLD:
                    if df["close"].iloc[-1] > ema.iloc[-1]:
                        trends[tf] = "long"
                    else:
                        trends[tf] = "short"
                else:
                    trends[tf] = "neutral"
                    
                if DEBUG:
                    print(f"[DEBUG TREND] {tf}: тренд={trends[tf]}, ADX={current_adx:.2f}")
            
            # 2. Принятие решения о глобальном тренде
            global_trend = None
            
            # Приоритет 1: D1 и 4H совпадают
            if trends["1d"] == trends["4h"] and trends["1d"] != "neutral":
                global_trend = trends["1d"]
                if DEBUG:
                    print(f"[DEBUG TREND] Глобальный тренд (совпадают D1+4H): {global_trend}")
            
            # Приоритет 2: только 4H имеет сильный тренд
            elif trends["4h"] != "neutral" and adx_values["4h"] >= ADX_THRESHOLD:
                global_trend = trends["4h"]
                if DEBUG:
                    print(f"[DEBUG TREND] Глобальный тренд (только 4H): {global_trend}")
            
            # Глобальный тренд не определен
            if global_trend is None:
                if DEBUG:
                    print("[DEBUG TREND] Глобальный тренд не определён")
                return None
            
            # 3. Проверка подтверждения на локальных ТФ
            confirmations = 0
            
            for tf in LOCAL_TFS:
                df = df_dict.get(tf)
                if df is None or df.empty or len(df) < ZLEMA_LENGTH * 3:
                    if DEBUG:
                        print(f"[DEBUG TREND] {tf}: недостаточно данных для подтверждения")
                    continue
                    
                # Приводим данные к float
                for col in ["close", "high", "low"]:
                    df[col] = df[col].astype(float)
                    
                # Рассчитываем индикаторы
                zlema = calculate_zlema(df["close"], ZLEMA_LENGTH)
                atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
                
                # Рассчитываем TRIX
                ema1 = df["close"].ewm(span=TRIX_WINDOW, adjust=False).mean()
                ema2 = ema1.ewm(span=TRIX_WINDOW, adjust=False).mean()
                ema3 = ema2.ewm(span=TRIX_WINDOW, adjust=False).mean()
                trix = (ema3 - ema3.shift(1)) / ema3.shift(1).abs() * 100
                
                # Проверяем волатильность
                volatility_ratio = atr.iloc[-1] / df["close"].iloc[-1]
                if volatility_ratio < MIN_VOLATILITY_RATIO:
                    if DEBUG:
                        print(f"[DEBUG TREND] {tf}: слишком низкая волатильность ({volatility_ratio:.4f})")
                    continue
                
                # Проверяем подтверждение тренда
                confirms_trend = False
                if global_trend == "long":
                    confirms_trend = (df["close"].iloc[-1] > zlema.iloc[-1] and 
                                    trix.iloc[-1] > 0)
                else:  # short
                    confirms_trend = (df["close"].iloc[-1] < zlema.iloc[-1] and 
                                    trix.iloc[-1] < 0)
                
                if confirms_trend:
                    confirmations += 1
                    if DEBUG:
                        print(f"[DEBUG TREND] {tf}: подтверждает {global_trend}")
                else:
                    if DEBUG:
                        print(f"[DEBUG TREND] {tf}: не подтверждает {global_trend}")
            
            # 4. Финальное решение
            if confirmations >= MIN_LOCAL_CONFIRMATIONS:
                if DEBUG:
                    print(f"[DEBUG TREND] Финальный тренд: {global_trend} (подтверждений: {confirmations})")
                return global_trend
            else:
                if DEBUG:
                    print(f"[DEBUG TREND] Недостаточно подтверждений: {confirmations}")
                return None
                
        except Exception as e:
            print(f"[ERROR] get_trend для {symbol}: {str(e)}")
            if config.get("DEBUG_TREND", False):
                import traceback
                traceback.print_exc()
            return None

    def get_trend(symbol: str, df_dict, length=20) -> Optional[str]:
        try:
            timeframes = ["5m", "15m", "30m", "1h", "2h", "4h", "1d"]
            weights = {
                "5m": 0.07,
                "15m": 0.1,
                "30m": 0.1,
                "1h": 0.15,
                "2h": 0.2,
                "4h": 0.2,
                "1d": 0.18
            }
            local_tfs = ["5m", "15m", "30m", "1h", "2h", "4h"]
            trends = {}
            
            # 1. Сбор данных по всем ТФ
            for tf in timeframes:

                df = df_dict.get(tf)

                if df is None or df.empty:
                    print(f"[SKIP TREND] {symbol} {tf}: нет данных")
                    trends[tf] = "neutral"
                    continue

                df["close"] = df["close"].astype(float)
                df["high"] = df["high"].astype(float)
                df["low"] = df["low"].astype(float)
                df["open"] = df["open"].astype(float)
                df["volume"] = df["volume"].astype(float)
                
                if len(df) < length * 3:
                    print(f"[SKIP ZLEMA] {symbol} {tf}: слишком мало свечей ({len(df)}), нужно ≥ {length * 3}")
                    trends[tf] = "neutral"
                    continue
                    
                # 2. Расчёт индикаторов
                zlema = calculate_zlema(df["close"], length)
                atr = AverageTrueRange(df["high"], df["low"], df["close"], window=length).average_true_range()
                volatility_ratio = atr.iloc[-1] / df["close"].iloc[-1]
                mult = 0.3 if volatility_ratio > 0.02 else 0.7
                volatility = atr.rolling(window=length*3).max() * mult
                
                # 3. Определение тренда
                trend = 0
                for i in range(length*3, len(df)):
                    price_acceleration = df["close"].iloc[i] / df["close"].iloc[i-3] - 1
                    if abs(price_acceleration) > 0.05:
                        trend = 1 if price_acceleration > 0 else -1
                    else:
                        price = df["close"].iloc[i]
                        upper_band = zlema.iloc[i] + volatility.iloc[i]
                        lower_band = zlema.iloc[i] - volatility.iloc[i]
                        if price > upper_band:
                            trend = 1
                        elif price < lower_band:
                            trend = -1
                            
                trends[tf] = "long" if trend == 1 else "short" if trend == -1 else "neutral"

            # 4. Приоритет D1 и H4
            if trends["1d"] == trends["4h"] and trends["1d"] in ["long", "short"]:
                return trends["1d"]

            # 5. Взвешенное голосование
            long_score = sum(weights[tf] for tf in local_tfs if trends[tf] == "long")
            short_score = sum(weights[tf] for tf in local_tfs if trends[tf] == "short")
            long_matches = sum(1 for tf in local_tfs if trends[tf] == "long")
            short_matches = sum(1 for tf in local_tfs if trends[tf] == "short")

            # 6. Логирование (опционально)
            if config.get("DEBUG_TREND", False):
                print(f"[DEBUG] Тренды для {symbol}:")
                for tf in timeframes:
                    print(f"{tf}: {trends[tf]} (вес {weights[tf]:.2f})")
                print(f"Итог: Long {long_score:.2f} ({long_matches} ТФ), Short {short_score:.2f} ({short_matches} ТФ)")

            # 7. Финальное решение
            score_margin = config.get("TREND_SCORE_MARGIN", 0.1)
            min_matches = config.get("MIN_TF_TREND_MATCHES", 4)
            
            if long_score > short_score + score_margin and long_matches >= min_matches:
                return "long"
            elif short_score > long_score + score_margin and short_matches >= min_matches:
                return "short"
            else:
                print(f"[FILTER TREND] {symbol}: неопределённый тренд (long {long_score:.1f} vs short {short_score:.1f})")
                return None

        except Exception as e:
            print(f"[ERROR] get_trend для {symbol}: {str(e)}")
            return None
    
    def get_trend_supertrend(symbol: str, df_dict, config: dict) -> Optional[str]:
        """
        Определяет направление тренда на основе анализа старших и младших таймфреймов.

        Args:
            symbol: Символ актива
            df_dict: Словарь с DataFrame для разных таймфреймов
            config: Словарь с настройками

        Returns:
            Optional[str]: 'long', 'short' или None (неопределенный тренд)
        """
        try:
            # Конфигурируемые параметры
            DEBUG = config.get("GT_DEBUG_TREND", False)
            EMA_FAST = config.get("GT_EMA_FAST", 9)
            EMA_SLOW = config.get("GT_EMA_SLOW", 26)
            ZLEMA_LENGTH = config.get("GT_ZLEMA_LENGTH", 20)
            MIN_VOLATILITY_RATIO = config.get("GT_MIN_VOLATILITY_RATIO", 0.001)
            MIN_LOCAL_CONFIRMATIONS = config.get("GT_MIN_LOCAL_CONFIRMATIONS", 2)
            SUPERTREND_ATR_PERIOD = config.get("GT_SUPERTREND_ATR_PERIOD", 10)
            SUPERTREND_MULTIPLIER = config.get("GT_SUPERTREND_MULTIPLIER", 3)

            GLOBAL_TFS = config.get("GT_GLOBAL_TFS", ["1d", "4h"])
            LOCAL_TFS = config.get("GT_LOCAL_TFS", ["1h", "15m", "5m"])

            trends = {}

            if DEBUG:
                print(f"[DEBUG GET_TREND_SUPERTREND] Анализ тренда для {symbol}")

            # 1. Глобальный тренд
            for tf in GLOBAL_TFS:
                df = df_dict.get(tf)
                if df is None or df.empty or len(df) < EMA_SLOW * 2:
                    if DEBUG:
                        print(f"[DEBUG GET_TREND_SUPERTREND] {tf}: недостаточно данных")
                    trends[tf] = "neutral"
                    continue

                for col in ["close", "high", "low", "open"]:
                    df[col] = df[col].astype(float)

                # EMA быстрый и медленный
                ema_fast = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
                ema_slow = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

                # Разворот EMA (slope)
                slope = ema_fast.iloc[-1] - ema_fast.iloc[-2]

                # Определяем глобальный тренд
                if ema_fast.iloc[-1] > ema_slow.iloc[-1] or slope > 0:
                    trends[tf] = "long"
                elif ema_fast.iloc[-1] < ema_slow.iloc[-1] or slope < 0:
                    trends[tf] = "short"
                else:
                    trends[tf] = "neutral"

                if DEBUG:
                    print(f"[DEBUG GET_TREND_SUPERTREND] {tf}: тренд={trends[tf]}, ema_slope={slope:.5f}")

            # 2. Принятие решения о глобальном тренде
            global_trend = None
            if trends["1d"] == trends["4h"] and trends["1d"] != "neutral":
                global_trend = trends["1d"]
                if DEBUG:
                    print(f"[DEBUG GET_TREND_SUPERTREND] Глобальный тренд (совпадают D1+4H): {global_trend}")
            elif trends["4h"] != "neutral":
                global_trend = trends["4h"]
                if DEBUG:
                    print(f"[DEBUG GET_TREND_SUPERTREND] Глобальный тренд (только 4H): {global_trend}")

            if global_trend is None:
                if DEBUG:
                    print("[DEBUG GET_TREND_SUPERTREND] Глобальный тренд не определён")
                return None

            # 3. Проверка подтверждения на локальных ТФ
            confirmations = 0
            USE_SUPERTREND = config.get("GT_USE_SUPERTREND", True)

            for tf in LOCAL_TFS:
                df = df_dict.get(tf)
                if df is None or df.empty or len(df) < ZLEMA_LENGTH * 3:
                    if DEBUG:
                        print(f"[DEBUG GET_TREND_SUPERTREND] {tf}: недостаточно данных для подтверждения")
                    continue

                for col in ["close", "high", "low"]:
                    df[col] = df[col].astype(float)

                zlema = calculate_zlema(df["close"], ZLEMA_LENGTH)
                atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

                volatility_ratio = atr.iloc[-1] / df["close"].iloc[-1]
                if volatility_ratio < MIN_VOLATILITY_RATIO:
                    if DEBUG:
                        print(f"[DEBUG GET_TREND_SUPERTREND] {tf}: слишком низкая волатильность ({volatility_ratio:.4f})")
                    continue

                # SuperTrend вместо TRIX
                if USE_SUPERTREND:
                    supertrend_dir = 1 if is_supertrend_up(
                        df, 
                        SUPERTREND_ATR_PERIOD, 
                        SUPERTREND_MULTIPLIER
                    ) else -1
                else:
                    # Альтернативная логика, если SuperTrend отключен
                    supertrend_dir = 1 if df["close"].iloc[-1] > zlema.iloc[-1] else -1

                confirms_trend = False
                if global_trend == "long":
                    confirms_trend = (df["close"].iloc[-1] > zlema.iloc[-1] and supertrend_dir > 0)
                else:  # short
                    confirms_trend = (df["close"].iloc[-1] < zlema.iloc[-1] and supertrend_dir < 0)

                if confirms_trend:
                    confirmations += 1
                    if DEBUG:
                        print(f"[DEBUG GET_TREND_SUPERTREND] {tf}: подтверждает {global_trend}")
                else:
                    if DEBUG:
                        print(f"[DEBUG GET_TREND_SUPERTREND] {tf}: не подтверждает {global_trend}")

            # 4. Финальное решение
            if confirmations >= MIN_LOCAL_CONFIRMATIONS:
                if DEBUG:
                    print(f"[DEBUG GET_TREND_SUPERTREND] Финальный тренд: {global_trend} (подтверждений: {confirmations})")
                return global_trend
            else:
                if DEBUG:
                    print(f"[DEBUG GET_TREND_SUPERTREND] Недостаточно подтверждений: {confirmations}")
                return None

        except Exception as e:
            print(f"[ERROR] get_trend для {symbol}: {str(e)}")
            if config.get("DEBUG_TREND", False):
                import traceback
                traceback.print_exc()
            return None
        
    # === Функция для оценки потенциала сигнала по нескольким ТФ ===
    def assess_signal_potential_multi_tf(symbol: str, df_dict, config: dict) -> float:
        """
        Оценивает силу сигнала по нескольким таймфреймам.
        Возвращает итоговый балл от 0 до 1.
        Теперь использует уже загруженные данные из df_dict.
        """
        tfs = config.get("POTENTIAL_TFS", ["15m"])
        weights = config.get("POTENTIAL_TF_WEIGHTS", [1.0])
        assert len(tfs) == len(weights), "Длины POTENTIAL_TFS и POTENTIAL_TF_WEIGHTS не совпадают"

        total_score = 0.0
        total_weight = 0.0

        for tf, weight in zip(tfs, weights):
            try:
                # Получаем данные из df_dict
                df = df_dict.get(tf)
                
                # Проверяем, что данные не None и не пустые
                if df is None or df.empty or len(df) < 50:
                    print(f"[POTENTIAL_FILTER] Недостаточно данных для {symbol} на {tf}")
                    continue

                # --- Momentum check ---
                recent_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
                momentum_score = 1.0 if abs(recent_change) < 0.02 else 0.3

                # --- OBV check ---
                df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
                obv_change = df["obv"].iloc[-1] - df["obv"].iloc[-5]
                obv_score = 1.0 if obv_change > 0 else 0.0

                # --- ATR check ---
                atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
                atr_val = atr.iloc[-1]
                atr_score = 1.0 if atr_val > df["close"].iloc[-1] * 0.0015 else 0.5

                # --- Финальный скор для одного ТФ ---
                tf_score = (momentum_score + obv_score + atr_score) / 3
                total_score += tf_score * weight
                total_weight += weight

            except Exception as e:
                print(f"[POTENTIAL_FILTER] Ошибка при расчёте по {tf}: {e}")
                continue

        return total_score / total_weight if total_weight > 0 else 0.0

    # S/D зоны функции
    # === Функция для проверки дивергенций на нескольких ТФ ===
    def detect_divergence_multi_tf(symbol: str, df_dict, config) -> dict:
        """
        Проверяет RSI и MACD дивергенции на заданных ТФ.
        Возвращает:
            {
                "rsi": {
                    "bullish": True/False,
                    "bearish": True/False,
                    "hidden_bullish": True/False,
                    "hidden_bearish": True/False
                },
                "macd": {
                    "bullish": True/False,
                    "bearish": True/False,
                    "hidden_bullish": True/False,
                    "hidden_bearish": True/False
                }
            }
        """
        result = {
            "rsi": {
                "bullish": False,
                "bearish": False,
                "hidden_bullish": False,
                "hidden_bearish": False
            },
            "macd": {
                "bullish": False,
                "bearish": False,
                "hidden_bullish": False,
                "hidden_bearish": False
            }
        }

        tfs = config.get("DIVERGENCE_TFS", ["15m"])
        lookback = config.get("DIVERGENCE_LOOKBACK", 5)
        check_rsi = config.get("CHECK_RSI_DIVERGENCE", True)
        check_macd = config.get("CHECK_MACD_DIVERGENCE", True)

        for tf in tfs:
            df = df_dict.get(tf)
            
            # Проверяем, что данные не None и не пустые
            if df is None or df.empty or len(df) < lookback + 10:
                print(f"[WARNING] Недостаточно данных для анализа дивергенции {symbol} на {tf}")
                continue

            try:

                if check_rsi:
                    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
                    curr_close = df["close"].iloc[-1]
                    prev_close = df["close"].iloc[-lookback]
                    curr_rsi = df["rsi"].iloc[-1]
                    prev_rsi = df["rsi"].iloc[-lookback]

                    # Классическая бычья
                    if curr_close < prev_close and curr_rsi > prev_rsi:
                        result["rsi"]["bullish"] = True
                    # Классическая медвежья
                    if curr_close > prev_close and curr_rsi < prev_rsi:
                        result["rsi"]["bearish"] = True
                    # Скрытая бычья (продолжение up-тренда)
                    if curr_close > prev_close and curr_rsi < prev_rsi:
                        result["rsi"]["hidden_bullish"] = True
                    # Скрытая медвежья (продолжение даун-тренда)
                    if curr_close < prev_close and curr_rsi > prev_rsi:
                        result["rsi"]["hidden_bearish"] = True

                if check_macd:
                    macd_line = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
                    signal_line = macd_line.ewm(span=9).mean()
                    df["macd_hist"] = macd_line - signal_line

                    curr_close = df["close"].iloc[-1]
                    prev_close = df["close"].iloc[-lookback]
                    curr_hist = df["macd_hist"].iloc[-1]
                    prev_hist = df["macd_hist"].iloc[-lookback]

                    # Классическая бычья
                    if curr_close < prev_close and curr_hist > prev_hist:
                        result["macd"]["bullish"] = True
                    # Классическая медвежья
                    if curr_close > prev_close and curr_hist < prev_hist:
                        result["macd"]["bearish"] = True
                    # Скрытая бычья
                    if curr_close > prev_close and curr_hist < prev_hist:
                        result["macd"]["hidden_bullish"] = True
                    # Скрытая медвежья
                    if curr_close < prev_close and curr_hist > prev_hist:
                        result["macd"]["hidden_bearish"] = True

            except Exception as e:
                print(f"[DIVERGENCE] Ошибка на {tf}: {e}")
                continue

        return result

    # === Функция для оценки реакции цены на уровни ===
    def calculate_price_reaction(df: pd.DataFrame, level: float, touch_index: int, tolerance: float = 0.005) -> str:
        """
        Определяет тип реакции цены на уровень поддержки/сопротивления.
        
        Args:
            df: DataFrame с данными
            level: Уровень поддержки/сопротивления
            touch_index: Индекс свечи, коснувшейся уровня
            tolerance: Допуск (0.5%)
            
        Returns:
            "bounce" - сильный отскок
            "consolidation" - консолидация
            "break" - пробой уровня
            "weak" - слабая реакция
        """
        if touch_index >= len(df) - 1:
            return "weak"
        
        # Определяем тип уровня
        is_resistance = df.iloc[touch_index]["high"] >= level
        is_support = df.iloc[touch_index]["low"] <= level
        
        # Анализируем последующие 3 свечи
        reaction_window = df.iloc[touch_index+1:touch_index+4]
        if len(reaction_window) < 2:
            return "weak"
        
        # Сильный отскок
        if is_resistance:
            if all(reaction_window["close"] < level * (1 - tolerance)):
                return "bounce"
        elif is_support:
            if all(reaction_window["close"] > level * (1 + tolerance)):
                return "bounce"
        
        # Пробой уровня
        if is_resistance:
            if any(reaction_window["close"] > level * (1 + tolerance)):
                return "break"
        elif is_support:
            if any(reaction_window["close"] < level * (1 - tolerance)):
                return "break"
        
        # Консолидация
        if all(abs(reaction_window["close"] - level) / level < tolerance):
            return "consolidation"
        
        return "weak"

    # === Функция для получения уровней поддержки/сопротивления ===
    def get_volume_at_level(df: pd.DataFrame, level: float, tolerance: float = 0.005) -> float:
        """
        Рассчитывает суммарный объем в зоне уровня.
        
        Args:
            df: DataFrame с данными
            level: Целевой уровень
            tolerance: Допуск (±0.5%)
            
        Returns:
            Суммарный объем в зоне
        """
        low_bound = level * (1 - tolerance)
        high_bound = level * (1 + tolerance)
        
        mask = (df["low"] >= low_bound) & (df["high"] <= high_bound)
        return df.loc[mask, "volume"].sum()

    # === Функция для проверки пробития уровня с подтверждением ===
    # Используется для фильтрации ложных пробитий
    def is_level_broken(df: pd.DataFrame, level: float, side: str, confirmation_bars: int = 2) -> bool:
        """
        Проверяет, был ли уровень пробит с подтверждением.
        
        Args:
            df: DataFrame с данными
            level: Проверяемый уровень
            side: 'support' или 'resistance'
            confirmation_bars: Количество свечей подтверждения
            
        Returns:
            True если уровень пробит с подтверждением
        """
        if len(df) < confirmation_bars + 1:
            return False
            
        # Проверяем последние N свечей
        recent = df.iloc[-confirmation_bars-1:-1]
        
        if side == "support":
            # Для поддержки: закрытие ниже уровня
            return any(recent["close"] < level)
        elif side == "resistance":
            # Для сопротивления: закрытие выше уровня
            return any(recent["close"] > level)
        return False

    # === Функция для получения улучшенных уровней поддержки/сопротивления ===
    def get_enhanced_support_resistance(df: pd.DataFrame, config: dict) -> dict:
        """
        Возвращает улучшенные данные об уровнях поддержки/сопротивления.
        
        Args:
            df: DataFrame с данными
            config: Конфигурация
            
        Returns:
            Словарь: {уровень: {type: str, touches: int, volume: float, strength: float}}
        """
        window = config.get("SD_SWING_WINDOW", 3)
        tolerance = config.get("SD_ZONE_TOLERANCE", 0.003)
        min_touches = config.get("SD_MIN_TOUCHES", 2)
        
        levels = {}
        price = df["close"].iloc[-1]
        
        # 1. Идентификация потенциальных уровней
        for i in range(window, len(df) - window):
            low_val = df["low"].iloc[i]
            high_val = df["high"].iloc[i]
            
            # Проверка на поддержку
            if all(low_val < df["low"].iloc[i-j] for j in range(1, window+1)) and \
            all(low_val < df["low"].iloc[i+j] for j in range(1, window+1)):
                level = low_val
                level_type = "support"
                
            # Проверка на сопротивление
            elif all(high_val > df["high"].iloc[i-j] for j in range(1, window+1)) and \
                all(high_val > df["high"].iloc[i+j] for j in range(1, window+1)):
                level = high_val
                level_type = "resistance"
            else:
                continue
                
            # Нормализация уровня
            level = round(level, 6)
            
            # Поиск существующего уровня в пределах допуска
            found = False
            for existing in list(levels.keys()):
                if abs(existing - level) / existing < tolerance:
                    level = existing  # Используем существующий уровень
                    found = True
                    break
                    
            if not found:
                levels[level] = {
                    "type": level_type,
                    "touches": 0,
                    "volume": 0,
                    "reactions": [],
                    "last_touched": len(df) - i
                }
                
            # Обновляем данные уровня
            levels[level]["touches"] += 1
            levels[level]["volume"] += get_volume_at_level(df, level, tolerance)
            levels[level]["reactions"].append(
                calculate_price_reaction(df, level, i, tolerance)
            )
            levels[level]["last_touched"] = min(levels[level]["last_touched"], len(df) - i)
        
        # 2. Фильтрация и расчет силы
        valid_levels = {}
        for level, data in levels.items():
            # Фильтр по минимальному количеству касаний
            if data["touches"] < min_touches:
                continue
                
            # Фильтр по свежести (последние 50 свечей)
            if data["last_touched"] > config.get("SD_LAST_TOUCHED", 50):
                continue
                
            # Расчет силы уровня
            bounce_count = data["reactions"].count("bounce")
            consolidation_count = data["reactions"].count("consolidation")
            strength = (
                data["touches"] * 0.4 +
                bounce_count * 0.5 +
                consolidation_count * 0.3 +
                min(data["volume"] / df["volume"].mean(), 5) * 0.3
            )
            
            # Применяем модификатор для текущей цены
            price_distance = abs(price - level) / price
            if price_distance < 0.03:  # 3%
                strength *= 1.5
            elif price_distance > 0.1:  # 10%
                strength *= 0.7
                
            data["strength"] = round(strength, 2)
            valid_levels[level] = data
        
        return valid_levels

    # TODO: оптимизировать - не пересчитывать уровни каждый раз
    # === Функция для проверки близости к значимым S/D уровням ===
    def check_proximity_to_sd_zone1(df: pd.DataFrame, side: str, config: dict) -> Optional[str]:
        """
        Улучшенная проверка близости к значимым S/D уровням.
        
        Args:
            df: DataFrame с данными
            side: 'long' или 'short'
            config: Конфигурация
            
        Returns:
            Причину блокировки или None
        """
        price = df["close"].iloc[-1]
        threshold = config.get("CONFIRMATION_SD_DISTANCE_THRESHOLD", 0.015)  # 1.5%
        min_strength = config.get("CONFIRMATION_SD_MIN_STRENGTH", 1.5)
        
        # Получаем улучшенные данные об уровнях
        levels = get_enhanced_support_resistance(df, config)
        
        if side == "long":
            # Фильтруем только сопротивления выше цены
            resistances = {
                lvl: data for lvl, data in levels.items() 
                if data["type"] == "resistance" and lvl > price
            }
            
            if not resistances:
                return None
                
            # Находим ближайшее сильное сопротивление
            nearest_resistance = min(resistances.keys(), key=lambda x: abs(x - price))
            data = resistances[nearest_resistance]
            
            # Проверяем силу уровня
            if data["strength"] < min_strength:
                return None
                
            # Проверяем расстояние
            distance_pct = abs(nearest_resistance - price) / price
            if distance_pct < threshold:
                # Проверка на пробой
                if is_level_broken(df, nearest_resistance, "resistance", 3):
                    return f"Сопротивление {nearest_resistance:.4f} недавно пробито"
                return f"Сильное сопротивление: {nearest_resistance:.4f} (сила: {data['strength']})"

        elif side == "short":
            # Фильтруем только поддержки ниже цены
            supports = {
                lvl: data for lvl, data in levels.items() 
                if data["type"] == "support" and lvl < price
            }
            
            if not supports:
                return None
                
            # Находим ближайшую сильную поддержку
            nearest_support = max(supports.keys(), key=lambda x: abs(x - price))
            data = supports[nearest_support]
            
            # Проверяем силу уровня
            if data["strength"] < min_strength:
                return None
                
            # Проверяем расстояние
            distance_pct = abs(price - nearest_support) / price
            if distance_pct < threshold:
                # Проверка на пробой
                if is_level_broken(df, nearest_support, "support", 3):
                    return f"Поддержка {nearest_support:.4f} недавно пробита"
                return f"Сильная поддержка: {nearest_support:.4f} (сила: {data['strength']})"

        return None
    
    def check_proximity_to_sd_zone(df: pd.DataFrame, side: str, config: dict) -> Optional[dict]:
        price = df["close"].iloc[-1]
        threshold = config.get("CONFIRMATION_SD_DISTANCE_THRESHOLD", 0.015)  # 1.5%
        min_strength = config.get("CONFIRMATION_SD_MIN_STRENGTH", 1.5)
        
        levels = get_enhanced_support_resistance(df, config)

        if side == "long":
            resistances = {
                lvl: data for lvl, data in levels.items() 
                if data["type"] == "resistance" and lvl > price
            }
            if not resistances:
                return None
            
            nearest_resistance = min(resistances.keys(), key=lambda x: abs(x - price))
            data = resistances[nearest_resistance]
            distance_pct = abs(nearest_resistance - price) / price

            if data["strength"] >= min_strength and distance_pct < threshold:
                broken = is_level_broken(df, nearest_resistance, "resistance", 3)
                return {
                    "type": "resistance",
                    "level": nearest_resistance,
                    "strength": data["strength"],
                    "broken": broken,
                    "text": f"{'Сопротивление недавно пробито' if broken else 'Сильное сопротивление: {:.4f} (сила: {})'.format(nearest_resistance, data['strength'])}"

                }

        elif side == "short":
            supports = {
                lvl: data for lvl, data in levels.items() 
                if data["type"] == "support" and lvl < price
            }
            if not supports:
                return None
            nearest_support = max(supports.keys(), key=lambda x: abs(x - price))
            data = supports[nearest_support]
            distance_pct = abs(price - nearest_support) / price

            if data["strength"] >= min_strength and distance_pct < threshold:
                broken = is_level_broken(df, nearest_support, "support", 3)
                return {
                    "type": "support",
                    "level": nearest_support,
                    "strength": data["strength"],
                    "broken": broken,
                    "text": "Поддержка недавно пробита" if broken else "Сильная поддержка: {:.4f} (сила: {})".format(nearest_support, data['strength'])
                }

        return None

    # === Функция для фильтра структурных паттернов ===
    def structural_patterns_filter(df, direction: str, tolerance: float = 0.003, mode: str = "soft"):
        """
        Фильтр структурных паттернов.
        Проверяет: W/M, BOS, Head&Shoulders, Triple Top/Bottom, Wedge, Cup&Handle, Flag, Triangle, Range.
        
        Args:
            df: DataFrame ['open','high','low','close']
            direction: 'long' или 'short'
            tolerance: допуск по % (по умолчанию 0.3%)
            mode: 'soft' или 'strict'
        
        Returns:
            score (0.0 или 1.0), patterns (list[str])
        """

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        patterns = []

        # swing highs/lows
        swing_highs, swing_lows = [], []
        for i in range(2, len(closes)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                swing_highs.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                swing_lows.append((i, lows[i]))

        # Таблица правил (русские названия)
        PATTERN_RULES = {
            "W (двойное дно)": "positive",
            "M (двойная вершина)": "positive",
            "Тройное дно": "negative",
            "Тройная вершина": "negative",
            "BOS вверх": "positive",
            "BOS вниз": "positive",
            "Голова и плечи": "positive",
            "Перевёрнутая ГиП": "positive",
            "Чашка с ручкой": "positive",
            "Бычий флаг": "positive",
            "Медвежий флаг": "positive",
            "Восходящий треугольник": "positive",
            "Нисходящий треугольник": "positive",
            "Флэт": "negative",
            "Восходящий клин": "negative",
            "Нисходящий клин": "positive"
        }

        # ================= PATTERN DETECTION =================
        # Double Bottom / Top
        if len(swing_lows) >= 2:
            p1, p2 = swing_lows[-2][1], swing_lows[-1][1]
            if abs(p1-p2)/p1 < tolerance and direction == "long":
                patterns.append("W (двойное дно)")
        if len(swing_highs) >= 2:
            p1, p2 = swing_highs[-2][1], swing_highs[-1][1]
            if abs(p1-p2)/p1 < tolerance and direction == "short":
                patterns.append("M (двойная вершина)")

        # Triple Top/Bottom
        if len(swing_highs) >= 3:
            p1, p2, p3 = [s[1] for s in swing_highs[-3:]]
            if abs(p1-p2)/p1 < tolerance and abs(p2-p3)/p2 < tolerance and direction == "short":
                patterns.append("Тройная вершина")
        if len(swing_lows) >= 3:
            p1, p2, p3 = [s[1] for s in swing_lows[-3:]]
            if abs(p1-p2)/p1 < tolerance and abs(p2-p3)/p2 < tolerance and direction == "long":
                patterns.append("Тройное дно")

        # BOS
        if swing_highs and swing_lows:
            last_high = swing_highs[-1][1]
            last_low = swing_lows[-1][1]
            price = closes[-1]
            if direction == "long" and price > last_high:
                patterns.append("BOS вверх")
            if direction == "short" and price < last_low:
                patterns.append("BOS вниз")

        # Head & Shoulders
        if len(swing_highs) >= 3:
            l, m, r = swing_highs[-3:]
            if m[1] > l[1] and m[1] > r[1] and direction == "short":
                patterns.append("Голова и плечи")
        if len(swing_lows) >= 3:
            l, m, r = swing_lows[-3:]
            if m[1] < l[1] and m[1] < r[1] and direction == "long":
                patterns.append("Перевёрнутая ГиП")

        # Cup&Handle
        if len(closes) > 30:
            left, bottom, right = min(closes[-30:-20]), min(closes[-20:-10]), min(closes[-10:])
            if abs(left-right)/left < tolerance and bottom < left and direction == "long":
                patterns.append("Чашка с ручкой")

        # Flag
        if len(closes) > 20:
            impulse = abs(closes[-20] - closes[-15]) / closes[-15]
            correction = (max(closes[-15:]) - min(closes[-15:])) / closes[-15]
            if impulse > 0.02 and correction < impulse:
                if direction == "long" and closes[-1] > closes[-15]:
                    patterns.append("Бычий флаг")
                if direction == "short" and closes[-1] < closes[-15]:
                    patterns.append("Медвежий флаг")

        # Triangle
        if len(closes) > 20:
            rng_high, rng_low = max(highs[-20:]), min(lows[-20:])
            if (rng_high - rng_low) / closes[-1] < 0.02:
                if direction == "long":
                    patterns.append("Восходящий треугольник")
                if direction == "short":
                    patterns.append("Нисходящий треугольник")

        # Range
        if len(closes) > 20:
            rng = (max(closes[-20:]) - min(closes[-20:])) / closes[-1]
            if rng < 0.015:
                patterns.append("Флэт")

        # Wedge
        if len(closes) > 20:
            highs_slope, lows_slope = highs[-1] - highs[-20], lows[-1] - lows[-20]
            if highs_slope < 0 and lows_slope > 0 and direction == "short":
                patterns.append("Восходящий клин")
            if highs_slope > 0 and lows_slope < 0 and direction == "long":
                patterns.append("Нисходящий клин")

        # ================= SCORING =================
        score = 0.0
        if mode == "soft":
            if patterns:
                score = 1.0
        elif mode == "strict":
            if any(PATTERN_RULES.get(p) == "negative" for p in patterns):
                score = 0.0  # фильтруем
            elif any(PATTERN_RULES.get(p) == "positive" for p in patterns):
                score = 1.0

        return score, patterns

    # === Функция для проверки фильтра SuperTrend ===
    def check_supertrend_filter(df, df_htf, config, reasons_long, reasons_short, passed_filters_long, passed_filters_short):
        atr_period = config.get("SUPERTREND_ATR_PERIOD", 10)
        multiplier = config.get("SUPERTREND_MULTIPLIER", 3.0)
        use_two_tf = config.get("SUPERTREND_USE_TWO_TF", True)
        increment = config.get("SUPERTREND_SCORE_INCREMENT", 1.0)

        st_main_up = is_supertrend_up(df, atr_period, multiplier)
        st_htf_up = is_supertrend_up(df_htf, atr_period, multiplier) if use_two_tf else None

        passed_long = False
        passed_short = False
        score_l = 0.0
        score_s = 0.0

        if use_two_tf:
            if st_main_up and st_htf_up:
                score_l += increment
                passed_long = True
                reasons_long.append("SuperTrend подтверждает лонг (оба ТФ ↑)")
                passed_filters_long.append(f"SuperTrend: score +{increment:.1f} (ltf=↑, htf=↑)")
            elif (not st_main_up) and (not st_htf_up):
                score_s += increment
                passed_short = True
                reasons_short.append("SuperTrend подтверждает шорт (оба ТФ ↓)")
                passed_filters_short.append(f"SuperTrend: score +{increment:.1f} (ltf=↓, htf=↓)")
            else:
                # Добавляем причины, когда фильтр не проходит
                if st_main_up != st_htf_up:
                    reasons_long.append("SuperTrend не подтверждает лонг (разные направления ТФ)")
                    reasons_short.append("SuperTrend не подтверждает шорт (разные направления ТФ)")
        else:
            if st_main_up:
                score_l += increment
                passed_long = True
                reasons_long.append("SuperTrend подтверждает лонг (осн. ТФ ↑)")
                passed_filters_long.append(f"SuperTrend: score +{increment:.1f} (ltf=↑)")
            else:
                score_s += increment
                passed_short = True
                reasons_short.append("SuperTrend подтверждает шорт (осн. ТФ ↓)")
                passed_filters_short.append(f"SuperTrend: score +{increment:.1f} (ltf=↓)")

        filter_result = {
            "supertrend_mode": "2TF" if use_two_tf else "1TF",
            "supertrend_ltf_value": int(st_main_up),
            "supertrend_ltf_config_atr": atr_period,
            "supertrend_ltf_config_mult": multiplier,
            "supertrend_ltf_passed": int(passed_long if st_main_up else passed_short),
            "supertrend_ltf_score": score_l if st_main_up else score_s,
            "supertrend_htf_value": int(st_htf_up) if st_htf_up is not None else None,
            "supertrend_htf_config_atr": atr_period if st_htf_up is not None else None,
            "supertrend_htf_config_mult": multiplier if st_htf_up is not None else None,
            "supertrend_htf_passed": int(passed_long if st_htf_up else passed_short) if st_htf_up is not None else None,
            "supertrend_htf_score": score_l if st_htf_up else score_s if st_htf_up is not None else None
        }

        return passed_long, passed_short, score_l, score_s, filter_result

    def check_adx_filter(ind, ind_htf, config, reasons_long, reasons_short, passed_filters_long, passed_filters_short, filters_results):
        """Проверка фильтра ADX"""
        adx_mode = config.get("ADX_MODE", "both")
        adx_threshold = config.get("ADX_THRESHOLD", 20)
        increment = config.get("ADX_SCORE_INCREMENT", 1.0)

        adx = ind["adx"]
        adx_htf = ind_htf["adx"] if adx_mode == "both" else None

        score_long_add = 0
        score_short_add = 0

        # Проверка наличия данных
        if adx is None or (adx_mode == "both" and adx_htf is None):
            reasons_long.append("Нет данных ADX")
            reasons_short.append("Нет данных ADX")
            filters_results.update({
                "adx_mode_config": adx_mode,
                "adx_ltf_value": None,
                "adx_ltf_config": adx_threshold,
                "adx_ltf_passed": 0,
                "adx_ltf_score": 0.0,
                "adx_htf_value": None,
                "adx_htf_config": None,
                "adx_htf_passed": None,
                "adx_htf_score": None
            })
        else:
            # Определяем логику проверки в зависимости от режима
            if adx_mode == "single":
                ltf_passed = 1 if adx >= adx_threshold else 0
                passed = ltf_passed == 1
                adx_repr = f"{adx:.2f}"
                htf_passed = None
            else:  # both
                ltf_passed = 1 if adx >= adx_threshold else 0
                htf_passed = 1 if adx_htf >= adx_threshold else 0
                passed = ltf_passed == 1 and htf_passed == 1
                adx_repr = f"{adx:.2f}/{adx_htf:.2f}"

            # Начисляем баллы и формируем сообщения
            if passed:
                score_long_add = increment
                score_short_add = increment
                passed_filters_long.append(f"ADX: score +{increment:.1f} (value={adx_repr})")
                passed_filters_short.append(f"ADX: score +{increment:.1f} (value={adx_repr})")
            else:
                reasons_long.append(f"ADX слабый ({adx_repr})")
                reasons_short.append(f"ADX слабый ({adx_repr})")

            # Формируем результат фильтра
            filter_entry = {
                "adx_mode_config": adx_mode,
                "adx_ltf_value": adx,
                "adx_ltf_config": adx_threshold,
                "adx_ltf_passed": ltf_passed,
                "adx_ltf_score": increment if passed else 0.0,
                "adx_htf_value": adx_htf if adx_mode == "both" else None,
                "adx_htf_config": adx_threshold if adx_mode == "both" else None,
                "adx_htf_passed": htf_passed if adx_mode == "both" else None,
                "adx_htf_score": increment if (adx_mode == "both" and passed) else None
            }

            filters_results.update(filter_entry)
        
        return score_long_add, score_short_add
   
    def check_cdv_filter(symbol, trend_global, config, reasons_long, reasons_short, passed_filters_long, passed_filters_short, filters_results):
        """Проверка фильтра CDV с расширенными признаками для ML"""
        try:
            tfs = config["MARKET_ANALYSIS_TF"]
            cdv_passed = True
            cdv_reasons = []
            score_long_add = 0
            score_short_add = 0

            # Данные для filters_results
            cdv_ltf_value = None
            cdv_ltf_volume = None
            cdv_ltf_passed = 0
            cdv_ltf_score = 0.0
            cdv_ltf_conflict = 0
            cdv_htf_value = None
            cdv_htf_volume = None
            cdv_htf_passed = 0
            cdv_htf_score = 0.0
            cdv_htf_conflict = 0

            side = trend_global
            if side not in ["long", "short"]:
                print(f"[CDV] {symbol}: пропуск CDV, неопределённый тренд")
                reasons_long.append("CDV пропущен — неопределённый тренд")
                reasons_short.append("CDV пропущен — неопределённый тренд")
            else:
                # Проверяем CDV по каждому TF
                for i, tf in enumerate(tfs):
                    ratio, vol = get_cdv_ratio(symbol, tf)
                    if ratio is None:
                        cdv_reasons.append(f"CDV {tf}: нет данных")
                        cdv_passed = False
                        continue

                    reason = ""
                    tf_passed = False
                    conflict_flag = 0

                    # Проверка направления
                    if (side == "long" and ratio >= config["CDV_MIN_THRESHOLD"]) or \
                    (side == "short" and ratio <= -config["CDV_MIN_THRESHOLD"]):
                        reason = f"CDV {tf}: {ratio:+.2%} (подтверждено)"
                        tf_passed = True
                    else:
                        reason = f"CDV {tf}: {ratio:+.2%} (слабый)"
                        cdv_passed = False

                    # Проверка согласованности между ТФ
                    if config.get("CDV_CHECK_MULTI_TF", False) and i == 0 and len(tfs) > 1:
                        ratio_next, _ = get_cdv_ratio(symbol, tfs[1])
                        if ratio_next is not None:
                            if (side == "long" and ratio_next < 0) or (side == "short" and ratio_next > 0):
                                cdv_passed = False
                                reason += f" | конфликт с {tfs[1]}"
                                tf_passed = False
                                conflict_flag = 1  # 🔹 добавляем флаг конфликта

                    # Начисление баллов
                    increment = config.get("CDV_SCORE_INCREMENT", 1.0)

                    if tf_passed:
                        if side == "long":
                            score_long_add += increment
                            passed_filters_long.append(f"CDV {tf}: score +{increment:.1f} (value={ratio:+.2%})")
                        else:
                            score_short_add += increment
                            passed_filters_short.append(f"CDV {tf}: score +{increment:.1f} (value={ratio:+.2%})")

                    # Сохраняем данные для filters_results
                    if i == 0:  # LTF
                        cdv_ltf_value = ratio
                        cdv_ltf_volume = vol
                        cdv_ltf_passed = 1 if tf_passed else 0
                        cdv_ltf_score = increment if tf_passed else 0.0
                        cdv_ltf_conflict = conflict_flag
                    elif i == 1:  # HTF
                        cdv_htf_value = ratio
                        cdv_htf_volume = vol
                        cdv_htf_passed = 1 if tf_passed else 0
                        cdv_htf_score = increment if tf_passed else 0.0
                        cdv_htf_conflict = conflict_flag

                    cdv_reasons.append(reason)

                # Добавляем причины в список
                if side == "long":
                    reasons_long.extend(cdv_reasons)
                else:
                    reasons_short.extend(cdv_reasons)

            # 🔹 Нормализованные значения
            def normalize(value):
                if value is None:
                    return None
                th = config["CDV_MIN_THRESHOLD"]
                return value / th if th != 0 else value

            # Добавляем результаты в filters_results
            filters_results.update({
                "cdv_ltf_value": cdv_ltf_value,
                "cdv_ltf_volume": cdv_ltf_volume,
                "cdv_ltf_passed": cdv_ltf_passed,
                "cdv_ltf_score": cdv_ltf_score,
                "cdv_ltf_conflict": cdv_ltf_conflict,
                "cdv_ltf_threshold_config": config["CDV_MIN_THRESHOLD"],
                "cdv_ltf_normalized": normalize(cdv_ltf_value),

                "cdv_htf_value": cdv_htf_value,
                "cdv_htf_volume": cdv_htf_volume,
                "cdv_htf_passed": cdv_htf_passed,
                "cdv_htf_score": cdv_htf_score,
                "cdv_htf_conflict": cdv_htf_conflict,
                "cdv_htf_threshold_config": config["CDV_MIN_THRESHOLD"],
                "cdv_htf_normalized": normalize(cdv_htf_value)
            })

            return score_long_add, score_short_add

        except Exception as e:
            print(f"[CDV] Ошибка для {symbol}: {e}")
            filters_results.update({
                "cdv_ltf_value": None,
                "cdv_ltf_volume": None,
                "cdv_ltf_passed": 0,
                "cdv_ltf_score": 0.0,
                "cdv_ltf_conflict": 0,
                "cdv_ltf_threshold_config": config["CDV_MIN_THRESHOLD"],
                "cdv_ltf_normalized": None,

                "cdv_htf_value": None,
                "cdv_htf_volume": None,
                "cdv_htf_passed": 0,
                "cdv_htf_score": 0.0,
                "cdv_htf_conflict": 0,
                "cdv_htf_threshold_config": config["CDV_MIN_THRESHOLD"],
                "cdv_htf_normalized": None
            })
            return 0, 0


    # === Функция для получения данных по монете ===
    async def check_signal(symbol, df, df_htf, trend_global):
        try:
            # Добавьте проверку на минимальное количество данных
            if df is None or len(df) < 50:
                return None
        
            if symbol in active_trades:
                print(f"[SKIP] {symbol} — уже есть активная сделка, сигнал пропущен")
                return None
            
            ind = calculate_indicators(df)
            ind_htf = calculate_indicators(df_htf)

            reasons_long = []
            reasons_short = []
            score_long = 0
            score_short = 0

            # filters_results = []  # сюда будем собирать (название, значение, passed/failed)


            filters_results = {
                "signal_id": f"{symbol}_{int(time.time())}",
                "symbol": symbol,
                "timestamp": datetime.utcnow().isoformat(),
                "signal_ltf": config["MARKET_ANALYSIS_TF"][0],
                "signal_htf": config["MARKET_ANALYSIS_TF"][1],
                "trend_global": trend_global,
            }

            # Добавьте эти списки для успешных фильтров
            passed_filters_long = []
            passed_filters_short = []


            # --- Фильтр потенциала движения --- DONE
            if config.get("USE_POTENTIAL_FILTER", True):
                potential_score = assess_signal_potential_multi_tf(symbol, df_dict, config)
                
                # Добавляем запись в filters_results
                filters_results.update({
                    "potential_value": potential_score,
                    "potential_passed": 1 if potential_score >= config.get("POTENTIAL_THRESHOLD", 0.5) else 0,
                    "potential_config": config.get('POTENTIAL_THRESHOLD', 0.5)
                })
                
                if potential_score < config.get("POTENTIAL_THRESHOLD", 0.5):
                    print(f"[POTENTIAL FILTER] {symbol}: Потенциал слабый ({potential_score:.2f}) — сигнал отклонён")
                    return None
                else:
                    # Потенциал достаточный, добавляем в passed_filters
                    passed_reason = f"POTENTIAL: score {potential_score:.2f} ≥ {config.get('POTENTIAL_THRESHOLD', 0.5)}"
                    passed_filters_long.append(passed_reason)
                    passed_filters_short.append(passed_reason)

            # --- CDV фильтр --- DONE
            if config.get("USE_CDV_FILTER", False):
                cdv_score_long, cdv_score_short = check_cdv_filter(
                    symbol, trend_global, config, reasons_long, reasons_short,
                    passed_filters_long, passed_filters_short, filters_results
                )
                score_long += cdv_score_long
                score_short += cdv_score_short

            # --- ATR фильтр --- DONE
            if config.get("USE_ATR_FILTER", True):
                current_atr = ind["atr"]
                last_close = ind["last_close"]
                
                min_atr_ratio = config.get("MIN_ATR_RATIO", 0.0007)
                atr_threshold = last_close * min_atr_ratio
                increment = config.get("ATR_SCORE_INCREMENT", 1.0)
                
                # Проверка фильтра ATR
                if current_atr < atr_threshold:
                    reason_msg = f"ATR {current_atr:.4f} < {atr_threshold:.4f}"
                    reasons_long.append(reason_msg)
                    reasons_short.append(reason_msg)
                    atr_passed, atr_score = 0, 0.0
                else:
                    reason_msg = f"ATR passed: score +{increment:.1f} (value={current_atr:.4f})"
                    passed_filters_long.append(reason_msg)
                    passed_filters_short.append(reason_msg)
                    score_long += increment
                    score_short += increment
                    atr_passed, atr_score = 1, increment
                
                # Добавляем запись в filters_results для БД
                filters_results.update({
                    "atr_ltf_value": current_atr,
                    "atr_ltf_config": min_atr_ratio,
                    "atr_ltf_passed": atr_passed,
                    "atr_ltf_score": atr_score
                })

            # --- volume фильтр --- DONE
            if config.get("USE_VOLUME_FILTER", True):
                volume = ind["volume"]  # Накопленный объем за N свечей
                volume_delta = ind["volume_delta"]  # Дельта накопленного объема
                vol_mean = ind["vol_mean"]
                vol_mean_long = ind["vol_mean_long"]
                normalized_body = ind["normalized_body"]

                increment = 0.0
                volume_delta_bonus = 0.0
                delta_bonus_applied = False

                # --- 1. Динамический абсолютный минимум ---
                min_abs_config = config.get("MIN_ABSOLUTE_VOLUME", 15000)
                min_abs_dynamic = max(min_abs_config, vol_mean_long * 0.2)
                tolerance = config.get("MIN_ABSOLUTE_VOLUME_TOLERANCE", 0.8)

                if volume < min_abs_dynamic:
                    if volume < min_abs_dynamic * tolerance:
                        # Жёсткий стоп: объём сильно ниже минимума
                        msg = (f"Накоп. объём {volume:.0f} < абсолютного минимума {min_abs_dynamic:.0f} "
                            f"({volume/min_abs_dynamic:.2%} от мин., порог tolerance={tolerance:.2f})")
                        reasons_long.append(msg)
                        passed_filters_long.append(f"Volume: score 0 (value={volume}, abs_min={min_abs_dynamic:.0f})")
                        passed_filters_short.append(f"Volume: score 0 (value={volume}, abs_min={min_abs_dynamic:.0f})")
                        return None  # Прерываем дальнейший анализ
                    else:
                        # Объём близок к минимуму — relaxed проход
                        msg = (f"Накоп. объём {volume:.0f} близок к абс. минимуму {min_abs_dynamic:.0f} "
                            f"({volume/min_abs_dynamic:.2%}, tolerance={tolerance:.2f})")
                        reasons_long.append(msg)
                        increment = config.get("VOLUME_RELAXED_SCORE_INCREMENT", 0.6)
                        score_long += increment
                        score_short += increment
                        passed_filters_long.append(f"Volume: score +{increment:.1f} (value={volume}, abs_min={min_abs_dynamic:.0f})")
                        passed_filters_short.append(f"Volume: score +{increment:.1f} (value={volume}, abs_min={min_abs_dynamic:.0f})")
                else:
                    msg = f"Накоп. объём {volume:.0f} ≥ абсолютного минимума {min_abs_dynamic:.0f}"
                    reasons_long.append(msg)

                # --- 2. Основная проверка объёма ---
                vol_multiplier = config.get("VOLUME_THRESHOLD_MULTIPLIER", 0.55)
                min_relaxed = config.get("MIN_VOLUME_RELAXED", 0.45)
                noise_vol_ratio = config.get("NOISE_VOL_RATIO", 0.6)
                noise_body_ratio = config.get("NOISE_BODY_RATIO", 0.3)

                # Фильтр шума: маленький объём И маленькое тело свечи
                noise_filter = (volume < vol_mean_long * noise_vol_ratio) and (normalized_body < noise_body_ratio)
                volume_threshold = vol_mean * vol_multiplier
                passed = volume >= volume_threshold and not noise_filter
                volume_relaxed = False

                # УСЛОВИЕ volume_delta > 0 УДАЛЕНО! Это была ошибка.
                # Проверяем relaxed проход, только если не сработал шум-фильтр
                if not passed and min_relaxed > 0 and not noise_filter:
                    if volume >= vol_mean * min_relaxed:
                        volume_relaxed = True
                        passed = True

                # --- 3. Бонус за положительную динамику ---
                volume_delta_bonus = 0.0
                delta_bonus_applied = False
                if passed and volume_delta > 0:
                    # Добавляем бонус за рост объема относительно предыдущего периода
                    volume_delta_bonus = config.get("VOLUME_DELTA_BONUS", 0.2)
                    delta_bonus_applied = True
                    msg = (f"Бонус за рост объёма: +{volume_delta_bonus:.1f} "
                        f"(дельта: {volume_delta:.0f})")
                    reasons_long.append(msg)

                # --- 4. Начисление баллов ---
                if passed:
                    increment = config.get("VOLUME_SCORE_INCREMENT", 1.0)
                    if volume_relaxed:
                        increment *= config.get("VOLUME_RELAXED_SCORE_INCREMENT", 0.6)
                    
                    # Добавляем бонус за дельту
                    increment += volume_delta_bonus

                    score_long += increment
                    score_short += increment
                    status = "relaxed" if volume_relaxed else "passed"

                    if volume_relaxed:
                        msg = (f"Накоп. объём {volume:.0f} (relaxed) ≥ {vol_mean*min_relaxed:.0f} "
                            f"({volume/vol_mean:.2%} от среднего {vol_mean:.0f}, порог relaxed={min_relaxed:.2f})")
                    else:
                        msg = (f"Накоп. объём {volume:.0f} ≥ {volume_threshold:.0f} "
                            f"({volume/vol_mean:.2%} от среднего {vol_mean:.0f}, mult={vol_multiplier:.2f})")
                    
                    reasons_long.append(msg)
                    
                    # Детализированное сообщение для passed_filters
                    detail_msg = f"{volume/vol_mean:.2%} от среднего"
                    if volume_relaxed:
                        detail_msg += f", relaxed ({min_relaxed:.2f})"
                    if delta_bonus_applied:
                        detail_msg += f", бонус за дельту +{volume_delta_bonus:.1f}"
                        
                    passed_filters_long.append(f"Volume: score +{increment:.1f} (value={volume}, {detail_msg})")
                    passed_filters_short.append(f"Volume: score +{increment:.1f} (value={volume}, {detail_msg})")
                else:
                    # Сигнал не прошёл фильтр объема
                    if noise_filter:
                        msg = (f"Фильтр шума: объём {volume:.0f} < {vol_mean_long*noise_vol_ratio:.0f} "
                            f"({volume/vol_mean_long:.2%} от долгоср. среднего) "
                            f"и тело свечи {normalized_body:.2f} < {noise_body_ratio:.2f}")
                    else:
                        msg = (f"Накоп. объём {volume:.0f} < порога {volume_threshold:.0f} "
                            f"({volume/vol_mean:.2%} от среднего {vol_mean:.0f}, mult={vol_multiplier:.2f})")
                    
                    reasons_long.append(msg)
                    
                    # Детализированное сообщение для passed_filters
                    if noise_filter:
                        detail_msg = f"шум (объем {volume/vol_mean_long:.2%}, тело {normalized_body:.2f})"
                    else:
                        detail_msg = f"{volume/vol_mean:.2%} от среднего"
                        
                    passed_filters_long.append(f"Volume: score 0 (value={volume}, {detail_msg})")
                    passed_filters_short.append(f"Volume: score 0 (value={volume}, {detail_msg})")
                
                filters_results.update({
                    "volume_value": volume,
                    "volume_delta": volume_delta,
                    "volume_mean": vol_mean,
                    "volume_mean_long": vol_mean_long,
                    "volume_normalized_body": normalized_body,
                    "volume_passed": 1 if passed else 0,
                    "volume_score": round(increment if passed else 0, 2),
                    "volume_min_abs_config": min_abs_config,
                    "volume_min_abs_dynamic_config": min_abs_dynamic,
                    "volume_tolerance": tolerance,
                    "volume_multiplier_config": config.get("VOLUME_THRESHOLD_MULTIPLIER", 0.55),
                    "volume_min_relaxed_config": config.get("MIN_VOLUME_RELAXED", 0.45),
                    "volume_noise_vol_ratio_config": config.get("NOISE_VOL_RATIO", 0.6),
                    "volume_noise_body_ratio_config": config.get("NOISE_BODY_RATIO", 0.3),
                    "volume_delta_bonus_config": volume_delta_bonus,
                })

            # --- ADX --- DONE
            if config.get("USE_ADX_FILTER", True):
                adx_score_long, adx_score_short = check_adx_filter(
                    ind, ind_htf, config, reasons_long, reasons_short,
                    passed_filters_long, passed_filters_short, filters_results
                )
                score_long += adx_score_long
                score_short += adx_score_short

            # --- TRIX --- DONE
            if config.get("USE_TRIX_FILTER", True):
                # Параметры
                trix_threshold = config.get("TRIX_MOMENTUM_THRESHOLD", 0.0005)
                trix_depth = config.get("TRIX_DEPTH_THRESHOLD", 0.002)
                
                # Текущие значения
                trix_main = ind["trix"]
                trix_htf = ind_htf["trix"]
                
                # Предыдущие значения
                prev_trix_main = ind.get("prev_trix")
                prev_trix_htf = ind_htf.get("prev_trix")
                
                # Расчет импульса
                trix_momentum_main = trix_main - prev_trix_main if prev_trix_main is not None else 0
                trix_momentum_htf = trix_htf - prev_trix_htf if prev_trix_htf is not None else 0
                
                # Ранний бычий сигнал (разворот в минусе)
                early_trix_bullish = (
                    trix_main < 0
                    and trix_momentum_main > trix_threshold
                    and trix_momentum_htf > trix_threshold * 0.5
                    and abs(trix_main) < trix_depth
                    and prev_trix_main is not None
                )
                
                # Ранний медвежий сигнал (разворот в плюсе)
                early_trix_bearish = (
                    trix_main > 0
                    and trix_momentum_main < -trix_threshold
                    and trix_momentum_htf < -trix_threshold * 0.5
                    and abs(trix_main) < trix_depth
                    and prev_trix_main is not None
                )
                
                # Классические условия
                trix_classic_long = trix_main > 0 and trix_htf > 0
                trix_classic_short = trix_main < 0 and trix_htf < 0
                
                # Комбинированные условия
                trix_long = trix_classic_long or early_trix_bullish
                trix_short = trix_classic_short or early_trix_bearish
                
                # Логика баллов и логирование
                increment = config.get("TRIX_SCORE_INCREMENT", 1.0)
                
                if trix_long:
                    score_long += increment
                    passed_filters_long.append(f"TRIX: score +{increment:.1f} (value={trix_main:.6f})")
                    
                    if early_trix_bullish:
                        reasons_long.append(
                            f"⚠️ РАННИЙ TRIX-ЛОНГ: curr={trix_main:.6f}, prev={prev_trix_main:.6f}, "
                            f"mom={trix_momentum_main:.6f}, htf={trix_htf:.6f}, htf_mom={trix_momentum_htf:.6f}"
                        )
                    else:
                        reasons_long.append(
                            f"TRIX подтверждает лонг: curr={trix_main:.6f}, prev={prev_trix_main:.6f}, "
                            f"mom={trix_momentum_main:.6f}, htf={trix_htf:.6f}, htf_mom={trix_momentum_htf:.6f}"
                        )
                else:
                    reasons_long.append(
                        f"TRIX не подтверждает лонг: curr={trix_main:.6f}, prev={prev_trix_main:.6f}, "
                        f"mom={trix_momentum_main:.6f}, htf={trix_htf:.6f}, htf_mom={trix_momentum_htf:.6f}"
                    )
                    passed_filters_long.append(f"TRIX: score 0 (value={trix_main:.6f})")
                
                if trix_short:
                    score_short += increment
                    passed_filters_short.append(f"TRIX: score +{increment:.1f} (value={trix_main:.6f})")
                    
                    if early_trix_bearish:
                        reasons_short.append(
                            f"⚠️ РАННИЙ TRIX-ШОРТ: curr={trix_main:.6f}, prev={prev_trix_main:.6f}, "
                            f"mom={trix_momentum_main:.6f}, htf={trix_htf:.6f}, htf_mom={trix_momentum_htf:.6f}"
                        )
                    else:
                        reasons_short.append(
                            f"TRIX подтверждает шорт: curr={trix_main:.6f}, prev={prev_trix_main:.6f}, "
                            f"mom={trix_momentum_main:.6f}, htf={trix_htf:.6f}, htf_mom={trix_momentum_htf:.6f}"
                        )
                else:
                    reasons_short.append(
                        f"TRIX не подтверждает шорт: curr={trix_main:.6f}, prev={prev_trix_main:.6f}, "
                        f"mom={trix_momentum_main:.6f}, htf={trix_htf:.6f}, htf_mom={trix_momentum_htf:.6f}"
                    )
                    passed_filters_short.append(f"TRIX: score 0 (value={trix_main:.6f})")
                
                filters_results.update({
                    "trix_ltf": trix_main,
                    "trix_htf": trix_htf,
                    "prev_trix_ltf": prev_trix_main,
                    "prev_trix_htf": prev_trix_htf,
                    "trix_momentum_ltf": trix_momentum_main,
                    "trix_momentum_htf": trix_momentum_htf,
                    "early_trix_bullish": early_trix_bullish,
                    "early_trix_bearish": early_trix_bearish,
                    "trix_classic_long": trix_classic_long,
                    "trix_classic_short": trix_classic_short,
                    "trix_long_passed": 1 if trix_long else 0,
                    "trix_short_passed": 1 if trix_short else 0,
                    "trix_long_score": round(increment if trix_long else 0, 2),
                    "trix_short_score": round(increment if trix_short else 0, 2),
                    "trix_threshold_config": trix_threshold,
                    "trix_depth_config": trix_depth,
                })

            # --- SuperTrend --- DONE
            if config.get("USE_SUPERTREND_FILTER", True):
                (passed_long_supertrend, 
                passed_short_supertrend, 
                score_l_supertrend, 
                score_s_supertrend, 
                filter_result) = check_supertrend_filter(
                    df, df_htf, config, 
                    reasons_long, reasons_short, 
                    passed_filters_long, passed_filters_short
                )
                
                score_long += score_l_supertrend
                score_short += score_s_supertrend
                # passed_filters_long.append(passed_long_supertrend)
                # passed_filters_short.append(passed_short_supertrend)
                filters_results.update(filter_result)

            # --- MACD --- DONE
            if config.get("USE_MACD_FILTER", True):
                # Параметры из конфига
                momentum_threshold = config.get("MACD_MOMENTUM_THRESHOLD", 0.0008)
                depth_threshold = config.get("MACD_DEPTH_THRESHOLD", 0.003)
                macd_threshold = config.get("MACD_THRESHOLD", 0.0025)
                
                # Текущие значения
                current_macd_diff = ind["macd_diff"]
                current_htf_macd_diff = ind_htf["macd_diff"]
                
                # Предыдущие значения
                prev_macd_diff = ind.get("prev_macd_diff")
                prev_htf_macd_diff = ind_htf.get("prev_macd_diff")
                
                # Расчет импульса
                macd_momentum = current_macd_diff - prev_macd_diff if prev_macd_diff is not None else 0
                htf_momentum = current_htf_macd_diff - prev_htf_macd_diff if prev_htf_macd_diff is not None else 0
                
                # Ранний бычий сигнал
                early_bullish = (
                    current_macd_diff < 0
                    and macd_momentum > momentum_threshold
                    and htf_momentum > momentum_threshold * 0.6
                    and abs(current_macd_diff) < depth_threshold
                    and prev_macd_diff is not None
                )
                
                # Ранний медвежий сигнал
                early_bearish = (
                    current_macd_diff > 0
                    and macd_momentum < -momentum_threshold
                    and htf_momentum < -momentum_threshold * 0.6
                    and abs(current_macd_diff) < depth_threshold
                    and prev_macd_diff is not None
                )
                
                # Комбинированные условия
                macd_long = (current_macd_diff > 0 and current_htf_macd_diff > 0) or \
                            (-macd_threshold <= current_macd_diff <= 0 and current_htf_macd_diff >= 0) or \
                            early_bullish
                
                macd_short = (current_macd_diff < 0 and current_htf_macd_diff < 0) or \
                            (-macd_threshold <= current_macd_diff <= 0 and current_htf_macd_diff <= 0) or \
                            early_bearish
                
                increment = config.get("MACD_SCORE_INCREMENT", 1.0)
                
                # Лонг
                if macd_long:
                    score_long += increment
                    passed_filters_long.append(f"MACD: score +{increment:.1f} (value={current_macd_diff:.6f})")
                    reasons_long.append(
                        f"MACD подтверждает лонг: diff={current_macd_diff:.6f}, htf_diff={current_htf_macd_diff:.6f}, "
                        f"prev={prev_macd_diff:.6f}, mom={macd_momentum:.6f}, htf_mom={htf_momentum:.6f}"
                    )
                    if early_bullish:
                        reasons_long.append(f"⚠️ РАННИЙ ЛОНГ: импульс {macd_momentum:.6f}")
                else:
                    passed_filters_long.append(f"MACD: (value={current_macd_diff:.6f})")
                    reasons_long.append(
                        f"MACD не подтверждает лонг: diff={current_macd_diff:.6f}, htf_diff={current_htf_macd_diff:.6f}, "
                        f"prev={prev_macd_diff:.6f}, mom={macd_momentum:.6f}, htf_mom={htf_momentum:.6f}"
                    )
                
                # Шорт
                if macd_short:
                    score_short += increment
                    passed_filters_short.append(f"MACD: score +{increment:.1f} (value={current_macd_diff:.6f})")
                    reasons_short.append(
                        f"MACD подтверждает шорт: diff={current_macd_diff:.6f}, htf_diff={current_htf_macd_diff:.6f}, "
                        f"prev={prev_macd_diff:.6f}, mom={macd_momentum:.6f}, htf_mom={htf_momentum:.6f}"
                    )
                    if early_bearish:
                        reasons_short.append(f"⚠️ РАННИЙ ШОРТ: импульс {macd_momentum:.6f}")
                else:
                    passed_filters_short.append(f"MACD: (value={current_macd_diff:.6f})")
                    reasons_short.append(
                        f"MACD не подтверждает шорт: diff={current_macd_diff:.6f}, htf_diff={current_htf_macd_diff:.6f}, "
                        f"prev={prev_macd_diff:.6f}, mom={macd_momentum:.6f}, htf_mom={htf_momentum:.6f}"
                    )

                # Добавляем итоговую запись в filters_results
                filters_results.update({
                    "macd_current_diff": current_macd_diff,
                    "macd_htf_diff": current_htf_macd_diff,
                    "macd_prev_diff": prev_macd_diff,
                    "macd_prev_htf_diff": prev_htf_macd_diff,
                    "macd_momentum": macd_momentum,
                    "macd_htf_momentum": htf_momentum,
                    "macd_momentum_threshold_config": momentum_threshold,
                    "macd_depth_threshold_config": depth_threshold,
                    "macd_threshold_config": macd_threshold,
                    "macd_score_increment": increment,
                    "macd_early_bullish": 1 if early_bullish else 0,
                    "macd_early_bearish": 1 if early_bearish else 0,
                    "macd_long": 1 if macd_long else 0,
                    "macd_short": 1 if macd_short else 0,
                    "macd_score_long": increment if macd_long else 0.0,
                    "macd_score_short": increment if macd_short else 0.0
                })
            
            # --- Стохастик --- DONE
            if config.get("USE_STOCH_FILTER", True):
                stoch = ind["stoch"]
                stoch_htf = ind_htf["stoch"]
                stoch_repr = f"{stoch:.2f}/{stoch_htf:.2f}"

                # Диапазоны
                stoch_range_long = config["STOCHASTIC_RANGE_LONG"]
                stoch_range_short = config["STOCHASTIC_RANGE_SHORT"]

                # Проверка вхождения в диапазон
                passed_long = (stoch_range_long[0] <= stoch <= stoch_range_long[1] and
                            stoch_range_long[0] <= stoch_htf <= stoch_range_long[1])

                passed_short = (stoch_range_short[0] <= stoch <= stoch_range_short[1] and
                                stoch_range_short[0] <= stoch_htf <= stoch_range_short[1])

                increment = config.get("STOCH_SCORE_INCREMENT", 1.0)

                # LONG
                if passed_long:
                    score_long += increment
                    passed_filters_long.append(f"Stoch: score +{increment:.1f} (value={stoch_repr})")
                else:
                    reasons_long.append(f"Stoch вне LONG-диапазона {stoch_range_long} (текущие: {stoch_repr})")

                # SHORT
                if passed_short:
                    score_short += increment
                    passed_filters_short.append(f"Stoch: score +{increment:.1f} (value={stoch_repr})")
                else:
                    reasons_short.append(f"Stoch вне SHORT-диапазона {stoch_range_short} (текущие: {stoch_repr})")

                filters_results.update({
                    "stoch_ltf": stoch,
                    "stoch_htf": stoch_htf,

                    "stoch_prev_ltf": ind.get("prev_stoch"),
                    "stoch_prev_htf": ind_htf.get("prev_stoch"),
                    "stoch_delta_ltf": ind.get("stoch_delta"),
                    "stoch_delta_htf": ind_htf.get("stoch_delta"),

                    "stoch_long_passed": 1 if passed_long else 0,
                    "stoch_short_passed": 1 if passed_short else 0,
                    "stoch_long_score": round(increment if passed_long else 0, 2),
                    "stoch_short_score": round(increment if passed_short else 0, 2),
                    "stoch_range_long_config": stoch_range_long,
                    "stoch_range_short_config": stoch_range_short,
                })

            # --- RSI --- DONE
            if config.get("USE_RSI_FILTER", True):
                rsi_range_long = config.get("RSI_RANGE_LONG", [38, 75])
                rsi_range_short = config.get("RSI_RANGE_SHORT", [35, 65])
                rsi_mode = config.get("RSI_MODE", "both")
                tolerance = 1e-9

                # Значения RSI
                rsi = ind["rsi"]
                rsi_htf = ind_htf["rsi"] if rsi_mode == "both" else None
                prev_rsi = ind.get("prev_rsi")
                prev_rsi_htf = ind_htf.get("prev_rsi") if rsi_mode == "both" else None
                rsi_delta = ind.get("rsi_delta")
                rsi_delta_htf = ind_htf.get("rsi_delta") if rsi_mode == "both" else None

                # Проверка наличия данных
                missing_data = False
                if rsi is None:
                    reasons_long.append("Отсутствует RSI на основном ТФ")
                    reasons_short.append("Отсутствует RSI на основном ТФ")
                    missing_data = True
                if rsi_mode == "both" and rsi_htf is None:
                    reasons_long.append("Отсутствует RSI на старшем ТФ")
                    reasons_short.append("Отсутствует RSI на старшем ТФ")
                    missing_data = True
                if missing_data:
                    filters_results.update({
                        "rsi_ltf": 0,
                        "rsi_htf": 0,
                        "rsi_prev_ltf": 0,
                        "rsi_prev_htf": 0,
                        "rsi_delta_ltf": 0,
                        "rsi_delta_htf": 0,
                        "rsi_mode_config": rsi_mode,
                        "rsi_long_passed": 0,
                        "rsi_short_passed": 0,
                        "rsi_long_score": 0,
                        "rsi_short_score": 0,
                        "rsi_range_long_config": rsi_range_long,
                        "rsi_range_short_config": rsi_range_short,
                        "rsi_tolerance": tolerance,
                    })
                    return  # или continue

                # Безопасное сравнение
                def in_range(value, range_list):
                    return (range_list[0] - tolerance) <= value <= (range_list[1] + tolerance)

                # Проверка условий
                if rsi_mode == "single":
                    passed_long = in_range(rsi, rsi_range_long)
                    passed_short = in_range(rsi, rsi_range_short)
                    rsi_repr = f"{rsi:.2f}"
                else:
                    passed_long = in_range(rsi, rsi_range_long) and in_range(rsi_htf, rsi_range_long)
                    passed_short = in_range(rsi, rsi_range_short) and in_range(rsi_htf, rsi_range_short)
                    rsi_repr = f"{rsi:.2f}/{rsi_htf:.2f}"

                increment = config.get("RSI_SCORE_INCREMENT", 1.0)

                # LONG
                if passed_long:
                    score_long += increment
                    passed_filters_long.append(f"RSI: score +{increment:.1f} (value={rsi_repr})")
                else:
                    passed_filters_long.append(f"RSI: score 0 (value={rsi_repr})")

                    reason = "RSI вне диапазона"
                    if rsi_mode == "both":
                        details = []
                        if not in_range(rsi, rsi_range_long):
                            details.append(f"основной ТФ ({rsi:.2f})")
                        if not in_range(rsi_htf, rsi_range_long):
                            details.append(f"старший ТФ ({rsi_htf:.2f})")
                        reason += " для лонга: " + ", ".join(details)
                    else:
                        reason += f" для лонга ({rsi:.2f})"
                    reasons_long.append(reason)

                # SHORT
                if passed_short:
                    score_short += increment
                    passed_filters_short.append(f"RSI: score +{increment:.1f} (value={rsi_repr})")
                else:
                    passed_filters_short.append(f"RSI: score 0 (value={rsi_repr})")

                    reason = "RSI вне диапазона"
                    if rsi_mode == "both":
                        details = []
                        if not in_range(rsi, rsi_range_short):
                            details.append(f"основной ТФ ({rsi:.2f})")
                        if not in_range(rsi_htf, rsi_range_short):
                            details.append(f"старший ТФ ({rsi_htf:.2f})")
                        reason += " для шорта: " + ", ".join(details)
                    else:
                        reason += f" для шорта ({rsi:.2f})"
                    reasons_short.append(reason)
                
                filters_results.update({
                    "rsi_ltf": rsi,
                    "rsi_htf": rsi_htf,
                    "rsi_prev_ltf": prev_rsi,
                    "rsi_prev_htf": prev_rsi_htf,
                    "rsi_delta_ltf": rsi_delta,
                    "rsi_delta_htf": rsi_delta_htf,
                    "rsi_mode_config": rsi_mode,
                    "rsi_long_passed": 1 if passed_long else 0,
                    "rsi_short_passed": 1 if passed_short else 0,
                    "rsi_long_score": round(increment if passed_long else 0, 2),
                    "rsi_short_score": round(increment if passed_short else 0, 2),
                    "rsi_range_long_config": rsi_range_long,
                    "rsi_range_short_config": rsi_range_short,
                    "rsi_tolerance": tolerance,
                })

            # --- EMA тренд --- DONE
            if config.get("USE_EMA_FILTER", True):
                ema_threshold = config.get("EMA_THRESHOLD", 0.005)
                ema_diff = ind["ema_fast"] - ind["ema_slow"]
                ema_diff_htf = ind_htf["ema_fast"] - ind_htf["ema_slow"]

                # Чёткие условия с порогом
                ema_long = (ema_diff > -ema_threshold) and (ema_diff_htf > 0)
                ema_short = (ema_diff < ema_threshold) and (ema_diff_htf < 0)

                # Дополнительное подтверждение для сильных трендов
                if ema_diff > 0 and ema_diff_htf > ema_threshold:
                    ema_long = True
                if ema_diff < 0 and ema_diff_htf < -ema_threshold:
                    ema_short = True

                increment = config.get("EMA_SCORE_INCREMENT", 1.0)

                # LONG
                if ema_long:
                    score_long += increment
                    passed_filters_long.append(f"EMA: score +{increment:.1f} (diff={ema_diff:.6f})")
                    reasons_long.append(
                        f"EMA подтверждает лонг: fast={ind['ema_fast']:.6f}, slow={ind['ema_slow']:.6f}, "
                        f"diff={ema_diff:.6f}, htf_diff={ema_diff_htf:.6f}, thr={ema_threshold:.6f}"
                    )
                    if -ema_threshold <= ema_diff <= 0:
                        reasons_long.append(f"⚠️ EMA лонг прошёл с допуском порога (diff={ema_diff:.6f}, thr={ema_threshold:.6f})")
                else:
                    passed_filters_long.append(f"EMA: score 0 (diff={ema_diff:.6f})")
                    reasons_long.append(
                        f"EMA не подтверждает лонг: fast={ind['ema_fast']:.6f}, slow={ind['ema_slow']:.6f}, "
                        f"diff={ema_diff:.6f}, htf_diff={ema_diff_htf:.6f}, thr={ema_threshold:.6f}"
                    )

                # SHORT
                if ema_short:
                    score_short += increment
                    passed_filters_short.append(f"EMA: score +{increment:.1f} (diff={ema_diff:.6f})")
                    reasons_short.append(
                        f"EMA подтверждает шорт: fast={ind['ema_fast']:.6f}, slow={ind['ema_slow']:.6f}, "
                        f"diff={ema_diff:.6f}, htf_diff={ema_diff_htf:.6f}, thr={ema_threshold:.6f}"
                    )
                    if 0 <= ema_diff <= ema_threshold:
                        reasons_short.append(f"⚠️ EMA шорт прошёл с допуском порога (diff={ema_diff:.6f}, thr={ema_threshold:.6f})")
                else:
                    passed_filters_short.append(f"EMA: score 0 (diff={ema_diff:.6f})")
                    reasons_short.append(
                        f"EMA не подтверждает шорт: fast={ind['ema_fast']:.6f}, slow={ind['ema_slow']:.6f}, "
                        f"diff={ema_diff:.6f}, htf_diff={ema_diff_htf:.6f}, thr={ema_threshold:.6f}"
                    )
                
                filters_results.update({
                    "ema_diff_ltf": ema_diff,
                    "ema_diff_htf": ema_diff_htf,
                    "ema_fast": ind["ema_fast"],
                    "ema_slow": ind["ema_slow"],
                    "ema_long_passed": 1 if ema_long else 0,
                    "ema_short_passed": 1 if ema_short else 0,
                    "ema_long_score": round(increment if ema_long else 0, 2),
                    "ema_short_score": round(increment if ema_short else 0, 2),
                    "ema_threshold_config": ema_threshold,
                })

            # --- Open Interest --- DONE
            if config.get("USE_OPEN_INTEREST", 1.0):
                try:
                    oi_change = ind["oi_change"]
                    min_oi_change = config.get("OI_CHANGE_THRESHOLD", 0.15)

                    side = trend_global
                    if side not in ["long", "short"]:
                        side = None  # неопределённый тренд, пропускаем начисление

                    if side in ["long", "short"]:
                        # Проверка направления
                        if side == "long":
                            if oi_change < min_oi_change:
                                reasons_long.append(f"OI change {oi_change:.2f}% < {min_oi_change}%")
                            passed_filters_long.append(f"OI change value={oi_change:.2f}%")
                        else:  # short
                            if oi_change > -min_oi_change:
                                reasons_short.append(f"OI change {oi_change:.2f}% > {-min_oi_change}%")
                            passed_filters_short.append(f"OI change value={oi_change:.2f}%")
                    
                    # Добавляем минимальную запись в filters_results
                    filters_results.update({
                        "oi_value": oi_change,
                        "oi_config_threshold": min_oi_change
                    })
                    
                except Exception as e:
                    print(f"[OI] Ошибка для {symbol}: {e}")
                    # Добавляем пустую запись в случае ошибки
                    filters_results.update({
                        "oi_value": None,
                        "oi_config_threshold": config.get("OI_CHANGE_THRESHOLD", 0.15)  # Получаем значение напрямую из конфига
                    })

            # === ФИЛЬТР ДИВЕРГЕНЦИЙ (RSI / MACD по нескольким ТФ) ===
            if config.get("USE_DIVERGENCE_FILTER", False):
                def calculate_divergence_scores(symbol, df_dict, config, reasons_long, reasons_short,
                                                passed_filters_long, passed_filters_short):
                    # --- Инициализация ---
                    div_long_passed = False
                    div_short_passed = False
                    div_long_score_total = 0.0
                    div_short_score_total = 0.0

                    # Частные скоры и типы
                    scores = {
                        "rsi_1h": {"long": 0.0, "short": 0.0, "type_long": None, "type_short": None},
                        "rsi_4h": {"long": 0.0, "short": 0.0, "type_long": None, "type_short": None},
                        "macd_1h": {"long": 0.0, "short": 0.0, "type_long": None, "type_short": None},
                        "macd_4h": {"long": 0.0, "short": 0.0, "type_long": None, "type_short": None},
                    }

                    # Веса из конфига
                    score_weight = {
                        "1h": config.get("DIVERGENCE_SCORE_1H", 0.25),
                        "4h": config.get("DIVERGENCE_SCORE_4H", 0.75)
                    }

                    # Получаем дивергенции по каждому ТФ
                    div_result = {}
                    for tf in config.get("DIVERGENCE_TFS", ["1h", "4h"]):
                        temp_cfg = config.copy()
                        temp_cfg["DIVERGENCE_TFS"] = [tf]
                        div_result[tf] = detect_divergence_multi_tf(symbol, df_dict, temp_cfg)

                    # Вспомогательная функция обработки одного индикатора
                    def handle_indicator(ind_name, tf, data, weight):
                        nonlocal div_long_passed, div_short_passed, div_long_score_total, div_short_score_total
                        long_score = 0.0
                        short_score = 0.0
                        type_long = None
                        type_short = None

                        # Проверка длинной позиции
                        if data.get("hidden_bullish"):
                            type_long = "hidden"
                            long_score = weight
                            div_long_passed = True
                            div_long_score_total += weight
                            passed_filters_long.append(f"{ind_name} {tf}: скрытая бычья дивергенция (+{weight:.2f})")
                        elif data.get("bullish"):
                            type_long = "regular"
                            long_score = weight
                            div_long_passed = True
                            div_long_score_total += weight
                            passed_filters_long.append(f"{ind_name} {tf}: регулярная бычья дивергенция (+{weight:.2f})")
                        else:
                            reasons_long.append(f"{ind_name} {tf}: дивергенций нет")

                        # Проверка короткой позиции
                        if data.get("hidden_bearish"):
                            type_short = "hidden"
                            short_score = weight
                            div_short_passed = True
                            div_short_score_total += weight
                            passed_filters_short.append(f"{ind_name} {tf}: скрытая медвежья дивергенция (+{weight:.2f})")
                        elif data.get("bearish"):
                            type_short = "regular"
                            short_score = weight
                            div_short_passed = True
                            div_short_score_total += weight
                            passed_filters_short.append(f"{ind_name} {tf}: регулярная медвежья дивергенция (+{weight:.2f})")
                        else:
                            reasons_short.append(f"{ind_name} {tf}: дивергенций нет")

                        return long_score, short_score, type_long, type_short

                    # Обрабатываем каждый индикатор и таймфрейм
                    for tf in ["4h", "1h"]:
                        if tf in div_result and config.get(f"DIVERGENCE_USE_{tf.upper()}", True):
                            for ind in ["rsi", "macd"]:
                                key = f"{ind}_{tf}"
                                l_score, s_score, t_long, t_short = handle_indicator(
                                    ind.upper(), tf, div_result[tf][ind], score_weight[tf]
                                )
                                scores[key]["long"] = l_score
                                scores[key]["short"] = s_score
                                scores[key]["type_long"] = t_long
                                scores[key]["type_short"] = t_short

                    # Итог по лонгу/шорту
                    score_long_add = div_long_score_total if div_long_passed else 0
                    score_short_add = div_short_score_total if div_short_passed else 0

                    if not div_long_passed:
                        reasons_long.append("Дивергенции RSI/MACD не подтвердили LONG")
                    if not div_short_passed:
                        reasons_short.append("Дивергенции RSI/MACD не подтвердили SHORT")

                    # Сохраняем результаты
                    filters_results.update({
                        "div_long_passed": 1 if div_long_passed else 0,
                        "div_short_passed": 1 if div_short_passed else 0,
                        "div_long_total_score": round(div_long_score_total, 2),
                        "div_short_total_score": round(div_short_score_total, 2),
                        "div_score_1h_config": score_weight["1h"],
                        "div_score_4h_config": score_weight["4h"],
                        "div_tf_config": config.get("DIVERGENCE_TFS", ["1h", "4h"]),
                        # RSI и MACD
                        **{f"{k}_long_score": round(v["long"], 2) for k, v in scores.items()},
                        **{f"{k}_short_score": round(v["short"], 2) for k, v in scores.items()},
                        **{f"{k}_long_type": v["type_long"] for k, v in scores.items()},
                        **{f"{k}_short_type": v["type_short"] for k, v in scores.items()},
                    })

                    return score_long_add, score_short_add

                # Вызов функции расчета дивергенций
                div_score_long, div_score_short = calculate_divergence_scores(
                    symbol, df_dict, config, reasons_long, reasons_short, passed_filters_long, passed_filters_short
                )
                score_long += div_score_long
                score_short += div_score_short


            # --- Фильтр рыночной структуры ---
            def candle_body(candle):
                return abs(candle["close"] - candle["open"])

            def is_bullish(candle):
                return candle["close"] > candle["open"]

            def is_bearish(candle):
                return candle["close"] < candle["open"]

            def get_shadows(candle):
                """Универсальный расчет теней свечи"""
                high, low = candle["high"], candle["low"]
                open_price, close_price = candle["open"], candle["close"]
                body = abs(close_price - open_price)
                
                upper_shadow = high - max(open_price, close_price)
                lower_shadow = min(open_price, close_price) - low
                
                return body, upper_shadow, lower_shadow

            def bullish_engulfing(df):
                if len(df) < 2: return False
                c1, c2 = df.iloc[-2], df.iloc[-1]
                return (is_bearish(c1) and is_bullish(c2) and 
                        c2["open"] < c1["close"] and 
                        c2["close"] > c1["open"])

            def bearish_engulfing(df):
                if len(df) < 2: return False
                c1, c2 = df.iloc[-2], df.iloc[-1]
                return (is_bullish(c1) and is_bearish(c2) and 
                        c2["open"] > c1["close"] and 
                        c2["close"] < c1["open"])

            def hammer(candle):
                body, upper, lower = get_shadows(candle)
                return (lower > 2 * body and 
                        upper < body and
                        body > 0)  # Защита от нулевого тела

            def inverted_hammer(candle):
                body, upper, lower = get_shadows(candle)
                return (upper > 2 * body and 
                        lower < body and
                        body > 0)

            def marubozu(candle, min_body=0.001, shadow_ratio=0.01):
                """Универсальная функция для Marubozu"""
                body, upper, lower = get_shadows(candle)
                if body < min_body: return False  # Слишком маленькое тело
                
                # Проверяем что тени меньше указанного процента от тела
                upper_ok = upper < shadow_ratio * body
                lower_ok = lower < shadow_ratio * body
                
                return upper_ok and lower_ok

            def marubozu_bullish(candle):
                return marubozu(candle) and is_bullish(candle)

            def marubozu_bearish(candle):
                return marubozu(candle) and is_bearish(candle)

            def shooting_star(candle):
                body, upper, lower = get_shadows(candle)
                return (upper > 2 * body and 
                        lower < body and
                        body > 0)

            def hanging_man(candle):
                return hammer(candle) and is_bearish(candle)

            def doji(candle, body_ratio=0.1):
                """Проверка с защитой от нулевого диапазона"""
                body = candle_body(candle)
                candle_range = candle["high"] - candle["low"]
                if candle_range <= 0:  # Защита от нулевого диапазона
                    return False
                return body <= body_ratio * candle_range

            def morning_star(df):
                if len(df) < 3: return False
                c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
                
                cond1 = is_bearish(c1)
                cond2 = doji(c2) or (abs(c2["close"] - c2["open"]) < 0.3 * candle_body(c1))
                midpoint = (c1["open"] + c1["close"]) / 2
                cond3 = is_bullish(c3) and c3["close"] > midpoint
                
                return cond1 and cond2 and cond3

            def evening_star(df):
                if len(df) < 3: return False
                c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
                
                cond1 = is_bullish(c1)
                cond2 = doji(c2) or (abs(c2["close"] - c2["open"]) < 0.3 * candle_body(c1))
                midpoint = (c1["open"] + c1["close"]) / 2
                cond3 = is_bearish(c3) and c3["close"] < midpoint
                
                return cond1 and cond2 and cond3

            def piercing_pattern(df):
                if len(df) < 2: return False
                c1, c2 = df.iloc[-2], df.iloc[-1]
                
                cond1 = is_bearish(c1)
                cond2 = is_bullish(c2)
                penetration_level = c1["close"] + 0.5 * (c1["open"] - c1["close"])
                cond3 = c2["close"] > penetration_level
                cond4 = c2["open"] < c1["close"]
                
                return cond1 and cond2 and cond3 and cond4

            def dark_cloud_cover(df):
                if len(df) < 2: return False
                c1, c2 = df.iloc[-2], df.iloc[-1]
                
                cond1 = is_bullish(c1)
                cond2 = is_bearish(c2)
                penetration_level = c1["close"] - 0.5 * (c1["close"] - c1["open"])
                cond3 = c2["open"] > c1["close"]
                cond4 = c2["close"] < penetration_level
                
                return cond1 and cond2 and cond3 and cond4

            def three_white_soldiers(df):
                if len(df) < 3: return False
                c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
                
                return (all(is_bullish(c) for c in [c1, c2, c3]) and
                        c2["open"] > c1["open"] and
                        c3["open"] > c2["open"] and
                        c2["close"] > c1["close"] and
                        c3["close"] > c2["close"])

            def three_black_crows(df):
                if len(df) < 3: return False
                c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
                
                return (all(is_bearish(c) for c in [c1, c2, c3]) and
                        c2["open"] < c1["open"] and
                        c3["open"] < c2["open"] and
                        c2["close"] < c1["close"] and
                        c3["close"] < c2["close"])

            def bullish_harami(df):
                if len(df) < 2: return False
                c1, c2 = df.iloc[-2], df.iloc[-1]
                body1 = abs(c1["close"] - c1["open"])
                body2 = abs(c2["close"] - c2["open"])
                return (is_bearish(c1) and 
                        is_bullish(c2) and
                        c2["open"] > c1["close"] and 
                        c2["close"] < c1["open"] and
                        body2 < 0.75 * body1)

            def bearish_harami(df):
                if len(df) < 2: return False
                c1, c2 = df.iloc[-2], df.iloc[-1]
                body1 = abs(c1["close"] - c1["open"])
                body2 = abs(c2["close"] - c2["open"])
                return (is_bullish(c1) and 
                        is_bearish(c2) and
                        c2["open"] < c1["close"] and 
                        c2["close"] > c1["open"] and
                        body2 < 0.75 * body1)

            def bullish_tasuki_gap(df):
                if len(df) < 3: return False
                c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
                gap = c2["low"] > c1["high"]
                return (is_bullish(c1) and 
                        is_bullish(c2) and
                        gap and
                        is_bearish(c3) and
                        c3["open"] < c2["close"] and
                        c3["close"] > c1["high"])

            def bearish_tasuki_gap(df):
                if len(df) < 3: return False
                c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
                gap = c2["high"] < c1["low"]
                return (is_bearish(c1) and 
                        is_bearish(c2) and
                        gap and
                        is_bullish(c3) and
                        c3["open"] > c2["close"] and
                        c3["close"] < c1["low"])

            def bullish_kicking(df):
                if len(df) < 2: return False
                c1, c2 = df.iloc[-2], df.iloc[-1]
                gap = c2["low"] > c1["high"]
                return (marubozu_bearish(c1) and 
                        marubozu_bullish(c2) and
                        gap)

            def bearish_kicking(df):
                if len(df) < 2: return False
                c1, c2 = df.iloc[-2], df.iloc[-1]
                gap = c2["high"] < c1["low"]
                return (marubozu_bullish(c1) and 
                        marubozu_bearish(c2) and
                        gap)

            def dragonfly_doji(candle, body_ratio=0.1):
                body = candle_body(candle)
                candle_range = candle["high"] - candle["low"]
                if candle_range <= 0: return False
                upper_shadow = candle["high"] - max(candle["close"], candle["open"])
                lower_shadow = min(candle["close"], candle["open"]) - candle["low"]
                return (body <= body_ratio * candle_range and
                        upper_shadow <= 0.1 * candle_range and
                        lower_shadow >= 0.6 * candle_range)

            def gravestone_doji(candle, body_ratio=0.1):
                body = candle_body(candle)
                candle_range = candle["high"] - candle["low"]
                if candle_range <= 0: return False
                upper_shadow = candle["high"] - max(candle["close"], candle["open"])
                lower_shadow = min(candle["close"], candle["open"]) - candle["low"]
                return (body <= body_ratio * candle_range and
                        lower_shadow <= 0.1 * candle_range and
                        upper_shadow >= 0.6 * candle_range)

            def long_legged_doji(candle, body_ratio=0.1):
                body = candle_body(candle)
                candle_range = candle["high"] - candle["low"]
                if candle_range <= 0: return False
                upper_shadow = candle["high"] - max(candle["close"], candle["open"])
                lower_shadow = min(candle["close"], candle["open"]) - candle["low"]
                return (body <= body_ratio * candle_range and
                        lower_shadow >= 0.3 * candle_range and
                        upper_shadow >= 0.3 * candle_range)

            def bullish_harami_cross(df):
                if len(df) < 2: return False
                c1, c2 = df.iloc[-2], df.iloc[-1]
                return (is_bearish(c1) and 
                        doji(c2) and
                        c2["high"] < c1["open"] and 
                        c2["low"] > c1["close"])

            def bearish_harami_cross(df):
                if len(df) < 2: return False
                c1, c2 = df.iloc[-2], df.iloc[-1]
                return (is_bullish(c1) and 
                        doji(c2) and
                        c2["high"] < c1["close"] and 
                        c2["low"] > c1["open"])

            # ======== Основной код фильтра ========  DONE
            if config.get("USE_MARKET_STRUCTURE_FILTER", False):
                if config.get("USE_MARKET_STRUCTURE_FILTER", False):
                    if config.get("USE_STRUCTURE_HH_HL_FILTER", False):
                        # Исправленная логика рыночной структуры
                        if len(df_htf) >= 2:
                            # Для лонга: более высокий максимум и минимум
                            hh = df_htf["high"].iloc[-1] > df_htf["high"].iloc[-2]
                            hl = df_htf["low"].iloc[-1] > df_htf["low"].iloc[-2]
                            long_struct = hh and hl

                            # Для шорта: более низкий максимум и минимум
                            lh = df_htf["high"].iloc[-1] < df_htf["high"].iloc[-2]
                            ll = df_htf["low"].iloc[-1] < df_htf["low"].iloc[-2]
                            short_struct = lh and ll
                        else:
                            long_struct = short_struct = False

                        structure_weight = config.get("MARKET_SCORE_STRUCTURE_INCREMENT", 0.5)

                        # Сохраняем результаты для последующего добавления в filters_results
                        market_structure_ltf_passed = 1 if long_struct else 0
                        market_structure_ltf_score = structure_weight if long_struct else 0.0
                        market_structure_htf_passed = 1 if short_struct else 0
                        market_structure_htf_score = structure_weight if short_struct else 0.0

                        if long_struct:
                            score_long += structure_weight
                        else:
                            reasons_long.append("Нет структуры HH/HL для лонга")
                        
                        if short_struct:
                            score_short += structure_weight
                        else:
                            reasons_short.append("Нет структуры LH/LL для шорта")

                        # Добавляем итоговую запись в filters_results
                        filters_results.update({
                            "market_structure_ltf_passed": market_structure_ltf_passed,
                            "market_structure_ltf_score": market_structure_ltf_score,
                            "market_structure_htf_passed": market_structure_htf_passed,
                            "market_structure_htf_score": market_structure_htf_score,
                            "market_structure_increment_config": config["MARKET_SCORE_STRUCTURE_INCREMENT"]
                        })


                pattern_names = {
                    "bullish_engulfing": "Бычье поглощение",
                    "hammer": "Молот",
                    "inverted_hammer": "Перевернутый молот",
                    "marubozu_bullish": "Бычья марубозу",
                    "morning_star": "Утренняя звезда",
                    "piercing_pattern": "Пробивной паттерн",
                    "three_white_soldiers": "Три белых солдата",
                    "bearish_engulfing": "Медвежье поглощение",
                    "shooting_star": "Падающая звезда",
                    "hanging_man": "Повешенный",
                    "marubozu_bearish": "Медвежья марубозу",
                    "evening_star": "Вечерняя звезда",
                    "dark_cloud_cover": "Темное облако",
                    "three_black_crows": "Три черные вороны",
                    "doji": "Доджи",
                    "bullish_harami": "Бычий харами",
                    "bearish_harami": "Медвежий харами",
                    "bullish_tasuki_gap": "Бычий Тасуки гэп",
                    "bearish_tasuki_gap": "Медвежий Тасуки гэп",
                    "bullish_kicking": "Бычий удар",
                    "bearish_kicking": "Медвежий удар",
                    "dragonfly_doji": "Доджи-стрекоза",
                    "gravestone_doji": "Доджи-надгробие",
                    "long_legged_doji": "Длинноногий доджи",
                    "bullish_harami_cross": "Бычий харами-крест",
                    "bearish_harami_cross": "Медвежий харами-крест"
                }
                
                # Обновленный список свечных паттернов с проверкой длины
                candle_checks = [
                    ("bullish_engulfing", bullish_engulfing, 1),
                    ("hammer", lambda df: hammer(df.iloc[-1]) if len(df) >= 1 else False, 1),
                    ("inverted_hammer", lambda df: inverted_hammer(df.iloc[-1]) if len(df) >= 1 else False, 1),
                    ("marubozu_bullish", lambda df: marubozu_bullish(df.iloc[-1]) if len(df) >= 1 else False, 1),
                    ("morning_star", morning_star, 1),
                    ("piercing_pattern", piercing_pattern, 1),
                    ("three_white_soldiers", three_white_soldiers, 1),
                    
                    ("bearish_engulfing", bearish_engulfing, -1),
                    ("shooting_star", lambda df: shooting_star(df.iloc[-1]) if len(df) >= 1 else False, -1),
                    ("hanging_man", lambda df: hanging_man(df.iloc[-1]) if len(df) >= 1 else False, -1),
                    ("marubozu_bearish", lambda df: marubozu_bearish(df.iloc[-1]) if len(df) >= 1 else False, -1),
                    ("evening_star", evening_star, -1),
                    ("dark_cloud_cover", dark_cloud_cover, -1),
                    ("three_black_crows", three_black_crows, -1),
                    ("bullish_harami", bullish_harami, 1),
                    ("bearish_harami", bearish_harami, -1),
                    ("bullish_tasuki_gap", bullish_tasuki_gap, 1),
                    ("bearish_tasuki_gap", bearish_tasuki_gap, -1),
                    ("bullish_kicking", bullish_kicking, 1),
                    ("bearish_kicking", bearish_kicking, -1),
                    ("dragonfly_doji", lambda df: dragonfly_doji(df.iloc[-1]) if len(df) >= 1 else False, 0),
                    ("gravestone_doji", lambda df: gravestone_doji(df.iloc[-1]) if len(df) >= 1 else False, 0),
                    ("long_legged_doji", lambda df: long_legged_doji(df.iloc[-1]) if len(df) >= 1 else False, 0),
                    ("bullish_harami_cross", bullish_harami_cross, 1),
                    ("bearish_harami_cross", bearish_harami_cross, -1),
                    
                    ("doji", lambda df: doji(df.iloc[-1]) if len(df) >= 1 else False, 0),
                ]

                # Лонг
                pattern_long_found = False
                # Шорт
                pattern_short_found = False

                confirming_zones = {"long": False, "short": False}
        
                # Проверяем наличие подтверждающих зон
                levels = get_enhanced_support_resistance(df, config)
                price = df["close"].iloc[-1]
                threshold = config.get("SD_ZONE_DISTANCE_THRESHOLD", 0.015)
                min_strength = config.get("SD_MIN_STRENGTH", 1.5)
                
                # Для бычьих паттернов ищем поддержку (demand)
                for level, data in levels.items():
                    if (data["type"] == "support" and 
                        abs(price - level) / price < threshold and 
                        data["strength"] >= min_strength):
                        confirming_zones["long"] = True
                        break
                
                # Для медвежьих паттернов ищем сопротивление (supply)
                for level, data in levels.items():
                    if (data["type"] == "resistance" and 
                        abs(price - level) / price < threshold and 
                        data["strength"] >= min_strength):
                        confirming_zones["short"] = True
                        break

                for name, func, direction in candle_checks:
                    # Переменные для сбора информации о свечных паттернах
                    candle_pattern_passed = 0
                    candle_pattern_score = 0.0
                    candle_pattern_sd_bonus = 0
                    candle_pattern_sd_bonus_score = 0.0
                    candle_pattern_sd_bonus_passed = 0

                    try:
                        result = func(df)
                        human_name = pattern_names.get(name, name)

                        if result:
                            # Бычий паттерн
                            if direction == 1:
                                has_confirming_zone = confirming_zones["long"]
                                score_to_add = config["CANDLE_PATTERN_WITH_SD_BONUS"] if has_confirming_zone else config["CANDLE_PATTERN_BASE_SCORE"]

                                if not pattern_long_found:
                                    score_long += score_to_add
                                    pattern_long_found = True
                                    reason_msg = f"Найден бычий паттерн: {human_name}"
                                    if has_confirming_zone:
                                        reason_msg += " (подтверждён S/D зоной)"
                                    reasons_long.append(reason_msg)
                                    passed_filters_long.append(f"Найден бычий паттерн: {human_name}: score +{score_to_add:.1f}")

                                    # Обновляем данные для итоговой записи
                                    candle_pattern_passed = 1
                                    candle_pattern_score = score_to_add
                                    if has_confirming_zone:
                                        candle_pattern_sd_bonus = 1
                                        candle_pattern_sd_bonus_score = config["CANDLE_PATTERN_WITH_SD_BONUS"] - config["CANDLE_PATTERN_BASE_SCORE"]
                                        candle_pattern_sd_bonus_passed = 1

                            # Медвежий паттерн
                            elif direction == -1:
                                has_confirming_zone = confirming_zones["short"]
                                score_to_add = config["CANDLE_PATTERN_WITH_SD_BONUS"] if has_confirming_zone else config["CANDLE_PATTERN_BASE_SCORE"]

                                if not pattern_short_found:
                                    score_short += score_to_add
                                    pattern_short_found = True
                                    reason_msg = f"Найден медвежий паттерн: {human_name}"
                                    if has_confirming_zone:
                                        reason_msg += " (подтверждён S/D зоной)"
                                    reasons_short.append(reason_msg)
                                    passed_filters_short.append(f"Найден медвежий паттерн: {human_name}: score +{score_to_add:.1f}")

                                    # Обновляем данные для итоговой записи
                                    candle_pattern_passed = 1
                                    candle_pattern_score = score_to_add
                                    if has_confirming_zone:
                                        candle_pattern_sd_bonus = 1
                                        candle_pattern_sd_bonus_score = config["CANDLE_PATTERN_WITH_SD_BONUS"] - config["CANDLE_PATTERN_BASE_SCORE"]
                                        candle_pattern_sd_bonus_passed = 1

                            # Нейтральные паттерны
                            else:
                                reasons_long.append(f"Найден паттерн: {human_name} ( нейтральный)")
                                reasons_short.append(f"Найден паттерн: {human_name} (нейтральный)")
                                
                        else:
                            # Для ненайденных паттернов не добавляем в passed_filters, только в filters_results для статистики
                            filters_results.update({
                                "candle_pattern_passed": 0,
                                "candle_pattern_score": 0,
                                "candle_pattern_sd_bonus": 0,
                                "candle_pattern_sd_bonus_score": 0,
                                "candle_pattern_sd_bonus_passed": 0,
                                "candle_pattern_with_sd_bonus_config": config["CANDLE_PATTERN_WITH_SD_BONUS"],
                                "candle_pattern_base_score_config": config["CANDLE_PATTERN_BASE_SCORE"]
                            })

                    except Exception as e:
                        print(f"[ERROR] Ошибка при проверке паттерна {name} для {symbol}: {e}")

                    # Добавляем итоговую запись в filters_results
                    filters_results.update({
                        "candle_pattern_passed": candle_pattern_passed,
                        "candle_pattern_score": candle_pattern_score,
                        "candle_pattern_sd_bonus": candle_pattern_sd_bonus,
                        "candle_pattern_sd_bonus_score": candle_pattern_sd_bonus_score,
                        "candle_pattern_sd_bonus_passed": candle_pattern_sd_bonus_passed,
                        "candle_pattern_with_sd_bonus_config": config["CANDLE_PATTERN_WITH_SD_BONUS"],
                        "candle_pattern_base_score_config": config["CANDLE_PATTERN_BASE_SCORE"]
                    })


                # После проверки всех паттернов, если не найдено ни одного паттерна для направления
                if not pattern_long_found and not any(r.startswith("Найден паттерн") for r in reasons_long):
                    reasons_long.append("Не найдено подтверждающих паттернов")
                    
                if not pattern_short_found and not any(r.startswith("Найден паттерн") for r in reasons_short):
                    reasons_short.append("Не найдено подтверждающих паттернов")

            # --- Фильтр структурных паттернов ---
            if config.get("USE_STRUCTURAL_FILTER", True):
                structural_passed = 0
                structural_score = 0.0
                structural_tolerance = config.get("STRUCTURAL_TOLERANCE", 0.003)
                structural_mode = config.get("STRUCTURAL_MODE", "strict")

                # Проверяем направление тренда
                if trend_global == "long":
                    score_val_long, patterns_long_found = structural_patterns_filter(
                        df,
                        direction="long",
                        tolerance=structural_tolerance,
                        mode=structural_mode
                    )

                    if score_val_long > 0:
                        inc = config.get("STRUCTURAL_SCORE_INCREMENT", 1.0)
                        score_long += inc
                        structural_passed = 1
                        structural_score = inc
                        passed_filters_long.append(
                            f"СТРУКТУРА: +{inc:.1f} (найдено: {', '.join(patterns_long_found)})"
                        )
                    else:
                        passed_filters_long.append(
                            f"СТРУКТУРА: (найдено: {', '.join(patterns_long_found) or 'нет'})"
                        )

                elif trend_global == "short":
                    score_val_short, patterns_short_found = structural_patterns_filter(
                        df,
                        direction="short",
                        tolerance=structural_tolerance,
                        mode=structural_mode
                    )

                    if score_val_short > 0:
                        inc = config.get("STRUCTURAL_SCORE_INCREMENT", 1.0)
                        score_short += inc
                        structural_passed = 1
                        structural_score = inc
                        passed_filters_short.append(
                            f"СТРУКТУРА: +{inc:.1f} (найдено: {', '.join(patterns_short_found)})"
                        )
                    else:
                        passed_filters_short.append(
                            f"СТРУКТУРА: (найдено: {', '.join(patterns_short_found) or 'нет'})"
                        )

                else:
                    # тренда нет — фильтр не применяется
                    passed_filters_long.append("СТРУКТУРА: пропуск (нет тренда)")
                    passed_filters_short.append("СТРУКТУРА: пропуск (нет тренда)")

                # Добавляем итоговую запись в filters_results
                filters_results.update({
                    "structural_passed": structural_passed,
                    "structural_score": structural_score,
                    "structural_tolerance": structural_tolerance,
                    "structural_mode_config": structural_mode
                })

            # --- Итог ---
            score_threshold = config.get("SIGNAL_SCORE_THRESHOLD", 6)

            is_valid = True
            final_signal = None

            # Определяем итоговый сигнал
            if score_long >= score_threshold and score_long >= score_short:
                print(f"[SIGNAL PASSED][{symbol}] LONG passed with score {score_long}. Filters: {', '.join(passed_filters_long)}")
                final_signal = "long"
            elif score_short >= score_threshold:
                print(f"[SIGNAL PASSED][{symbol}] SHORT passed with score {score_short}. Filters: {', '.join(passed_filters_short)}")
                final_signal = "short"
            else:
                print(f"[SIGNAL FAILED][{symbol}] Лонг: {reasons_long} | Шорт: {reasons_short}")
                is_valid = False
            # --- Фильтр по зонам S/D ---    
            if final_signal is not None and config.get("USE_SD_ZONE_FILTER", True):
                sd_reason = check_proximity_to_sd_zone(df, final_signal, config)
                if sd_reason:
                    print(f"[FILTER GLOBAL S/D] {symbol}: {sd_reason['text']} — сигнал {final_signal} отклонён ")
                    filters_results.update({
                        "sd_zone_type": sd_reason["type"],
                        "sd_zone_level": sd_reason["level"],
                        "sd_zone_strength": sd_reason["strength"],
                        "sd_zone_broken": 1 if sd_reason["broken"] else 0
                    })
                    is_valid = False

            # Контроль глобального тренда
            if config.get("USE_GLOBAL_TREND_FILTER", True):
                if trend_global not in ["long", "short"]:
                    print("[FILTER GLOBAL TREND] Глобальный тренд не определён — сигнал отклонён")
                    is_valid = False
                if final_signal != trend_global:
                    print(f"[FILTER GLOBAL TREND] Контртренд: тренд {trend_global}, сигнал {final_signal} — отклонено")
                    is_valid = False

            print(f"[DEBUG] Score long: {score_long}, short: {score_short}, threshold: {score_threshold}")

            filters_results.update({
                "scores_config": score_threshold,
                "final_scores": score_long if final_signal == 'long' else score_short,
                "signal_passed": 1 if is_valid else 0,
            })

            # Сохраняем исходное значение is_valid до ML
            original_is_valid = is_valid

            # --- ML оценка ---
            ml_out = {}
            ml_allowed = ML_CONFIG.get("enabled", False) and final_signal is not None and trend_global in ["long", "short"]

            if ml_allowed:
                # ML оценивает сигналы с определенным трендом
                ml_out = ml_handle_signal(filters_results, original_is_valid)
                
                # Сохраняем ML результат для БД
                if ml_out and "ml_result" in ml_out:
                    filters_results.update(ml_out["ml_result"])
                
                filters_results["ml_allowed"] = 1
                
                # --- Решение на основе режима работы ML ---
                if ML_CONFIG.get("enabled", False):
                    mode = ML_CONFIG.get("mode", "disabled")
                    ml_action = ml_out.get("ml_action", "neutral")
                    
                    print(f"[ML MODE] {symbol}: режим {mode}, действие ML: {ml_action}")
                    
                    if mode == "advisory":
                        # ML только советует, не меняет решение
                        pass
                    elif mode == "hybrid":
                        # Комбинированное решение: ML может переопределить
                        if ml_action == "approve" and not original_is_valid:
                            print(f"[ML OVERRIDE] {symbol}: ML одобрила отклоненный сигнал")
                            is_valid = True
                        elif ml_action == "reject" and original_is_valid:
                            print(f"[ML OVERRIDE] {symbol}: ML отклонила валидный сигнал")
                            is_valid = False
                    elif mode == "autonomous":
                        # Полностью автономный режим
                        if ml_action == "approve":
                            is_valid = True
                        elif ml_action == "reject":
                            is_valid = False
                        # При neutral оставляем текущее is_valid
            else:
                # Заполняем default значения для всех ML полей
                filters_results.update({
                    "ml_mode": "disabled",
                    "ml_decision": "neutral",
                    "ml_comment": "",
                    "ml_predicted_class": 0,
                    "ml_predicted_label": "",
                    "ml_confidence": 0.0,
                    "ml_strength": 0.0,
                    "ml_probas": "[]",
                    "ml_allowed": 0
                })

            # Обновляем signal_passed после возможного изменения ML
            filters_results["signal_passed"] = 1 if is_valid else 0

            # --- Сохраняем в БД только если глобальный тренд определён ---
            if trend_global in ["long", "short"]:
                try:
                    await db_stats.add_filter_result(filters_results)
                    print(f"[DB SUCCESS] {symbol}: данные сохранены в БД")
                except Exception as e:
                    print(f"[DB ERROR] {symbol}: ошибка сохранения в БД: {e}")
            else:
                print(f"[SKIP DB] Сигнал {filters_results.get('signal_id')} не записан — глобальный тренд не определён")

            if not is_valid:
                return None

            return {
                "signal_id": filters_results["signal_id"],
                "side": final_signal,
                "filters_results": filters_results,
                "ml_out": ml_out
            }
        
        except Exception as e:
            print(f"[ERROR] Signal calculation for {symbol}: {e}")
            return None

    # === Функция для получения текущей цены с учётом last_prices ===
    def get_live_price(symbol, fallback=None):
        data = last_prices.get(symbol)
        if data:
            return data if isinstance(data, float) else data.get("price", fallback)
        return fallback

    # === Функция для получения количества знаков после запятой в цене ===
    def get_decimal_places(price):
        s = str(price)
        if '.' in s:
            decimals = len(s.split('.')[1].rstrip('0'))
            return max(4, decimals)
        return 4
    
    # === СТАБЫ для поиска FVG и S/D зон ===
    def find_nearest_fvg(entry, signal, df):
        try:
            # Явная проверка на пустой DataFrame
            if df is None or df.empty:
                return None

            """
            Поиск ближайшего FVG (Fair Value Gap) для TP1.
            FVG (gap) — разрыв между high предыдущей свечи и low следующей (long),
            либо между low предыдущей и high следующей (short).
            Возвращает float (уровень gap) или None.
            """
            if df is None or len(df) < 5:
                return None
            fvg_levels = []
            for i in range(1, len(df) - 1):
                prev = df.iloc[i - 1]
                curr = df.iloc[i]
                next_ = df.iloc[i + 1]
                # Для long ищем gap вниз (между prev high и next low)
                if signal == "long":
                    if prev["high"] < next_["low"]:
                        gap_low = prev["high"]
                        gap_high = next_["low"]
                        gap_size = (gap_high - gap_low) / gap_low
                        if gap_low > entry and gap_size > 0.002:  # хотя бы 0.2%
                            # 25% ширины FVG для TP1 (от нижней границы)
                            tp1_fvg = gap_low + 0.20 * (gap_high - gap_low)
                            fvg_levels.append(tp1_fvg)

                # Для short ищем gap вверх (между prev low и next high)
                elif signal == "short":
                    if prev["low"] > next_["high"]:
                        gap_high = prev["low"]
                        gap_low = next_["high"]
                        gap_size = (gap_high - gap_low) / gap_high
                        if gap_high < entry and gap_size > 0.002:  # хотя бы 0.2%
                            # 25% ширины FVG для TP1 (от верхней границы вниз)
                            tp1_fvg = gap_high - 0.20 * (gap_high - gap_low)
                            fvg_levels.append(tp1_fvg)
            # Возвращаем ближайший к entry по направлению сделки
            if not fvg_levels:
                return None
            if signal == "long":
                return min(fvg_levels)
            else:
                return max(fvg_levels)
        except Exception as e:
            print(f"[ERROR] В функции find_nearest_fvg: {e}")
            traceback.print_exc()
            return None


    # === Функция для поиска ближайшей S/D зоны ===
    def find_nearest_sd_zone(entry, signal, df):
        try:
            """
            Поиск ближайшей S/D (Supply/Demand) зоны для TP3.
            Ищет диапазон консолидации (узкий диапазон, много касаний).
            Возвращает float (граница зоны) или None.
            """
            if df is None or len(df) < 20:
                return None
            
            window = 10
            threshold = 0.003  # 0.3% ширина зоны
            zones = []
            for i in range(len(df) - window):
                win = df.iloc[i:i+window]
                zone_low = win["low"].min()
                zone_high = win["high"].max()
                width = (zone_high - zone_low) / zone_low if zone_low else 0
                if width < threshold:
                    zones.append((zone_low, zone_high))  # сохраняем как диапазон

            # Отфильтровать зоны по направлению и entry
            filtered_zones = []
            for zl, zh in zones:
                center = (zl + zh) / 2
                if signal == "long" and center > entry:
                    filtered_zones.append((zl, zh))
                elif signal == "short" and center < entry:
                    filtered_zones.append((zl, zh))

            if not filtered_zones:
                return None

            # Вернуть ближнюю границу зоны
            if signal == "long":
                return min(filtered_zones, key=lambda z: z[0])[0]  # ближайшая нижняя граница
            else:
                return max(filtered_zones, key=lambda z: z[1])[1]  # ближайшая верхняя граница
        except Exception as e:
            print(f"[ERROR] В функции find_nearest_sd_zone: {e}")
            traceback.print_exc()
            return None

    # === Расчёта тейков с гибкими таймфреймами ===
    def calculate_targets(entry_zone_min, entry_zone_max, atr, signal, symbol, decimal_places, config, df_dict, verbose=False):
        """
        Расчет тейков (гибко, любое количество TP).
        Возвращает список: [(tp1, tp1_type), (tp2, tp2_type), ...].
        Все TP гарантированно ≥ min_tp_dist от предыдущего.
        """
        try:
            
            # Получаем названия таймфреймов из конфига
            tf_fvg_name = config.get("TP_FVG_TF", "15m")
            tf_swing_name = config.get("TP_SWING_TF", "30m")
            tf_sd_name = config.get("TP_SD_TF", "1h")
            
            # Безопасное получение DataFrame с явной проверкой
            df_fvg = df_dict.get(tf_fvg_name)
            if df_fvg is None or df_fvg.empty:
                df_fvg = None
                print(f"[WARNING] Нет данных FVG для {symbol} на таймфрейме {tf_fvg_name}")
            
            df_swing = df_dict.get(tf_swing_name)
            if df_swing is None or df_swing.empty:
                df_swing = None
                print(f"[WARNING] Нет данных Swing для {symbol} на таймфрейме {tf_swing_name}")
            
            df_sd = df_dict.get(tf_sd_name)
            if df_sd is None or df_sd.empty:
                df_sd = None
                print(f"[WARNING] Нет данных S/D для {symbol} на таймфрейме {tf_sd_name}")
            
            swing_lookback = config.get("SWING_LOOKBACK", 30)
            tp_count = config.get("TP_COUNT", 5)

            # Получаем уровни свингов с обработкой пустых данных
            highs_swing, lows_swing, _, _ = get_swing_levels_advanced(
                symbol=symbol,
                df=df_swing,
                lookback=swing_lookback,
                left=config.get("SWING_LEFT", 4),
                right=config.get("SWING_RIGHT", 4),
                volume_mult=config.get("SWING_VOLUME_MULT", 1.2),
                min_atr_ratio=config.get("SWING_MIN_ATR_RATIO", 0.001),
                min_price_dist=config.get("SWING_MIN_PRICE_DIST", 0.004)
            )

            atr_mults = config.get("TP_ATR", [0.5, 1.0, 1.5, 2.0, 3.0])
            if len(atr_mults) < tp_count:
                step = atr_mults[-1] - atr_mults[-2] if len(atr_mults) > 1 else 0.5
                while len(atr_mults) < tp_count:
                    atr_mults.append(round(atr_mults[-1] + step, 6))

            def atr_fallback(mult, anchor):
                if signal == "long":
                    return round(anchor + atr * mult, decimal_places)
                else:
                    return round(anchor - atr * mult, decimal_places)

            min_tp_dist = atr * config.get("MIN_TP_DIST_MULT", 1.0)
            entry_anchor = entry_zone_max if signal == "long" else entry_zone_min

            targets = []

            # ========== TP1 ==========
            fvg_level = find_nearest_fvg(entry_zone_min if signal == "long" else entry_zone_max, signal, df_fvg)
            if fvg_level is not None:
                tp1, tp1_type = round(fvg_level, decimal_places), "fvg"
            else:
                if signal == "long":
                    swings_above = [lvl for lvl in highs_swing if lvl > entry_zone_max]
                    if swings_above:
                        tp1, tp1_type = round(min(swings_above), decimal_places), "swing"
                    else:
                        tp1, tp1_type = atr_fallback(atr_mults[0], entry_zone_max), "atr"
                else:
                    swings_below = [lvl for lvl in lows_swing if lvl < entry_zone_min]
                    if swings_below:
                        tp1, tp1_type = round(max(swings_below), decimal_places), "swing"
                    else:
                        tp1, tp1_type = atr_fallback(atr_mults[0], entry_zone_min), "atr"

            # Проверка TP1 на дистанцию от входа
            if (signal == "long" and tp1 - entry_anchor < min_tp_dist) or \
            (signal == "short" and entry_anchor - tp1 < min_tp_dist):
                tp1, tp1_type = atr_fallback(max(atr_mults[0], 1.0), entry_anchor), "atr"

            targets.append((tp1, tp1_type))

            # ========== TP2..TPN ==========
            for i in range(2, tp_count + 1):
                if i == 2:
                    # TP2 ищем swing/SD или ATR fallback
                    sd_level = find_nearest_sd_zone(entry_zone_min if signal == "long" else entry_zone_max, signal, df_sd)
                    if signal == "long":
                        swings_above = [lvl for lvl in highs_swing if lvl > entry_zone_max and lvl != tp1]
                        if swings_above:
                            tp, tp_type = round(min(swings_above), decimal_places), "swing"
                        elif sd_level is not None:
                            tp, tp_type = round(sd_level, decimal_places), "sd"
                        else:
                            tp, tp_type = atr_fallback(atr_mults[1], entry_anchor), "atr"
                    else:
                        swings_below = [lvl for lvl in lows_swing if lvl < entry_zone_min and lvl != tp1]
                        if swings_below:
                            tp, tp_type = round(max(swings_below), decimal_places), "swing"
                        elif sd_level is not None:
                            tp, tp_type = round(sd_level, decimal_places), "sd"
                        else:
                            tp, tp_type = atr_fallback(atr_mults[1], entry_anchor), "atr"
                else:
                    mult = atr_mults[i - 1]
                    tp = atr_fallback(mult, entry_anchor)
                    tp_type = "atr"

                # Проверка минимальной дистанции от предыдущего TP
                prev_lvl, _ = targets[-1]
                if (signal == "long" and tp - prev_lvl < min_tp_dist):
                    tp = round(prev_lvl + min_tp_dist, decimal_places)
                elif (signal == "short" and prev_lvl - tp < min_tp_dist):
                    tp = round(prev_lvl - min_tp_dist, decimal_places)

                targets.append((tp, tp_type))

            if verbose:
                for i, (lvl, t_type) in enumerate(targets, 1):
                    print(f"TP{i}: {lvl} (type: {t_type})")

            return targets
        
        except Exception as e:
            print(f"[ERROR] В функции calculate_targets для {symbol}: {e}")
            traceback.print_exc()
            
            # Fallback: возвращаем тейки на основе ATR
            atr_mults = config.get("TP_ATR", [0.5, 1.0, 1.5, 2.0, 3.0])
            entry_anchor = entry_zone_max if signal == "long" else entry_zone_min
            
            targets = []
            for i, mult in enumerate(atr_mults):
                if signal == "long":
                    tp = round(entry_anchor + atr * mult, decimal_places)
                else:
                    tp = round(entry_anchor - atr * mult, decimal_places)
                targets.append((tp, "atr_fallback"))
            
            return targets

    # === Функция для получения уровней swing high/low с дополнительными фильтрами ===
    def get_swing_levels_advanced(
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        lookback: int = 100,
        left: int = 2,
        right: int = 2,
        volume_mult: float = 1.2,
        min_atr_ratio: float = 0.001,
        min_price_dist: float = 0.004
    ):
        """
        Возвращает списки swing_highs и swing_lows [(i, high), ...], а также последний high/low.
        """

        if df is None or df.empty:
            print(f"[WARNING] Пустой DataFrame для {symbol} в get_swing_levels_advanced")
            return [], [], None, None
        try:
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)

            df["atr"] = df["high"] - df["low"]
            df["atr_sma"] = df["atr"].rolling(window=10).mean()
            df["vol_sma"] = df["volume"].rolling(window=10).mean()

            current_price = df["close"].iloc[-1]

            swing_highs = []
            swing_lows = []

            for i in range(left, len(df) - right):
                high = df["high"].iloc[i]
                low = df["low"].iloc[i]
                close = df["close"].iloc[i]
                volume = df["volume"].iloc[i]
                atr_ratio = df["atr_sma"].iloc[i] / close if close != 0 else 0
                price_dist_high = abs(high - current_price) / current_price
                price_dist_low = abs(low - current_price) / current_price

                # Swing High
                if all(high > df["high"].iloc[i - j] for j in range(1, left + 1)) and \
                all(high > df["high"].iloc[i + j] for j in range(1, right + 1)) and \
                volume > df["vol_sma"].iloc[i] * volume_mult and \
                atr_ratio > min_atr_ratio and \
                price_dist_high > min_price_dist:
                    swing_highs.append((i, high))

                # Swing Low
                if all(low < df["low"].iloc[i - j] for j in range(1, left + 1)) and \
                all(low < df["low"].iloc[i + j] for j in range(1, right + 1)) and \
                volume > df["vol_sma"].iloc[i] * volume_mult and \
                atr_ratio > min_atr_ratio and \
                price_dist_low > min_price_dist:
                    swing_lows.append((i, low))

            highs = [h for i, h in swing_highs]
            lows = [l for i, l in swing_lows]

            last_high = highs[-1] if highs else df["high"].max()
            last_low = lows[-1] if lows else df["low"].min()

            return highs, lows, last_high, last_low

        except Exception as e:
            print(f"[ERROR] get_swing_levels_advanced({symbol}): {e}")
            return [], [], None, None

    # === Функция для поиска зон консолидации ===
    def find_consolidation_zone(df, window=30, consolidation_thresh=0.005):
        closes = df['close'].iloc[-window:]
        low = closes.min()
        high = closes.max()
        width = (high - low) / low

        if width <= consolidation_thresh:
            return low, high
        return None

    # === Функция для расчёта стоп-лосса с учётом свингов и ATR === НЕ ТЕСТИРОВАНА
    # Использует get_swing_levels_advanced для получения свингов
    # Добавляет буфер ATR для защиты от рыночного шума
    # Проверяет, что стоп находится в диапазоне 5-7% от цены входа
    # Возвращает уровень стопа с округлением до нужного количества знаков после запятой
 
    # === Функция для расчёта стоп-лосса с учётом свингов и ATR === ТЕСТИРУЕМ, типа дохуя умная
    def calculate_stop(entry_zone_min, entry_zone_max, atr, signal, symbol, decimal_places, config, df_dict: dict):
        """
        Умный стоп:
        - Для BTC: большой ATR → стоп будет шире.
        - Для мелких альтов: ATR маленький → стоп будет уже, но не меньше 3%.
        - Если есть свинг в диапазоне ATR±% – ставим по нему.
        """
        try:
            # Параметры из конфига
            atr_mult = config.get("STOP_ATR_MULTIPLIER", 1.5)
            min_atr_mult = config.get("STOP_MIN_ATR_MULTIPLIER", 1.0)
            max_atr_mult = config.get("STOP_MAX_ATR_MULTIPLIER", 3.0)
            min_dist_pct = config.get("STOP_MIN_DISTANCE_PCT", 0.03)
            max_dist_pct = config.get("STOP_MAX_DISTANCE_PCT", 0.07)
            atr_buffer_mult = config.get("STOP_ATR_BUFFER_MULT", 0.5)

            # Получаем свинги
            tf_stop_name = config.get("STOP_TF", "15m")
            stop_tf = df_dict.get(tf_stop_name)
            
            # Явная проверка на пустой DataFrame
            if stop_tf is None or stop_tf.empty:
                print(f"[WARNING] Нет таймфрейма для стопа {symbol} на {tf_stop_name}")
                # Возвращаем стоп по умолчанию на основе ATR
                if signal == "long":
                    return round(entry_zone_min - atr * atr_mult, decimal_places)
                else:
                    return round(entry_zone_max + atr * atr_mult, decimal_places)

            highs, lows, _, _ = get_swing_levels_advanced(
                symbol,
                df=stop_tf,
                lookback=config.get("SWING_LOOKBACK", 30),
                left=config.get("SWING_LEFT", 4),
                right=config.get("SWING_RIGHT", 4),
                volume_mult=config.get("SWING_VOLUME_MULT", 1.2),
                min_atr_ratio=config.get("SWING_MIN_ATR_RATIO", 0.001),
                min_price_dist=config.get("SWING_MIN_PRICE_DIST", 0.004)
            )

            entry_mid = (entry_zone_min + entry_zone_max) / 2

            if signal == "long":
                # 1. Рассчитываем ATR-стоп (базовый)
                atr_stop = entry_mid - (atr * atr_mult)
                
                # 2. Минимальный стоп (3% или min ATR)
                min_stop = entry_mid * (1 - min_dist_pct)
                min_atr_stop = entry_mid - (atr * min_atr_mult)
                final_min_stop = max(min_stop, min_atr_stop)  # Выбираем больший
                
                # 3. Максимальный стоп (7% или max ATR)
                max_stop = entry_mid * (1 - max_dist_pct)
                max_atr_stop = entry_mid - (atr * max_atr_mult)
                final_max_stop = min(max_stop, max_atr_stop)  # Выбираем меньший
                
                # 4. Корректируем ATR-стоп под диапазон
                atr_stop = max(atr_stop, final_min_stop)  # Не меньше минимума
                atr_stop = min(atr_stop, final_max_stop)  # Не больше максимума
                
                # 5. Ищем свинги в диапазоне [final_min_stop, final_max_stop]
                valid_lows = [low for low in lows if final_min_stop <= low <= final_max_stop]
                
                if valid_lows:
                    # Берём ближайший свинг (самый высокий low)
                    swing_stop = max(valid_lows)
                    # Добавляем буфер ATR
                    final_stop = swing_stop - (atr * atr_buffer_mult)
                else:
                    # Если свингов нет – используем ATR-стоп
                    final_stop = atr_stop

            else:  # short
                # 1. Рассчитываем ATR-стоп (базовый)
                atr_stop = entry_mid + (atr * atr_mult)
                
                # 2. Минимальный стоп (3% или min ATR)
                min_stop = entry_mid * (1 + min_dist_pct)
                min_atr_stop = entry_mid + (atr * min_atr_mult)
                final_min_stop = min(min_stop, min_atr_stop)  # Выбираем меньший
                
                # 3. Максимальный стоп (7% или max ATR)
                max_stop = entry_mid * (1 + max_dist_pct)
                max_atr_stop = entry_mid + (atr * max_atr_mult)
                final_max_stop = max(max_stop, max_atr_stop)  # Выбираем больший
                
                # 4. Корректируем ATR-стоп под диапазон
                atr_stop = min(atr_stop, final_min_stop)  # Не меньше минимума
                atr_stop = max(atr_stop, final_max_stop)  # Не больше максимума
                
                # 5. Ищем свинги в диапазоне [final_min_stop, final_max_stop]
                valid_highs = [high for high in highs if final_min_stop <= high <= final_max_stop]
                
                if valid_highs:
                    # Берём ближайший свинг (самый низкий high)
                    swing_stop = min(valid_highs)
                    # Добавляем буфер ATR
                    final_stop = swing_stop + (atr * atr_buffer_mult)
                else:
                    # Если свингов нет – используем ATR-стоп
                    final_stop = atr_stop

            return round(final_stop, decimal_places)
            
        except Exception as e:
            print(f"[ERROR] В функции calculate_stop для {symbol}: {e}")
            import traceback
            traceback.print_exc()
            
            # Всегда возвращаем значение по умолчанию в случае ошибки
            atr_mult = config.get("STOP_ATR_MULTIPLIER", 1.5)
            if signal == "long":
                return round(entry_zone_min - atr * atr_mult, decimal_places)
            else:
                return round(entry_zone_max + atr * atr_mult, decimal_places)

    # === Функция для получения лучшей точки входа ===
    def get_entry_point(symbol, df, df_htf, side: str, config: dict) -> dict | None:
        try:
            # Явная проверка на пустые DataFrame
            if df is None or df.empty or df_htf is None or df_htf.empty:
                print(f"[WARNING] Пустой DataFrame для {symbol} в get_entry_point")
                return None
            """
            Возвращает оптимальную точку входа или None, если подходящих зон нет.
            """
            current_price = df["close"].iloc[-1]
            threshold = config.get("ENTRY_DISTANCE_THRESHOLD", 0.01)
            weights = config.get("ENTRY_CANDIDATE_WEIGHTS", {
                "fvg": 10,
                "sd": 9,
                "swing": 8,
                "accumulation": 7
            })

            candidates = []

            # === 1. FVG ===
            fvg_price = find_nearest_fvg(entry=current_price, signal=side, df=df_htf)
            if fvg_price:
                distance = abs(fvg_price - current_price) / current_price
                if distance <= threshold:
                    score = weights["fvg"] - distance * 100
                    print(f"[ENTRY][FVG] Принят: {fvg_price:.6f} (расстояние {distance*100:.2f}%)")
                    candidates.append({
                        "type": "fvg",
                        "price": fvg_price,
                        "reason": f"FVG в {distance*100:.2f}% от цены",
                        "score": score
                    })
                else:
                    print(f"[ENTRY][FVG] Отклонён: {fvg_price:.6f} (слишком далеко: {distance*100:.2f}%)")

            # === 2. SD ===
            sd_price = find_nearest_sd_zone(entry=current_price, signal=side, df=df_htf)
            if sd_price:
                distance = abs(sd_price - current_price) / current_price
                if distance <= threshold:
                    score = weights["sd"] - distance * 100
                    print(f"[ENTRY][SD] Принят: {sd_price:.6f} (расстояние {distance*100:.2f}%)")
                    candidates.append({
                        "type": "sd",
                        "price": sd_price,
                        "reason": f"S/D зона в {distance*100:.2f}% от цены",
                        "score": score
                    })
                else:
                    print(f"[ENTRY][SD] Отклонён: {sd_price:.6f} (слишком далеко: {distance*100:.2f}%)")

            # === 3. Swing ===
            # swing = get_swing_levels_advanced(df_htf, side, config)
            stop_tf = df_dict[config["STOP_TF"]]
            swing = get_swing_levels_advanced(
                symbol=symbol,
                df=stop_tf,
                lookback=config.get("SWING_LOOKBACK", 100),
                left=config.get("SWING_LEFT", 2),
                right=config.get("SWING_RIGHT", 2),
                volume_mult=config.get("SWING_VOLUME_MULT", 1.2),
                min_atr_ratio=config.get("SWING_MIN_ATR_RATIO", 0.001),
                min_price_dist=config.get("SWING_MIN_PRICE_DIST", 0.004)
            )

            if swing:
                swing_low, swing_high = swing[2], swing[3]
                if side == "long" and swing_low:
                    swing_price = swing_low
                elif side == "short" and swing_high:
                    swing_price = swing_high
                else:
                    swing_price = None

                if swing_price:
                    distance = abs(swing_price - current_price) / current_price
                    if distance <= threshold:
                        score = weights["swing"] - distance * 100
                        print(f"[ENTRY][SWING] Принят: {swing_price:.6f} (расстояние {distance*100:.2f}%)")
                        candidates.append({
                            "type": "swing",
                            "price": swing_price,
                            "reason": f"Swing уровень в {distance*100:.2f}% от цены",
                            "score": score
                        })
                    else:
                        print(f"[ENTRY][SWING] Отклонён: {swing_price:.6f} (слишком далеко: {distance*100:.2f}%)")

            # === 4. Accumulation ===
            acc = find_consolidation_zone(
                df_htf,
                window=config.get("CONSOLIDATION_WINDOW", 20),
                consolidation_thresh=config.get("CONSOLIDATION_THRESHOLD", 0.0025)
            )
            if acc and isinstance(acc, tuple) and len(acc) == 2:
                acc_low, acc_high = acc
                if acc_low <= current_price <= acc_high:
                    acc_price = (acc_low + acc_high) / 2
                    distance = abs(acc_price - current_price) / current_price
                    if distance <= threshold:
                        score = weights["accumulation"] - distance * 100
                        print(f"[ENTRY][ACC] Принят midpoint: {acc_price:.6f} (отклонение {distance*100:.2f}%)")
                        candidates.append({
                            "type": "accumulation",
                            "price": acc_price,
                            "reason": f"Цена внутри зоны накопления ({distance*100:.2f}% отклонение)",
                            "score": score
                        })
                    else:
                        print(f"[ENTRY][ACC] Отклонён: цена {current_price:.6f} вне зоны ({acc_low:.6f} – {acc_high:.6f})")

            # === Выбор лучшего кандидата ===
            if candidates:
                best = sorted(candidates, key=lambda x: x["score"], reverse=True)[0]
                print(f"[ENTRY] ✅ Выбран тип: {best['type'].upper()} | Цена: {best['price']:.6f} | Причина: {best['reason']}")
                return best

            print(f"[ENTRY] ❌ Fallback — зон нет или все отклонены. Вход по close: {current_price:.6f}")
            return None  # нет подходящих зон
        except Exception as e:
            print(f"[ERROR] В функции get_entry_point для {symbol}: {e}")
            traceback.print_exc()
            return None

    # --- Инициализация клиента Binance ---
    while True:
        try:
            # --- Оценка волатильности рынка ---
            atr_values = []
            price_values = []

            print("[VOLATILITY] Начало оценки волатильности рынка...")

            for base_symbol in ["BTCUSDT", "ETHUSDT"]:
                print(f"[VOLATILITY] Обработка символа: {base_symbol}")
                try:
                    klines = client.futures_klines(symbol=base_symbol, interval=config["MARKET_ANALYSIS_TF"][0], limit=50)
                    df = pd.DataFrame(klines, columns=[
                        "t", "open", "high", "low", "close", "volume", "close_time", "q", "n",
                        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
                    ])
                    print(f"[VOLATILITY] Получено {len(klines)} свечей для {base_symbol}")
                    df["close"] = df["close"].astype(float)
                    df["high"] = df["high"].astype(float)
                    df["low"] = df["low"].astype(float)
                    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
                    atr_val = atr.iloc[-1]
                    price_val = df["close"].iloc[-1]
                    atr_values.append(atr_val)
                    price_values.append(price_val)
                    
                    print(f"[VOLATILITY] {base_symbol} | ATR: {atr_val:.6f}, Цена: {price_val:.2f}")

                except Exception as e:
                    print(f"[ERROR] Не удалось получить ATR для {base_symbol}: {e}")

            avg_atr = sum(atr_values) / len(atr_values) if atr_values else 0
            avg_price = sum(price_values) / len(price_values) if price_values else 1
            atr_ratio = avg_atr / avg_price if avg_price else 0

            print(f"[VOLATILITY] Средний ATR: {avg_atr:.6f}, Средняя цена: {avg_price:.2f}, ATR/Цена: {atr_ratio:.6f}")

            now = time.time()

            # --- Сообщения о волатильности ---
            print(f"[VOLATILITY] Состояние: last_vol_msg_time={last_vol_msg_time}, last_vol_state={last_vol_state}, now={now}")

            if atr_ratio > 0.005:
                high_volatility = True
                low_volatility = False

                print("[VOLATILITY] Обнаружена высокая волатильность (atr_ratio > 0.005)")

                if now - last_vol_msg_time > 1800 or last_vol_state != "high":
                    print("[VOLATILITY] Отправка сообщения о турбулентности")
                    await send_message("⚠️ BTC/ETH в турбулентности. Возможны ложные сигналы. Ждем стабилизации.")
                    last_vol_msg_time = now
                    last_vol_state = "high"
                else:
                    print("[VOLATILITY] Сообщение о турбулентности НЕ отправлено (таймер или статус не изменились)")

            elif atr_ratio < 0.0005:
                low_volatility = True
                high_volatility = False

                print("[VOLATILITY] Обнаружен флэт (atr_ratio < 0.0005)")
                if now - last_vol_msg_time > 1800 or last_vol_state != "low":
                    print("[VOLATILITY] Отправка сообщения о флэте")
                    await send_message("ℹ️ BTC/ETH во флэте. Возможна слабая активность по альтам. Ждем волатильности.")
                    last_vol_msg_time = now
                    last_vol_state = "low"
                else:
                    print("[VOLATILITY] Сообщение о флэте НЕ отправлено (таймер или статус не изменились)")

            else:
                print("[VOLATILITY] Волатильность в пределах нормы")
                high_volatility = False
                low_volatility = False
                last_vol_state = "normal"

            for symbol in config["MARKET_ANALYSIS_SYMBOLS"]:
                try:
                    trade = active_trades.get(symbol)
                    last_price = None

                    # Загружаем все данные одним вызовом
                    df_dict = load_df(
                        symbol=symbol,
                        timeframes=config["REQUIRED_TFS"],
                        client=client,
                        limit=100
                    )
                    
                    # Проверяем наличие основных таймфреймов
                    market_tfs = config["MARKET_ANALYSIS_TF"]
                    if not all(tf in df_dict for tf in market_tfs):
                        continue
                        
                    # Используем данные из df_dict
                    df = df_dict[market_tfs[0]]
                    df_htf = df_dict[market_tfs[1]]

                    # Получаем последнюю цену
                    last_price = last_prices.get(symbol)
                    if last_price is None:
                        try:
                            ticker = client.futures_symbol_ticker(symbol=symbol)
                            last_price = float(ticker["price"])
                            last_prices[symbol] = last_price
                        except Exception as e:
                            print(f"[ERROR] Не удалось получить цену {symbol}: {e}")

                    link = f"https://www.tradingview.com/chart/?symbol=BINANCE%3A{symbol}.P"

                    if trade and last_price:
                        # --- Логика активации и закрытия сделки ---
                        if not trade.get("activated", False):
                            klines = client.futures_klines(symbol=symbol, interval=config["MARKET_ANALYSIS_TF"][0], limit=2)
                            last_kline = klines[-1]
                            last_high = float(last_kline[2])
                            last_low = float(last_kline[3])

                            # Проверяем попадание в диапазон входа
                            if trade["side"] == "long":
                                entry_reached = last_low <= trade["entry_zone_max"] and last_high >= trade["entry_zone_min"]
                            else:
                                entry_reached = last_high >= trade["entry_zone_min"] and last_low <= trade["entry_zone_max"]

                            if entry_reached:
                                print(f"[ACTIVATION] {symbol}: вход в диапазон ({trade['entry_zone_min']} - {trade['entry_zone_max']}), цена {last_price}")

                                trade["activated"] = True
                                daily_stats["trades_opened"] += 1
                                remove_new_tag(symbol, trade) # Удаляем тег "новый" при активации


                                # Корректируем цену входа, чтобы она всегда была в диапазоне
                                if trade["side"] == "short":
                                    activation_price = min(max(last_price, trade["entry_zone_min"]), trade["entry_zone_max"])
                                else:
                                    activation_price = max(min(last_price, trade["entry_zone_max"]), trade["entry_zone_min"])

                                trade["entry_real"] = activation_price
                                stop = trade["stop"]

                                try:
                                    await db_stats.update_signal_result(
                                        signal_id=trade["trade_id"],
                                        signal_activated=1,
                                        entry_price=trade["entry_real"]
                                    )
                                except Exception as e:
                                    print(f"[DB ERROR] update on activation failed for {trade['trade_id']}: {e}")

                                open_msg = (
                                    f"🚀 <a href='{link}'>{symbol}</a> | ID: <code>{trade['trade_id']}</code>\n"
                                    f"📌 Сделка открыта по цене: <code>{activation_price}</code>\n"
                                    f"🛑 Стоп: <code>{stop}</code>\n"
                                    f"Время: {datetime.now(kyiv_tz).strftime('%H:%M:%S')}"
                                )
                                open_message_id = await send_message(open_msg, reply_to_message_id=trade["message_id"])
                                trade["open_message_id"] = open_message_id
                            elif time.time() - trade["opened_at"] > 1800:
                                daily_stats["expired"] += 1
                                remove_new_tag(symbol, trade) # Удаляем тег "новый" при активации
                                result_msg = (
                                    f"🕒 <a href='{link}'>{symbol}</a> | ID: <code>{trade['trade_id']}</code>\n"
                                    f"Вход не выполнен — сигнал снят по таймауту (30 мин)."
                                )
                                await send_message(result_msg, reply_to_message_id=trade["message_id"])
                                active_trades.pop(symbol, None)

                                try:
                                    await db_stats.update_signal_result(
                                        signal_id=trade["trade_id"],
                                        signal_activated=0
                                    )
                                except Exception as e:
                                    print(f"[DB ERROR] update on expiry failed for {trade['trade_id']}: {e}")

                                retracement_alerts_sent.pop(symbol, None)
                                continue
                        elif trade.get("activated", False):
                            minutes_in_trade = int((time.time() - trade["opened_at"]) // 60)
                            result_msg = None

                            # Получаем последнюю свечу (1 свечу с нужным таймфреймом)
                            klines = client.futures_klines(symbol=symbol, interval=config["MARKET_ANALYSIS_TF"][0], limit=1)
                            last_kline = klines[-1]
                            candle_high = float(last_kline[2])
                            candle_low = float(last_kline[3])

                            print(f"[CHECK STOP] {symbol} | side: {trade['side']} | last_price: {last_price} | stop: {trade['stop']} | candle_high: {candle_high} | candle_low: {candle_low}")

                            # ==========================
                            # Вспомогательные функции
                            # ==========================
                            def price_reached(side, price, target):
                                """Проверка достижения цели в зависимости от направления сделки"""
                                return price >= target if side == "long" else price <= target

                            def back_to_entry(side, price, entry):
                                """Проверка возврата к цене входа (безубыток)"""
                                return price <= entry if side == "long" else price >= entry

                            def stop_triggered(side, candle_low, candle_high, stop):
                                """Проверка срабатывания стопа"""
                                if side == "long":
                                    return candle_low <= stop
                                else:
                                    return candle_high >= stop


                            # ==========================
                            # Сообщения по тейкам
                            # ==========================
                            TP_MESSAGES = {
                                1: {
                                    "hit": "🎯 Первая цель достигнута: <code>{price}</code> (частичный тейк 20%, стоп на твх)\n"
                                        "🚀 Отлично! Вперёд к следующей цели: <code>{next}</code>",
                                    "breakeven": "ℹ️ Сделка закрыта в <b>безубыток</b> после TP1\n"
                                                "💪 TP1 выполнена, TP2 не реализована — но ты уже на коне!"
                                },
                                2: {
                                    "hit": "🎯 Вторая цель достигнута: <code>{price}</code> (частичный тейк 20%)\n"
                                        "🔥 Продолжаем! Следующая цель: <code>{next}</code>",
                                    "breakeven": "ℹ️ Сделка закрыта в <b>безубыток</b> после TP2\n"
                                                "💪 TP2 достигнута, TP3 не реализована — держим планку!"
                                },
                                3: {
                                    "hit": "🎯 Третья цель достигнута: <code>{price}</code> (частичный тейк 20%)\n"
                                        "✨ На пути к финалу: <code>{next}</code>",
                                    "breakeven": "ℹ️ Сделка закрыта в <b>безубыток</b> после TP3\n"
                                                "💪 TP3 выполнена, TP4 не реализована — почти у цели!"
                                },
                                4: {
                                    "hit": "🎯 Четвёртая цель достигнута: <code>{price}</code> (частичный тейк 20%)\n"
                                        "🏁 Финальная цель впереди: <code>{next}</code>",
                                    "breakeven": "ℹ️ Сделка закрыта в <b>безубыток</b> после TP4\n"
                                                "💪 TP4 выполнена, TP5 не реализована — ещё чуть-чуть!"
                                },
                                5: {
                                    "final": (
                                        "🎯 Пятая цель достигнута: <code>{price}</code>\n"
                                        "💰 Сделка закрыта полностью — ты теперь официально на пути к финансовой свободе! 🏆\n"
                                        "🚀 Молодец, продолжай в том же духе! \n"
                                    )
                                }
                            }


                            # ==========================
                            # Основная логика
                            # ==========================
                            side = trade["side"]
                            result_msg = None
                            reply_id = trade.get("open_message_id") or trade["message_id"]

                            # ---- Цели TP1–TP5 ----
                            for i in range(1, 6):
                                prev_key = f"tp{i-1}_reached" if i > 1 else None
                                curr_key = f"tp{i}_reached"

                                # --- Проверка достижения цели ---
                                if (i == 1 and not trade.get("partial_taken") and price_reached(side, last_price, trade[f"target{i}"])) \
                                or (i > 1 and trade.get(prev_key) and not trade.get(curr_key) and price_reached(side, last_price, trade[f"target{i}"])):

                                    trade[curr_key] = True
                                    daily_stats[f"tp{i}_hit"] += 1

                                    # --- синхронизируем in-memory флаги ---
                                    trade[f"take{i}_hit"] = 1
                                    prev_target = trade.get("target", 0)
                                    write_target_to_db = False
                                    if prev_target == 0:
                                        trade["target"] = 1
                                        write_target_to_db = True

                                    # --- обновление БД (пишем target только если это первый тейк) ---
                                    try:
                                        kwargs = {f"take{i}_hit": 1}
                                        if write_target_to_db:
                                            kwargs["target"] = 1
                                        await db_stats.update_signal_result(signal_id=trade["trade_id"], **kwargs)
                                    except Exception as e:
                                        print(f"[DB ERROR] update TP{i} for {trade['trade_id']} failed: {e}")

                                    update_profit_loss_from_trade(symbol, trade, "win", target_idx=i)

                                    if i == 1:
                                        trade["partial_taken"] = True

                                    if i < 5:  # промежуточные цели
                                        await send_message(
                                            f"✅ <a href='{link}'>{symbol}</a> | ID: <code>{trade['trade_id']}</code>\n"
                                            + TP_MESSAGES[i]["hit"].format(price=last_price, next=trade[f"target{i+1}"]),
                                            reply_to_message_id=reply_id
                                        )
                                    else:  # финальная цель TP5
                                        result_msg = (
                                            f"✅ <a href='{link}'>{symbol}</a> | ID: <code>{trade['trade_id']}</code>\n"
                                            + TP_MESSAGES[i]["final"].format(price=last_price)
                                            + f"\n⏱️ Время в сделке: {minutes_in_trade} мин"
                                        )

                                    break

                            # ---- Проверка возврата к entry (безубыток) ----
                            entry_price = trade.get("entry_real", trade["entry"])
                            for i in range(1, 5):  # до TP4 включительно
                                curr_key = f"tp{i}_reached"
                                next_key = f"tp{i+1}_reached"
                                if trade.get(curr_key) and not trade.get(next_key) and back_to_entry(side, last_price, entry_price):
                                    result_msg = (
                                        f"ℹ️ <a href='{link}'>{symbol}</a> | ID: <code>{trade['trade_id']}</code>\n"
                                        + TP_MESSAGES[i]["breakeven"]
                                        + f"\n⏱️ Время в сделке: {minutes_in_trade} мин"
                                    )
                                    daily_stats[f"closed_breakeven_after_tp{i}"] += 1
                                    active_trades.pop(symbol, None)
                                    retracement_alerts_sent.pop(symbol, None)
                                    break

                            # ---- Проверка стопа ----
                            if stop_triggered(side, candle_low, candle_high, trade["stop"]):
                                daily_stats["stopped_out"] += 1
                                update_profit_loss_from_trade(symbol, trade, "loss", target_idx=None)

                                recently_stopped[symbol] = time.time()
                                stop_price = candle_low if side == "long" else candle_high
                                
                                try:
                                    # проверяем, был ли взят хотя бы один тейк
                                    took_any_take = any([
                                        trade.get("take1_hit"),
                                        trade.get("take2_hit"),
                                        trade.get("take3_hit"),
                                        trade.get("take4_hit"),
                                        trade.get("take5_hit"),
                                    ])

                                    final_target = 1 if took_any_take else 0

                                    await db_stats.update_signal_result(
                                        signal_id=trade["trade_id"],
                                        stop_loss_hit=1,
                                        target=final_target
                                    )
                                except Exception as e:
                                    print(f"[DB ERROR] update stop for {trade['trade_id']} failed: {e}")
                                
                                result_msg = (
                                    f"🛑 <a href='{link}'>{symbol}</a> | ID: <code>{trade['trade_id']}</code>\n"
                                    f"Стоп достигнут ({stop_price})\n"
                                    f"Время в сделке: {minutes_in_trade} мин"
                                )

                            # ---- Финальное сообщение (если есть) ----
                            if result_msg:
                                await send_message(result_msg, reply_to_message_id=reply_id)
                                active_trades.pop(symbol, None)
                                retracement_alerts_sent.pop(symbol, None)
                                continue

                    # --- Если сделки нет — ищем новую, только если нет турбулентности/флэта ---
                    if symbol not in active_trades:
                        cooldown = config.get("COOLDOWN_AFTER_STOP", 900)
                        if symbol in recently_stopped and time.time() - recently_stopped[symbol] < cooldown:
                            print(f"[SKIP] {symbol}: недавно вышла по стопу, ждём паузу")
                            continue

                        # Используем данные из df_dict
                        df = df_dict[config["MARKET_ANALYSIS_TF"][0]]
                        df_htf = df_dict[config["MARKET_ANALYSIS_TF"][1]]

                        
                        # signal = check_signal(df, df_htf)

                        methods = {
                            "ZLEMA": lambda: get_trend(symbol, df_dict, length=20),
                            "TRIX":  lambda: get_trend_trix(symbol, df_dict, config),
                            "TREND":  lambda: get_trend_supertrend(symbol, df_dict, config),
                        }

                        method = config.get("USE_TREND_METHOD", "ZLEMA")
                        trend_global = methods.get(method, lambda: None)()

                        signal = await check_signal(symbol, df, df_htf, trend_global)
                        if not signal:
                            continue

                        # Получите filters_results и ml_out из результата сигнала
                        filters_results = signal.get("filters_results", {})
                        ml_out = signal.get("ml_out", {})
                        
                        ind = calculate_indicators(df)

                        # last_close = df["close"].iloc[-1]
                        # last_close = ind["last_close"]
                        last_close = get_live_price(symbol, ind["last_close"])
                        if last_close is None:
                            last_close = ind["last_close"]

                        decimal_places = get_decimal_places(last_close)
                        side = signal["side"]
                        trend = "🟢 Лонг" if side == "long" else "🔻 Шорт"
    

                        entry_zone_k = config.get("ENTRY_ZONE_WIDTH_FRACTAL", 0.0007)
                        atr = ind["atr"]

                        entry_data = get_entry_point(symbol, df, df_htf, side, config)
                        if entry_data:
                            entry = round(entry_data["price"], decimal_places)
                            entry_type = entry_data["type"]
                        else:
                            entry = round(float(last_close), decimal_places)
                            entry_type = "fallback"


                        # --- Выбор функции расчёта тейков по TP_MODE ---
                        entry_zone_min = round(entry - entry * entry_zone_k, decimal_places)
                        entry_zone_max = round(entry + entry * entry_zone_k, decimal_places)

                        stop = calculate_stop(
                            entry_zone_min, 
                            entry_zone_max, 
                            atr, 
                            side, 
                            symbol, 
                            decimal_places, 
                            config=config,
                            df_dict=df_dict
                        )

                        # Получаем список всех тейков
                        targets = calculate_targets(
                            entry_zone_min,
                            entry_zone_max,
                            atr,
                            side,
                            symbol,
                            decimal_places,
                            config=config,
                            df_dict=df_dict,
                            verbose=False
                        )

                        # Распаковываем первые 3 тейка в именованные переменные
                        (target1, tp1_type), (target2, tp2_type), (target3, tp3_type) = targets[:3]

                        # trade_id = hashlib.md5(f"{symbol}{entry}{target1}{target2}{stop}{time.time()}".encode()).hexdigest()[:8]
                        trade_id = signal["signal_id"]
                        signal_time = datetime.now(kyiv_tz).strftime('%H:%M:%S')

                        # --- Получаем информацию о индикаторах ---
                        ind = calculate_indicators(df)
                        ind_htf = calculate_indicators(df_htf)

                        # --- Формируем сообщение о сигнале ---
                        icons = {
                            "long": "🟢",
                            "short": "🔴"
                        }

                        # Получаем иконки для тренда и сигнала
                        trend_result = icons.get(str(trend_global), "❌")
                        signal_result = icons.get(str(side), "❌")


                        # --- Текст дивергенций ---
                        divergence_text = ""
                        if config.get("USE_DIVERGENCE_TAG", True):
                            # Получаем все дивергенции одним вызовом функции
                            # divs = detect_divergence_multi_tf(symbol, df_dict, config)
                            
                            # Списки для каждого типа дивергенции и индикатора
                            rsi_bullish_tfs = []
                            rsi_bearish_tfs = []
                            rsi_hidden_bullish_tfs = []
                            rsi_hidden_bearish_tfs = []

                            macd_bullish_tfs = []
                            macd_bearish_tfs = []
                            macd_hidden_bullish_tfs = []
                            macd_hidden_bearish_tfs = []

                            # Проверяем каждый таймфрейм из конфига
                            for tf in config.get("DIVERGENCE_TFS", ["1h", "4h"]):
                                # Создаем временный конфиг только с одним таймфреймом
                                temp_config = config.copy()
                                temp_config["DIVERGENCE_TFS"] = [tf]
                                
                                # Получаем дивергенции для конкретного таймфрейма
                                tf_divs = detect_divergence_multi_tf(symbol, df_dict, temp_config)

                                # Заполняем списки для RSI
                                if tf_divs["rsi"]["bullish"]:
                                    rsi_bullish_tfs.append(tf)
                                if tf_divs["rsi"]["bearish"]:
                                    rsi_bearish_tfs.append(tf)
                                if tf_divs["rsi"]["hidden_bullish"]:
                                    rsi_hidden_bullish_tfs.append(tf)
                                if tf_divs["rsi"]["hidden_bearish"]:
                                    rsi_hidden_bearish_tfs.append(tf)

                                # Заполняем списки для MACD
                                if tf_divs["macd"]["bullish"]:
                                    macd_bullish_tfs.append(tf)
                                if tf_divs["macd"]["bearish"]:
                                    macd_bearish_tfs.append(tf)
                                if tf_divs["macd"]["hidden_bullish"]:
                                    macd_hidden_bullish_tfs.append(tf)
                                if tf_divs["macd"]["hidden_bearish"]:
                                    macd_hidden_bearish_tfs.append(tf)

                            # Формируем текст с дивергенциями
                            if (rsi_bullish_tfs or rsi_bearish_tfs or rsi_hidden_bullish_tfs or rsi_hidden_bearish_tfs or
                                macd_bullish_tfs or macd_bearish_tfs or macd_hidden_bullish_tfs or macd_hidden_bearish_tfs):
                                
                                divergence_text = "\n\n<b>Divers:</b>\n"
                                
                                if rsi_bullish_tfs:
                                    divergence_text += f"RSI бычья ({', '.join(rsi_bullish_tfs)})\n"
                                if rsi_bearish_tfs:
                                    divergence_text += f"RSI медвежья ({', '.join(rsi_bearish_tfs)})\n"
                                if rsi_hidden_bullish_tfs:
                                    divergence_text += f"RSI скрытая бычья ({', '.join(rsi_hidden_bullish_tfs)})\n"
                                if rsi_hidden_bearish_tfs:
                                    divergence_text += f"RSI скрытая медвежья ({', '.join(rsi_hidden_bearish_tfs)})\n"
                                    
                                if macd_bullish_tfs:
                                    divergence_text += f"MACD бычья ({', '.join(macd_bullish_tfs)})\n"
                                if macd_bearish_tfs:
                                    divergence_text += f"MACD медвежья ({', '.join(macd_bearish_tfs)})\n"
                                if macd_hidden_bullish_tfs:
                                    divergence_text += f"MACD скрытая бычья ({', '.join(macd_hidden_bullish_tfs)})\n"
                                if macd_hidden_bearish_tfs:
                                    divergence_text += f"MACD скрытая медвежья ({', '.join(macd_hidden_bearish_tfs)})\n"
                                
                                divergence_text = divergence_text.strip()


                        # Вычисляем стрелки для объема
                        def get_volume_arrow(current_vol, mean_vol):
                            if current_vol > mean_vol:
                                return "↑"  # Стрелка вверх
                            elif current_vol < mean_vol:
                                return "↓"  # Стрелка вниз
                            return "→"     # Если равны

                        # Вычисляем стрелки для TRIX
                        def get_trix_arrow(current_trix, prev_trix):
                            if prev_trix is None:
                                return ""  # Нет предыдущего значения
                            if current_trix > prev_trix:
                                return "↑"  # Бычий тренд
                            elif current_trix < prev_trix:
                                return "↓"  # Медвежий тренд
                            return "→"     # Без изменений

                        # Формируем блок индикаторов с новыми стрелками
                        st_arrow_ltf = st_arrow_htf = ""
                        if config.get("USE_SUPERTREND_FILTER", False): 
                            atr_period = config.get("SUPERTREND_ATR_PERIOD", 10)
                            multiplier = config.get("SUPERTREND_MULTIPLIER", 3)

                            def get_supertrend_arrow(current_df, prev_df=None, atr_period=atr_period, multiplier=multiplier):
                                """
                                Возвращает стрелку направления SuperTrend:
                                ↑ если тренд вверх, ↓ если вниз, → если нет изменений
                                prev_df можно передать для сравнения с предыдущим состоянием (опционально)
                                """
                                is_up = is_supertrend_up(current_df, atr_period, multiplier)

                                if prev_df is not None:
                                    prev_up = is_supertrend_up(prev_df, atr_period, multiplier)
                                    if is_up and not prev_up:
                                        return "↑"
                                    elif not is_up and prev_up:
                                        return "↓"
                                    else:
                                        return "→"
                                else:
                                    return "↑" if is_up else "↓"
                            st_arrow_ltf = get_supertrend_arrow(df)
                            st_arrow_htf = get_supertrend_arrow(df_htf)

                        # Словарь фильтров: ключ = имя фильтра в CONFIG, value = формат строки
                        filters_info = ""
                        if config.get("SHOW_FILTERS_DETAILS", True):

                            def build_combined_filters_info(ind_ltf, ind_htf, symbol, side, tf_ltf, tf_htf):
                                """Объединяет фильтры двух ТФ в единый табличный вид (30m | 2h)."""
                                lines = [f"\n<b>📊 Фильтры ({tf_ltf} | {tf_htf}):</b>"]

                                # === базовые фильтры ===
                                def safe(f, d, default="—"):
                                    try:
                                        return f(d)
                                    except Exception:
                                        return default

                                if config.get("USE_ATR_FILTER", False):
                                    lines.append(f"ATR: <code>{ind_ltf['atr']:.6f}</code> | <code>{ind_htf['atr']:.6f}</code>")

                                if config.get("USE_VOLUME_FILTER", False):
                                    arrow_l = get_volume_arrow(ind_ltf["volume"], ind_ltf["vol_mean"])
                                    arrow_h = get_volume_arrow(ind_htf["volume"], ind_htf["vol_mean"])
                                    lines.append(
                                        f"Объём: <code>{ind_ltf['volume']:.2f}</code>{arrow_l} | "
                                        f"<code>{ind_htf['volume']:.2f}</code>{arrow_h}"
                                    )

                                if config.get("USE_TRIX_FILTER", False):
                                    arr_l = get_trix_arrow(ind_ltf["trix"], ind_ltf.get("prev_trix"))
                                    arr_h = get_trix_arrow(ind_htf["trix"], ind_htf.get("prev_trix"))
                                    lines.append(f"TRIX: <code>{ind_ltf['trix']:.6f}</code>{arr_l} | <code>{ind_htf['trix']:.6f}</code>{arr_h}")

                                if config.get("USE_SUPERTREND_FILTER", False):
                                    lines.append( f"ST: <code>{st_arrow_ltf}</code> / <code>{st_arrow_htf}</code>")

                                if config.get("USE_ADX_FILTER", False):
                                    lines.append(f"ADX: <code>{ind_ltf['adx']:.2f}</code> | <code>{ind_htf['adx']:.2f}</code>")

                                if config.get("USE_MACD_FILTER", False):
                                    lines.append(
                                        f"MACD: <code>{ind_ltf['macd']:.6f}</code> / {ind_ltf['macd_signal']:.6f} | "
                                        f"<code>{ind_htf['macd']:.6f}</code> / {ind_htf['macd_signal']:.6f}"
                                    )

                                if config.get("USE_STOCH_FILTER", False):
                                    lines.append(f"Stoch: <code>{ind_ltf['stoch']:.2f}</code> | <code>{ind_htf['stoch']:.2f}</code>")

                                if config.get("USE_RSI_FILTER", False):
                                    lines.append(f"RSI: <code>{ind_ltf['rsi']:.2f}</code> | <code>{ind_htf['rsi']:.2f}</code>")

                                if config.get("USE_EMA_FILTER", False):
                                    lines.append(
                                        f"EMA: <code>{ind_ltf['ema_fast']:.4f}</code> / {ind_ltf['ema_slow']:.4f} | "
                                        f"<code>{ind_htf['ema_fast']:.4f}</code> / {ind_htf['ema_slow']:.4f}"
                                    )

                                # === CDV ===
                                if config.get("USE_CDV_FILTER", False):
                                    ratio_l, vol_l = get_cdv_ratio(symbol, config['MARKET_ANALYSIS_TF'][0])
                                    ratio_h, vol_h = get_cdv_ratio(symbol, config['MARKET_ANALYSIS_TF'][1])

                                    def cdv_icon(r, side):
                                        if side == "long":
                                            return "🟢 Buy" if r >= CONFIG["CDV_MIN_THRESHOLD"] else "🟡 Sell"
                                        else:
                                            return "🔴 Sell" if r <= -CONFIG["CDV_MIN_THRESHOLD"] else "🟡 Buy"

                                    if ratio_l is not None and ratio_h is not None:
                                        lines.append(
                                            f"CDV: {cdv_icon(ratio_l, side)} {ratio_l:+.2%} | "
                                            f"{cdv_icon(ratio_h, side)} {ratio_h:+.2%}"
                                        )

                                return "\n".join(lines)

                            # === Выводим объединённый блок ===
                            filters_info = build_combined_filters_info(ind, ind_htf, symbol, side, config['MARKET_ANALYSIS_TF'][0], config['MARKET_ANALYSIS_TF'][1])
                            filters_info += f"\n{divergence_text}\n\n<b>Глобальный тренд:</b> <code>{trend_result}</code>\n<b>Результат фильтров:</b> <code>{signal_result}</code>\n"

                        # Формируем основную часть сообщения
                        msg = (
                            f"🔔 #new\n"
                            f"{trend} ({config['MARKET_ANALYSIS_TF'][0]})\n"
                            f"📊 <a href='{link}'>{symbol}</a> | <code>{symbol}</code>\n"
                            f"ID сделки: <code>{trade_id}</code>\n"
                            f"📍 Цена: <code>{entry:.{decimal_places}f}</code> ({entry_type})\n"
                            f"🎯 Вход: <code>{entry_zone_min}</code> – <code>{entry_zone_max}</code>\n"
                            f"🏁 Цель 1 ({tp1_type}): <code>{target1}</code>\n"
                            f"🏁 Цель 2 ({tp2_type}): <code>{target2}</code>\n"
                            f"🏁 Цель 3 ({tp3_type}): <code>{target3}</code>\n"
                        )

                        # Добавляем дополнительные тейки (4 и 5) если они есть
                        if len(targets) > 3:
                            # Получаем четвертый тейк
                            target4, tp4_type = targets[3]
                            msg += f"🏁 Цель 4 ({tp4_type}): <code>{target4}</code>\n"
                            
                        if len(targets) > 4:
                            # Получаем пятый тейк
                            target5, tp5_type = targets[4]
                            msg += f"🏁 Цель 5 ({tp5_type}): <code>{target5}</code>\n"

                        ml_text = ""
                        if ML_CONFIG.get("enabled", True):
                            try:
                                ml_text, ema_wait = generate_signal_text(filters_results, ml_out, targets, stop, entry_zone_min, entry_zone_max)

                                if ema_wait:
                                    ema_fast = ind["ema_fast"]
                                    ema_slow = ind["ema_slow"]
                                    atr = ind["atr"]
                                    
                                    if ema_fast and atr:
                                        trend_strength = abs(ema_fast - ema_slow) / ema_slow if ema_slow else 0
                                        k = 0.5 if trend_strength >= 0.002 else 0.25
                                        
                                        entry_zone_min = round(ema_fast - atr * k, decimal_places)
                                        entry_zone_max = round(ema_fast + atr * k, decimal_places)

                                    stop = calculate_stop(
                                        entry_zone_min, 
                                        entry_zone_max, 
                                        atr, 
                                        side, 
                                        symbol, 
                                        decimal_places, 
                                        config=config,
                                        df_dict=df_dict
                                    )

                            except Exception as e:
                                print(f"[ERROR] ML text generation failed: {e}")
                                ml_text = "🤖 ML: прогноз недоступен"

                        # Завершаем сообщение
                        msg += (
                            f"🛑 Стоп: <code>{stop}</code>\n"
                            f"⏱️ Время сигнала: <b>{signal_time}</b>\n"
                            f"{filters_info}"
                            f"\n"
                        )

                        if ML_CONFIG.get("enabled", True):
                            msg += f"🤖 ML Прогноз:\n{ml_text}"


                        try:
                            update_kwargs = {
                                "signal_id": trade_id,
                                "entry_zone_min": entry_zone_min, 
                                "entry_zone_max": entry_zone_max,
                                "take1_profit": target1,
                                "take2_profit": target2,
                                "take3_profit": target3,
                                "stop_loss": stop,
                            }

                            if len(targets) > 3:  # если есть TP4
                                update_kwargs["take4_profit"] = targets[3][0]

                            if len(targets) > 4:  # если есть TP5
                                update_kwargs["take5_profit"] = targets[4][0]

                            await db_stats.update_signal_result(**update_kwargs)

                        except Exception as e:
                            print(f"[DB ERROR] Не удалось записать уровни для сигнала {trade_id}: {e}")


                        message_id = await send_message(msg)
                        daily_stats["signals_sent"] += 1

                        # Создаем базовую структуру сделки
                        trade_data = {
                            "trade_id": trade_id,
                            "symbol": symbol,
                            "side": side,
                            "entry": entry,
                            "entry_real": None,
                            "entry_zone_min": entry_zone_min,
                            "entry_zone_max": entry_zone_max,
                            "target1": target1,
                            "target2": target2,
                            "target3": target3,
                            "stop": stop,
                            "status": "open",
                            "opened_at": time.time(),
                            "message_id": message_id,
                            "activated": False,
                            "open_message_id": None,
                            "partial_taken": False,
                            "tp2_reached": False,
                            "tp3_reached": False,
                            "tp4_reached": False,
                            "tp5_reached": False,
                            "original_message_text": msg,
                            "entry_type": entry_type
                        }

                        # Добавляем 4-й и 5-й тейки если они существуют
                        if len(targets) > 3:
                            trade_data["target4"] = targets[3][0]  # Цена 4-го тейка
                            #trade_data["tp4_type"] = targets[3][1]  # Тип 4-го тейка

                        if len(targets) > 4:
                            trade_data["target5"] = targets[4][0]  # Цена 5-го тейка
                            #trade_data["tp5_type"] = targets[4][1]  # Тип 5-го тейка

                        # Сохраняем сделку
                        active_trades[symbol] = trade_data
                        all_symbols_ever.add(symbol)


                    print(f"[DEBUG] Данные для сохранения: {active_trades[symbol]}")
                    await asyncio.sleep(config.get("PER_SYMBOL_DELAY", 0.5))

                except Exception as e:
                    print(f"[ERROR] Обработка символа {symbol} прервана: {e}")
                    traceback.print_exc()
                    continue  # Переходим к следующему символу

        except Exception as e:
            print(f"[ERROR][MARKET_ANALYSIS] {e}")

        await asyncio.sleep(config["MARKET_ANALYSIS_SEND"])

# === Статистика по сделкам ===
async def daily_report_loop():
    global pinned_stats_message_id

    # Первичная отправка и закрепление
    try:
        msg_id = bot.send_message(
            TG_CHAT_ID, get_stats_message(),
            parse_mode='HTML',
            disable_web_page_preview=True
        ).message_id
        bot.pin_chat_message(TG_CHAT_ID, msg_id)
        pinned_stats_message_id = msg_id
    except Exception as e:
        print(f"[ERROR] Не удалось отправить/закрепить статистику: {e}")

    while True:
        try:
            now = datetime.now(kyiv_tz)

            # === Сброс статистики в 00:00
            if now.hour == 0 and now.minute < 2:
                for key in daily_stats:
                    daily_stats[key] = 0

            # === Обновление закреплённого сообщения
            if pinned_stats_message_id:
                try:
                    bot.edit_message_text(
                        chat_id=TG_CHAT_ID,
                        message_id=pinned_stats_message_id,
                        text=get_stats_message(),
                        parse_mode='HTML',
                        disable_web_page_preview=True
                    )
                except telebot.apihelper.ApiTelegramException as e:
                    if "message is not modified" not in str(e):
                        print(f"[ERROR] Обновление статистики: {e}")

        except Exception as e:
            print(f"[ERROR] Обновление статистики: {e}")

        await asyncio.sleep(3600)  # Обновлять каждый час

# === Глобальная переменная для времени запуска ===
BOT_START_TIME = datetime.now(kyiv_tz)

# === Функция для формирования сообщения со статистикой ===
def get_stats_message():
    now = datetime.now(kyiv_tz)
    start_date = BOT_START_TIME.strftime('%Y-%m-%d')
    current_date = now.strftime('%Y-%m-%d')
    
    if start_date == current_date:
        date_range = current_date
    else:
        date_range = f"{start_date} - {current_date}"

    unique_symbols_count = len(daily_stats["retracement_unique_symbols"])
    active_count = sum(1 for trade in active_trades.values() if trade.get("activated", False))

    total_alerts = (
        daily_stats['retracement_level1_alerts'] +
        daily_stats['retracement_level2_alerts']
    )

    return (
        f"📊 <b>Статистика сигналов</b> ({date_range})\n\n"
        f"📥 Сигналов отправлено: <b>{daily_stats['signals_sent']}</b>\n"
        f"✅ Сделок активировано: <b>{daily_stats['trades_opened']}</b>\n"
        f"📈 Активных сделок: <b>{active_count}</b>\n"
        f"🎯 TP1 достигнуто: <b>{daily_stats['tp1_hit']}</b>\n"
        f"🎯 TP2 достигнуто: <b>{daily_stats['tp2_hit']}</b>\n"
        f"🎯 TP3 достигнуто: <b>{daily_stats['tp3_hit']}</b>\n"
        f"🎯 TP4 достигнуто: <b>{daily_stats['tp4_hit']}</b>\n"
        f"🏆 TP5 достигнуто: <b>{daily_stats['tp5_hit']}</b>\n"
        f"🛑 Безубытков после TP1: <b>{daily_stats['closed_breakeven_after_tp1']}</b>\n"
        f"🛑 Безубытков после TP2: <b>{daily_stats['closed_breakeven_after_tp2']}</b>\n"
        f"🛑 Безубытков после TP3: <b>{daily_stats['closed_breakeven_after_tp3']}</b>\n"
        f"🛑 Безубытков после TP4: <b>{daily_stats['closed_breakeven_after_tp4']}</b>\n"
        f"🛑 Стопов: <b>{daily_stats['stopped_out']}</b>\n"
        f"⌛ Не отработали за 30 мин: <b>{daily_stats['expired']}</b>\n\n"
        f"🟡 <b>Риск-метрика (сигналы разворота)</b>\n"
        f"🔔 Предупреждений: <b>{total_alerts}</b>\n"
        f"📈 Уникальных символов: <b>{unique_symbols_count}</b>\n\n"

        f"🥈 Профит - 10х: <b>+{daily_stats['profit_10x']:.2f}%</b>\n"
        f"🥇 Профит - 20х: <b>+{daily_stats['profit_20x']:.2f}%</b>\n"
        f"💸 Убыток - 10х: <b>-{daily_stats['loss_10x']:.2f}%</b>\n"
        f"💸 Убыток - 20х: <b>-{daily_stats['loss_20x']:.2f}%</b>\n"
    )

# === Инициализация баз данных ===
async def on_startup():
    await db_stats.init()


# === Функции для отправки/редактирования/закрепления рыночной сводки ===
async def edit_message(msg, message_id):
    try:
        bot.edit_message_text(
            chat_id=TG_CHAT_ID,
            message_id=message_id,
            text=msg,
            parse_mode='HTML',
            disable_web_page_preview=True
        )
    except Exception as e:
        print(f"[ERROR] edit_message: {e}")

async def pin_message(message_id):
    try:
        bot.pin_chat_message(TG_CHAT_ID, message_id, disable_notification=True)
    except Exception as e:
        print(f"[ERROR] pin_message: {e}")

# === Глобальное состояние для закреплённой рыночной сводки ===
market_summary_state = {}

async def main():
    # await on_startup()

    # Запускаем обновление символов если включен динамический режим
    if CONFIG["USE_DYNAMIC_SYMBOLS"]:
        asyncio.create_task(update_symbols_loop())
    
    # Запускаем WebSocket-клиент
    asyncio.create_task(binance_websocket_client())

    if CONFIG.get("USE_CDV_FILTER", False):
        asyncio.create_task(cdv_websocket_client())


    # Создаем задачи, но не включаем мониторинг откатов, если он отключен
    tasks = [
        market_analysis_loop(send_message, client, CONFIG),
        daily_report_loop(),
        session_monitor_loop(send_message, edit_message, pin_message, CONFIG, client)
        
    ]
    
    # Добавляем мониторинг откатов только если он включен
    if CONFIG.get("RETRACEMENT_ALERTS_ENABLED", True):
        tasks.append(retracement_monitor_loop())
    
    await asyncio.gather(*tasks)

def start_polling():
    try:
        print("[BOT] Запущен Telegram polling (для /stats)")
        bot.infinity_polling()
    except Exception as e:
        print(f"[ERROR] polling: {e}")

if __name__ == "__main__":
    try:
        loop = asyncio.new_event_loop()            # ← ✅ создаёшь глобальный loop
        asyncio.set_event_loop(loop)               # ← ✅ регистрируешь его как текущий
        threading.Thread(target=start_polling, daemon=True).start()
        loop.run_until_complete(main())            # ← запускаешь async main в этом loop
    except Exception as e:
        print(f"[FATAL] Бот упал: {e}")