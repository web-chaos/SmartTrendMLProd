import aiosqlite
import asyncio
import os
from datetime import datetime
import csv
import io
from typing import Optional
import json 

DB_PATH = "ml_trading.db"
TABLE_NAME = "ml_trading_data"

db_stats_enabled = False

class DBStats:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    async def init(self):
        # Создаем таблицу с полной структурой
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT NOT NULL UNIQUE DEFAULT '',
                symbol TEXT NOT NULL DEFAULT '',
                timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                signal_ltf TEXT NOT NULL DEFAULT '',
                signal_htf TEXT NOT NULL DEFAULT '',
                trend_global TEXT DEFAULT '',

                oi_value REAL DEFAULT 0.0,
                oi_config_threshold REAL DEFAULT 0.0,

                atr_ltf_value REAL DEFAULT 0.0,
                atr_ltf_config REAL DEFAULT 0.0,
                atr_ltf_passed INTEGER DEFAULT 0,
                atr_ltf_score REAL DEFAULT 0.0,

                adx_mode_config TEXT NOT NULL DEFAULT '',
                adx_ltf_value REAL DEFAULT 0.0,
                adx_ltf_config REAL DEFAULT 0.0,
                adx_ltf_passed INTEGER DEFAULT 0,
                adx_ltf_score REAL DEFAULT 0.0,
                adx_htf_value REAL DEFAULT 0.0,
                adx_htf_config REAL DEFAULT 0.0,
                adx_htf_passed INTEGER DEFAULT 0,
                adx_htf_score REAL DEFAULT 0.0,

                supertrend_mode TEXT NOT NULL DEFAULT '',
                supertrend_ltf_value REAL DEFAULT 0.0,
                supertrend_ltf_config_atr REAL DEFAULT 0.0,
                supertrend_ltf_config_mult REAL DEFAULT 0.0,
                supertrend_ltf_passed INTEGER DEFAULT 0,
                supertrend_ltf_score REAL DEFAULT 0.0,
                supertrend_htf_value REAL DEFAULT 0.0,
                supertrend_htf_config_atr REAL DEFAULT 0.0,
                supertrend_htf_config_mult REAL DEFAULT 0.0,
                supertrend_htf_passed INTEGER DEFAULT 0,
                supertrend_htf_score REAL DEFAULT 0.0,

                cdv_ltf_value REAL DEFAULT 0.0,
                cdv_ltf_volume REAL DEFAULT 0.0,
                cdv_ltf_passed INTEGER DEFAULT 0,
                cdv_ltf_score REAL DEFAULT 0.0,
                cdv_ltf_conflict INTEGER DEFAULT 0,
                cdv_ltf_threshold_config REAL DEFAULT 0.0,
                cdv_ltf_normalized REAL DEFAULT 0.0,
                cdv_htf_value REAL DEFAULT 0.0,
                cdv_htf_volume REAL DEFAULT 0.0,
                cdv_htf_passed INTEGER DEFAULT 0,
                cdv_htf_score REAL DEFAULT 0.0,
                cdv_htf_conflict INTEGER DEFAULT 0,
                cdv_htf_threshold_config REAL DEFAULT 0.0,
                cdv_htf_normalized REAL DEFAULT 0.0,

                potential_value REAL DEFAULT 0.0,
                potential_passed INTEGER DEFAULT 0,
                potential_config REAL DEFAULT 0.0,

                market_structure_ltf_passed INTEGER DEFAULT 0,
                market_structure_ltf_score REAL DEFAULT 0.0,
                market_structure_htf_passed INTEGER DEFAULT 0,
                market_structure_htf_score REAL DEFAULT 0.0,
                market_structure_increment_config REAL DEFAULT 0.0,

                candle_pattern_passed INTEGER DEFAULT 0,
                candle_pattern_score REAL DEFAULT 0.0,
                candle_pattern_sd_bonus INTEGER DEFAULT 0,
                candle_pattern_sd_bonus_score INTEGER DEFAULT 0.0,
                candle_pattern_sd_bonus_passed INTEGER DEFAULT 0,
                candle_pattern_with_sd_bonus_config REAL DEFAULT 0.0,
                candle_pattern_base_score_config REAL DEFAULT 0.0,

                structural_passed INTEGER DEFAULT 0,
                structural_score REAL DEFAULT 0.0,
                structural_tolerance INTEGER DEFAULT 0,
                structural_mode_config TEXT NOT NULL DEFAULT '',

                macd_current_diff REAL DEFAULT 0.0,
                macd_htf_diff REAL DEFAULT 0.0,
                macd_prev_diff REAL DEFAULT 0.0,
                macd_prev_htf_diff REAL DEFAULT 0.0,
                macd_momentum REAL DEFAULT 0.0,
                macd_htf_momentum REAL DEFAULT 0.0,
                macd_momentum_threshold_config REAL DEFAULT 0.0,
                macd_depth_threshold_config REAL DEFAULT 0.0,
                macd_threshold_config REAL DEFAULT 0.0,
                macd_score_increment REAL DEFAULT 0.0,
                macd_early_bullish INTEGER DEFAULT 0,
                macd_early_bearish INTEGER DEFAULT 0,
                macd_long INTEGER DEFAULT 0,
                macd_short INTEGER DEFAULT 0,
                macd_score_long REAL DEFAULT 0.0,
                macd_score_short REAL DEFAULT 0.0,

                volume_value REAL DEFAULT 0.0,
                volume_delta REAL DEFAULT 0.0,
                volume_mean REAL DEFAULT 0.0,
                volume_mean_long REAL DEFAULT 0.0,
                volume_normalized_body REAL DEFAULT 0.0,
                volume_passed INTEGER DEFAULT 0,
                volume_score REAL DEFAULT 0.0,
                volume_min_abs_config REAL DEFAULT 0.0,
                volume_min_abs_dynamic_config REAL DEFAULT 0.0,
                volume_tolerance REAL DEFAULT 0.0,
                volume_multiplier_config REAL DEFAULT 0.0,
                volume_min_relaxed_config REAL DEFAULT 0.0,
                volume_noise_vol_ratio_config REAL DEFAULT 0.0,
                volume_noise_body_ratio_config REAL DEFAULT 0.0,
                volume_delta_bonus_config REAL DEFAULT 0.0,

                trix_ltf REAL DEFAULT 0.0,
                trix_htf REAL DEFAULT 0.0,
                prev_trix_ltf REAL DEFAULT 0.0,
                prev_trix_htf REAL DEFAULT 0.0,
                trix_momentum_ltf REAL DEFAULT 0.0,
                trix_momentum_htf REAL DEFAULT 0.0,
                early_trix_bullish INTEGER DEFAULT 0,
                early_trix_bearish INTEGER DEFAULT 0,
                trix_classic_long INTEGER DEFAULT 0,
                trix_classic_short INTEGER DEFAULT 0,
                trix_long_passed INTEGER DEFAULT 0,
                trix_short_passed INTEGER DEFAULT 0,
                trix_long_score REAL DEFAULT 0.0,
                trix_short_score REAL DEFAULT 0.0,
                trix_threshold_config REAL DEFAULT 0.0,
                trix_depth_config REAL DEFAULT 0.0,

                stoch_ltf REAL DEFAULT 0.0,
                stoch_htf REAL DEFAULT 0.0,
                stoch_prev_ltf REAL DEFAULT 0.0,
                stoch_prev_htf REAL DEFAULT 0.0,
                stoch_delta_ltf REAL DEFAULT 0.0,
                stoch_delta_htf REAL DEFAULT 0.0,
                stoch_long_passed INTEGER DEFAULT 0,
                stoch_short_passed INTEGER DEFAULT 0,
                stoch_long_score REAL DEFAULT 0.0,
                stoch_short_score REAL DEFAULT 0.0,
                stoch_range_long_config REAL DEFAULT 0.0,
                stoch_range_short_config REAL DEFAULT 0.0,

                rsi_ltf REAL DEFAULT 0.0,
                rsi_htf REAL DEFAULT 0.0,
                rsi_prev_ltf REAL DEFAULT 0.0,
                rsi_prev_htf REAL DEFAULT 0.0,
                rsi_delta_ltf REAL DEFAULT 0.0,
                rsi_delta_htf REAL DEFAULT 0.0,
                rsi_mode_config REAL DEFAULT 0.0,
                rsi_long_passed INTEGER DEFAULT 0,
                rsi_short_passed INTEGER DEFAULT 0,
                rsi_long_score REAL DEFAULT 0.0,
                rsi_short_score REAL DEFAULT 0.0,
                rsi_range_long_config REAL DEFAULT 0.0,
                rsi_range_short_config REAL DEFAULT 0.0,
                rsi_tolerance REAL DEFAULT 0.0,

                ema_diff_ltf REAL DEFAULT 0.0,
                ema_diff_htf REAL DEFAULT 0.0,
                ema_fast REAL DEFAULT 0.0,
                ema_slow REAL DEFAULT 0.0,
                ema_long_passed INTEGER DEFAULT 0,
                ema_short_passed INTEGER DEFAULT 0,
                ema_long_score REAL DEFAULT 0.0,
                ema_short_score REAL DEFAULT 0.0,
                ema_threshold_config REAL DEFAULT 0.0,

                div_long_passed INTEGER DEFAULT 0,
                div_short_passed INTEGER DEFAULT 0,
                div_long_total_score REAL DEFAULT 0.0,
                div_short_total_score REAL DEFAULT 0.0,
                div_score_1h_config REAL DEFAULT 0.0,
                div_score_4h_config REAL DEFAULT 0.0,
                div_tf_config TEXT DEFAULT '1h,4h',
                rsi_1h_long_score REAL DEFAULT 0.0,
                rsi_1h_short_score REAL DEFAULT 0.0,
                rsi_4h_long_score REAL DEFAULT 0.0,
                rsi_4h_short_score REAL DEFAULT 0.0,
                rsi_1h_long_type TEXT DEFAULT NULL,
                rsi_1h_short_type TEXT DEFAULT NULL,
                rsi_4h_long_type TEXT DEFAULT NULL,
                rsi_4h_short_type TEXT DEFAULT NULL,
                macd_1h_long_score REAL DEFAULT 0.0,
                macd_1h_short_score REAL DEFAULT 0.0,
                macd_4h_long_score REAL DEFAULT 0.0,
                macd_4h_short_score REAL DEFAULT 0.0,
                macd_1h_long_type TEXT DEFAULT NULL,
                macd_1h_short_type TEXT DEFAULT NULL,
                macd_4h_long_type TEXT DEFAULT NULL,
                macd_4h_short_type TEXT DEFAULT NULL,

                sd_zone_type TEXT DEFAULT NULL,
                sd_zone_level REAL DEFAULT 0.0,
                sd_zone_strength REAL DEFAULT 0.0,
                sd_zone_broken INTEGER DEFAULT 0,

                entry_zone_min REAL DEFAULT 0.0,
                entry_zone_max REAL DEFAULT 0.0,
                entry_price REAL DEFAULT 0.0,
                take1_profit REAL DEFAULT 0.0,                
                take2_profit REAL DEFAULT 0.0,                
                take3_profit REAL DEFAULT 0.0,                
                take4_profit REAL DEFAULT 0.0,                
                take5_profit REAL DEFAULT 0.0,
                stop_loss INTEGER DEFAULT 0,             

                signal_passed INTEGER DEFAULT 0,
                scores_config REAL DEFAULT 0.0,
                final_scores REAL DEFAULT 0.0,
                target INTEGER DEFAULT 0,
                take1_hit INTEGER DEFAULT 0,
                take2_hit INTEGER  DEFAULT 0,
                take3_hit INTEGER  DEFAULT 0,
                take4_hit INTEGER  DEFAULT 0,
                take5_hit INTEGER  DEFAULT 0,
                stop_loss_hit INTEGER DEFAULT 0,
                ml_changed_signal_passed INTEGER DEFAULT 0,

                signal_activated INTEGER DEFAULT 0,
                ml_predicted_class INTEGER DEFAULT NULL,
                ml_predicted_label TEXT DEFAULT '', 
                ml_confidence REAL DEFAULT 0.0,
                ml_strength REAL DEFAULT 0.0,
                ml_mode TEXT DEFAULT 'advisory',
                ml_decision TEXT DEFAULT 'neutral',
                ml_comment TEXT DEFAULT '',
                ml_allowed INTEGER DEFAULT NULL
            );
            """)
            await db.commit()

    async def add_filter_result(self, result_dict: dict):
        global db_stats_enabled
        if not db_stats_enabled:
            return

        # Преобразуем все списки в JSON-строки
        for k, v in result_dict.items():
            if isinstance(v, list):
                result_dict[k] = json.dumps(v)

        columns = ", ".join(result_dict.keys())
        placeholders = ", ".join("?" for _ in result_dict)
        values = tuple(result_dict.values())

        async with aiosqlite.connect(self.db_path) as db:
            try:
                await db.execute(
                    f"INSERT OR REPLACE INTO {TABLE_NAME} ({columns}) VALUES ({placeholders})",
                    values
                )
                await db.commit()
            except Exception as e:
                print(f"[DB ERROR] Не удалось вставить запись: {e}")


    async def update_signal_result(
        self,
        signal_id: str,
        target: Optional[int] = None,
        take1_hit: Optional[int] = None,
        take2_hit: Optional[int] = None,
        take3_hit: Optional[int] = None,
        take4_hit: Optional[int] = None,
        take5_hit: Optional[int] = None,
        stop_loss_hit: Optional[int] = None,
        signal_activated: Optional[int] = None,
        entry_zone_min: Optional[int] = None,
        entry_zone_max: Optional[int] = None,
        entry_price: Optional[int] = None,
        take1_profit: Optional[int] = None,
        take2_profit: Optional[int] = None,
        take3_profit: Optional[int] = None,
        take4_profit: Optional[int] = None,
        take5_profit: Optional[int] = None,
        stop_loss: Optional[int] = None
    ):
        """
        Обновляет поля результата сделки по уникальному signal_id.
        Передавать None для полей, которые не нужно менять.
        """
        global db_stats_enabled
        if not db_stats_enabled:
            return

        try:
            # собираем только те поля, которые переданы
            fields_to_update = {}
            if target is not None:
                fields_to_update["target"] = target
            if take1_hit is not None:
                fields_to_update["take1_hit"] = take1_hit
            if take2_hit is not None:
                fields_to_update["take2_hit"] = take2_hit
            if take3_hit is not None:
                fields_to_update["take3_hit"] = take3_hit
            if take4_hit is not None:
                fields_to_update["take4_hit"] = take4_hit
            if take5_hit is not None:
                fields_to_update["take5_hit"] = take5_hit
            if stop_loss_hit is not None:
                fields_to_update["stop_loss_hit"] = stop_loss_hit
            if signal_activated is not None:
                fields_to_update["signal_activated"] = signal_activated
            if entry_zone_min is not None:
                fields_to_update["entry_zone_min"] = entry_zone_min
            if entry_zone_max is not None:
                fields_to_update["entry_zone_max"] = entry_zone_max
            if entry_price is not None:
                fields_to_update["entry_price"] = entry_price
            if take1_profit is not None:
                fields_to_update["take1_profit"] = take1_profit
            if take2_profit is not None:
                fields_to_update["take2_profit"] = take2_profit
            if take3_profit is not None:
                fields_to_update["take3_profit"] = take3_profit
            if take4_profit is not None:
                fields_to_update["take4_profit"] = take4_profit
            if take5_profit is not None:
                fields_to_update["take5_profit"] = take5_profit
            if stop_loss is not None:
                fields_to_update["stop_loss"] = stop_loss

            if not fields_to_update:
                return  # нечего обновлять

            set_clause = ", ".join(f"{k} = ?" for k in fields_to_update.keys())
            values = list(fields_to_update.values())
            values.append(signal_id)  # для WHERE

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    f"UPDATE {TABLE_NAME} SET {set_clause} WHERE signal_id = ?",
                    values
                )
                await db.commit()

        except Exception as e:
            print(f"[DB ERROR] Не удалось обновить signal_id {signal_id}: {e}")

        try:
            # собираем только те поля, которые переданы
            fields_to_update = {}
            if target is not None:
                fields_to_update["target"] = target
            if take1_hit is not None:
                fields_to_update["take1_hit"] = take1_hit
            if take2_hit is not None:
                fields_to_update["take2_hit"] = take2_hit
            if take3_hit is not None:
                fields_to_update["take3_hit"] = take3_hit
            if take4_hit is not None:
                fields_to_update["take4_hit"] = take4_hit
            if take5_hit is not None:
                fields_to_update["take5_hit"] = take5_hit
            if stop_loss_hit is not None:
                fields_to_update["stop_loss_hit"] = stop_loss_hit
            if signal_activated is not None:
                fields_to_update["signal_activated"] = signal_activated
            if entry_zone_min is not None:
                fields_to_update["entry_zone_min"] = entry_zone_min
            if entry_zone_max is not None:
                fields_to_update["entry_zone_max"] = entry_zone_max
            if entry_price is not None:
                fields_to_update["entry_price"] = entry_price
            if take1_profit is not None:
                fields_to_update["take1_profit"] = take1_profit
            if take2_profit is not None:
                fields_to_update["take2_profit"] = take2_profit
            if take3_profit is not None:
                fields_to_update["take3_profit"] = take3_profit
            if take4_profit is not None:
                fields_to_update["take4_profit"] = take4_profit
            if take5_profit is not None:
                fields_to_update["take5_profit"] = take5_profit
            if stop_loss is not None:
                fields_to_update["stop_loss"] = stop_loss

            if not fields_to_update:
                return  # нечего обновлять

            set_clause = ", ".join(f"{k} = ?" for k in fields_to_update.keys())
            values = list(fields_to_update.values())
            values.append(signal_id)  # для WHERE

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    f"UPDATE {TABLE_NAME} SET {set_clause} WHERE signal_id = ?",
                    values
                )
                await db.commit()

        except Exception as e:
            print(f"[DB ERROR] Не удалось обновить signal_id {signal_id}: {e}")

    async def enable_filter_stats(self):
        global db_stats_enabled
        db_stats_enabled = True

    async def disable_filter_stats(self):
        global db_stats_enabled
        db_stats_enabled = False

    async def clear_filter_stats(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(f"DELETE FROM {TABLE_NAME}")
            await db.commit()

    async def export_filter_stats_csv(self, bot, chat_id, symbol=None, limit=3000):
        query = f"SELECT * FROM {TABLE_NAME}"
        params = []
        if symbol:
            query += " WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?"
            query += " WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?"
            params = [symbol, limit]
        else:
            query += " ORDER BY timestamp DESC LIMIT ?"
            query += " ORDER BY timestamp DESC LIMIT ?"
            params = [limit]

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                if not rows:
                    bot.send_message(chat_id, "Нет данных для экспорта.")
                    return
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                if not rows:
                    bot.send_message(chat_id, "Нет данных для экспорта.")
                    return

                col_names = [description[0] for description in cursor.description]
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(col_names)
                writer.writerows(rows)
                output.seek(0)
                csv_file = io.BytesIO(output.getvalue().encode("utf-8"))
                csv_file.name = "ml_trading_data.csv"
                bot.send_document(chat_id, csv_file)


# Экземпляр
db_stats = DBStats()
