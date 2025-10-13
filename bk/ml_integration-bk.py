# ml_integration.py (важная часть)
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import joblib, json, os

# конфиг (или загружай из main/config)
ML_CONFIG = {
    "enabled": True,
    "mode": "advisory",  # off | advisory | hybrid | autonomous
    "model_path": "models/lgbm_v1.pkl",
    "features_path": "models/feature_order.json",
    "confidence_threshold": 0.6,  # порог минимальной уверенности для принятия активного решения
    "hybrid_override_threshold": 0.7,  # порог для переубедить бота
}

ml_model = None
ml_features = []
def init_model():
    global ml_model, ml_features
    if not ML_CONFIG.get("enabled", False) or ML_CONFIG["mode"] == "off":
        ml_model = None
        return False
    try:
        ml_model = joblib.load(ML_CONFIG["model_path"])
        with open(ML_CONFIG["features_path"], "r", encoding="utf-8") as f:
            ml_features = json.load(f)
        print("[ML] loaded", ML_CONFIG["model_path"])
        return True
    except Exception as e:
        print("[ML] load error:", e)
        ml_model = None
        return False

def analyze_signal_row(filters_results: Dict) -> Optional[Dict]:
    """Возвращает словарь с ml_result или None, если нет уверенного прогноза."""
    if ml_model is None:
        return None
    try:
        print(f"[ML DEBUG] Получено {len(filters_results)} полей в filters_results")

        row = {f: filters_results.get(f, 0) for f in ml_features}
        X = pd.DataFrame([row]).fillna(0).astype(float)
        probs = ml_model.predict_proba(X)[0]
        pred_class = int(np.argmax(probs))
        conf = float(np.max(probs))
        return {
            "ml_predicted_class": pred_class,
            "ml_confidence": conf,
            "ml_probas": probs.tolist()
        }
    except Exception as e:
        print("[ML] analyze error:", e)
        return None

def ml_handle_signal(filters_results: Dict, passed_by_bot: bool) -> Dict:
    """
    Возвращает:
      {
        "ml_result": {...}   # поля анализа для сохранения в БД
        "ml_action": "approve"|"reject"|"neutral"  # действие ML по сигналу
        "ml_reason": str
      }
    Не меняет ничего в основном потоке — только возвращает решение.
    """

    print(f"[ML] Обработка сигнала: passed_by_bot={passed_by_bot}")

    # Инициализируем out с новой структурой ml_result
    out = {
        "ml_result": {
            "ml_mode": ML_CONFIG["mode"],
            "ml_decision": "neutral", 
            "ml_comment": "",
            "ml_predicted_class": 0,
            "ml_predicted_label": "",
            "ml_confidence": 0.0,
            "ml_strength": 0.0,
            "ml_probas": "[]"
        },
        "ml_action": "neutral", 
        "ml_reason": ""
    }

    # быстрый fail-safe
    if not ML_CONFIG.get("enabled", False) or ML_CONFIG["mode"] == "off" or ml_model is None:
        out["ml_reason"] = "ML_OFF"
        # Обновляем ml_result в соответствии с решением
        out["ml_result"]["ml_decision"] = out["ml_action"]
        out["ml_result"]["ml_comment"] = out["ml_reason"]
        return out
    
    print(f"[ML] Mode: {ML_CONFIG['mode']}, Model loaded: {ml_model is not None}, Bot decision: {'approved' if passed_by_bot else 'rejected'}")

    ml = analyze_signal_row(filters_results)

    if ml:
        print(f"[ML] Prediction: class={ml['ml_predicted_class']}, confidence={ml['ml_confidence']:.3f}")
    else:
        print("[ML] No prediction available")

    if not ml:
        out["ml_reason"] = "NO_CONFIDENT_PREDICTION"
        # Обновляем ml_result в соответствии с решением
        out["ml_result"]["ml_decision"] = out["ml_action"]
        out["ml_result"]["ml_comment"] = out["ml_reason"]
        return out

    # положим результат для БД
    ml_label = f"TP{ml['ml_predicted_class']}" if ml["ml_predicted_class"] > 0 else "STOP"
    ml_strength = round(ml["ml_confidence"] * 100, 2)
    
    # Обновляем ml_result с прогнозами модели
    out["ml_result"].update({
        "ml_predicted_class": int(ml["ml_predicted_class"]),
        "ml_predicted_label": ml_label,
        "ml_confidence": float(ml["ml_confidence"]),
        "ml_strength": ml_strength,
        "ml_probas": json.dumps(ml["ml_probas"])
    })

    mode = ML_CONFIG["mode"]
    # decision logic
    if mode == "advisory":
        out["ml_action"] = "neutral"
        out["ml_reason"] = "ADVISORY_NEUTRAL"
        # Обновляем ml_result в соответствии с решением
        out["ml_result"]["ml_decision"] = out["ml_action"]
        out["ml_result"]["ml_comment"] = out["ml_reason"]
        return out

    if mode == "hybrid":
        # если бот отклонил сигнал, но ML очень уверен и предсказывает TP2+
        if not passed_by_bot and ml["ml_confidence"] >= ML_CONFIG["hybrid_override_threshold"] and ml["ml_predicted_class"] >= 2:
            out["ml_action"] = "approve"
            out["ml_reason"] = "HYBRID_OVERRIDE_APPROVE"
        # если бот разрешил, но ML уверен что стоп — ML может отклонить
        elif passed_by_bot and ml["ml_confidence"] >= ML_CONFIG["hybrid_override_threshold"] and ml["ml_predicted_class"] == 0:
            out["ml_action"] = "reject"
            out["ml_reason"] = "HYBRID_OVERRIDE_REJECT"
        else:
            out["ml_action"] = "neutral"
            out["ml_reason"] = "HYBRID_NO_OVERRIDE"
        
        # Обновляем ml_result в соответствии с решением
        out["ml_result"]["ml_decision"] = out["ml_action"]
        out["ml_result"]["ml_comment"] = out["ml_reason"]
        return out

    if mode == "autonomous":
        # полностью доверяем модели при уверенности >= threshold
        if ml["ml_confidence"] >= ML_CONFIG["confidence_threshold"]:
            if ml["ml_predicted_class"] == 0:
                out["ml_action"] = "reject"
                out["ml_reason"] = "AUTONOMOUS_REJECT_STOP"
            elif ml["ml_predicted_class"] >= 2:
                out["ml_action"] = "approve"
                out["ml_reason"] = "AUTONOMOUS_APPROVE_TP2P"
            else:
                out["ml_action"] = "neutral"
                out["ml_reason"] = "AUTONOMOUS_NEUTRAL_TP1"
        else:
            out["ml_action"] = "neutral"
            out["ml_reason"] = "AUTONOMOUS_LOW_CONFIDENCE"
        
        # Обновляем ml_result в соответствии с решением
        out["ml_result"]["ml_decision"] = out["ml_action"]
        out["ml_result"]["ml_comment"] = out["ml_reason"]
        return out

    out["ml_reason"] = "UNKNOWN_MODE"
    # Обновляем ml_result в соответствии с решением
    out["ml_result"]["ml_decision"] = out["ml_action"]
    out["ml_result"]["ml_comment"] = out["ml_reason"]
    return out

def generate_signal_text_base(filters_results: dict, ml_output: dict) -> str:
    """
    Формирует человекочитаемый прогноз по сигналу с учетом ML и фильтров.
    Поддерживает до 5 TP (тейков).
    """
    # --- 1. Базовые данные ---
    entry_low = filters_results.get("entry_low", 0)
    entry_high = filters_results.get("entry_high", 0)
    stop = filters_results.get("stop", 0)
    tps = filters_results.get("tps", [])  # список целей TP1–TP5, может быть меньше 5
    global_trend = filters_results.get("global_trend", "neutral")

    # --- 2. Комбинированная сила ---
    final_score = filters_results.get("final_score", 0)
    ml_conf = ml_output.get("ml_result", {}).get("ml_confidence", 0)
    strength = round((0.6 * final_score + 0.4 * ml_conf) * 100, 2)

    # --- 3. Словесное описание силы ---
    if strength < 40:
        desc, emoji = "слабый / сомнительный сценарий", "🔴"
    elif strength < 60:
        desc, emoji = "нейтральный / требующий подтверждения", "🟠"
    elif strength < 80:
        desc, emoji = "средне-сильный бычий / медвежий сценарий", "🟢"
    else:
        desc, emoji = "мощный импульсный сценарий", "🟣"

    # --- 4. Прогноз движения ---
    ml_class = ml_output.get("ml_result", {}).get("ml_predicted_class", 0)
    ml_label = ml_output.get("ml_result", {}).get("ml_predicted_label", "STOP")
    
    if ml_class == 0:
        forecast_text = f"STOP — сигнал на отклонение"
    else:
        # Безопасное формирование tps_text
        tps_display = tps[:ml_class]  # берем только нужное количество целей
        if tps_display:
            tps_text = " → ".join([f"{tp:.4f}" for tp in tps_display])
            forecast_text = (
                f"В ближайшие 2–4 часа вероятен небольшой откат к нижней границе входа "
                f"{entry_low:.4f}-{entry_high:.4f}, после подтверждения — движение к целям TP1–TP{ml_class} ({tps_text})."
            )
        else:
            forecast_text = "Цели не определены"

    # --- 5. Итоговая рекомендация ---
    action = ml_output.get("ml_action", "neutral")
    ml_reason = ml_output.get("ml_reason", "")
    
    # Преобразуем коды причин в читаемый текст
    reason_map = {
        "ML_OFF": "ML выключена",
        "NO_CONFIDENT_PREDICTION": "нет уверенного прогноза",
        "ADVISORY_NEUTRAL": "режим советника",
        "HYBRID_OVERRIDE_APPROVE": "ML переубедила бота",
        "HYBRID_OVERRIDE_REJECT": "ML переубедила бота", 
        "HYBRID_NO_OVERRIDE": "ML не вмешалась",
        "AUTONOMOUS_REJECT_STOP": "автономное решение",
        "AUTONOMOUS_APPROVE_TP2P": "автономное решение",
        "AUTONOMOUS_NEUTRAL_TP1": "автономное решение",
        "AUTONOMOUS_LOW_CONFIDENCE": "низкая уверенность",
        "UNKNOWN_MODE": "неизвестный режим"
    }
    
    readable_reason = reason_map.get(ml_reason, ml_reason)
    
    if action == "approve":
        rec = f"Входить после подтверждения — {readable_reason}"
    elif action == "reject":
        rec = f"Сигнал отклонён — {readable_reason}"
    else:
        rec = f"Ждать подтверждения — {readable_reason}"

    # --- 6. Краткий итог одной строкой ---
    if ml_class > 0 and tps:
        last_tp = f"{tps[min(ml_class-1, len(tps)-1)]:.4f}"
        short_summary = f"🔮 Сила {strength}% — ждём ретест {entry_low:.4f}, потом рост к TP{ml_class} ({last_tp})"
    else:
        short_summary = f"🔮 Сила {strength}% — {desc}"

    # --- 7. Формируем текст ---
    message = (
        f"📊 Итоговая сила\n"
        f"{emoji} {strength}% ({desc})\n\n"
        f"🔮 Прогноз\n"
        f"{forecast_text}\n\n"
        f"✅ Итоговая рекомендация\n"
        f"{rec}\n\n"
        f"{short_summary}"
    )

    return message


def generate_signal_text(filters_results: dict, ml_output: dict, targets: tuple, stop: float) -> str:
    # --- 1. Вход и сила ---
    entry_low = filters_results.get("entry_low", 0)
    entry_high = filters_results.get("entry_high", 0)
    final_score = filters_results.get("final_score", 0)
    ml_conf = ml_output.get("ml_result", {}).get("ml_confidence", 0)
    strength = round((0.6 * final_score + 0.4 * ml_conf) * 100, 2)

    # словесное описание силы
    signal_type = filters_results.get("trend_global", "long")  # long / short

    if strength < 40:
        desc, emoji = "слабый / сомнительный сценарий", "🔴"
    elif strength < 60:
        desc, emoji = "нейтральный / требующий подтверждения", "🟠"
    elif strength < 80:
        desc = "средне-сильный бычий сценарий" if signal_type=="long" else "средне-сильный медвежий сценарий"
        emoji = "🟢"
    else:
        desc = "мощный импульсный бычий сценарий" if signal_type=="long" else "мощный импульсный медвежий сценарий"
        emoji = "🟣"

    # --- 2. Тейки ---
    # берем только цены из targets
    tps = [t[0] for t in targets]
    ml_class = ml_output.get("ml_result", {}).get("ml_predicted_class", 0)
    
    if ml_class == 0:
        tps_text = "STOP — сигнал на отклонение"
    else:
        tps_display = tps[:ml_class]
        if tps_display:
            tps_text = f"{tps_display[0]:.4f}"
            if len(tps_display) > 1:
                tps_text += f" → {tps_display[-1]:.4f}"
        else:
            tps_text = "Цели не определены"

    # --- 3. ML прогноз ---
    ml_action = ml_output.get("ml_action", "neutral")
    ml_conf_pct = round(ml_conf * 100, 1)
    if ml_action == "approve":
        ml_text = f"вход возможен (уверенность {ml_conf_pct}%)"
    elif ml_action == "reject":
        ml_text = f"сигнал отклонён (уверенность {ml_conf_pct}%)"
    else:
        ml_text = f"нейтральный, не даёт однозначного сигнала (уверенность {ml_conf_pct}%)"

    # --- 4. Рекомендация ---
    if ml_class == 0 or ml_action == "reject":
        rec = "сигнал отклонён"
    elif ml_action == "approve":
        rec = "входить после подтверждения роста"
    else:
        rec = "подождать подтверждения роста на графике"

    # --- 5. Формируем текст ---
    message = (
        f"🔮 Сила: {strength}% ({desc})\n"
        f"📈 Вход: {entry_low:.4f}–{entry_high:.4f}\n"
        f"🎯 Цели: {tps_text}\n"
        f"🛑 Стоп: {stop:.4f}\n"
        f"💡 ML: {ml_text}\n"
        f"✅ Рекомендация: {rec}"
    )

    return message


# HYBRID режим:
# HYBRID_OVERRIDE_APPROVE    - ML переубедила на approve
# HYBRID_OVERRIDE_REJECT     - ML переубедила на reject  
# HYBRID_NO_OVERRIDE         - ML не вмешалась

# AUTONOMOUS режим:
# AUTONOMOUS_APPROVE_TP2P    - автоодобрение TP2+
# AUTONOMOUS_REJECT_STOP     - автоотклонение STOP
# AUTONOMOUS_NEUTRAL_TP1     - нейтрально из-за TP1
# AUTONOMOUS_LOW_CONFIDENCE  - нейтрально из-за низкой уверенности

# ОБЩИЕ:
# ML_OFF                     - ML выключена
# NO_CONFIDENT_PREDICTION    - нет уверенного прогноза
# UNKNOWN_MODE               - неизвестный режим