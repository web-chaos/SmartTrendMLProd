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


def generate_signal_text_v2(filters_results: dict, ml_output: dict) -> str:
    entry_low = filters_results.get("entry_low", 0)
    entry_high = filters_results.get("entry_high", 0)
    stop = filters_results.get("stop", 0)
    tps = filters_results.get("tps", [])
    global_trend = filters_results.get("global_trend", "neutral")

    # --- Сила сигнала по фильтрам ---
    filter_score = filters_results.get("final_score", 0)
    ml_conf = ml_output.get("ml_result", {}).get("ml_confidence", 0)

    combined_strength = round(0.6*filter_score + 0.4*ml_conf, 2)
    
    # --- Словесное описание силы ---
    if combined_strength < 40:
        strength_desc, emoji = "слабый / сомнительный", "🔴"
    elif combined_strength < 60:
        strength_desc, emoji = "средний / требующий подтверждения", "🟠"
    elif combined_strength < 80:
        strength_desc, emoji = "сильный сценарий", "🟢"
    else:
        strength_desc, emoji = "очень сильный импульс", "🟣"

    # --- Сигнал фильтров ---
    signal_side = filters_results.get("signal_passed", 0)
    signal_text = "STOP" if signal_side == 0 else filters_results.get("final_signal", "NEUTRAL")

    # --- ML рекомендация ---
    ml_action = ml_output.get("ml_action", "neutral")
    ml_reason = ml_output.get("ml_reason", "")
    reason_map = {
        "ADVISORY_NEUTRAL": "режим советника",
        "HYBRID_OVERRIDE_APPROVE": "ML подтвердила сигнал",
        "HYBRID_OVERRIDE_REJECT": "ML отклонила сигнал",
        "AUTONOMOUS_APPROVE_TP2P": "ML автономно одобрила",
        "AUTONOMOUS_REJECT_STOP": "ML автономно отклонила"
    }
    ml_comment = reason_map.get(ml_reason, ml_reason)

    if ml_action == "approve":
        rec = f"Можно входить — {ml_comment}"
    elif ml_action == "reject":
        rec = f"Сигнал отклонён — {ml_comment}"
    else:
        rec = f"Ждать подтверждения — {ml_comment}"

    # --- Формируем прогноз по TP ---
    if tps and signal_side:
        tps_text = " → ".join([f"{tp:.4f}" for tp in tps])
        forecast_text = f"После подтверждения движение к целям: {tps_text}. Стоп: {stop:.4f}"
    else:
        forecast_text = "Цели не определены"

    # --- Итоговое сообщение ---
    message = (
        f"📊 Итоговая сила\n"
        f"{emoji} {combined_strength}% ({strength_desc})\n\n"
        f"🔮 Сигнал фильтров: {signal_text}\n"
        f"🔮 Прогноз ML: {rec}\n\n"
        f"📈 Прогноз движения:\n"
        f"{forecast_text}\n"
    )

    return message

def generate_signal_text_v3(filters_results: dict, ml_output: dict) -> str:
    entry_low = filters_results.get("entry_low", 0)
    entry_high = filters_results.get("entry_high", 0)
    stop = filters_results.get("stop", 0)
    tps = filters_results.get("tps", [])
    global_trend = filters_results.get("global_trend", "neutral")

    # --- Сила сигнала по фильтрам ---
    filter_score = filters_results.get("final_score", 0)
    ml_conf = ml_output.get("ml_result", {}).get("ml_confidence", 0)

    combined_strength = round(0.6 * filter_score + 0.4 * ml_conf, 2)
    
    # --- Словесное описание силы ---
    if combined_strength < 40:
        strength_desc, emoji = "слабый / сомнительный", "🔴"
    elif combined_strength < 60:
        strength_desc, emoji = "средний / требующий подтверждения", "🟠"
    elif combined_strength < 80:
        strength_desc, emoji = "сильный сценарий", "🟢"
    else:
        strength_desc, emoji = "очень сильный импульс", "🟣"

    # --- Сигнал фильтров ---
    signal_side = filters_results.get("signal_passed", 0)
    signal_text = "STOP" if signal_side == 0 else filters_results.get("final_signal", "NEUTRAL")

    # --- ML рекомендация ---
    ml_action = ml_output.get("ml_action", "neutral")
    ml_reason = ml_output.get("ml_reason", "")
    reason_map = {
        "ADVISORY_NEUTRAL": "режим советника",
        "HYBRID_OVERRIDE_APPROVE": "ML подтвердила сигнал",
        "HYBRID_OVERRIDE_REJECT": "ML отклонила сигнал",
        "AUTONOMOUS_APPROVE_TP2P": "ML автономно одобрила",
        "AUTONOMOUS_REJECT_STOP": "ML автономно отклонила"
    }
    ml_comment = reason_map.get(ml_reason, ml_reason)

    if ml_action == "approve":
        rec = f"Можно входить — {ml_comment}"
    elif ml_action == "reject":
        rec = f"Сигнал отклонён — {ml_comment}"
    else:
        rec = f"Ждать подтверждения — {ml_comment}"

    # --- Прогноз движения по TP ---
    if tps and signal_side:
        tps_text = " → ".join([f"{tp:.4f}" for tp in tps])
        forecast_text = f"После подтверждения движение к целям: {tps_text}. Стоп: {stop:.4f}"
    else:
        forecast_text = "Цели не определены"

    # --- Итоговое сообщение ---
    message = (
        f"📊 Итоговая сила\n"
        f"{emoji} {combined_strength}% ({strength_desc})\n\n"
        f"🔮 Сигнал фильтров: {signal_text}\n"
        f"🔮 Прогноз ML: {rec}\n\n"
        f"📈 Прогноз движения:\n"
        f"{forecast_text}\n"
    )

    return message


def generate_signal_text_v4(filters_results: dict, ml_output: dict, targets=None, stop=None) -> str:
    entry_low = filters_results.get("entry_low", 0)
    entry_high = filters_results.get("entry_high", 0)
    global_trend = filters_results.get("global_trend", "neutral")

    # --- Сила сигнала по фильтрам ---
    filter_score = filters_results.get("final_score", 0)
    ml_conf = ml_output.get("ml_result", {}).get("ml_confidence", 0)
    combined_strength = round(0.6*filter_score + 0.4*ml_conf, 2)

    # --- Словесное описание силы ---
    if combined_strength < 40:
        strength_desc, emoji = "слабый / сомнительный", "🔴"
    elif combined_strength < 60:
        strength_desc, emoji = "средний / требующий подтверждения", "🟠"
    elif combined_strength < 80:
        strength_desc, emoji = "сильный сценарий", "🟢"
    else:
        strength_desc, emoji = "очень сильный импульс", "🟣"

    # --- Сигнал фильтров ---
    signal_side = filters_results.get("signal_passed", 0)
    final_signal = filters_results.get("final_signal", "NEUTRAL")
    signal_text = "STOP" if signal_side == 0 else final_signal.upper()

    # --- ML рекомендация ---
    ml_action = ml_output.get("ml_action", "neutral")
    ml_reason = ml_output.get("ml_reason", "")
    reason_map = {
        "ADVISORY_NEUTRAL": "режим советника",
        "HYBRID_OVERRIDE_APPROVE": "ML подтвердила сигнал",
        "HYBRID_OVERRIDE_REJECT": "ML отклонила сигнал",
        "AUTONOMOUS_APPROVE_TP2P": "ML автономно одобрила",
        "AUTONOMOUS_REJECT_STOP": "ML автономно отклонила"
    }
    ml_comment = reason_map.get(ml_reason, ml_reason)

    if ml_action == "approve":
        rec = f"Можно входить — {ml_comment}"
    elif ml_action == "reject":
        rec = f"Сигнал отклонён — {ml_comment}"
    else:
        rec = f"Ждать подтверждения — {ml_comment}"

    # --- EMA подтверждение ---
    ema_fast = filters_results.get("ema_fast", 0)
    ema_slow = filters_results.get("ema_slow", 0)
    confirmed = False
    if final_signal == "long" and ema_fast > ema_slow:
        confirmed = True
    elif final_signal == "short" and ema_fast < ema_slow:
        confirmed = True

    if confirmed:
        rec = f"Подтверждено по EMA ({ema_fast:.4f} vs {ema_slow:.4f}) — можно входить"

    # --- Формируем прогноз по TP ---
    forecast_text = "Цели не определены"
    if targets and signal_side:
        tp_texts = []
        for i, (tp_val, tp_type) in enumerate(targets, start=1):
            tp_texts.append(f"TP{i} ({tp_type}): {tp_val}")
        forecast_text = "После подтверждения движение к целям:\n" + " → ".join(tp_texts)
        if stop:
            forecast_text += f"\nСтоп: {stop}"

    # --- Итоговое сообщение ---
    message = (
        f"📊 Итоговая сила\n"
        f"{emoji} {combined_strength}% ({strength_desc})\n\n"
        f"🔮 Сигнал фильтров: {signal_text}\n"
        f"🔮 Прогноз ML: {rec}\n\n"
        f"📈 Прогноз движения:\n"
        f"{forecast_text}\n"
    )

    return message

def generate_signal_text5(filters_results, ml_out, targets=None, stop=None):
    """
    Создаёт человеко-понятный ML блок прогноза.
    """
    try:
        ema_fast = filters_results.get("ema_fast")
        ema_slow = filters_results.get("ema_slow")
        strength = ml_out.get("final_score", 0)

        # Определяем подтверждение по EMA
        confirmed = ema_fast is not None and ema_slow is not None and ema_fast > ema_slow
        confirm_text = "EMA_fast > EMA_slow (восходящий импульс)" if confirmed else "EMA_fast < EMA_slow (нисходящая тенденция)"

        # Определяем категорию силы
        if strength >= 0.8:
            emoji, desc = "🟣", "очень сильный импульс"
        elif strength >= 0.6:
            emoji, desc = "🟢", "сильный сигнал"
        elif strength >= 0.4:
            emoji, desc = "🟡", "умеренно уверенный сценарий"
        elif strength > 0:
            emoji, desc = "🟠", "сомнительный сигнал"
        else:
            emoji, desc = "🔴", "отсутствие подтверждения"

        # Потенциал движения
        tp_target = targets[2][0] if targets and len(targets) >= 3 else (targets[-1][0] if targets else None)
        if tp_target:
            potential_text = f"📈 Потенциал роста до TP3 ({tp_target})"
        else:
            potential_text = "📈 Цели не определены"

        # Стоп
        risk_text = f"⚠️ Возможное снижение до стопа ({stop})" if stop else ""

        # Формируем связный текст
        return (
            f"📊 Итоговая сила: {emoji} {strength:.2f} ({desc})\n"
            f"🔮 Подтверждение по EMA: {'✅' if confirmed else '❌'} {confirm_text}\n"
            # f"{potential_text}\n"
            # f"{risk_text}"
        )

    except Exception as e:
        print(f"[ML ERROR] generate_signal_text: {e}")
        return "🤖 Прогноз недоступен"

def generate_signal_text(filters_results: dict, ml_output: dict) -> str:
    """
    Генерирует связный прогноз движения с конкретным сценарием.
    ML даёт силу сигнала, фильтры дают TP и стоп.
    """
    entry_low = filters_results.get("entry_low", 0)
    entry_high = filters_results.get("entry_high", 0)
    stop = filters_results.get("stop", 0)
    tps = filters_results.get("tps", [])
    global_trend = filters_results.get("global_trend", "neutral")
    
    ema_fast = filters_results.get("ema_fast", 0)
    ema_slow = filters_results.get("ema_slow", 0)

    # --- Сила сигнала ---
    filter_score = filters_results.get("final_score", 0)
    ml_conf = ml_output.get("ml_result", {}).get("ml_confidence", 0)
    combined_strength = round(0.6*filter_score + 0.4*ml_conf, 2)

    # --- Описание силы ---
    if combined_strength < 40:
        strength_desc, emoji = "слабый / сомнительный", "🔴"
    elif combined_strength < 60:
        strength_desc, emoji = "средний / требующий подтверждения", "🟠"
    elif combined_strength < 80:
        strength_desc, emoji = "сильный сценарий", "🟢"
    else:
        strength_desc, emoji = "очень сильный импульс", "🟣"

    # --- Импульс по EMA ---
    if ema_fast > ema_slow:
        impulse_text = "восходящий импульс (EMA_fast > EMA_slow)"
    elif ema_fast < ema_slow:
        impulse_text = "нисходящий импульс (EMA_fast < EMA_slow)"
    else:
        impulse_text = "флет / неопределённый импульс"

    # --- Сигнал фильтров ---
    signal_side = filters_results.get("signal_passed", 0)
    signal_text = "STOP" if signal_side == 0 else filters_results.get("final_signal", "NEUTRAL")

    # --- ML рекомендация ---
    ml_action = ml_output.get("ml_action", "neutral")
    ml_reason = ml_output.get("ml_reason", "")
    reason_map = {
        "ADVISORY_NEUTRAL": "режим советника",
        "HYBRID_OVERRIDE_APPROVE": "ML подтвердила сигнал",
        "HYBRID_OVERRIDE_REJECT": "ML отклонила сигнал",
        "AUTONOMOUS_APPROVE_TP2P": "ML автономно одобрила",
        "AUTONOMOUS_REJECT_STOP": "ML автономно отклонила"
    }
    ml_comment = reason_map.get(ml_reason, ml_reason)
    if ml_action == "approve":
        rec = f"Можно входить — {ml_comment}"
    elif ml_action == "reject":
        rec = f"Сигнал отклонён — {ml_comment}"
    else:
        rec = f"Ждать подтверждения — {ml_comment}"

    # --- Формируем сценарий движения ---
    if tps and signal_side:
        tps_text = " → ".join([f"{tp:.4f}" for tp in tps])
        forecast_text = (
            f"Вероятно, сначала откат к нижней границе входа {entry_low:.4f}-{entry_high:.4f} "
            f"({impulse_text}), затем движение к целям TP1–TP{len(tps)}: {tps_text}. "
            f"Стоп: {stop:.4f}"
        )
        short_summary = f"🔮 Сила {combined_strength}% — ретест {entry_low:.4f}, рост к TP{len(tps)} ({tps[-1]:.4f})"
    else:
        forecast_text = "Цели не определены"
        short_summary = f"🔮 Сила {combined_strength}% — {strength_desc}"

    # --- Финальный текст ---
    message = (
        f"📊 Итоговая сила\n"
        f"{emoji} {combined_strength}% ({strength_desc})\n\n"
        f"🔮 Сигнал фильтров: {signal_text}\n"
        f"🔮 Прогноз ML: {rec}\n\n"
        f"📈 Сценарий движения:\n"
        f"{forecast_text}\n\n"
        f"{short_summary}"
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