import asyncio
from datetime import datetime, time, timezone, timedelta
import pytz

print("[SESSION_MONITOR] Инициализация модуля мониторинга торговых сессий")

# Временные зоны
TZ_KYIV = pytz.timezone("Europe/Kiev")
TZ_LONDON = pytz.timezone("Europe/London")
TZ_NY = pytz.timezone("America/New_York")
TZ_SYD = pytz.timezone("Australia/Sydney")
TZ_BRINKS = pytz.timezone("UTC")  # как и обсуждали выше

# Фиксированные сессии в UTC (время торговли по TradingView)
SESSIONS = [
    {"name": "Тихоокеанская (Сидней)", "start": time(22, 0), "end": time(6, 0), "tz": TZ_SYD},
    {"name": "Азиатская (Токио)", "start": time(0, 0), "end": time(6, 0), "tz": pytz.timezone("Asia/Tokyo")},
    {"name": "Азиатская (Гонконг, Сингапур)", "start": time(1, 30), "end": time(9, 0), "tz": pytz.timezone("Asia/Hong_Kong")},
    {"name": "Европейская (Лондон)", "start": time(8, 0), "end": time(16, 30), "tz": TZ_LONDON},
    {"name": "Американская (Нью-Йорк)", "start": time(13, 30), "end": time(20, 0), "tz": TZ_NY},
    # Brinks-сессии
    {"name": "EU Brinks", "start": time(8, 0), "end": time(9, 0), "tz": TZ_BRINKS},
    {"name": "US Brinks", "start": time(14, 0), "end": time(15, 0), "tz": TZ_BRINKS},

]

# Проверка, находится ли время now в диапазоне start-end с учётом перехода через полночь
def time_in_range(start: time, end: time, now: time) -> bool:
    if start <= end:
        return start <= now < end
    return now >= start or now < end  # Через полночь

# Конвертирует время из UTC в Киевское, возвращает строку "HH:MM–HH:MM по Киеву"
def format_kyiv_range(start_utc: time, end_utc: time) -> str:
    today_utc = datetime.now(timezone.utc).date()
    dt_start_utc = datetime.combine(today_utc, start_utc, tzinfo=timezone.utc)
    dt_end_utc = datetime.combine(today_utc, end_utc, tzinfo=timezone.utc)
    if start_utc > end_utc:
        dt_end_utc += timedelta(days=1)  # Если диапазон через полночь

    dt_start_kyiv = dt_start_utc.astimezone(TZ_KYIV)
    dt_end_kyiv = dt_end_utc.astimezone(TZ_KYIV)

    return f"{dt_start_kyiv.strftime('%H:%M')}–{dt_end_kyiv.strftime('%H:%M')} по Киеву"

def fetch_symbol_info(client):
    try:
        symbols = ["BTCUSDT", "ETHUSDT"]
        msg = ""
        for sym in symbols:
            ticker = client.futures_mark_price(symbol=sym)
            price = float(ticker["markPrice"])

            klines = client.futures_klines(symbol=sym, interval="1h", limit=1)
            vol = float(klines[0][5])

            vol_str = (f"{vol/1_000_000_000:.1f}B" if vol > 1_000_000_000 else
                       f"{vol/1_000_000:.1f}M" if vol > 1_000_000 else
                       f"{vol/1_000:.1f}K")
            msg += f"\n📊 {sym[:-4]}: ${price:,.0f} | Объём (1ч): {vol_str}"
        return msg
    except Exception as e:
        print(f"[SESSION_MONITOR][FETCH][ERROR] {e}")
        return ""

async def session_monitor_loop(send_message, edit_message, pin_message, config, client):
    if not config.get("ENABLE_SESSION_MONITOR", True):
        print("[SESSION_MONITOR] Отключено через конфиг")
        return

    interval = config.get("SESSION_MONITOR_INTERVAL", 60)
    pin_enabled = config.get("SESSION_PIN_MESSAGE", True)
    show_symbols = config.get("SHOW_SESSION_SYMBOL_STATUS", True)

    session_state = {"last_active_names": set(), "message_id": None}

    while True:
        try:
            now_utc = datetime.now(timezone.utc).time()
            active_sessions = []

            for session in SESSIONS:
                if time_in_range(session["start"], session["end"], now_utc):
                    session_copy = session.copy()
                    session_copy["range"] = format_kyiv_range(session["start"], session["end"])
                    active_sessions.append(session_copy)

            # Если состав активных сессий изменился — обновляем сообщение
            current_names = set(s["name"] for s in active_sessions)
            if current_names != session_state["last_active_names"]:
                session_state["last_active_names"] = current_names

                # if not active_sessions:
                #     msg = "❌ Сейчас ни одна крупная торговая сессия не активна."
                # else:
                #     msg = ""
                #     for sess in active_sessions:
                #         msg += f"🕒 <b>{sess['name']} торговая сессия</b>\n⏰ {sess['range']}\n\n"

                #     msg += "📍 Следи за рынком в часы активности."
                #     if show_symbols:
                #         msg += fetch_symbol_info(client)
                if not active_sessions:
                    msg = "❌ Сейчас ни одна крупная торговая сессия не активна."
                else:
                    msg = ""
                    for sess in active_sessions:
                        # Особое оформление для Brinks
                        if sess["name"] == "EU Brinks":
                            msg += f"⚠️ <b>{sess['name']} — всплеск ликвидности (высокая активность)</b>\n⏰ {sess['range']}\n\n"
                        elif sess["name"] == "US Brinks":
                            msg += f"⚠️ <b>{sess['name']} — расчёты и клиринг (повышенная волатильность)</b>\n⏰ {sess['range']}\n\n"
                        else:
                            msg += f"🕒 <b>{sess['name']} торговая сессия</b>\n⏰ {sess['range']}\n\n"

                    msg += "📍 Следи за рынком в часы активности."
                    if show_symbols:
                        msg += fetch_symbol_info(client)

                # --- ДОБАВЛЕНИЕ БЛОКА С КОШЕЛЬКАМИ ---
                if config.get("ENABLE_WALLETS", False):
                    wallets = config.get("WALLETS", {})
                    trc_wallet = wallets.get("USDT_TRC20")
                    ton_wallet_data = wallets.get("USDT_TON")

                    if trc_wallet and isinstance(ton_wallet_data, dict):
                        ton_address = ton_wallet_data.get("address")
                        ton_memo = ton_wallet_data.get("memo")

                        if ton_address and ton_memo:
                            msg += (
                                "\n\n💳 <b>Кошельки для донатов:</b>\n"
                                f"USDT (TRC-20): <code>{trc_wallet}</code>\n"
                                f"USDT (TON): <code>{ton_address}</code>\n"
                                f"🔖 Memo: <code>{ton_memo}</code>\n"
                                "❗ <i>Не забудьте указать Memo при переводе в TON!</i>\n"
                                "🙏 Спасибо за поддержку!"
                            )


                if session_state["message_id"] and pin_enabled:
                    try:
                        await edit_message(msg, session_state["message_id"])
                    except Exception as e:
                        print(f"[SESSION_MONITOR][EDIT][ERROR] {e}")
                        new_id = await send_message(msg)
                        if pin_enabled:
                            await pin_message(new_id)
                        session_state["message_id"] = new_id
                else:
                    new_id = await send_message(msg)
                    if pin_enabled:
                        await pin_message(new_id)
                    session_state["message_id"] = new_id

        except Exception as e:
            print(f"[SESSION_MONITOR][ERROR] {e}")

        await asyncio.sleep(interval)