import time
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import math

class BinanceTrader:
    def __init__(self, client: Client, config: dict, **kwargs):
        self.client = client
        self.config = config
        self.active_orders = {}
        self.daily_stats = kwargs.get('daily_stats')
        self.on_trade_update = kwargs.get('on_trade_update')

    def get_account_info(self):
        """Получение информации о счете (синхронно)."""
        try:
            account = self.client.futures_account()
            balance = next(item for item in account['assets'] if item['asset'] == 'USDT')
            positions = account['positions']
            return {
                'balance': float(balance.get('walletBalance', 0.0)),
                'available_balance': float(balance.get('availableBalance', 0.0)),
                'positions': positions
            }
        except BinanceAPIException as e:
            print(f"[TRADING][API] Ошибка получения данных счета: code={getattr(e, 'code', '')} msg={getattr(e, 'message', e)}")
            return None
        except Exception as e:
            print(f"[TRADING] Ошибка получения данных счета: {e}")
            return None

    def calculate_position_size(self, symbol: str, stop_price: float, entry_price: float):
        """Расчет размера позиции (возвращает quantity в лотах)."""
        try:
            account_info = self.get_account_info()
            if not account_info:
                return None

            balance = account_info['available_balance']

            if self.config.get("POSITION_SIZE_TYPE", "percentage") == "percentage":
                position_value = balance * (self.config.get("POSITION_SIZE_PERCENTAGE", 1.0) / 100.0)
            else:
                position_value = self.config.get("POSITION_SIZE_FIXED", 50.0)

            # Учитываем риск на сделку
            risk_amount = balance * (self.config.get("RISK_PER_TRADE", 1.0) / 100.0)
            price_diff = abs(entry_price - stop_price)
            # защитный кейс: если price_diff слишком мал - не делим на ноль
            risk_position_size = (risk_amount / price_diff) if price_diff > 0 else float('inf')

            # Берем минимальный размер из двух подходов (в USD)
            position_size_usd = min(position_value, risk_position_size)

            # Получаем информацию о символе для точного расчета
            symbol_info = self.client.futures_exchange_info()
            for s in symbol_info.get('symbols', []):
                if s['symbol'] == symbol:
                    lot_size_filter = next((f for f in s['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                    if not lot_size_filter:
                        print(f"[TRADING] Не найден LOT_SIZE для {symbol}")
                        return None

                    step_size = float(lot_size_filter['stepSize'])
                    min_qty = float(lot_size_filter['minQty'])

                    # Рассчитываем количество контрактов (quantity)
                    raw_qty = position_size_usd / entry_price if entry_price > 0 else 0.0

                    # Округляем вниз к шагу
                    # чтобы избежать ошибок с float precision, используем деление и floor
                    qty_steps = math.floor(raw_qty / step_size)
                    quantity = qty_steps * step_size

                    # Защита: не ниже min_qty
                    if quantity < min_qty:
                        if raw_qty >= min_qty:
                            quantity = min_qty
                        else:
                            # позиция слишком мала
                            print(f"[TRADING] Вычисленный объём меньше min_qty ({quantity} < {min_qty}), отклонено.")
                            return None

                    # округляем до подходящего количества знаков, соответствующих step_size
                    # вычислим количество десятичных знаков
                    decimals = max(0, int(round(-math.log10(step_size)))) if step_size < 1 else 0
                    quantity = round(quantity, decimals)

                    return quantity

            print(f"[TRADING] Информация по символу {symbol} не найдена в exchange_info.")
            return None

        except BinanceAPIException as e:
            print(f"[TRADING][API] Ошибка расчёта размера позиции: code={getattr(e, 'code', '')} msg={getattr(e, 'message', e)}")
            return None
        except Exception as e:
            print(f"[TRADING] Ошибка расчета размера позиции: {e}")
            return None

    def set_leverage(self, symbol: str, leverage: int):
        """Установка плеча (с учётом MAX_LEVERAGE)."""
        try:
            leverage_to_set = int(min(leverage, self.config.get("MAX_LEVERAGE", leverage)))
            res = self.client.futures_change_leverage(symbol=symbol, leverage=leverage_to_set)
            print(f"[TRADING] Плечо установлено: {symbol} {leverage_to_set}x -> {res}")
            return True
        except BinanceAPIException as e:
            print(f"[TRADING][API] Ошибка установки плеча для {symbol}: code={getattr(e, 'code', '')} msg={getattr(e, 'message', e)}")
            return False
        except Exception as e:
            print(f"[TRADING] Ошибка установки плеча: {e}")
            return False

    def set_margin_type(self, symbol: str):
        """Установка типа маржи (ISOLATED / CROSS)."""
        try:
            margin_type = self.config.get("MARGIN_TYPE", "ISOLATED")
            # Binance может вернуть ошибку если тип уже тот же — ловим и игнорируем
            res = self.client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
            print(f"[TRADING] MarginType установлен: {symbol} {margin_type} -> {res}")
            return True
        except BinanceAPIException as e:
            # если ошибка 'No need to change margin type.' или подобная — можно игнорировать
            print(f"[TRADING][API] Ошибка установки margin type для {symbol}: code={getattr(e, 'code', '')} msg={getattr(e, 'message', e)}")
            return False
        except Exception as e:
            print(f"[TRADING] Ошибка установки margin type: {e}")
            return False

    def create_order(self, symbol: str, side: str, quantity: float,
                     entry_price: float, stop_loss: float, take_profits: list):
        """
        Создание позиции: main order, SL, TP(ы).
        - side: 'long' или 'short'
        - take_profits: список кортежей (tp_price, note) или [(tp_price, ...), ...]
        """
        try:
            if quantity is None or quantity <= 0:
                print("[TRADING] Неверный quantity, отмена создания ордера.")
                return None

            # Устанавливаем маржу и плечо
            self.set_margin_type(symbol)
            leverage = self.config.get("LEVERAGE", 1)
            self.set_leverage(symbol, leverage)

            bin_side = 'BUY' if side == 'long' else 'SELL'
            opposite_side = 'SELL' if bin_side == 'BUY' else 'BUY'

            # Создаём основную позицию
            if self.config.get("USE_MARKET_ORDERS", True):
                try:
                    order = self.client.futures_create_order(
                        symbol=symbol,
                        side=bin_side,
                        type='MARKET',
                        quantity=quantity
                    )
                except BinanceAPIException as e:
                    print(f"[TRADING][API] Ошибка создания MARKET ордера: code={getattr(e,'code','')} msg={getattr(e,'message',e)}")
                    return None
            else:
                try:
                    order = self.client.futures_create_order(
                        symbol=symbol,
                        side=bin_side,
                        type='LIMIT',
                        timeInForce='GTC',
                        quantity=quantity,
                        price=str(entry_price)
                    )
                except BinanceAPIException as e:
                    print(f"[TRADING][API] Ошибка создания LIMIT ордера: code={getattr(e,'code','')} msg={getattr(e,'message',e)}")
                    return None

            # Получаем цену исполнения: предпочтительно из позиции
            avg_price = None
            try:
                # Небольшая пауза, чтобы биржа обновила позицию
                time.sleep(0.2)
                pos_info = self.client.futures_position_information(symbol=symbol)
                pos = next((p for p in pos_info if float(p.get('positionAmt', 0)) != 0), None)
                if pos:
                    avg_price = float(pos.get('entryPrice', 0.0)) if pos.get('entryPrice') and float(pos.get('entryPrice', 0.0)) > 0 else None
            except Exception:
                avg_price = None

            # fallback — попытка взять среднюю цену из fills (если есть)
            if not avg_price:
                fills = order.get('fills', []) if isinstance(order, dict) else []
                if fills:
                    qty_sum = sum(float(f['qty']) for f in fills)
                    if qty_sum > 0:
                        avg_price = sum(float(f['price']) * float(f['qty']) for f in fills) / qty_sum

            # если всё ещё нет — используем переданный entry_price
            if not avg_price:
                avg_price = entry_price

            # Создаём стоп-лосс: сначала отменим старые стопы для позиции (чтобы не было конфликтов)
            try:
                open_orders = self.client.futures_get_open_orders(symbol=symbol)
                for o in open_orders:
                    # отменяем только стоповые ордера, которые относятся к закрытию позиции
                    if o.get('type') in ('STOP_MARKET', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT_MARKET'):
                        try:
                            self.client.futures_cancel_order(symbol=symbol, orderId=o['orderId'])
                        except Exception:
                            pass
            except Exception:
                pass

            # Для STOP_MARKET обычно достаточно параметра closePosition=True (без quantity)
            try:
                sl_payload = {
                    'symbol': symbol,
                    'side': opposite_side,
                    'type': 'STOP_MARKET',
                    'stopPrice': str(round(stop_loss, 6)),
                    'closePosition': True
                }
                sl_order = self.client.futures_create_order(**sl_payload)
            except BinanceAPIException as e:
                print(f"[TRADING][API] Ошибка создания SL: code={getattr(e,'code','')} msg={getattr(e,'message',e)}")
                sl_order = None
            except Exception as e:
                print(f"[TRADING] Ошибка создания SL: {e}")
                sl_order = None

            # Создаем тейк-профиты (частичные)
            tp_orders = []
            if self.config.get("PARTIAL_TAKE_PROFIT", True) and take_profits:
                close_percents = self.config.get("PARTIAL_CLOSE_PERCENTS", [])
                for i, tp_item in enumerate(take_profits):
                    tp_price = tp_item[0] if isinstance(tp_item, (list, tuple)) else tp_item
                    if i < len(close_percents):
                        tp_qty = round(quantity * (close_percents[i] / 100.0), 8)
                        if tp_qty <= 0:
                            continue
                        try:
                            tp_payload = {
                                'symbol': symbol,
                                'side': opposite_side,
                                'type': 'TAKE_PROFIT_MARKET',
                                'stopPrice': str(round(tp_price, 6)),
                                'quantity': tp_qty
                            }
                            tp_order = self.client.futures_create_order(**tp_payload)
                            tp_orders.append(tp_order)
                        except BinanceAPIException as e:
                            print(f"[TRADING][API] Ошибка создания TP #{i+1}: code={getattr(e,'code','')} msg={getattr(e,'message',e)}")
                        except Exception as e:
                            print(f"[TRADING] Ошибка создания TP #{i+1}: {e}")
            else:
                # Один тейк на всю позицию (если указан take_profits)
                if take_profits:
                    tp_price = take_profits[-1][0] if isinstance(take_profits[-1], (list, tuple)) else take_profits[-1]
                    try:
                        tp_payload = {
                            'symbol': symbol,
                            'side': opposite_side,
                            'type': 'TAKE_PROFIT_MARKET',
                            'stopPrice': str(round(tp_price, 6)),
                            'quantity': quantity
                        }
                        tp_order = self.client.futures_create_order(**tp_payload)
                        tp_orders.append(tp_order)
                    except BinanceAPIException as e:
                        print(f"[TRADING][API] Ошибка создания TP: code={getattr(e,'code','')} msg={getattr(e,'message',e)}")
                    except Exception as e:
                        print(f"[TRADING] Ошибка создания TP: {e}")

            result = {
                'main_order': order,
                'sl_order': sl_order,
                'tp_orders': tp_orders,
                'entry_price': avg_price,
                'quantity': quantity
            }

            # Сохраняем в active_orders (по symbol) — поверхностно
            self.active_orders[symbol] = result

            return result

        except BinanceAPIException as e:
            print(f"[TRADING][API] Ошибка создания ордера: code={getattr(e,'code','')} msg={getattr(e,'message',e)}")
            return None
        except Exception as e:
            print(f"[TRADING] Ошибка создания ордера: {e}")
            return None

    def close_position(self, symbol: str):
        """Закрытие позиции (определяем сторону автоматически)."""
        try:
            position_info = self.client.futures_position_information(symbol=symbol)
            position = next((p for p in position_info if float(p.get('positionAmt', 0)) != 0), None)
            if not position:
                print(f"[TRADING] Позиция для {symbol} не найдена.")
                return None

            side = 'SELL' if float(position.get('positionAmt')) > 0 else 'BUY'
            try:
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    closePosition=True
                )
                # удаляем из active_orders если есть
                if symbol in self.active_orders:
                    del self.active_orders[symbol]
                return order
            except BinanceAPIException as e:
                print(f"[TRADING][API] Ошибка закрытия позиции: code={getattr(e,'code','')} msg={getattr(e,'message',e)}")
                return None
        except Exception as e:
            print(f"[TRADING] Ошибка закрытия позиции: {e}")
            return None

    def close_partial_position(self, symbol: str, quantity: float):
        """Частичное закрытие позиции"""
        try:
            position_info = self.client.futures_position_information(symbol=symbol)
            position = next((p for p in position_info if float(p.get('positionAmt', 0)) != 0), None)
            if not position:
                print(f"[TRADING] Позиция для {symbol} не найдена.")
                return None

            # Определяем сторону для закрытия
            current_position = float(position.get('positionAmt'))
            if current_position > 0:  # long позиция
                side = 'SELL'
                close_quantity = min(quantity, current_position)
            else:  # short позиция  
                side = 'BUY'
                close_quantity = min(quantity, abs(current_position))

            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=close_quantity
            )
            
            return order
            
        except Exception as e:
            print(f"[TRADING] Ошибка частичного закрытия: {e}")
            return None

    def modify_stop_loss(self, symbol: str, new_stop_price: float):
        """Изменение стоп-лосса: отменяем старые STOP_MARKET и ставим новый."""
        try:
            # Отменяем старые стоп-лоссы
            try:
                open_orders = self.client.futures_get_open_orders(symbol=symbol)
                for order in open_orders:
                    if order.get('type') == 'STOP_MARKET':
                        try:
                            self.client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                        except Exception:
                            pass
            except Exception:
                pass

            position_info = self.client.futures_position_information(symbol=symbol)
            position = next((p for p in position_info if float(p.get('positionAmt', 0)) != 0), None)

            if position:
                quantity = abs(float(position.get('positionAmt')))
                side = 'SELL' if float(position.get('positionAmt')) > 0 else 'BUY'
                # для closePosition=True quantity можно не указывать
                try:
                    new_sl_order = self.client.futures_create_order(
                        symbol=symbol,
                        side=side,
                        type='STOP_MARKET',
                        stopPrice=str(round(new_stop_price, 6)),
                        closePosition=True
                    )
                    return new_sl_order
                except BinanceAPIException as e:
                    print(f"[TRADING][API] Ошибка создания нового SL: code={getattr(e,'code','')} msg={getattr(e,'message',e)}")
                    return None
                except Exception as e:
                    print(f"[TRADING] Ошибка создания нового SL: {e}")
                    return None

            print(f"[TRADING] Нет открытой позиции для {symbol}, SL не изменён.")
            return None

        except Exception as e:
            print(f"[TRADING] Ошибка изменения стоп-лосса: {e}")
            return None