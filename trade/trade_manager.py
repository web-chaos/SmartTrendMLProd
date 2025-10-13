import asyncio
from datetime import datetime
from binance.client import Client

class TradeManager:
    def __init__(self, client: Client, config: dict, binance_trader, daily_stats: dict):
        self.client = client
        self.config = config
        self.trader = binance_trader
        self.daily_stats = daily_stats
        self.active_trades = {}
        
    async def can_open_trade(self, symbol: str):
        """Проверка возможности открытия сделки"""
        if not self.config["TRADING_ENABLED"]:
            return False, "Торговля отключена"
            
        if len(self.active_trades) >= self.config["MAX_ACTIVE_TRADES"]:
            return False, "Достигнут лимит активных сделок"
            
        if symbol in self.active_trades:
            return False, "Сделка уже активна"
            
        # Проверяем баланс
        account_info = await self.trader.get_account_info()
        if not account_info or account_info['available_balance'] < 10:  # Минимум 10 USDT
            return False, "Недостаточно средств"
            
        return True, "OK"
    
    async def open_trade(self, symbol: str, signal_data: dict, trade_data: dict):
        """Открытие сделки"""
        try:
            # Проверяем возможность открытия
            can_open, reason = await self.can_open_trade(symbol)
            if not can_open:
                return False, reason
            
            # Рассчитываем размер позиции
            quantity = self.trader.calculate_position_size(
                symbol, trade_data['stop'], trade_data['entry']
            )
            
            if not quantity:
                return False, "Ошибка расчета размера позиции"
            
            # Создаем ордера
            order_result = self.trader.create_order(
                symbol=symbol,
                side=signal_data['side'],
                quantity=quantity,
                entry_price=trade_data['entry'],
                stop_loss=trade_data['stop'],
                take_profits=trade_data['targets']
            )
            
            if not order_result:
                return False, "Ошибка создания ордера"
            
            # Сохраняем информацию о сделке
            trade_info = {
                'symbol': symbol,
                'side': signal_data['side'],
                'entry_price': order_result['entry_price'],
                'quantity': quantity,
                'stop_loss': trade_data['stop'],
                'take_profits': trade_data['targets'],
                'orders': order_result,
                'opened_at': datetime.now(),
                'status': 'open',
                'tp_hit': [False] * len(trade_data['targets'])
            }
            
            self.active_trades[symbol] = trade_info
            
            return True, "Сделка открыта"
            
        except Exception as e:
            return False, f"Ошибка открытия сделки: {str(e)}"
    
    async def monitor_trades(self):
        """Мониторинг активных сделок"""
        while True:
            try:
                for symbol, trade in list(self.active_trades.items()):
                    if trade['status'] != 'open':
                        continue
                    
                    # Получаем текущую цену
                    ticker = self.client.futures_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    
                    # Проверяем тейк-профиты
                    for i, (tp_price, tp_type) in enumerate(trade['take_profits']):
                        if not trade['tp_hit'][i]:
                            if (trade['side'] == 'long' and current_price >= tp_price) or \
                               (trade['side'] == 'short' and current_price <= tp_price):
                                trade['tp_hit'][i] = True
                                print(f"[TRADE] TP{i+1} достигнут для {symbol}")
                                
                                # Если это первый TP, закрываем часть позиции
                                if i == 0:
                                    await self.close_partial_position(symbol, f"tp{i+1}", close_percent=30)
                                
                                # Переносим стоп в безубыток после TP1
                                if i == 0 and self.config["BREAKEVEN_AFTER_TP1"]:
                                    await self.move_to_breakeven(symbol, trade)
                    
                    # Проверяем стоп-лосс
                    if (trade['side'] == 'long' and current_price <= trade['stop_loss']) or \
                       (trade['side'] == 'short' and current_price >= trade['stop_loss']):
                        await self.close_trade(symbol, "stop_loss")
                
                await asyncio.sleep(5)  # Проверяем каждые 5 секунд
                
            except Exception as e:
                print(f"[TRADE MONITOR] Ошибка: {e}")
                await asyncio.sleep(10)
    
    async def move_to_breakeven(self, symbol: str, trade: dict):
        """Перенос стопа в безубыток"""
        try:
            if trade['side'] == 'long':
                new_stop = trade['entry_price'] * 0.999  # Чуть ниже цены входа
            else:
                new_stop = trade['entry_price'] * 1.001  # Чуть выше цены входа
            
            await self.trader.modify_stop_loss(symbol, new_stop)
            trade['stop_loss'] = new_stop
            print(f"[TRADE] Стоп перенесен в безубыток для {symbol}")
            
        except Exception as e:
            print(f"[TRADE] Ошибка переноса стопа: {e}")
    
    async def close_partial_position(self, symbol: str, reason: str, close_percent: float):
        """Частичное закрытие позиции"""
        try:
            if symbol not in self.active_trades:
                return False
                
            trade = self.active_trades[symbol]
            quantity_to_close = trade['quantity'] * (close_percent / 100.0)
            
            # Закрываем часть позиции
            result = self.trader.close_partial_position(symbol, quantity_to_close)
            if result:
                # Обновляем статистику
                if reason.startswith("tp"):
                    tp_number = int(reason.replace("tp", ""))
                    self.daily_stats[f"tp{tp_number}_hit"] += 1
                
                print(f"[TRADE] Частично закрыта позиция {symbol}: {close_percent}% - {reason}")
                return True
                
        except Exception as e:
            print(f"[TRADE] Ошибка частичного закрытия: {e}")
            return False
    
    async def close_trade(self, symbol: str, reason: str):
        """Закрытие сделки"""
        try:
            result = self.trader.close_position(symbol, reason) 
            if result:
                self.active_trades[symbol]['status'] = 'closed'
                self.active_trades[symbol]['closed_at'] = datetime.now()
                self.active_trades[symbol]['close_reason'] = reason
                print(f"[TRADE] Сделка закрыта: {symbol} - {reason}")
                    
            return result
            
        except Exception as e:
            print(f"[TRADE] Ошибка закрытия сделки: {e}")
            return None
    
    def get_active_trades_count(self):
        """Количество активных сделок"""
        return len([t for t in self.active_trades.values() if t['status'] == 'open'])