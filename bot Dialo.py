import os
import time
import ccxt
import openai
import dotenv
import schedule
import pandas as pd
from transformers import pipeline, AutoTokenizer
from ta.trend import EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ccxt.base.errors import InsufficientFunds, ExchangeError

# Carga de variables de entorno
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Modelo local de fallback
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
tokenizer.pad_token = tokenizer.eos_token
local_model = pipeline('text-generation', model='microsoft/DialoGPT-small', tokenizer=tokenizer, device='cpu', truncation=True)

# Configuración de exchange en modo swap
exchange = getattr(ccxt, os.getenv("EXCHANGE", "bingx"))({
    'apiKey': os.getenv("EXCHANGE_API_KEY"),
    'secret': os.getenv("EXCHANGE_API_SECRET"),
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

# Detectar símbolo swap correcto
exchange.load_markets()
symbol = next(
    (s for s, m in exchange.markets.items() if m.get('swap') and m['base'] == 'BTC' and m['quote'] == 'USDT'),
    'BTC/USDT'
)
print(f"⚙️ Mercado swap: {symbol}")

# Estado de posición
current_position = None
position_size = 0.0
entry_price = None
stop_loss_price = None
take_profit_price = None

# DCA variables
max_dca_entries = 2
accumulation_threshold = 0.015  # 1.5%

# Parámetros
timeframes = {'1h': '1h', '4h': '4h', '1d': '1d'}
leverage = 7
capital_per_trade = 5

def fetch_ohlcv(tf, limit=100) -> pd.DataFrame:
    df = pd.DataFrame(exchange.fetch_ohlcv(symbol, tf, limit=limit), columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.set_index('timestamp')

def compute_indicators(df: pd.DataFrame) -> dict:
    h, l, c = df['high'], df['low'], df['close']
    ema50 = EMAIndicator(c, 50).ema_indicator().iloc[-1]
    sma20 = SMAIndicator(c, 20).sma_indicator().iloc[-1]
    rsi14 = RSIIndicator(c, 14).rsi().iloc[-1]
    atr14 = AverageTrueRange(high=h, low=l, close=c, window=14).average_true_range().iloc[-1]
    sh, sl = h.max(), l.min()
    fibs = {
        'fib236': sh - (sh - sl) * 0.236,
        'fib382': sh - (sh - sl) * 0.382,
        'fib50':  sh - (sh - sl) * 0.5,
        'fib618': sh - (sh - sl) * 0.618
    }
    return {'ema50': ema50, 'sma20': sma20, 'rsi14': rsi14, 'atr14': atr14, **fibs}

def print_swap_balance():
    bal = exchange.fetch_balance({'type':'swap'})
    t = bal['total'].get('USDT',0)
    f = bal['free'].get('USDT',0)
    print(f"💰 Total USDT: {t:.4f} | Libre: {f:.4f}")

def sync_position():
    global current_position, position_size, entry_price, stop_loss_price, take_profit_price
    try:
        pos = exchange.fetch_positions([symbol])[0]
        amt = float(pos.get('contracts', 0) or pos.get('positionAmt', 0))
        entry_price = float(pos.get('entryPrice', pos.get('info', {}).get('entryPrice',0)))
        if amt > 0:
            current_position = 'LONG'
            position_size = amt
        elif amt < 0:
            current_position = 'SHORT'
            position_size = abs(amt)
        else:
            current_position = None
        if current_position:
            df_h = fetch_ohlcv(timeframes['1h'])
            atr1h = compute_indicators(df_h)['atr14']
            sl_dist = atr1h * 1.5
            tp_dist = atr1h * 3.0
            if current_position == 'LONG':
                stop_loss_price = entry_price - sl_dist
                take_profit_price = entry_price + tp_dist
            else:
                stop_loss_price = entry_price + sl_dist
                take_profit_price = entry_price - tp_dist
    except Exception:
        current_position = None
        print("⚠️ Error sincronizando posición inicial")

def ask_model(prompt: str) -> str:
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{'role':'system','content':'Eres un bot de trading experto.'}, {'role':'user','content':prompt}]
        )
        text = resp.choices[0].message.content.upper()
    except Exception:
        print("🔄 Fallback local")
        text = local_model(prompt, max_new_tokens=50)[0]['generated_text'].upper()
    for o in ('LONG','SHORT','NEUTRAL'):
        if o in text:
            return o
    return 'NEUTRAL'

def build_prompt(all_closes: dict, all_inds: dict) -> str:
    lines = ["Analiza BTC/USDT en 1h,4h,1d con indicadores y Fibonacci:"]
    for tf in ('1h','4h','1d'):
        inds = all_inds[tf]
        closes = all_closes[tf][-5:]  # last 5 closes
        lines.append(f"-- {tf} -- Cierres: {closes} EMA50={inds['ema50']:.2f} RSI14={inds['rsi14']:.1f}")
    lines.append("Devuelve LONG, SHORT o NEUTRAL.")
    return "\n".join(lines)

def manage_position(price: float) -> bool:
    global current_position
    if current_position is None:
        return False
    if (current_position=='LONG' and (price>=take_profit_price or price<=stop_loss_price)) or \
       (current_position=='SHORT' and (price<=take_profit_price or price>=stop_loss_price)):
        side = 'sell' if current_position=='LONG' else 'buy'
        exchange.create_market_order(symbol, side, position_size)
        print(f"🔒 Cierre {current_position} SL/TP @{price:.2f}")
        current_position = None
        return True
    return False

def place_order(signal: str, price: float, atr: float) -> None:
    global current_position, position_size, entry_price, stop_loss_price, take_profit_price
    bal = exchange.fetch_balance({'type': 'swap'})
    max_exposure = bal['total'].get('USDT', 0) * 0.5

    # DCA logic
    dca_unit = (capital_per_trade * leverage) / price
    if signal == 'LONG' and current_position == 'LONG':
        price_drop = (entry_price - price) / entry_price
        current_dca_entries = int(round(position_size / dca_unit))
        if price_drop > accumulation_threshold and current_dca_entries < max_dca_entries and (position_size * price) < max_exposure:
            add_size = dca_unit
            try:
                exchange.create_market_buy_order(symbol, add_size)
                print(f"💹 Añadido LONG DCA @{price:.2f} extra={add_size:.6f}")
                prev_cost = entry_price * position_size
                position_size += add_size
                entry_price = (prev_cost + price * add_size) / position_size
                sl = atr * 1.5
                tp = atr * 3.0
                stop_loss_price = entry_price - sl
                take_profit_price = entry_price + tp
                print(f"🔁 SL DCA: {stop_loss_price:.2f} | TP DCA: {take_profit_price:.2f}")
            except InsufficientFunds:
                print("❌ Sin saldo para acumular LONG.")
            return

    if signal == 'SHORT' and current_position == 'SHORT':
        price_rise = (price - entry_price) / entry_price
        current_dca_entries = int(round(position_size / dca_unit))
        if price_rise > accumulation_threshold and current_dca_entries < max_dca_entries and (position_size * price) < max_exposure:
            add_size = dca_unit
            try:
                exchange.create_market_sell_order(symbol, add_size)
                print(f"💹 Añadido SHORT DCA @{price:.2f} extra={add_size:.6f}")
                prev_cost = entry_price * position_size
                position_size += add_size
                entry_price = (prev_cost + price * add_size) / position_size
                sl = atr * 1.5
                tp = atr * 3.0
                stop_loss_price = entry_price + sl
                take_profit_price = entry_price - tp
                print(f"🔁 SL DCA: {stop_loss_price:.2f} | TP DCA: {take_profit_price:.2f}")
            except InsufficientFunds:
                print("❌ Sin saldo para acumular SHORT.")
            return

    if manage_position(price):
        return
    if signal in ('LONG','SHORT') and current_position is None:
        position_size = (capital_per_trade * leverage) / price
        entry_price = price
        sl = atr * 1.5
        tp = atr * 3.0
        if signal=='LONG':
            stop_loss_price = price - sl
            take_profit_price = price + tp
            exchange.create_market_buy_order(symbol, position_size)
        else:
            stop_loss_price = price + sl
            take_profit_price = price - tp
            exchange.create_market_sell_order(symbol, position_size)
        print(f"🚀 Apertura {signal} @{price:.2f} size={position_size:.6f}")
        # Intenta poner SL/TP por orden
        try:
            exchange.create_order(symbol,'STOP_MARKET', 'sell' if signal=='LONG' else 'buy', position_size, None, {'stopPrice':stop_loss_price})
            exchange.create_order(symbol,'TAKE_PROFIT_MARKET','sell' if signal=='LONG' else 'buy', position_size, None, {'stopPrice':take_profit_price})
            print(f"🔒 SL @ {stop_loss_price:.2f} | 🎯 TP @ {take_profit_price:.2f}")
        except ExchangeError as e:
            print(f"⚠️ Error SL/TP: {e}")
        current_position = signal

def job() -> None:
    global stop_loss_price, take_profit_price
    print_swap_balance()
    sync_position()
    # Si posición abierta y no hay SL/TP colocados, colocarlos
    if current_position in ('LONG','SHORT') and (stop_loss_price is None or take_profit_price is None):
        df_h = fetch_ohlcv(timeframes['1h'])
        atr = compute_indicators(df_h)['atr14']
        sl_dist = atr * 1.5
        tp_dist = atr * 3.0
        if current_position == 'LONG':
            stop_loss_price_calc = entry_price - sl_dist
            take_profit_price_calc = entry_price + tp_dist
            stop_params = {'stopPrice': stop_loss_price_calc, 'positionSide': 'LONG'}
            tp_params = {'stopPrice': take_profit_price_calc, 'positionSide': 'LONG'}
            exchange.create_order(symbol, 'STOP_MARKET', 'sell', position_size, None, stop_params)
            exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell', position_size, None, tp_params)
            print(f"🔒 SL creado al reiniciar @ {stop_loss_price_calc:.2f} | 🎯 TP creado al reiniciar @ {take_profit_price_calc:.2f}")
        else:
            stop_loss_price_calc = entry_price + sl_dist
            take_profit_price_calc = entry_price - tp_dist
            stop_params = {'stopPrice': stop_loss_price_calc, 'positionSide': 'SHORT'}
            tp_params = {'stopPrice': take_profit_price_calc, 'positionSide': 'SHORT'}
            exchange.create_order(symbol, 'STOP_MARKET', 'buy', position_size, None, stop_params)
            exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'buy', position_size, None, tp_params)
            print(f"🔒 SL creado al reiniciar @ {stop_loss_price_calc:.2f} | 🎯 TP creado al reiniciar @ {take_profit_price_calc:.2f}")
        stop_loss_price = stop_loss_price_calc
        take_profit_price = take_profit_price_calc
    # Mostrar SL/TP de posición actual si existe
    if current_position in ('LONG','SHORT') and stop_loss_price and take_profit_price:
        print(f"🔒 SL actual: {stop_loss_price:.2f} | 🎯 TP actual: {take_profit_price:.2f}")
    all_closes, all_inds = {}, {}
    for tf in ('1h','4h','1d'):
        df = fetch_ohlcv(timeframes[tf])
        all_closes[tf] = df['close'].tolist()
        all_inds[tf] = compute_indicators(df)
    signal = ask_model(build_prompt(all_closes, all_inds))
    print(f"🗒️ Señal: {signal}")
    last_price = all_closes['1h'][-1]  # list, safe indexing
    place_order(signal, last_price, all_inds['1h']['atr14'])

# Scheduler: revisar cada 30 minutos
schedule.every(30).minutes.do(job)

if __name__ == '__main__':
    sync_position()
    print(f"🤖 Bot iniciado en {symbol}. Posición: {current_position}")
    try:
        while True:
            schedule.run_pending()
            time.sleep(5)
    except KeyboardInterrupt:
        print("⏹️ Bot detenido por usuario.")
        exit(0)
