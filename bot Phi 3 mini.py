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

# === CONFIGURACIÓN ===

# 1) .env con tus claves
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 2) Modelo local: Phi-3 Mini
HF_MODEL = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
tokenizer.pad_token = tokenizer.eos_token
local_model = pipeline('text-generation', model=HF_MODEL, tokenizer=tokenizer, device='cpu', truncation=True)

# 3) Exchange BingX (swap/futuros)
exchange = getattr(ccxt, os.getenv("EXCHANGE", "bingx"))({
    'apiKey': os.getenv("EXCHANGE_API_KEY"),
    'secret': os.getenv("EXCHANGE_API_SECRET"),
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

exchange.load_markets()
symbol = next(
    (s for s, m in exchange.markets.items() if m.get('swap') and m['base'] == 'BTC' and m['quote'] == 'USDT'),
    'BTC/USDT'
)
print(f"⚙️ Mercado swap: {symbol}")

# === ESTADO ===
current_position = None
position_size = 0.0
entry_price = None
stop_loss_price = None
take_profit_price = None

# Parámetros
timeframes = {'1h': '1h', '4h': '4h', '1d': '1d'}
leverage = 7
capital_per_trade = 5
DCA_LIMIT = 3           # Máximo 3 compras DCA
DCA_DROP_PCT = 1.5      # Añade DCA si el precio se mueve > 1.5% en tu contra
dca_count = 0           # Nº de entradas DCA realizadas

# === FUNCIONES AUXILIARES ===

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
    global current_position, position_size, entry_price, stop_loss_price, take_profit_price, dca_count
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
        dca_count = 0
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
        print("🔄 Fallback local (Phi-3 Mini)")
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

def dca_logic(price):
    global entry_price, dca_count, position_size
    if current_position == 'LONG' and dca_count < DCA_LIMIT:
        drop_pct = (entry_price - price) / entry_price * 100
        if drop_pct >= DCA_DROP_PCT:
            dca_size = (capital_per_trade * leverage) / price
            try:
                exchange.create_market_buy_order(symbol, dca_size)
                position_size += dca_size
                dca_count += 1
                entry_price = (entry_price * dca_count + price) / (dca_count + 1)
                print(f"🔁 DCA LONG: añadido {dca_size:.6f}@{price:.2f} (Total entradas: {dca_count+1})")
            except (InsufficientFunds, ExchangeError) as e:
                print(f"❌ Error en DCA LONG: {e}")
    if current_position == 'SHORT' and dca_count < DCA_LIMIT:
        up_pct = (price - entry_price) / entry_price * 100
        if up_pct >= DCA_DROP_PCT:
            dca_size = (capital_per_trade * leverage) / price
            try:
                exchange.create_market_sell_order(symbol, dca_size)
                position_size += dca_size
                dca_count += 1
                entry_price = (entry_price * dca_count + price) / (dca_count + 1)
                print(f"🔁 DCA SHORT: añadido {dca_size:.6f}@{price:.2f} (Total entradas: {dca_count+1})")
            except (InsufficientFunds, ExchangeError) as e:
                print(f"❌ Error en DCA SHORT: {e}")

def place_order(signal: str, price: float, atr: float) -> None:
    global current_position, position_size, entry_price, stop_loss_price, take_profit_price
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
        exchange.create_order(symbol,'STOP_MARKET', 'sell' if signal=='LONG' else 'buy', position_size, None, {'stopPrice':stop_loss_price})
        exchange.create_order(symbol,'TAKE_PROFIT_MARKET','sell' if signal=='LONG' else 'buy', position_size, None, {'stopPrice':take_profit_price})
        print(f"🔒 SL @ {stop_loss_price:.2f} | 🎯 TP @ {take_profit_price:.2f}")
        current_position = signal

def job() -> None:
    global stop_loss_price, take_profit_price
    print_swap_balance()
    sync_position()
    # Si hay posición abierta y no hay SL/TP, ponlos
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
    last_price = all_closes['1h'][-1]  # Último precio 1h
    place_order(signal, last_price, all_inds['1h']['atr14'])
    # Lógica DCA para ambas direcciones
    dca_logic(last_price)

# === SCHEDULER: cada 30 minutos ===
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
