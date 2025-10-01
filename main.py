import os
from flask import Flask, jsonify, send_from_directory
import pandas as pd
import numpy as np
import time
import ccxt

app = Flask(__name__)

CRYPTOS = ["xrp", "xlm", "hbar", "dovu", "xdc", "shx", "velo", "xdb", "xpl", "doge", "zbcn", "paw", "dag", "xpr", "qubic"]
SYMBOL_MAP = {
    "xrp": "XRP/USDT", "xlm": "XLM/USDT", "hbar": "HBAR/USDT", "dovu": "DOVU/USDT",
    "xdc": "XDC/USDT", "shx": "SHX/USDT", "velo": "VELO/USDT", "xdb": "XDB/USDT",
    "xpl": "XPL/USDT", "doge": "DOGE/USDT", "zbcn": "ZBCN/USDT", "paw": "PAW/USDT",
    "dag": "DAG/USDT", "xpr": "XPR/USDT", "qubic": "QUBIC/USDT"
}

exchange = ccxt.mexc({
    'enableRateLimit': True,
    'timeout': 10000,
    'options': {'adjustForTimeDifference': True}
})

def fetch_ohlcv_safe(symbol, timeframe='1d', limit=500, max_retries=3):
    for attempt in range(max_retries):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception:
            time.sleep(3)
    return []

def validate_data(df):
    df.dropna(inplace=True)
    df = df[df['close'] > 0]
    return df

def calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def ichimoku_conversion_line(high, low, period=9):
    return (high.rolling(period).max() + low.rolling(period).min()) / 2

def ichimoku_base_line(high, low, period=26):
    return (high.rolling(period).max() + low.rolling(period).min()) / 2

def volume_weighted_average_price(high, low, close, volume):
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()

def sma_indicator(series, window):
    return series.rolling(window).mean()

def rsi(series, window):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window).mean()
    avg_loss = pd.Series(loss).rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def generate_alert(df):
    last = df.iloc[-1]
    alerts = []

    if last['tenkan'] > last['kijun'] and last['rsi'] > 50 and last['volume'] > last['volume_ma']:
        alerts.append("🚀 Cruce alcista Tenkan/Kijun con volumen fuerte y RSI positivo")
    if last['tenkan'] < last['kijun'] and last['rsi'] < 50:
        alerts.append("📉 Cruce bajista Tenkan/Kijun con debilidad de momentum")
    if last['close'] > last['senkou_a'] and last['close'] > last['senkou_b']:
        alerts.append("📈 Precio sobre la nube: tendencia alcista confirmada")
    if last['close'] < last['senkou_a'] and last['close'] < last['senkou_b']:
        alerts.append("⚠️ Precio bajo la nube: tendencia bajista")
    if last['senkou_a'] > last['senkou_b']:
        alerts.append("🌤️ Nube proyectada alcista")
    if last['senkou_a'] < last['senkou_b']:
        alerts.append("🌧️ Nube proyectada bajista")
    if last['rsi'] > 70:
        alerts.append("🔺 RSI en sobrecompra, posible corrección")
    if last['rsi'] < 30:
        alerts.append("🔻 RSI en sobreventa, posible rebote")
    if last['atr'] > df['atr'].rolling(20).mean().iloc[-1]:
        alerts.append("🔥 Volatilidad elevada")

    return alerts

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/<symbol>_data")
def serve_data(symbol):
    symbol = symbol.lower()
    if symbol not in SYMBOL_MAP:
        return jsonify([])

    raw = fetch_ohlcv_safe(SYMBOL_MAP[symbol])
    if not raw:
        return jsonify([])

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = validate_data(df)

    if len(df) < 50:
        return jsonify([])

    df['tenkan'] = ichimoku_conversion_line(df['high'], df['low'], 9)
    df['kijun'] = ichimoku_base_line(df['high'], df['low'], 26)
    df['senkou_a'] = df['tenkan'].shift(26)
    df['senkou_b'] = df['kijun'].shift(26)
    df['cloud'] = df['senkou_a'] > df['senkou_b']
    df['vwap'] = volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(5).std()
    df['volume_ma'] = df['volume'].rolling(5).mean()
    df['sma_50'] = sma_indicator(df['close'], 50)
    df['sma_200'] = sma_indicator(df['close'], 200)
    df['atr'] = calculate_atr(df, 14)
    df['rsi'] = rsi(df['close'], 14)
    df['ewo'] = df['close'].rolling(5).mean() - df['close'].rolling(35).mean()

    df.fillna(0, inplace=True)
    df = df.round(6)

    signals = generate_alert(df)

    return jsonify({
        "data": df.to_dict(orient="records"),
        "alerts": signals,
        "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
