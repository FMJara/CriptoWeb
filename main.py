from flask import Flask, jsonify, send_from_directory, request
import pandas as pd
import numpy as np
import time
import ccxt

app = Flask(__name__)

# Configuración de símbolos y timeframes
CRYPTOS = ["xrp", "xlm", "hbar", "dovu", "xdc", "shx", "velo", "xdb", "xpl", "doge", "zbcn", "paw", "dag", "xpr", "qubic"]
SYMBOL_MAP = {
    "xrp": "XRP/USDT", "xlm": "XLM/USDT", "hbar": "HBAR/USDT", "dovu": "DOVU/USDT",
    "xdc": "XDC/USDT", "shx": "SHX/USDT", "velo": "VELO/USDT", "xdb": "XDB/USDT",
    "xpl": "XPL/USDT", "doge": "DOGE/USDT", "zbcn": "ZBCN/USDT", "paw": "PAW/USDT",
    "dag": "DAG/USDT", "xpr": "XPR/USDT", "qubic": "QUBIC/USDT"
}
VALID_TIMEFRAMES = ["1d", "1h"]

# Exchange MEXC vía ccxt
exchange = ccxt.mexc({
    'enableRateLimit': True,
    'timeout': 10000,
    'options': {'adjustForTimeDifference': True}
})

# -----------------------------
# Utilidades de datos e indicadores
# -----------------------------

def fetch_ohlcv_paged(symbol, timeframe="1h", target=1000, max_retries=3):
    """Paginación automática para traer más de 1000 velas."""
    all_data = []
    ms_per_candle = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - target * ms_per_candle

    while len(all_data) < target:
        batch = []
        for attempt in range(max_retries):
            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=1000, since=since)
                break
            except Exception:
                time.sleep(2)
                batch = []
        if not batch:
            break
        all_data.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000:
            break
    return all_data[-target:]

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
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    vwap = cum_tp_vol / cum_vol
    return vwap.fillna(method='ffill').fillna(method='bfill')

def sma_indicator(series, window):
    return series.rolling(window).mean()

def rsi(series, window):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=series.index).rolling(window).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return pd.Series(rsi_val, index=series.index).fillna(50)

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

# -----------------------------
# Rutas Flask
# -----------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/<symbol>_data")
def serve_data(symbol):
    symbol = symbol.lower()
    timeframe = request.args.get("timeframe", "1h")
    if timeframe not in VALID_TIMEFRAMES:
        timeframe = "1h"
    if symbol not in SYMBOL_MAP:
        return jsonify({"data": [], "alerts": [], "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")})

    # Objetivos de histórico fijo
    if timeframe == "1h":
        target = 24 * 180   # 6 meses ≈ 4320 velas
    else:
        target = 365 * 6    # 6 años ≈ 2190 velas

    raw = fetch_ohlcv_paged(SYMBOL_MAP[symbol], timeframe=timeframe, target=target)
    if not raw:
        return jsonify({"data": [], "alerts": [], "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")})

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = validate_data(df)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    if len(df) < 50:
        return jsonify({"data": [], "alerts": [], "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")})

    # Indicadores
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

    # Escalados para compatibilidad con frontend
    df['atr_scaled'] = df['atr'] * 10
    df['ewo_scaled'] = df['ewo'] * 100

    df.fillna(0, inplace=True)
    df = df.round(6)

    # Alertas
    signals = generate_alert(df)

    # Respuesta en el formato que espera tu index.html
    payload = df[[
        "timestamp", "close", "tenkan", "kijun", "senkou_a", "senkou_b",
        "vwap", "sma_50", "sma_200", "rsi", "atr_scaled", "ewo_scaled"
    ]].rename(columns={
        "atr_scaled": "atr",
        "ewo_scaled": "ewo"
    }).to_dict(orient="records")

    return jsonify({
        "data": payload,
        "alerts": signals,
        "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    # En Render, host 0.0.0.0 y puerto 5000 funcionan bien en free
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
