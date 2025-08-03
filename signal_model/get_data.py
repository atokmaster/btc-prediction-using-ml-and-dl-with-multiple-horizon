import os
import requests
import pandas as pd
from datetime import datetime
import ta

# === Konfigurasi path output ===
output_dir = 'btc_realtime'
os.makedirs(output_dir, exist_ok=True)

# === Interval yang digunakan
intervals = {
    '1h': '1h',
    '4h': '4h',
    '1d': '1d'
}

# === Ambil data dari Binance API
def get_binance_data(symbol="BTCUSDT", interval="1h", limit=200):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"❌ Gagal ambil data {interval}: {response.text}")
        return None

    try:
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
        df = df.sort_values('datetime')
        return df

    except Exception as e:
        print(f"❌ Error parsing data Binance: {e}")
        return None

# === Hitung fitur teknikal
def add_technical_features(df):
    df = df.copy()

    df['ema12'] = df['close'].ewm(span=12).mean()
    df['ema26'] = df['close'].ewm(span=26).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema_diff'] = df['ema12'] - df['ema26']

    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df['close'])
    df['bbp'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    df['boll_pct_b'] = bb.bollinger_pband()

    df['volatility'] = df['close'].pct_change().rolling(10).std()
    df['roc'] = ta.momentum.ROCIndicator(df['close']).roc()
    df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()

    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    return df

# === Loop ambil dan simpan semua interval
for key, interval in intervals.items():
    try:
        df_raw = get_binance_data(interval=interval)
        if df_raw is None or df_raw.empty:
            raise ValueError(f"Data kosong untuk interval {key}")

        df_feat = add_technical_features(df_raw)
        if df_feat is None or df_feat.empty:
            raise ValueError(f"Fitur gagal dihitung untuk {key}")

        save_path = os.path.join(output_dir, f'btc_{key}_latest.xlsx')
        df_feat.to_excel(save_path, index=False)

        print(f"✅ {key.upper()} data saved: {save_path} (rows: {len(df_feat)})")

    except Exception as e:
        print(f"❌ Error untuk interval {key}: {e}")
