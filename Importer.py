import ccxt
from datetime import datetime, timedelta
import time
import pandas as pd
from pathlib import Path
from ta.trend import PSARIndicator, EMAIndicator, CCIIndicator
from ta.momentum import StochasticOscillator, RSIIndicator, StochRSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.volatility import average_true_range
from ST_Config import *
from Mfun import *

def fetch_data(exchange='binance', cryptos=['BTC/USDT'], sample_freq='1m', since_hours=48, page_limit=1000, max_retries = 3):

    since = (datetime.today() - timedelta(hours=since_hours) - timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%S')
    print('Begin download...')

    for market_symbol in cryptos:

        # Select exchange
        exchange = getattr(ccxt, exchange)({'enableRateLimit': True, })

        # Convert since from string to milliseconds
        since = exchange.parse8601(since)

        # Preload all markets from the exchange
        exchange.load_markets()

        # Define page_limit in milliseconds
        earliest_timestamp = exchange.milliseconds()
        ms_timeframe = exchange.parse_timeframe(sample_freq) * 1000 # exchange.parse_timeframe parses to the equivalent in seconds of the timeframe we use
        t_delta = page_limit * ms_timeframe
        all_ohlcv = []
        num_retries = 0
        fetch_since = earliest_timestamp - t_delta

        while True:

            try:
                num_retries += 1
                ohlcv = exchange.fetch_ohlcv(market_symbol, sample_freq, fetch_since, page_limit)

            except Exception as e:
                print(str(e))
                time.sleep(5)
                print('Retrying...')
                if num_retries > max_retries:
                    print('Could not connect with exchange. Exiting...')
                    exit()
                continue

            earliest_timestamp = ohlcv[0][0]
            all_ohlcv = ohlcv + all_ohlcv
            # if we have reached the checkpoint
            if fetch_since < since:
                break
            fetch_since = earliest_timestamp - t_delta

        ohlcv = exchange.filter_by_since_limit(all_ohlcv, since, None, key=0)
        df = pd.DataFrame(ohlcv)

        if market_symbol == cryptos[0]:

            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], unit='ms')
            df.rename(columns={0: 'Datetime', 1: 'Open', 2: 'High', 3: 'Low', 4: 'Close', 5: 'Volume'}, inplace=True)
            df = df.set_index('Datetime')
            dfx = df.copy()

        else:

            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], unit='ms')
            df.rename(columns={0: 'Datetime', 1: 'Open', 2: 'High', 3: 'Low', 4: 'Close', 5: 'Volume'}, inplace=True)
            df = df.set_index('Datetime')
            dfx = pd.merge(dfx, df, on=['Datetime'])

    dfx = dfx.loc[:, ~dfx.columns.duplicated()]
    dfx = dfx[~dfx.index.duplicated(keep='first')]

    print(f'Finished')
    return dfx

def import_csv(loc_folder, filename):

    read_file = f'{loc_folder}/{filename}'
    df = pd.read_csv(read_file, index_col='Datetime', parse_dates=True)
    return df

def calculate_technical_indicators(df, w, w0, w1, w2, w3,resample_size):

    df = df.resample(resample_size).agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).fillna(
        method='ffill')

    # Calculate Stochastic Oscillator
    stochastic = StochasticOscillator(df["high"], df["low"], df["close"], window=w0)
    df["so"] = stochastic.stoch()

    # Calculate Percentage Volume Oscillator (PVO) manually
    ema_short = df["volume"].rolling(window=w1, min_periods=w).mean()
    ema_long = df["volume"].rolling(window=w0, min_periods=w).mean()
    pvo = (ema_short - ema_long) / ema_long
    df["pvo"] = pvo

    # Calculate Exponential Moving Average (EMA)
    ema = EMAIndicator(df["close"], window=w2)
    df['ema'] = (df['close'] - (ema.ema_indicator())) / df['close']

    # Calculate Stochastic RSI
    rsi = RSIIndicator(df["close"], window=w1)
    df['rsi'] = rsi.rsi()
    stoch_rsi = StochRSIIndicator(rsi.rsi(), window=w1, smooth1=w2, smooth2=w3)
    df["srsi"] = stoch_rsi.stochrsi()

    # Calculate Commodity Channel Index (CCI)
    cci = CCIIndicator(df["high"], df["low"], df["close"], window=w0)
    df["cci"] = cci.cci()

    # Calculate Parabolic SAR (PSAR)
    df['psar'] = (df['close'] - (get_psar(df, iaf=0.0002, maxaf=0.2))) / df['close']

    # Calculate the VWAP
    df['vwap'] = (df['close'] - ((df['volume'] * df['close']).cumsum() / df['volume'].cumsum())) / df['close']

    return df

def Import(name='candles_t.csv', cryptos=['BTC/TUSD'], sample_freq='1s', since=48):
    print(f'{cryptos[0]}')
    dfx = fetch_data(exchange='binance', cryptos=cryptos, sample_freq=sample_freq, since_hours=since, page_limit=1000)
    dfx.to_csv(name)
    print(f'{cryptos[0]} Historical Length: {len(dfx)}')


'''
The following functio will import historical prices from binance based on the sample_freq. The parameter "since" 
determines the look back period expressed in hours and the parameter "cryptos" is the trading pair we are looking for.
1 week = 168 hours
1 month = 732 hours
1 semester = 4380 hours
1 year = 8760 hours

'''

import_name = f'{import_pair.replace("/", "")}-{import_sample}-Price Data-{import_desc}.csv'
Import(name=import_name, cryptos=[import_pair], sample_freq=f'{import_sample}', since=import_since)
