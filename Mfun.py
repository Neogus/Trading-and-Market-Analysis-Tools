from datetime import datetime, timedelta
import math
import random
import ccxt
import pandas as pd
import numpy as np
import time
from ta.volatility import AverageTrueRange, BollingerBands, DonchianChannel, KeltnerChannel, UlcerIndex
from ta.momentum import RSIIndicator, StochRSIIndicator, TSIIndicator, UltimateOscillator, StochasticOscillator, \
    KAMAIndicator, ROCIndicator, AwesomeOscillatorIndicator, WilliamsRIndicator, PercentagePriceOscillator, \
    PercentageVolumeOscillator
from ta.trend import MACD, ADXIndicator, AroonIndicator, CCIIndicator, DPOIndicator, IchimokuIndicator, \
    KSTIndicator, MassIndex, STCIndicator, TRIXIndicator, VortexIndicator, PSARIndicator
from ta.volume import AccDistIndexIndicator, ChaikinMoneyFlowIndicator, EaseOfMovementIndicator, ForceIndexIndicator, \
    MFIIndicator, OnBalanceVolumeIndicator, VolumePriceTrendIndicator, NegativeVolumeIndexIndicator
from ST_Config import *

ic_index = ['uo_1', 'uo_2', 'uo_3', 'uo_s', 'uo_b',
                     'so_w', 'so_sw', 'so_s', 'so_b', 'ao_1',
                     'ao_2', 'wi_w', 'wi_s', 'wi_b', 'srsi_w', 'srsi_kw', 'srsi_dw', 'srsi_s', 'srsi_b', 'po_sw',
                     'po_fw', 'po_sg', 'pvo_sw', 'pvo_fw', 'pvo_sg', 'ai_w', 'macd_fw', 'macd_sw', 'macd_sg', 'cci_w',
                     'cci_b',
                     'cci_s', 'ii_1', 'ii_2', 'ii_3', 'kst_r1', 'kst_r2', 'kst_r3', 'kst_r4', 'kst_1',
                     'kst_2', 'kst_3', 'kst_4', 'kst_ns', 'psar_st', 'psar_ms', 'stc_fw', 'stc_sw',
                     'stc_c', 'stc_s1', 'stc_s2', 'vi_w', 'eom_w', 'sma_w', 'sma_fw', 'ema_w', 'ema_fw', 'wma_w',
                     'wma_fw', 'kst_b',
                     'kst_s', 'stc_ll', 'stc_hl', 'adi_fw', 'adi_sw', 'adi_sg', 'obv_fw', 'obv_sw', 'obv_sg', 'eom_sma',
                     'vpt_sma',
                     'vi2_w', 'obv2_fw', 'obv2_sw', 'obv2_sg', 'ao2_1', 'ao2_2', 'macd2_fw', 'macd2_sw',
                     'macd2_sg', 'po2_sw', 'po2_fw', 'po2_sg', 'pvo2_sw', 'pvo2_fw', 'pvo2_sg', 'psar2_st', 'psar2_ms',
                     'stc2_fw',
                     'stc2_sw', 'stc2_c', 'stc2_s1', 'stc2_s2', 'stc2_ll', 'stc2_hl', 'adi2_fw', 'adi2_sw', 'adi2_sg',
                     'sma2_w',
                     'sma2_fw', 'ema2_w', 'ema2_fw', 'wma2_w', 'wma2_fw']

c_index = ['Criteria', 'Score', 'Alpha', 'Alpha Trade Rate', 'Alpha Max Days', 'Max Min Delta', 'Beta', 'Transactions', 'Accuracy', 'Av. Acc', 'Av.Acc.Error',
           'Final Equity','Market Return', 'Zone', 'Signal'] + ic_index


dfa_types = {'uo_1': int, 'uo_2': int, 'uo_3': int,
             'uo_s': int, 'uo_b': int, 'so_w': int, 'so_sw': int, 'so_s': int, 'so_b': int,
             'ao_1': int, 'ao_2': int, 'wi_w': int, 'wi_s': int, 'wi_b': int,
             'srsi_w': int, 'srsi_kw': int, 'srsi_dw': int, 'po_sw': int, 'po_fw': int,
             'po_sg': int, 'pvo_sw': int, 'pvo_fw': int, 'pvo_sg': int, 'ai_w': int, 'macd_fw': int,
             'macd_sw': int, 'macd_sg': int, 'cci_w': int, 'cci_b': int, 'cci_s': int, 'ii_1': int,
             'ii_2': int, 'ii_3': int, 'kst_r1': int, 'kst_r2': int, 'kst_r3': int, 'kst_r4': int,
             'kst_1': int, 'kst_2': int, 'kst_3': int, 'kst_4': int, 'kst_ns': int, 'stc_fw': int,
             'stc_sw': int, 'stc_c': int, 'stc_s1': int, 'stc_s2': int, 'vi_w': int, 'eom_w': int,
             'sma_w': int, 'sma_fw': int, 'wma_w': int, 'wma_fw': int, 'ema_w': int, 'ema_fw': int,
             'kst_b': int, 'kst_s': int, 'stc_ll': int, 'stc_hl': int, 'adi_fw': int, 'adi_sw': int,
             'adi_sg': int, 'obv_fw': int, 'obv_sw': int, 'obv_sg': int, 'eom_sma': int, 'vpt_sma': int,
             'vi2_w': int, 'obv2_fw': int, 'obv2_sw': int, 'obv2_sg': int, 'ao2_1': int, 'ao2_2': int,
             'macd2_fw': int, 'macd2_sw': int, 'macd2_sg': int, 'po2_sw': int, 'po2_fw': int, 'po2_sg': int,
             'pvo2_sw': int, 'pvo2_fw': int, 'pvo2_sg': int, 'stc2_fw': int, 'stc2_sw': int, 'stc2_c': int,
             'stc2_s1': int, 'stc2_s2': int, 'adi2_fw': int, 'adi2_sw': int, 'adi2_sg': int,
             'sma2_w': int, 'sma2_fw': int, 'ema2_w': int, 'ema2_fw': int, 'wma2_w': int, 'wma2_fw': int}

dfa_0 = [7, 14, 28, 60, 40,
                      # 'uo_1', 'uo_2', 'uo_3', 'uo_s', 'uo_b',
                      14, 3, 85, 15, 5,
                      # 'so_w', 'so_sw', 'so_s', 'so_b', 'ao_1',
                      34, 14, -20, -80, 14, 3, 3, 0.8, 0.2, 26,
                      # 'ao_2', 'wi_w', 'wi_s', 'wi_b', 'srsi_w', 'srsi_kw', 'srsi_dw', 'srsi_s', 'srsi_b', 'po_sw',
                      12, 9, 26, 12, 9,
                      # 'po_fw', 'po_sg', 'pvo_sw', 'pvo_fw', 'pvo_sg',
                      25, 12, 26, 9, 20, -110,
                      #  'ai_w', 'macd_fw', 'macd_sw', 'macd_sg', 'cci_w', 'cci_b',
                      110, 10, 30, 60, 10, 15, 20, 30, 10,
                      # 'cci_s', 'ii_1', 'ii_2', 'ii_3', 'kst_r1', 'kst_r2', 'kst_r3', 'kst_r4', 'kst_1',
                      10, 10, 15, 9, 0.02, 0.2, 23, 50,
                      # 'kst_2', 'kst_3', 'kst_4', 'kst_ns', 'psar_st', 'psar_ms', 'stc_fw', 'stc_sw',
                      10, 3, 3, 14, 14,
                      # 'stc_c', 'stc_s1', 'stc_s2', 'vi_w', 'eom_w',
                      15, 5, 15, 5, 15, 5, -80,
                      #  'sma_w','sma_fw', 'ema_w', 'ema_fw', 'wma_w', 'wma_fw', 'trix_sw', 'tsi_s', 'tsi_sig', 'ki_s', 'ki_b', 'kst_b'
                      75, 20, 80, 12, 26, 9, 12, 26, 9, 7, 7,
                      # 'kst_s', 'stc_ll', 'stc_hl', 'adi_fw', 'adi_sw', 'adi_sg', 'obv_fw', 'obv_sw', 'obv_sg', 'eom_sma', 'vpt_sma'
                      14, 12, 26, 9, 5, 34, 12, 26,
                      # 'vi2_w', 'obv2_fw', 'obv2_sw', 'obv2_sg', 'ao2_1', 'ao2_2', 'macd2_fw', 'macd2_sw',
                      9, 26, 12, 9, 26, 12, 9, 0.02, 0.2, 23,
                      # 'macd2_sg', 'po2_sw', 'po2_fw', 'po2_sg', 'pvo2_sw', 'pvo2_fw', 'pvo2_sg', 'psar2_st', 'psar2_ms', 'stc2_fw',
                      50, 10, 3, 3, 20, 80, 12, 26, 9, 15,
                      # 'stc2_sw', 'stc2_c', 'stc2_s1', 'stc2_s2', 'stc2_ll', 'stc2_hl', 'adi2_fw', 'adi2_sw', 'adi2_sg', 'sma2_w',
                      5, 15, 5, 15, 5]
                      # 'sma2_fw', 'ema2_w', 'ema2_fw', 'wma2_w', 'wma2_fw'



def log_status(dfs, cdf, price_p, stop_r, current_price):
    if int(dfs['Status'][0][0]) in [-1, 0, 3]:
        logger.info(f'Status: {dfs} \nSignals: {cdf[0]} Price Point: {price_p} - Stop Loss: {round(price_p + stop_r,2)} - Current Price: {current_price}')
    elif int(dfs['Status'][0][0]) in [1, 2, -2]:
        logger.info(f'Status: {dfs} \nSignals: {cdf[0]} Price Point: {price_p} - Stop Loss: {round(price_p - stop_r, 2)} - Current Price: {current_price}')



def retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    num_retries = 0
    try:
        num_retries += 1
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        # print('Fetched', len(ohlcv), symbol, 'candles from', exchange.iso8601 (ohlcv[0][0]), 'to', exchange.iso8601 (ohlcv[-1][0]))
        return ohlcv
    except Exception:
        if num_retries > max_retries:
            raise  # Exception('Failed to fetch', timeframe, symbol, 'OHLCV in', max_retries, 'attempts')


def scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    earliest_timestamp = exchange.milliseconds()
    timeframe_duration_in_seconds = exchange.parse_timeframe(timeframe)
    timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
    timedelta = limit * timeframe_duration_in_ms
    all_ohlcv = []
    while True:
        try:
            fetch_since = earliest_timestamp - timedelta
            ohlcv = retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe,
                                      fetch_since, limit)
            # if we have reached the beginning of history
            if ohlcv is None:
                # print('Historical Dataset is empty. Retrying...')
                logger.info('Historical Dataset is empty. Retrying...')
                time.sleep(10)
                continue
            if ohlcv[0][0] >= earliest_timestamp:
                break
            earliest_timestamp = ohlcv[0][0]
            all_ohlcv = ohlcv + all_ohlcv
            # print(len(all_ohlcv), 'candles in total from', exchange.iso8601(all_ohlcv[0][0]), 'to', exchange.iso8601(all_ohlcv[-1][0]))
            # if we have reached the checkpoint
            if fetch_since < since:
                break
        except:
            # print('Something went wrong while trying to fetch historical data!')
            logger.info('Something went wrong while trying to fetch historical data!')
            time.sleep(60)
            continue
    return exchange.filter_by_since_limit(all_ohlcv, since, None, key=0)

def scrape_candles_to_csv(exchange_id, max_retries, symbol,
                          timeframe, since, limit):
    # instantiate the exchange by id
    exchange = getattr(ccxt, exchange_id)({
        'enableRateLimit': True,
    })
    # convert since from string to milliseconds integer if needed
    if isinstance(since, str):
        since = exchange.parse8601(since)
    # preload all markets from the exchange
    load_loop = 0
    while load_loop == 0:
        try:
            exchange.load_markets()
            load_loop = 1
        except:
            time.sleep(60)
    # fetch all candles
    ohlcv = scrape_ohlcv(exchange, max_retries, symbol, timeframe, since,
                         limit)
    # Creates Dataframe and save it to csv file
    return pd.DataFrame(ohlcv)
    # print('Saved', len(ohlcv), 'candles from', exchange.iso8601(ohlcv[0][0]), 'to', exchange.iso8601(ohlcv[-1][0]), 'to', filename)


def fetch_data(exchange='binance',
               cryptos=['BTC/BUSD'],
               sample_freq='1d',
               since_hours=48,
               page_limit=1000):
    since = (datetime.today() - timedelta(hours=since_hours) -
             timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%S')

    for market_symbol in cryptos:
        df = scrape_candles_to_csv(exchange_id=exchange,
                                   max_retries=3,
                                   symbol=market_symbol,
                                   timeframe=sample_freq,
                                   since=since,
                                   limit=page_limit)
        if market_symbol == cryptos[0]:

            df[0] = pd.to_datetime(df[0], unit='ms')
            df.rename(columns={
                0: 'Datetime',
                1: 'Open',
                2: 'High',
                3: 'Low',
                4: 'Close',
                5: 'Volume'
            },
                inplace=True)
            df = df.set_index('Datetime')
            dfx = df.copy()

        else:

            df[0] = pd.to_datetime(df[0], unit='ms')
            df.rename(columns={
                0: 'Datetime',
                1: 'Open',
                2: 'High',
                3: 'Low',
                4: 'Close',
                5: 'Volume'
            },
                inplace=True)
            df = df.set_index('Datetime')
            dfx = pd.merge(dfx, df, on=['Datetime'])

    dfx = dfx.loc[:, ~dfx.columns.duplicated()]
    dfx = dfx[~dfx.index.duplicated(keep='first')]

    return dfx


def import_csv(loc_folder, filename):
    read_file = f'{loc_folder}{filename}'
    df = pd.read_csv(read_file, index_col='Datetime', parse_dates=True)
    return df

def round_up(num, dec):
    num = math.ceil(num * 10 ** dec) / 10 ** dec
    return num

def round_down(num, dec):
    num = math.floor(num * 10 ** dec) / 10 ** dec
    return num

def append_to_col(df, col, val):
    idx = df[col].last_valid_index()
    df.loc[idx + 1 if idx is not None else 0, col] = val
    return df

def get_dic(df, indicator, crypto, time_frame, str_dic, pos, idx_max):
    if indicator == 'Null':
        str_dic[f'{crypto}{time_frame}'][pos][0] = None
        str_dic[f'{crypto}{time_frame}'][pos][1] = None
    else:
        final = ' ' + indicator[:-2].lower()
        str_dic[f'{crypto}{time_frame}'][pos][0] = indicator
        str_dic[f'{crypto}{time_frame}'][pos][1] = list(
            df.loc[idx_max].filter(like=f'{final}'))
    return str_dic

def shuffle(li):
    random.shuffle(li)
    return li


def check_duplicates(lis):
    if len(lis) == len(set(lis)):
        return False
    else:
        return True


def get_ret_1(df, ind_list, lim, stop=0, atr=500):

    df.loc[(df.iloc[:, ind_list[0]] + df.iloc[:, ind_list[1]] == len(ind_list) - lim), 'Trade'] = df.Close
    df.loc[(df.iloc[:, ind_list[0]] + df.iloc[:, ind_list[1]] == -len(ind_list) + lim), 'Trade'] = -df.Close
    df.fillna(0, inplace=True)  # Fills with 0 the missing values in "Trade".

    # The following is the common procedure to clean the dataset to a format to calculate the percent changes.

    df = df[df['Trade'] != 0]
    df.loc[df.Trade.shift(1).apply(np.sign) == df.Trade.apply(np.sign), 'Trade'] = 0
    df = df[df['Trade'] != 0]
    if len(df[df['Trade'] < 0]) < 3:
        ret = pd.Series([], dtype='float64')
        return ret

    df = df.loc[(df['Trade'] < 0).idxmax():]
    perc = df['Trade'].abs()
    ret_f = perc.pct_change()[1::2]  # Only Buy Long, no short
    ret = perc.pct_change()[1:]  # Buy long/ Sell short
    ret.loc[ret.reset_index().index % 2 != 0] = ret.iloc[:] * -1  # Buy long/ Sell short
    return ret

def get_ret_2(df, ind_list, lim, stop=0, atr=500):

        df.loc[(df.iloc[:, ind_list[0]] + df.iloc[:, ind_list[1]] == len(ind_list) - lim), 'Trade'] = df.Close
        df.loc[(df.iloc[:, ind_list[0]] + df.iloc[:, ind_list[1]] == -len(ind_list) + lim), 'Trade'] = -df.Close
        df.fillna(0, inplace=True)  # Fills with 0 the missing values in "Trade".
        trade = df['Trade']
        trade = trade[trade != 0]
        trade.loc[trade.shift(1).apply(np.sign) == trade.apply(np.sign)] = 0
        trade = trade[trade != 0]

        if len(trade[trade < 0]) < 3:
            ret = pd.Series([], dtype='float64')
            return ret

        # Stop loss with fixed percentage
        for h in range(3):
            df = df.merge(trade.rename('T1'), left_index=True, right_index=True, how='left')
            df['T1'] = df['T1'].fillna(method='ffill')
            #df.dropna(inplace=True)
            df.loc[(df.T1 < 0) & (df.Close < df.T1.abs() * stop), 'Trade'] = df['Close']  # Applies the stop loss condition on long
            df.loc[(df.T1 > 0) & (df.Close > df.T1.abs() / stop), 'Trade'] = df['Close']  # Applies the stop loss condition on short
            df.drop(columns=['T1'], inplace=True)
            trade = df['Trade']
            trade = trade[trade != 0]
            trade.loc[trade.shift(1).apply(np.sign) == trade.apply(np.sign)] = 0
            trade = trade[trade != 0]

        # The following is the common procedure to clean the dataset to a format to calculate the percent changes.
        df = df[df['Trade'] != 0]
        df.loc[df.Trade.shift(1).apply(np.sign) == df.Trade.apply(np.sign), 'Trade'] = 0
        df = df[df['Trade'] != 0]
        df = df.loc[(df['Trade'] < 0).idxmax():]
        perc = df['Trade'].abs()
        #ret_f = perc.pct_change()[1::2]  # Only Buy Long, no short
        ret = perc.pct_change()[1:]  # Buy long/ Sell short
        # ret_f = ret.copy()
        ret.loc[ret.reset_index().index % 2 != 0] = ret.iloc[:] * -1  # Buy long/ Sell short
        return ret

def get_ret_3(df, ind_list, lim, stop=0, tr=500):


    df.loc[(df.iloc[:, ind_list[0]] + df.iloc[:, ind_list[1]] == len(ind_list) - lim), 'Trade'] = df.Close
    df.loc[(df.iloc[:, ind_list[0]] + df.iloc[:, ind_list[1]] == -len(ind_list) + lim), 'Trade'] = -df.Close
    df.fillna(0, inplace=True)  # Fills with 0 the missing values in "Trade".
    trade = df['Trade']
    trade = trade[trade != 0]
    trade.loc[trade.shift(1).apply(np.sign) == trade.apply(np.sign)] = 0
    trade = trade[trade != 0]

    if len(trade[trade < 0]) < 3:
        ret = pd.Series([], dtype='float64')
        return ret

    # Stop Loss using TR:
    df['TR1'] = tr(df, tr)
    for h in range(4):
        df = df.merge(trade.rename('T1'), left_index=True, right_index=True, how='left')
        df.loc[df.T1.notnull(), 'TR'] = df.TR1
        df = df.fillna(method='ffill')
        df.loc[(df.T1 < 0) & (
                    df.Close < df.T1.abs() - df.TR), 'Trade'] = df.Close  # Applies the stop loss condition on long
        df.loc[(df.T1 > 0) & (
                    df.Close > df.T1.abs() + df.TR), 'Trade'] = -df.Close  # Applies the stop loss condition on short
        df.drop(columns=['T1', 'TR'], inplace=True)
        trade = df['Trade']
        trade = trade[trade != 0]
        trade.loc[trade.shift(1).apply(np.sign) == trade.apply(np.sign)] = 0
        trade = trade[trade != 0]

    # The following is the common procedure to clean the dataset to a format to calculate the percent changes.
    df = df[df['Trade'] != 0]
    df.loc[df.Trade.shift(1).apply(np.sign) == df.Trade.apply(np.sign), 'Trade'] = 0
    df = df[df['Trade'] != 0]
    df = df.loc[(df['Trade'] < 0).idxmax():]
    perc = df['Trade'].abs()
    ret_f = perc.pct_change()[1::2]  # Only Buy Long, no short
    ret = perc.pct_change()[1:]  # Buy long/ Sell short
    ret.loc[ret.reset_index().index % 2 != 0] = ret.iloc[:] * -1  # Buy long/ Sell short
    return ret


def dt_etl(dt):
    dt = dt.astype({
        'uo_1': int, 'uo_2': int, 'uo_3': int,
        'uo_s': int, 'uo_b': int, 'so_w': int, 'so_sw': int, 'so_s': int, 'so_b': int,
        'ao_1': int, 'ao_2': int, 'wi_w': int, 'wi_s': int, 'wi_b': int,
        'srsi_w': int, 'srsi_kw': int, 'srsi_dw': int, 'po_sw': int, 'po_fw': int,
        'po_sg': int, 'pvo_sw': int, 'pvo_fw': int, 'pvo_sg': int, 'ai_w': int, 'macd_fw': int,
        'macd_sw': int, 'macd_sg': int, 'cci_w': int, 'cci_b': int, 'cci_s': int, 'ii_1': int,
        'ii_2': int, 'ii_3': int, 'kst_r1': int, 'kst_r2': int, 'kst_r3': int, 'kst_r4': int,
        'kst_1': int, 'kst_2': int, 'kst_3': int, 'kst_4': int, 'kst_ns': int, 'stc_fw': int,
        'stc_sw': int, 'stc_c': int, 'stc_s1': int, 'stc_s2': int, 'vi_w': int, 'eom_w': int,
        'sma_w': int, 'sma_fw': int, 'wma_w': int, 'wma_fw': int, 'ema_w': int, 'ema_fw': int,
        'kst_b': int, 'kst_s': int, 'stc_ll': int, 'stc_hl': int, 'adi_fw': int, 'adi_sw': int,
        'adi_sg': int, 'obv_fw': int, 'obv_sw': int, 'obv_sg': int, 'eom_sma': int, 'vpt_sma': int})
    dt = dt[
        ['Criteria', 'Score', 'Alpha', 'Alpha Trade Rate', 'Alpha Max Days', 'Max Min Delta', 'Beta', 'Transactions',
         'Accuracy', 'Av. Acc', 'Av.Acc.Error',
         'Final Equity',
         'Market Return',
         'Zone', 'Signal', 'uo_1', 'uo_2', 'uo_3', 'uo_s', 'uo_b',
         'so_w', 'so_sw', 'so_s', 'so_b', 'ao_1',
         'ao_2', 'wi_w', 'wi_s', 'wi_b', 'srsi_w', 'srsi_kw', 'srsi_dw', 'srsi_s', 'srsi_b', 'po_sw',
         'po_fw', 'po_sg', 'pvo_sw', 'pvo_fw', 'pvo_sg', 'ai_w', 'macd_fw', 'macd_sw', 'macd_sg', 'cci_w', 'cci_b',
         'cci_s', 'ii_1', 'ii_2', 'ii_3', 'kst_r1', 'kst_r2', 'kst_r3', 'kst_r4', 'kst_1',
         'kst_2', 'kst_3', 'kst_4', 'kst_ns', 'psar_st', 'psar_ms', 'stc_fw', 'stc_sw',
         'stc_c', 'stc_s1', 'stc_s2', 'vi_w', 'eom_w', 'sma_w', 'sma_fw', 'ema_w', 'ema_fw', 'wma_w', 'wma_fw',
         'kst_b',
         'kst_s', 'stc_ll', 'stc_hl', 'adi_fw', 'adi_sw', 'adi_sg', 'obv_fw', 'obv_sw', 'obv_sg', 'eom_sma',
         'vpt_sma'
         ]]
    dt.columns = [f' {x}' for x in
                  dt.columns]  # Add a space in front of each column's name to help later in selecting some indicators by name.
    return dt

def get_psar(df, iaf=0.02, maxaf=0.2):
    length = len(df)
    high = df['High']
    low = df['Low']
    df['PSAR'] = df['Close'].copy()
    bull = True
    af = iaf
    hp = high.iloc[0]
    lp = low.iloc[0]

    for i in range(2, length):
        if bull:
            df.PSAR.iloc[i] = df.PSAR.iloc[i - 1] + af * (hp - df.PSAR.iloc[i - 1])
        else:
            df.PSAR.iloc[i] = df.PSAR.iloc[i - 1] + af * (lp - df.PSAR.iloc[i - 1])

        reverse = False

        if bull:
            if low.iloc[i] < df.PSAR.iloc[i]:
                bull = False
                reverse = True
                df.PSAR.iloc[i] = hp
                lp = low.iloc[i]
                af = iaf
        else:
            if high.iloc[i] > df.PSAR.iloc[i]:
                bull = True
                reverse = True
                df.PSAR.iloc[i] = lp
                hp = high.iloc[i]
                af = iaf

        if not reverse:
            if bull:
                if high.iloc[i] > hp:
                    hp = high.iloc[i]
                    af = min(af + iaf, maxaf)
                if low.iloc[i - 1] < df.PSAR.iloc[i]:
                    df.PSAR.iloc[i] = low[i - 1]
                if low.iloc[i - 2] < df.PSAR.iloc[i]:
                    df.PSAR.iloc[i] = low.iloc[i - 2]
            else:
                if low.iloc[i] < lp:
                    lp = low.iloc[i]
                    af = min(af + iaf, maxaf)
                if high.iloc[i - 1] > df.PSAR.iloc[i]:
                    df.PSAR.iloc[i] = high.iloc[i - 1]
                if high.iloc[i - 2] > df.PSAR.iloc[i]:
                    df.PSAR.iloc[i] = high.iloc[i - 2]
    return df.PSAR

def tr(df, window=14):
    tr = df['High'].rolling(window).max() - df['Low'].rolling(window).min()
    return tr

def atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)

    atr = true_range.rolling(window).sum() / window
    return atr

def rsi_s(df, args):  # rsi_w (Is not Used)
    rsi = RSIIndicator(close=df['Close'], window=args[0])
    df['RSI'] = rsi.rsi()
    df.loc[(df.RSI > args[2]), 'RSI_Z'] = 1
    df.loc[(df.RSI < args[1]), 'RSI_Z'] = -1
    df.drop(columns=['RSI'], inplace=True)
    return df


def tsi_s(df, args):  # tsi_fw, tsi_fw, tsi_sig, tsi_s
    tsi = TSIIndicator(close=df['Close'],
                       window_slow=args[0],
                       window_fast=args[1])
    df['TSI'] = tsi.tsi()
    df['TSI_S'] = df['TSI'].ewm(args[2],
                                min_periods=0,
                                adjust=False,
                                ignore_na=False).mean()
    df.loc[(df.TSI > args[3]), 'TSI_Z'] = 1
    df.loc[(df.TSI < -args[3]), 'TSI_Z'] = -1
    df.loc[(df.TSI < 0), 'TSI_T'] = 1
    df.loc[(df.TSI >= 0), 'TSI_T'] = -1
    df.loc[(df.TSI.shift(1) > df.TSI_S.shift(1)) & (df.TSI < df.TSI_S),
           'TSI_A'] = 1
    df.loc[(df.TSI.shift(1) < df.TSI_S.shift(1)) & (df.TSI > df.TSI_S),
           'TSI_A'] = -1
    df.drop(columns=['TSI', 'TSI_S'], inplace=True)
    return df


def kst_s(
        df, args
):  # kst_r1, kst_r2, kst_r3, kst_r4, kst_1, kst_2, kst_3, kst_4, kst_ns, kst_b, kst_s
    kst = KSTIndicator(close=df['Close'],
                       roc1=args[0],
                       roc2=args[1],
                       roc3=args[2],
                       roc4=args[3],
                       window1=args[4],
                       window2=args[5],
                       window3=args[6],
                       window4=args[7],
                       nsig=args[8])
    df['KST'] = kst.kst()
    df['KST_S'] = kst.kst_sig()
    df['KST_H'] = kst.kst_diff()
    df.loc[df.KST_H >= 0, 'KST_T'] = -1
    df.loc[df.KST_H < 0, 'KST_T'] = 1
    df.loc[(df.KST.shift(1) > df.KST_S.shift(1)) & (df.KST < df.KST_S) &
           (df.KST.shift(1) > args[10]), 'KST_A'] = 1
    df.loc[(df.KST.shift(1) < df.KST_S.shift(1)) & (df.KST > df.KST_S) &
           (df.KST.shift(1) < args[9]), 'KST_A'] = -1
    df.drop(columns=['KST', 'KST_S', 'KST_H'], inplace=True)
    return df


def stc_s(df, args):  # stc_fw, stc_sw, stc_c, stc_s1, stc_s2, stc_ll, stc_hl
    stc = STCIndicator(close=df['Close'],
                       window_fast=args[0],
                       window_slow=args[1],
                       cycle=args[2],
                       smooth1=args[3],
                       smooth2=args[4])
    df['STC'] = stc.stc()
    df.loc[df.STC < args[6], 'STC_T'] = 1
    df.loc[df.STC > args[5], 'STC_T'] = -1
    df.loc[df.STC < args[6], 'STC2_T'] = 1
    df.loc[df.STC > args[5], 'STC2_T'] = -1
    df.loc[(df.STC.shift(1) > args[6]) & (df.STC < args[6]), 'STC_A'] = 1
    df.loc[(df.STC.shift(1) < args[5]) & (df.STC > args[5]), 'STC_A'] = -1
    df.drop(columns=['STC'], inplace=True)
    return df


def cmf_s(df, args):  # cmf_w, cmf_l
    cmf = ChaikinMoneyFlowIndicator(high=df['High'],
                                    low=df['Low'],
                                    close=df['Close'],
                                    volume=df['Volume'],
                                    window=args[0])
    df['CMF'] = cmf.chaikin_money_flow()
    df.loc[(df.CMF > args[1]), 'CMF_Z'] = 1
    df.loc[(df.CMF < -args[1]), 'CMF_Z'] = -1
    df.loc[(df.CMF.shift(1) > 0) & (df.CMF < 0), 'CMF_A'] = 1
    df.loc[(df.CMF.shift(1) < 0) & (df.CMF > 0), 'CMF_A'] = -1
    df.drop(columns=['CMF'], inplace=True)
    return df


def fi_s(df, args):  # fi_w, std_m
    fi = ForceIndexIndicator(close=df['Close'],
                             volume=df['Volume'],
                             window=args[0])
    df['FI'] = fi.force_index()
    fi_s = int(df[df['FI'] > 0]['FI'].mean() +
               3.5 * df[df['FI'] > 0]['FI'].mean() * args[1])
    fi_b = int(df[df['FI'] < 0]['FI'].mean() +
               2.5 * df[df['FI'] < 0]['FI'].mean() * args[1])
    df.loc[df.FI > fi_s, 'FI_Z'] = 1
    df.loc[df.FI < fi_b, 'FI_Z'] = -1
    df.loc[(df.FI.shift(1) > fi_s) & (df.FI < fi_s), 'FI_A'] = 1
    df.loc[(df.FI.shift(1) < fi_b) & (df.FI > fi_b), 'FI_A'] = -1
    df.drop(columns=['FI'], inplace=True)
    return df


def mfi_s(df, args):  # mfi_w, mfi_b, mfi_s
    mfi = MFIIndicator(high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       volume=df['Volume'],
                       window=args[0])
    df['MFI'] = mfi.money_flow_index()
    df.loc[(df.MFI > args[2]), 'MFI_Z'] = 1
    df.loc[(df.MFI < args[1]), 'MFI_Z'] = -1
    df.drop(columns=['MFI'], inplace=True)
    return df


def uo_s(df, args):  # uo_1, uo_2, uo_3, uo_s, uo_b
    uo = UltimateOscillator(high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            window1=args[0],
                            window2=args[1],
                            window3=args[2])
    df['UO'] = uo.ultimate_oscillator()
    df.loc[(df.UO > args[3]), 'UO_Z'] = 1
    df.loc[(df.UO < args[4]), 'UO_Z'] = -1
    df.drop(columns=['UO'], inplace=True)
    return df


def ki_s(df, args):  # ki_w, ki_p1, ki_p2, ki_p3, ki_b, ki_s
    ki = KAMAIndicator(close=df['Close'],
                       window=args[0],
                       pow1=args[1],
                       pow2=args[2])
    ki_sig = KAMAIndicator(close=df['Close'],
                           window=args[0],
                           pow1=args[3],
                           pow2=args[2])
    df['KI'] = ki.kama()
    df['KIS'] = ki_sig.kama()
    df.loc[(df.Close > df.KI * (1 + args[5])), 'KI_Z'] = 1
    df.loc[(df.Close < df.KI * (1 - args[4])), 'KI_Z'] = -1
    df.loc[(df.Close.shift(1) > df.KI.shift(1)) & (df.Close < df.KI *
                                                   (1 - args[4])), 'KI_T'] = 1
    df.loc[(df.Close.shift(1) < df.KI.shift(1)) & (df.Close > df.KI *
                                                   (1 + args[5])), 'KI_T'] = -1
    df['KI_T'] = df['KI_T'].fillna(method='ffill')
    df.loc[(df.KI > df.KIS.shift(1)) & (df.KI < df.KIS), 'KI_A'] = 1
    df.loc[(df.KI < df.KIS.shift(1)) & (df.KI > df.KIS), 'KI_A'] = -1
    df.drop(columns=['KI', 'KIS'], inplace=True)
    return df


def roc_s(df, args):  # roc_w, roc_b, roc_s
    roc = ROCIndicator(close=df['Close'], window=args[0])
    df['ROC'] = roc.roc()
    df.loc[(df.ROC > args[2]), 'ROC_Z'] = 1
    df.loc[(df.ROC < args[1]), 'ROC_Z'] = -1
    df.loc[(df.ROC < 0), 'ROC_T'] = 1
    df.loc[(df.ROC >= 0), 'ROC_T'] = -1
    df.drop(columns=['ROC'], inplace=True)
    return df


def ao_s(df, args):  # ao_1, ao_2
    ao = AwesomeOscillatorIndicator(high=df['High'],
                                    low=df['Low'],
                                    window1=args[0],
                                    window2=args[1])
    df['AO'] = ao.awesome_oscillator()
    df.loc[(df.AO < 0), 'AO_T'] = 1
    df.loc[(df.AO >= 0), 'AO_T'] = -1
    df.loc[(df.AO < 0), 'AO2_T'] = 1
    df.loc[(df.AO >= 0), 'AO2_T'] = -1
    df.loc[((df.AO.shift(1) > 0) & (df.AO < 0)) |
           ((df.AO < 0) & (df.AO.shift(2) < df.AO.shift(1)) &
            (df.AO.shift(1) > df.AO)), 'AO_A'] = 1
    df.loc[((df.AO.shift(1) < 0) & (df.AO > 0)) |
           ((df.AO > 0) & (df.AO.shift(2) > df.AO.shift(1)) &
            (df.AO.shift(1) < df.AO)), 'AO_A'] = -1
    df.drop(columns=['AO'], inplace=True)
    return df


def wi_s(df, args):  # wi_w	wi_s wi_b
    wi = WilliamsRIndicator(high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            lbp=args[0])
    df['WI'] = wi.williams_r()
    df.loc[(df.WI > args[1]), 'WI_Z'] = 1
    df.loc[(df.WI < args[2]), 'WI_Z'] = -1
    df.drop(columns=['WI'], inplace=True)
    return df


def srsi_s(df, args):  # srsi_w srsi_kw	srsi_dw	srsi_s srsi_b
    srsi = StochRSIIndicator(close=df['Close'],
                             window=args[0],
                             smooth1=args[1],
                             smooth2=args[2])
    df['SRSI'] = srsi.stochrsi()
    df['SRSI_K'] = srsi.stochrsi_k()
    df['SRSI_D'] = srsi.stochrsi_d()
    df.loc[(df.SRSI > args[3]), 'SRSI_Z'] = 1
    df.loc[(df.SRSI < args[4]), 'SRSI_Z'] = -1
    df.loc[(df.SRSI_K.shift(1) > df.SRSI_D.shift(1)) & (df.SRSI_K < df.SRSI_D),
           'SRSI_A'] = 1
    df.loc[(df.SRSI_K.shift(1) < df.SRSI_D.shift(1)) & (df.SRSI_K > df.SRSI_D),
           'SRSI_A'] = -1
    df.drop(columns=['SRSI_K', 'SRSI_D', 'SRSI'], inplace=True)
    return df


def po_s(df, args):  # po_sw po_fw po_sg

    po = PercentagePriceOscillator(close=df['Close'],
                                   window_slow=args[0],
                                   window_fast=args[1],
                                   window_sign=args[2])
    df['PO'] = po.ppo()
    df['PO_S'] = po.ppo_signal()
    df['PO_H'] = po.ppo_hist()
    df.loc[(df.PO_H < 0), 'PO_T'] = 1
    df.loc[(df.PO_H >= 0), 'PO_T'] = -1
    df.loc[(df.PO_H < 0), 'PO2_T'] = 1
    df.loc[(df.PO_H >= 0), 'PO2_T'] = -1
    df.loc[(df.PO.shift(1) > df.PO_S.shift(1)) & (df.PO < df.PO_S), 'PO_A'] = 1
    df.loc[(df.PO.shift(1) < df.PO_S.shift(1)) & (df.PO > df.PO_S),
           'PO_A'] = -1
    df.drop(columns=['PO', 'PO_S', 'PO_H'], inplace=True)
    return df


def pvo_s(df, args):  # pvo_sw	pvo_fw	pvo_sg
    pvo = PercentageVolumeOscillator(volume=df['Volume'],
                                     window_slow=args[0],
                                     window_fast=args[1],
                                     window_sign=args[2])
    df['PVO'] = pvo.pvo()
    df['PVO_S'] = pvo.pvo_signal()
    df['PVO_H'] = pvo.pvo_hist()
    df.loc[(df.PVO_H < 0), 'PVO_T'] = 1
    df.loc[(df.PVO_H >= 0), 'PVO_T'] = -1
    df.loc[(df.PVO_H < 0), 'PVO2_T'] = 1
    df.loc[(df.PVO_H >= 0), 'PVO2_T'] = -1
    df.loc[(df.PVO.shift(1) > df.PVO_S.shift(1)) & (df.PVO < df.PVO_S),
           'PVO_A'] = 1
    df.loc[(df.PVO.shift(1) < df.PVO_S.shift(1)) & (df.PVO > df.PVO_S),
           'PVO_A'] = -1
    df.drop(columns=['PVO', 'PVO_S', 'PVO_H'], inplace=True)
    return df


def atr_s(df, args):  # atr_w atr_l rsi_w rsi_b rsi_s
    atr = AverageTrueRange(high=df['High'],
                           low=df['Low'],
                           close=df['Close'],
                           window=args[0])
    rsi = RSIIndicator(close=df['Close'], window=args[2])
    df['RSI'] = rsi.rsi()
    df['ATR'] = atr.average_true_range()
    df.loc[((df.ATR.shift(1) * (args[0] - 1) + df.ATR) / args[0] > args[1]) &
           (df.RSI > args[4]), 'ATR_Z'] = 1
    df.loc[((df.ATR.shift(1) * (args[0] - 1) + df.ATR) / args[0] > args[1]) &
           (df.RSI < args[3]), 'ATR_Z'] = -1
    df.drop(columns=['ATR', 'RSI'], inplace=True)
    return df


def bb_s(df, args):  # bb_w	bb_d bb_l
    bb = BollingerBands(close=df['Close'], window=args[0], window_dev=args[1])
    adx = ADXIndicator(high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       window=args[0])
    df['BB_H'] = bb.bollinger_hband_indicator()
    df['BB_L'] = bb.bollinger_lband_indicator()
    df['BB_W'] = bb.bollinger_wband()
    df['ADX_P'] = adx.adx_pos()
    df['ADX_N'] = adx.adx_neg()
    df.loc[(df.ADX_P < df.ADX_N), 'ADX_T'] = 1
    df.loc[(df.ADX_P > df.ADX_N), 'ADX_T'] = -1
    df.loc[(df.BB_W > args[2]), 'BB_Z'] = 1
    df.loc[(df.BB_W > args[2]), 'BB_Z'] = -1
    df.loc[(df.Close >= df.BB_H) & (df.ADX_T > 0), 'BB_T'] = 1
    df.loc[(df.Close <= df.BB_L) & (df.ADX_T < 0), 'BB_T'] = -1
    df['BB_T'] = df['BB_T'].fillna(method='ffill')
    df.loc[(df.Close >= df.BB_H), 'BB_A'] = 1
    df.loc[(df.Close <= df.BB_L), 'BB_A'] = -1
    df.drop(columns=['BB_H', 'BB_L', 'BB_W', 'ADX_P', 'ADX_N', 'ADX_T'],
            inplace=True)
    return df


def kc_s(df, args):  # kc_w	kc_a
    kc = KeltnerChannel(high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        window=args[0],
                        window_atr=args[1])
    df['KC_H'] = kc.keltner_channel_hband_indicator()
    df['KC_L'] = kc.keltner_channel_lband_indicator()
    df.loc[df.KC_L > 0, 'KC_T'] = 1
    df.loc[df.KC_H > 0, 'KC_T'] = -1
    df['KC_T'] = df['KC_T'].fillna(method='ffill')
    df.drop(columns=['KC_H', 'KC_L'], inplace=True)
    return df


def dc_s(df, args):  # dc_w	dc_l
    dc = DonchianChannel(high=df['High'],
                         low=df['Low'],
                         close=df['Close'],
                         window=args[0])
    adx = ADXIndicator(high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       window=args[0])
    df['DC_H'] = dc.donchian_channel_hband()
    df['DC_L'] = dc.donchian_channel_lband()
    df['DC_W'] = dc.donchian_channel_wband()
    df['ADX_P'] = adx.adx_pos()
    df['ADX_N'] = adx.adx_neg()
    df.loc[(df.ADX_P < df.ADX_N), 'ADX_T'] = 1
    df.loc[(df.ADX_P > df.ADX_N), 'ADX_T'] = -1
    df.loc[(df.DC_W.shift(1) < args[1]), 'DC_Z'] = 1
    df.loc[(df.DC_W.shift(1) > args[1]), 'DC_Z'] = -1
    df.loc[(df.High >= df.DC_H) & (df.ADX_T > 0), 'DC_T'] = 1
    df.loc[(df.Low <= df.DC_L) & (df.ADX_T < 0), 'DC_T'] = -1
    df['DC_T'] = df['DC_T'].fillna(method='ffill')
    df.loc[(df.Close >= df.DC_H), 'DC_A'] = 1
    df.loc[(df.Close <= df.DC_L), 'DC_A'] = -1
    df.drop(columns=['DC_H', 'DC_L', 'DC_W', 'ADX_P', 'ADX_N', 'ADX_T'],
            inplace=True)
    return df


def ui_s(df, args):  # ui_w	ui_b ui_s
    ui = UlcerIndex(close=df['Close'], window=args[0])
    df['UI'] = ui.ulcer_index()
    df.loc[(df.UI < args[2]), 'UI_Z'] = 1
    df.loc[(df.UI > args[1]), 'UI_Z'] = -1
    df.drop(columns=['UI'], inplace=True)
    return df


def adx_s(df, args):  # adx_w	adx_l
    adx = ADXIndicator(high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       window=args[0])
    df['ADX'] = adx.adx()
    df['ADX_P'] = adx.adx_pos()
    df['ADX_N'] = adx.adx_neg()
    df.loc[(df.ADX_P < df.ADX_N), 'ADX_T'] = 1
    df.loc[(df.ADX_P > df.ADX_N), 'ADX_T'] = -1
    df.loc[(df.ADX_T > 0) & df.ADX > args[1], 'ADX_Z'] = 1
    df.loc[(df.ADX_T < 0) & df.ADX > args[1], 'ADX_Z'] = -1
    df.drop(columns=['ADX', 'ADX_P', 'ADX_N'], inplace=True)
    return df


def ai_s(df, args):  # ai_w
    ai = AroonIndicator(close=df['Close'], window=args[0])
    df['AI_U'] = ai.aroon_up()
    df['AI_D'] = ai.aroon_down()
    df.loc[(df.AI_U.shift(1) > df.AI_D.shift(1)) & (df.AI_U < df.AI_D),
           'AI_A'] = 1
    df.loc[(df.AI_U.shift(1) < df.AI_D.shift(1)) & (df.AI_U > df.AI_D),
           'AI_A'] = -1
    df.drop(columns=['AI_U', 'AI_D'], inplace=True)
    return df


def macd_s(df, args):  # macd_fw macd_sw macd_sg
    macd = MACD(close=df['Close'],
                window_fast=args[0],
                window_slow=args[1],
                window_sign=args[2])
    df['MACD'] = macd.macd()
    df['MACD_S'] = macd.macd_signal()
    df['MACD_H'] = macd.macd_diff()
    df.loc[(df.MACD_H < 0), 'MACD_T'] = 1
    df.loc[(df.MACD_H >= 0), 'MACD_T'] = -1
    df.loc[(df.MACD_H < 0), 'MACD2_T'] = 1
    df.loc[(df.MACD_H >= 0), 'MACD2_T'] = -1
    df.loc[(df.MACD.shift(1) > df.MACD_S.shift(1)) & (df.MACD < df.MACD_S),
           'MACD_A'] = 1
    df.loc[(df.MACD.shift(1) < df.MACD_S.shift(1)) & (df.MACD > df.MACD_S),
           'MACD_A'] = -1
    df.drop(columns=['MACD', 'MACD_S', 'MACD_H'], inplace=True)
    return df


def cci_s(df, args):  # cci_w cci_b cci_s
    cci = CCIIndicator(high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       window=args[0],
                       constant=0.015)
    df['CCI'] = cci.cci()
    df.loc[df.CCI > args[2], 'CCI_Z'] = 1
    df.loc[df.CCI < args[1], 'CCI_Z'] = -1
    df.loc[(df.CCI.shift(1) > args[2]) & (df.CCI < args[2]), 'CCI_A'] = 1
    df.loc[(df.CCI.shift(1) < args[1]) & (df.CCI > args[1]), 'CCI_A'] = -1
    df.drop(columns=['CCI'], inplace=True)
    return df


def dpo_s(df, args):  # dpo_w, dpo_s
    dpo = DPOIndicator(close=df['Close'], window=args[0])
    df['DPO'] = dpo.dpo()
    df.loc[(df.DPO > args[1]), 'DPO_Z'] = 1
    df.loc[(df.DPO < -args[1]), 'DPO_Z'] = -1
    df.drop(columns=['DPO'], inplace=True)
    return df


def mi_s(df, args):  # mi_fw	mi_sw std_m
    mi = MassIndex(high=df['High'],
                   low=df['Low'],
                   window_fast=args[0],
                   window_slow=args[1])
    df['MI'] = mi.mass_index()
    mi_s = df.MI.mean() + df.MI.std() * args[2]
    df.loc[df.MI > mi_s, 'MI_Z'] = 1
    df.loc[df.MI < mi_s, 'MI_Z'] = -1
    df.drop(columns=['MI'], inplace=True)
    return df


def ii_s(df, args):  # ii_1 ii_2 ii_3
    ii = IchimokuIndicator(high=df['High'],
                           low=df['Low'],
                           window1=args[0],
                           window2=args[1],
                           window3=args[2])
    df['II'] = ii.ichimoku_conversion_line()
    df['II_S'] = ii.ichimoku_base_line()
    df['II_LA'] = ii.ichimoku_a()
    df['II_LB'] = ii.ichimoku_b()
    df.loc[(df.II > df.II_S), 'II_Z'] = 1
    df.loc[(df.II < df.II_S), 'II_Z'] = -1
    df.loc[(df.Close < df.II_S), 'II_T'] = 1
    df.loc[(df.Close >= df.II_S), 'II_T'] = -1
    df.loc[(df.II.shift(1) > df.II_S.shift(1)) & (df.II < df.II_S) &
           (df.High > df.II_LA), 'II_A'] = 1
    df.loc[(df.II.shift(1) < df.II_S.shift(1)) & (df.II > df.II_S) &
           (df.Low < df.II_LB), 'II_A'] = -1
    df.drop(columns=['II', 'II_S', 'II_LA', 'II_LB'], inplace=True)
    return df


def trix_s(df, args):  # trix_w	trix_sw
    trix = TRIXIndicator(close=df['Close'], window=args[0])
    df['TRIX'] = trix.trix()
    df['TRIX_S'] = df['TRIX'].ewm(span=args[1], adjust=False).mean()
    df.loc[df.TRIX < 0, 'TRIX_T'] = 1
    df.loc[df.TRIX >= 0, 'TRIX_T'] = -1
    df.loc[(df.TRIX.shift(1) > df.TRIX_S.shift(1)) & (df.TRIX < df.TRIX_S),
           'TRIX_A'] = 1
    df.loc[(df.TRIX.shift(1) < df.TRIX_S.shift(1)) & (df.TRIX > df.TRIX_S),
           'TRIX_A'] = -1
    df.drop(columns=['TRIX', 'TRIX_S'], inplace=True)
    return df


def vi_s(df, args):  # vi_w
    vi = VortexIndicator(high=df['High'],
                         low=df['Low'],
                         close=df['Close'],
                         window=args[0])
    df['VI_P'] = vi.vortex_indicator_pos()
    df['VI_N'] = vi.vortex_indicator_neg()
    df['VI_H'] = vi.vortex_indicator_diff()
    df.loc[(df.VI_P < df.VI_N), 'VI_T'] = 1
    df.loc[(df.VI_P > df.VI_N), 'VI_T'] = -1
    df.loc[(df.VI_P < df.VI_N), 'VI2_T'] = 1
    df.loc[(df.VI_P > df.VI_N), 'VI2_T'] = -1
    df.loc[(df.VI_P.shift(1) > df.VI_N.shift(1)) & (df.VI_P < df.VI_N),
           'VI_A'] = 1
    df.loc[(df.VI_P.shift(1) < df.VI_N.shift(1)) & (df.VI_P > df.VI_N),
           'VI_A'] = -1
    df.drop(columns=['VI_H', 'VI_P', 'VI_N'], inplace=True)
    return df


def get_dfa(div, mul):
    ran_dic = {'w7': [int(x * 4 / div) * mul + 3 for x in range(div)],
               'w10': [int(x * 9 / div) * mul + 1 for x in range(div)],
               'w14': [int(x * 11 / div) * mul + 3 for x in range(div)],
               'w28': [int(x * 25 / div) * mul + 3 for x in range(div)],
               'w20': [int(x * 19 / div) * mul + 1 for x in range(div)],
               'wn20': [-1 * (int(x * 19 / div) * mul + 1) for x in range(div)],
               'w40': [int(x * 40 / div) * mul for x in range(div)],
               'w5_15': [int(x * 10 / div) * mul + 5 for x in range(div)],
               'w5_20': [int(x * 15 / div) * mul + 5 for x in range(div)],
               'w3_30': [int(x * 27 / div) * mul + 3 for x in range(div)],
               'w5_40': [int(x * 35 / div) * mul + 5 for x in range(div)],
               'w5_50': [int(x * 45 / div) * mul + 5 for x in range(div)],
               'w7_21': [int(x * 14 / div) * mul + 7 for x in range(div)],
               'w7_28': [int(x * 21 / div) * mul + 7 for x in range(div)],
               'w10_30': [int(x * 20 / div) * mul + 10 for x in range(div)],
               'w10_40': [int(x * 30 / div) * mul + 10 for x in range(div)],
               'w10_50': [int(x * 40 / div) * mul + 10 for x in range(div)],
               'w20_40': [int(x * 20 / div) * mul + 20 for x in range(div)],
               'w14_56': [int(x * 42 / div) * mul + 14 for x in range(div)],
               'w25_50': [int(x * 25 / div) * mul + 25 for x in range(div)],
               'w25_75': [int(x * 50 / div) * mul + 25 for x in range(div)],
               'w100': [int(x * 99 / div) * mul + 1 for x in range(div)],
               'w50_90': [int(x * 40 / div) * mul + 50 for x in range(div)],
               'w60_90': [int(x * 30 / div) * mul + 60 for x in range(div)],
               'w60_100': [int(x * 40 / div) * mul + 60 for x in range(div)],
               'w80_100': [int(x * 20 / div) * mul + 80 for x in range(div)],
               'wn80_100': [(int(x * 20 / div) * mul + 80) * -1 for x in range(div)],
               'w060_100': [round(x * 0.4 / div + 0.6, 2) for x in range(div)],
               'w040': [round(x * 0.4 / div, 2) for x in range(div)],
               'w90_130': [int(x * 40 / div) * mul + 90 for x in range(div)],
               'wn90_130': [-1 * (int(x * 40 / div) * mul + 90) for x in range(div)],
               'w010': [round(x * 0.09 / div, 2) * mul + 0.01 for x in range(div)],
               'w005': [round(x * 0.04 / div, 2) * mul + 0.01 for x in range(div)],
               'w05': [round(x * 0.4 / div, 1) + 0.1 for x in range(div)],
               'wn150': [int(x * 150 / div) * mul * (-1) for x in range(div)],
               'w15_150': [int(x * 135 / div) * mul + 15 for x in range(div)]}

    p_dic = {
        'uo_1': list(shuffle(ran_dic['w14'])),
        'uo_2': list(shuffle(ran_dic['w7_28'])),
        'uo_3': list(shuffle(ran_dic['w14_56'])),
        'uo_s': list(shuffle(ran_dic['w50_90'])),
        'uo_b': list(shuffle(ran_dic['w10_50'])),
        'so_w': list(shuffle(ran_dic['w28'])),
        'so_sw': list(shuffle(ran_dic['w7'])),
        'so_s': list(shuffle(ran_dic['w60_90'])),
        'so_b': list(shuffle(ran_dic['w5_40'])),
        'ao_1': list(shuffle(ran_dic['w10'])),
        'ao_2': list(shuffle(ran_dic['w14_56'])),
        'wi_w': list(shuffle(ran_dic['w7_28'])),
        'wi_s': list(shuffle(ran_dic['wn20'])),
        'wi_b': list(shuffle(ran_dic['wn80_100'])),
        'srsi_w': list(shuffle(ran_dic['w7_28'])),
        'srsi_kw': list(shuffle(ran_dic['w7'])),
        'srsi_dw': list(shuffle(ran_dic['w7'])),
        'srsi_s': list(shuffle(ran_dic['w060_100'])),
        'srsi_b': list(shuffle(ran_dic['w040'])),
        'po_sw': list(shuffle(ran_dic['w14_56'])),
        'po_fw': list(shuffle(ran_dic['w7_28'])),
        'po_sg': list(shuffle(ran_dic['w14'])),
        'pvo_sw': list(shuffle(ran_dic['w14_56'])),
        'pvo_fw': list(shuffle(ran_dic['w7_28'])),
        'pvo_sg': list(shuffle(ran_dic['w14'])),
        'ai_w': list(shuffle(ran_dic['w10_40'])),
        'macd_fw': list(shuffle(ran_dic['w5_20'])),
        'macd_sw': list(shuffle(ran_dic['w14_56'])),
        'macd_sg': list(shuffle(ran_dic['w14'])),
        'cci_w': list(shuffle(ran_dic['w10_30'])),
        'cci_b': list(shuffle(ran_dic['wn90_130'])),
        'cci_s': list(shuffle(ran_dic['w90_130'])),
        'ii_1': list(shuffle(ran_dic['w5_15'])),
        'ii_2': list(shuffle(ran_dic['w10_40'])),
        'ii_3': list(shuffle(ran_dic['w25_75'])),
        'kst_r1': list(shuffle(ran_dic['w5_15'])),
        'kst_r2': list(shuffle(ran_dic['w7_21'])),
        'kst_r3': list(shuffle(ran_dic['w10_30'])),
        'kst_r4': list(shuffle(ran_dic['w20_40'])),
        'kst_1': list(shuffle(ran_dic['w5_15'])),
        'kst_2': list(shuffle(ran_dic['w5_15'])),
        'kst_3': list(shuffle(ran_dic['w5_15'])),
        'kst_4': list(shuffle(ran_dic['w7_21'])),
        'kst_ns': list(shuffle(ran_dic['w5_15'])),
        'psar_st': list(shuffle(ran_dic['w005'])),
        'psar_ms': list(shuffle(ran_dic['w05'])),
        'stc_fw': list(shuffle(ran_dic['w10_30'])),
        'stc_sw': list(shuffle(ran_dic['w25_75'])),
        'stc_c': list(shuffle(ran_dic['w5_15'])),
        'stc_s1': list(shuffle(ran_dic['w7'])),
        'stc_s2': list(shuffle(ran_dic['w7'])),
        'vi_w': list(shuffle(ran_dic['w28'])),
        'eom_w': list(shuffle(ran_dic['w28'])),
        'sma_w': list(shuffle(ran_dic['w100'])),
        'sma_fw': list(shuffle(ran_dic['w3_30'])),
        'ema_w': list(shuffle(ran_dic['w100'])),
        'ema_fw': list(shuffle(ran_dic['w3_30'])),
        'wma_w': list(shuffle(ran_dic['w100'])),
        'wma_fw': list(shuffle(ran_dic['w3_30'])),
        'kst_b': list(shuffle(ran_dic['wn150'])),
        'kst_s': list(shuffle(ran_dic['w15_150'])),
        'stc_ll': list(shuffle(ran_dic['w40'])),
        'stc_hl': list(shuffle(ran_dic['w60_100'])),
        'adi_fw': list(shuffle(ran_dic['w5_20'])),
        'adi_sw': list(shuffle(ran_dic['w10_40'])),
        'adi_sg': list(shuffle(ran_dic['w14'])),
        'obv_fw': list(shuffle(ran_dic['w5_20'])),
        'obv_sw': list(shuffle(ran_dic['w10_40'])),
        'obv_sg': list(shuffle(ran_dic['w14'])),
        'eom_sma': list(shuffle(ran_dic['w14'])),
        'vpt_sma': list(shuffle(ran_dic['w14'])),
        'vi2_w': list(shuffle(ran_dic['w28'])),
        'obv2_fw': list(shuffle(ran_dic['w5_20'])),
        'obv2_sw': list(shuffle(ran_dic['w10_40'])),
        'obv2_sg': list(shuffle(ran_dic['w14'])),
        'ao2_1': list(shuffle(ran_dic['w10'])),
        'ao2_2': list(shuffle(ran_dic['w14_56'])),
        'macd2_fw': list(shuffle(ran_dic['w5_20'])),
        'macd2_sw': list(shuffle(ran_dic['w14_56'])),
        'macd2_sg': list(shuffle(ran_dic['w14'])),
        'po2_sw': list(shuffle(ran_dic['w14_56'])),
        'po2_fw': list(shuffle(ran_dic['w7_28'])),
        'po2_sg': list(shuffle(ran_dic['w14'])),
        'pvo2_sw': list(shuffle(ran_dic['w14_56'])),
        'pvo2_fw': list(shuffle(ran_dic['w7_28'])),
        'pvo2_sg': list(shuffle(ran_dic['w14'])),
        'psar2_st': list(shuffle(ran_dic['w005'])),
        'psar2_ms': list(shuffle(ran_dic['w05'])),
        'stc2_fw': list(shuffle(ran_dic['w10_30'])),
        'stc2_sw': list(shuffle(ran_dic['w25_75'])),
        'stc2_c': list(shuffle(ran_dic['w5_15'])),
        'stc2_s1': list(shuffle(ran_dic['w7'])),
        'stc2_s2': list(shuffle(ran_dic['w7'])),
        'stc2_ll': list(shuffle(ran_dic['w40'])),
        'stc2_hl': list(shuffle(ran_dic['w60_100'])),
        'adi2_fw': list(shuffle(ran_dic['w5_20'])),
        'adi2_sw': list(shuffle(ran_dic['w5_40'])),
        'adi2_sg': list(shuffle(ran_dic['w14'])),
        'sma2_w': list(shuffle(ran_dic['w100'])),
        'sma2_fw': list(shuffle(ran_dic['w3_30'])),
        'ema2_w': list(shuffle(ran_dic['w100'])),
        'ema2_fw': list(shuffle(ran_dic['w3_30'])),
        'wma2_w': list(shuffle(ran_dic['w100'])),
        'wma2_fw': list(shuffle(ran_dic['w3_30']))}
    dfa = pd.DataFrame(p_dic)
    dfa = dfa.astype(dfa_types)
    dfa.loc[0] = dfa_0
    return dfa

def stop_fun(stop_loss):

    if isinstance(stop_loss, str) and stop_loss == 'tr':
        return get_ret_3, stop_loss
    elif isinstance(stop_loss, int) or isinstance(stop_loss, float):
        stop = 1 - stop_loss / 100
        if stop_loss == 0:
            return get_ret_1, stop
        else:
            return get_ret_2, stop

def save(loc_folder, result_filename, dfr, dfa, df, n, acc, equity, score, alpha, alpha_rate, delta, minmax, criteria,
         market, beta, avg_acc,
         avg_acc_e, ind_list, y):
    current_time = pd.Timestamp('now')
    new_row = {'Criteria': criteria, 'Score': round(score, 6), 'Alpha': round(alpha * 100, 2),
               'Alpha Trade Rate': alpha_rate * 100,
               'Alpha Max Days': round(delta, 2), 'Max Min Delta': round(minmax, 2), 'Beta': round(beta, 4),
               'Transactions': n, 'Accuracy': round(acc, 2), 'Av. Acc': round(avg_acc, 2),
               'Av.Acc.Error': round(avg_acc_e, 2),
               'Final Equity': round(equity, 2), 'Market Return': market, 'Zone': df.columns[ind_list[0]],
               'Signal': df.columns[ind_list[1]]}

    new_row = pd.Series(new_row)
    new_row = pd.concat([new_row, dfa.iloc[y, :]])
    dfr.loc[current_time] = new_row

    dfr = dfr.reindex(c_index, axis=1)
    dfr.to_csv(f'{loc_folder}{result_filename}.csv')
    dfr.to_pickle(f'{loc_folder}{result_filename}.pkl')
    print(f'\nA better strategy was found!')
    return dfr


def psar_s(df, args):  # psar_st	psar_ms
    df['PSAR'] = get_psar(df, args[0], args[1])

    df.loc[df.PSAR > df.Close, 'PSAR_T'] = 1
    df.loc[df.PSAR < df.Close, 'PSAR_T'] = -1
    df.loc[df.PSAR > df.Close, 'PSAR2_T'] = 1
    df.loc[df.PSAR < df.Close, 'PSAR2_T'] = -1
    df = df.fillna(0)
    df.loc[(df.PSAR_T.shift(1) < 0) & (df.PSAR_T > 0), 'PSAR_A'] = 1
    df.loc[(df.PSAR_T.shift(1) > 0) & (df.PSAR_T < 0), 'PSAR_A'] = -1
    df.drop(columns=['PSAR'], inplace=True)
    return df


def adi_s(df, args):  # adi_fw	adi_sw	adi_sg
    adi = AccDistIndexIndicator(high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                volume=df['Volume'])
    df['ADI'] = adi.acc_dist_index()
    df['ADI_MACD'] = df.ADI.ewm(
        span=args[0], adjust=False).mean() - df.ADI.ewm(span=args[1],
                                                        adjust=False).mean()
    df['ADI_MACD_S'] = df.ADI_MACD.rolling(args[2]).mean()
    df.loc[df.ADI_MACD < df.ADI_MACD_S, 'ADI_T'] = 1
    df.loc[df.ADI_MACD > df.ADI_MACD_S, 'ADI_T'] = -1
    df.loc[df.ADI_MACD < df.ADI_MACD_S, 'ADI2_T'] = 1
    df.loc[df.ADI_MACD > df.ADI_MACD_S, 'ADI2_T'] = -1
    df.loc[(df.ADI_MACD.shift(1) > df.ADI_MACD_S.shift(1)) &
           (df.ADI_MACD < df.ADI_MACD_S), 'ADI_A'] = 1
    df.loc[(df.ADI_MACD.shift(1) < df.ADI_MACD_S.shift(1)) &
           (df.ADI_MACD > df.ADI_MACD_S), 'ADI_A'] = -1
    df.drop(columns=['ADI', 'ADI_MACD', 'ADI_MACD_S'], inplace=True)
    return df


def obv_s(df, args):  # obv_fw obv_sw obv_sg
    obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
    df['OBV'] = obv.on_balance_volume()
    df['OBV_MACD'] = df.OBV.ewm(
        span=args[0], adjust=False).mean() - df.OBV.ewm(span=args[1],
                                                        adjust=False).mean()
    df['OBV_MACD_S'] = df.OBV_MACD.rolling(args[2]).mean()
    df.loc[df.OBV_MACD < df.OBV_MACD_S, 'OBV_T'] = 1
    df.loc[df.OBV_MACD > df.OBV_MACD_S, 'OBV_T'] = -1
    df.loc[df.OBV_MACD < df.OBV_MACD_S, 'OBV2_T'] = 1
    df.loc[df.OBV_MACD > df.OBV_MACD_S, 'OBV2_T'] = -1
    df.loc[(df.OBV_MACD.shift(1) > df.OBV_MACD_S.shift(1)) &
           (df.OBV_MACD < df.OBV_MACD_S), 'OBV_A'] = 1
    df.loc[(df.OBV_MACD.shift(1) < df.OBV_MACD_S.shift(1)) &
           (df.OBV_MACD > df.OBV_MACD_S), 'OBV_A'] = -1
    df.drop(columns=[
        'OBV',
        'OBV_MACD',
        'OBV_MACD_S',
    ], inplace=True)
    return df


def eom_s(df, args):  # eom_w eom_sma
    eom = EaseOfMovementIndicator(high=df['High'],
                                  low=df['Low'],
                                  volume=df['Volume'],
                                  window=args[0])
    df['EOM'] = eom.ease_of_movement()
    df['EOM_S'] = df['EOM'].rolling(args[1]).mean()
    df.loc[(df.EOM.shift(1) > df.EOM_S) & (df.EOM < df.EOM_S), 'EOM_A'] = 1
    df.loc[(df.EOM.shift(1) < df.EOM_S) & (df.EOM > df.EOM_S), 'EOM_A'] = -1
    df.drop(columns=['EOM', 'EOM_S'], inplace=True)
    return df


def vpt_s(df, args):  # vpt_sma adx_l
    vpt = VolumePriceTrendIndicator(close=df['Close'], volume=df['Volume'])
    adx = ADXIndicator(high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       window=args[0])
    df['VPT'] = vpt.volume_price_trend()
    df['ADX'] = adx.adx()
    df['VPT_S'] = df['VPT'].rolling(args[0]).mean()
    df.loc[(df.VPT.shift(1) > df.VPT_S) & (df.VPT < df.VPT_S), 'VPT_A'] = 1
    df.loc[(df.VPT.shift(1) < df.VPT_S) & (df.VPT > df.VPT_S), 'VPT_A'] = -1
    df.drop(columns=['VPT', 'VPT_S', 'ADX'], inplace=True)
    return df


def sma_s(df, args):  # sma_w sma_fw
    df['SMA'] = df['Close'].rolling(args[0]).mean()
    df['SMA_S'] = df['Close'].rolling(args[1]).mean()
    df.loc[(df.SMA_S > df.SMA), 'SMA_T'] = 1
    df.loc[(df.SMA_S < df.SMA), 'SMA_T'] = -1
    df.loc[(df.SMA_S > df.SMA), 'SMA2_T'] = 1
    df.loc[(df.SMA_S < df.SMA), 'SMA2_T'] = -1
    df.loc[(df.SMA_S.shift(1) > df.SMA.shift(1)) & (df.SMA_S < df.SMA), 'SMA_A'] = 1
    df.loc[(df.SMA_S.shift(1) < df.SMA.shift(1)) & (df.SMA_S > df.SMA), 'SMA_A'] = -1
    df.drop(columns=['SMA', 'SMA_S'], inplace=True)
    return df


def ema_s(df, args):  # ema_w, ema_fw
    df['EMA'] = df['Close'].ewm(span=args[0],
                                min_periods=0,
                                adjust=False,
                                ignore_na=False).mean()
    df['EMA_S'] = df['Close'].ewm(span=args[1], min_periods=0, adjust=False, ignore_na=False).mean()
    df.loc[(df.EMA_S > df.EMA), 'EMA_T'] = 1
    df.loc[(df.EMA_S < df.EMA), 'EMA_T'] = -1
    df.loc[(df.EMA_S.shift(1) > df.EMA.shift(1)) & (df.EMA_S < df.EMA), 'EMA_A'] = 1
    df.loc[(df.EMA_S.shift(1) < df.EMA.shift(1)) & (df.EMA_S > df.EMA), 'EMA_A'] = -1
    df.loc[(df.EMA_S > df.EMA), 'EMA2_T'] = 1
    df.loc[(df.EMA_S < df.EMA), 'EMA2_T'] = -1
    df.drop(columns=['EMA', 'EMA_S'], inplace=True)
    return df

def rsi_s(df, args):  # rsi_w (Is not Used)
    rsi = RSIIndicator(close=df['Close'], window=args[0])
    df['RSI'] = rsi.rsi()
    df.loc[(df.RSI > args[2]), 'RSI_Z'] = 1
    df.loc[(df.RSI < args[1]), 'RSI_Z'] = -1
    df.drop(columns=['RSI'], inplace=True)
    return df


def tsi_s(df, args):  # tsi_sw, tsi_fw, tsi_sig, tsi_s
    tsi = TSIIndicator(close=df['Close'],
                       window_slow=args[0],
                       window_fast=args[1])
    df['TSI'] = tsi.tsi()
    df['TSI_S'] = df['TSI'].ewm(args[2],
                                min_periods=0,
                                adjust=False,
                                ignore_na=False).mean()
    df.loc[(df.TSI > args[3]), 'TSI_Z'] = 1
    df.loc[(df.TSI < -args[3]), 'TSI_Z'] = -1
    df.loc[(df.TSI < 0), 'TSI_T'] = 1
    df.loc[(df.TSI >= 0), 'TSI_T'] = -1
    df.loc[(df.TSI.shift(1) > df.TSI_S.shift(1)) & (df.TSI < df.TSI_S),
           'TSI_A'] = 1
    df.loc[(df.TSI.shift(1) < df.TSI_S.shift(1)) & (df.TSI > df.TSI_S),
           'TSI_A'] = -1
    df.drop(columns=['TSI', 'TSI_S'], inplace=True)
    return df


def kst_s(
        df, args
):  # kst_r1, kst_r2, kst_r3, kst_r4, kst_1, kst_2, kst_3, kst_4, kst_ns, kst_b, kst_s
    kst = KSTIndicator(close=df['Close'],
                       roc1=args[0],
                       roc2=args[1],
                       roc3=args[2],
                       roc4=args[3],
                       window1=args[4],
                       window2=args[5],
                       window3=args[6],
                       window4=args[7],
                       nsig=args[8])
    df['KST'] = kst.kst()
    df['KST_S'] = kst.kst_sig()
    df['KST_H'] = kst.kst_diff()
    df.loc[df.KST_H >= 0, 'KST_T'] = -1
    df.loc[df.KST_H < 0, 'KST_T'] = 1
    df.loc[(df.KST.shift(1) > df.KST_S.shift(1)) & (df.KST < df.KST_S) &
           (df.KST.shift(1) > args[10]), 'KST_A'] = 1
    df.loc[(df.KST.shift(1) < df.KST_S.shift(1)) & (df.KST > df.KST_S) &
           (df.KST.shift(1) < args[9]), 'KST_A'] = -1
    df.drop(columns=['KST', 'KST_S', 'KST_H'], inplace=True)
    return df


def stc_s(df, args):  # stc_fw, stc_sw, stc_c, stc_s1, stc_s2, stc_ll, stc_hl
    stc = STCIndicator(close=df['Close'],
                       window_fast=args[0],
                       window_slow=args[1],
                       cycle=args[2],
                       smooth1=args[3],
                       smooth2=args[4])
    df['STC'] = stc.stc()
    df.loc[df.STC < args[6], 'STC_T'] = 1
    df.loc[df.STC > args[5], 'STC_T'] = -1
    df.loc[df.STC < args[6], 'STC2_T'] = 1
    df.loc[df.STC > args[5], 'STC2_T'] = -1
    df.loc[(df.STC.shift(1) > args[6]) & (df.STC < args[6]), 'STC_A'] = 1
    df.loc[(df.STC.shift(1) < args[5]) & (df.STC > args[5]), 'STC_A'] = -1
    df.drop(columns=['STC'], inplace=True)
    return df


def cmf_s(df, args):  # cmf_w, cmf_l
    cmf = ChaikinMoneyFlowIndicator(high=df['High'],
                                    low=df['Low'],
                                    close=df['Close'],
                                    volume=df['Volume'],
                                    window=args[0])
    df['CMF'] = cmf.chaikin_money_flow()
    df.loc[(df.CMF > args[1]), 'CMF_Z'] = 1
    df.loc[(df.CMF < -args[1]), 'CMF_Z'] = -1
    df.loc[(df.CMF.shift(1) > 0) & (df.CMF < 0), 'CMF_A'] = 1
    df.loc[(df.CMF.shift(1) < 0) & (df.CMF > 0), 'CMF_A'] = -1
    df.drop(columns=['CMF'], inplace=True)
    return df


def fi_s(df, args):  # fi_w, std_m
    fi = ForceIndexIndicator(close=df['Close'],
                             volume=df['Volume'],
                             window=args[0])
    df['FI'] = fi.force_index()
    fi_s = int(df[df['FI'] > 0]['FI'].mean() +
               3.5 * df[df['FI'] > 0]['FI'].mean() * args[1])
    fi_b = int(df[df['FI'] < 0]['FI'].mean() +
               2.5 * df[df['FI'] < 0]['FI'].mean() * args[1])
    df.loc[df.FI > fi_s, 'FI_Z'] = 1
    df.loc[df.FI < fi_b, 'FI_Z'] = -1
    df.loc[(df.FI.shift(1) > fi_s) & (df.FI < fi_s), 'FI_A'] = 1
    df.loc[(df.FI.shift(1) < fi_b) & (df.FI > fi_b), 'FI_A'] = -1
    df.drop(columns=['FI'], inplace=True)
    return df


def mfi_s(df, args):  # mfi_w, mfi_b, mfi_s
    mfi = MFIIndicator(high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       volume=df['Volume'],
                       window=args[0])
    df['MFI'] = mfi.money_flow_index()
    df.loc[(df.MFI > args[2]), 'MFI_Z'] = 1
    df.loc[(df.MFI < args[1]), 'MFI_Z'] = -1
    df.drop(columns=['MFI'], inplace=True)
    return df


def uo_s(df, args):  # uo_1, uo_2, uo_3, uo_s, uo_b
    uo = UltimateOscillator(high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            window1=args[0],
                            window2=args[1],
                            window3=args[2])
    df['UO'] = uo.ultimate_oscillator()
    df.loc[(df.UO > args[3]), 'UO_Z'] = 1
    df.loc[(df.UO < args[4]), 'UO_Z'] = -1
    df.drop(columns=['UO'], inplace=True)
    return df


def so_s(df, args):  # so_w	so_sw so_s so_b
    so = StochasticOscillator(high=df['High'],
                              low=df['Low'],
                              close=df['Close'],
                              window=args[0],
                              smooth_window=args[1])
    df['SO'] = so.stoch()
    df['SOS'] = so.stoch_signal()
    df.loc[(df.SO > args[2]), 'SO_Z'] = 1
    df.loc[(df.SO < args[3]), 'SO_Z'] = -1
    df.loc[(df.SO.shift(1) > df.SOS.shift(1)) & (df.SO < df.SOS), 'SO_A'] = 1
    df.loc[(df.SO.shift(1) < df.SOS.shift(1)) & (df.SO > df.SOS), 'SO_A'] = -1
    df.drop(columns=['SO', 'SOS'], inplace=True)
    return df


def ki_s(df, args):  # ki_w, ki_p1, ki_p2, ki_p3, ki_b, ki_s
    ki = KAMAIndicator(close=df['Close'],
                       window=args[0],
                       pow1=args[1],
                       pow2=args[2])
    ki_sig = KAMAIndicator(close=df['Close'],
                           window=args[0],
                           pow1=args[3],
                           pow2=args[2])
    df['KI'] = ki.kama()
    df['KIS'] = ki_sig.kama()
    df.loc[(df.Close > df.KI * (1 + args[5])), 'KI_Z'] = 1
    df.loc[(df.Close < df.KI * (1 - args[4])), 'KI_Z'] = -1
    df.loc[(df.Close.shift(1) > df.KI.shift(1)) & (df.Close < df.KI *
                                                   (1 - args[4])), 'KI_T'] = 1
    df.loc[(df.Close.shift(1) < df.KI.shift(1)) & (df.Close > df.KI *
                                                   (1 + args[5])), 'KI_T'] = -1
    df['KI_T'] = df['KI_T'].fillna(method='ffill')
    df.loc[(df.KI > df.KIS.shift(1)) & (df.KI < df.KIS), 'KI_A'] = 1
    df.loc[(df.KI < df.KIS.shift(1)) & (df.KI > df.KIS), 'KI_A'] = -1
    df.drop(columns=['KI', 'KIS'], inplace=True)
    return df


def roc_s(df, args):  # roc_w, roc_b, roc_s
    roc = ROCIndicator(close=df['Close'], window=args[0])
    df['ROC'] = roc.roc()
    df.loc[(df.ROC > args[2]), 'ROC_Z'] = 1
    df.loc[(df.ROC < args[1]), 'ROC_Z'] = -1
    df.loc[(df.ROC < 0), 'ROC_T'] = 1
    df.loc[(df.ROC >= 0), 'ROC_T'] = -1
    df.drop(columns=['ROC'], inplace=True)
    return df


def ao_s(df, args):  # ao_1, ao_2
    ao = AwesomeOscillatorIndicator(high=df['High'],
                                    low=df['Low'],
                                    window1=args[0],
                                    window2=args[1])
    df['AO'] = ao.awesome_oscillator()
    df.loc[(df.AO < 0), 'AO_T'] = 1
    df.loc[(df.AO >= 0), 'AO_T'] = -1
    df.loc[(df.AO < 0), 'AO2_T'] = 1
    df.loc[(df.AO >= 0), 'AO2_T'] = -1
    df.loc[((df.AO.shift(1) > 0) & (df.AO < 0)) |
           ((df.AO < 0) & (df.AO.shift(2) < df.AO.shift(1)) &
            (df.AO.shift(1) > df.AO)), 'AO_A'] = 1
    df.loc[((df.AO.shift(1) < 0) & (df.AO > 0)) |
           ((df.AO > 0) & (df.AO.shift(2) > df.AO.shift(1)) &
            (df.AO.shift(1) < df.AO)), 'AO_A'] = -1
    df.drop(columns=['AO'], inplace=True)
    return df


def wi_s(df, args):  # wi_w	wi_s wi_b
    wi = WilliamsRIndicator(high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            lbp=args[0])
    df['WI'] = wi.williams_r()
    df.loc[(df.WI > args[1]), 'WI_Z'] = 1
    df.loc[(df.WI < args[2]), 'WI_Z'] = -1
    df.drop(columns=['WI'], inplace=True)
    return df


def srsi_s(df, args):  # srsi_w srsi_kw	srsi_dw	srsi_s srsi_b
    srsi = StochRSIIndicator(close=df['Close'],
                             window=args[0],
                             smooth1=args[1],
                             smooth2=args[2])
    df['SRSI'] = srsi.stochrsi()
    df['SRSI_K'] = srsi.stochrsi_k()
    df['SRSI_D'] = srsi.stochrsi_d()
    df.loc[(df.SRSI > args[3]), 'SRSI_Z'] = 1
    df.loc[(df.SRSI < args[4]), 'SRSI_Z'] = -1
    df.loc[(df.SRSI_K.shift(1) > df.SRSI_D.shift(1)) & (df.SRSI_K < df.SRSI_D),
           'SRSI_A'] = 1
    df.loc[(df.SRSI_K.shift(1) < df.SRSI_D.shift(1)) & (df.SRSI_K > df.SRSI_D),
           'SRSI_A'] = -1
    df.drop(columns=['SRSI_K', 'SRSI_D', 'SRSI'], inplace=True)
    return df


def po_s(df, args):  # po_sw po_fw po_sg

    po = PercentagePriceOscillator(close=df['Close'],
                                   window_slow=args[0],
                                   window_fast=args[1],
                                   window_sign=args[2])
    df['PO'] = po.ppo()
    df['PO_S'] = po.ppo_signal()
    df['PO_H'] = po.ppo_hist()
    df.loc[(df.PO_H < 0), 'PO_T'] = 1
    df.loc[(df.PO_H >= 0), 'PO_T'] = -1
    df.loc[(df.PO_H < 0), 'PO2_T'] = 1
    df.loc[(df.PO_H >= 0), 'PO2_T'] = -1
    df.loc[(df.PO.shift(1) > df.PO_S.shift(1)) & (df.PO < df.PO_S), 'PO_A'] = 1
    df.loc[(df.PO.shift(1) < df.PO_S.shift(1)) & (df.PO > df.PO_S),
           'PO_A'] = -1
    df.drop(columns=['PO', 'PO_S', 'PO_H'], inplace=True)
    return df


def pvo_s(df, args):  # pvo_sw	pvo_fw	pvo_sg
    pvo = PercentageVolumeOscillator(volume=df['Volume'],
                                     window_slow=args[0],
                                     window_fast=args[1],
                                     window_sign=args[2])
    df['PVO'] = pvo.pvo()
    df['PVO_S'] = pvo.pvo_signal()
    df['PVO_H'] = pvo.pvo_hist()
    df.loc[(df.PVO_H < 0), 'PVO_T'] = 1
    df.loc[(df.PVO_H >= 0), 'PVO_T'] = -1
    df.loc[(df.PVO_H < 0), 'PVO2_T'] = 1
    df.loc[(df.PVO_H >= 0), 'PVO2_T'] = -1
    df.loc[(df.PVO.shift(1) > df.PVO_S.shift(1)) & (df.PVO < df.PVO_S),
           'PVO_A'] = 1
    df.loc[(df.PVO.shift(1) < df.PVO_S.shift(1)) & (df.PVO > df.PVO_S),
           'PVO_A'] = -1
    df.drop(columns=['PVO', 'PVO_S', 'PVO_H'], inplace=True)
    return df


def atr_s(df, args):  # atr_w atr_l rsi_w rsi_b rsi_s
    atr = AverageTrueRange(high=df['High'],
                           low=df['Low'],
                           close=df['Close'],
                           window=args[0])
    rsi = RSIIndicator(close=df['Close'], window=args[2])
    df['RSI'] = rsi.rsi()
    df['ATR'] = atr.average_true_range()
    df.loc[((df.ATR.shift(1) * (args[0] - 1) + df.ATR) / args[0] > args[1]) &
           (df.RSI > args[4]), 'ATR_Z'] = 1
    df.loc[((df.ATR.shift(1) * (args[0] - 1) + df.ATR) / args[0] > args[1]) &
           (df.RSI < args[3]), 'ATR_Z'] = -1
    df.drop(columns=['ATR', 'RSI'], inplace=True)
    return df


def bb_s(df, args):  # bb_w	bb_d bb_l
    bb = BollingerBands(close=df['Close'], window=args[0], window_dev=args[1])
    adx = ADXIndicator(high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       window=args[0])
    df['BB_H'] = bb.bollinger_hband_indicator()
    df['BB_L'] = bb.bollinger_lband_indicator()
    df['BB_W'] = bb.bollinger_wband()
    df['ADX_P'] = adx.adx_pos()
    df['ADX_N'] = adx.adx_neg()
    df.loc[(df.ADX_P < df.ADX_N), 'ADX_T'] = 1
    df.loc[(df.ADX_P > df.ADX_N), 'ADX_T'] = -1
    df.loc[(df.BB_W > args[2]), 'BB_Z'] = 1
    df.loc[(df.BB_W > args[2]), 'BB_Z'] = -1
    df.loc[(df.Close >= df.BB_H) & (df.ADX_T > 0), 'BB_T'] = 1
    df.loc[(df.Close <= df.BB_L) & (df.ADX_T < 0), 'BB_T'] = -1
    df['BB_T'] = df['BB_T'].fillna(method='ffill')
    df.loc[(df.Close >= df.BB_H), 'BB_A'] = 1
    df.loc[(df.Close <= df.BB_L), 'BB_A'] = -1
    df.drop(columns=['BB_H', 'BB_L', 'BB_W', 'ADX_P', 'ADX_N', 'ADX_T'],
            inplace=True)
    return df


def kc_s(df, args):  # kc_w	kc_a
    kc = KeltnerChannel(high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        window=args[0],
                        window_atr=args[1])
    df['KC_H'] = kc.keltner_channel_hband_indicator()
    df['KC_L'] = kc.keltner_channel_lband_indicator()
    df.loc[df.KC_L > 0, 'KC_T'] = 1
    df.loc[df.KC_H > 0, 'KC_T'] = -1
    df['KC_T'] = df['KC_T'].fillna(method='ffill')
    df.drop(columns=['KC_H', 'KC_L'], inplace=True)
    return df


def dc_s(df, args):  # dc_w	dc_l
    dc = DonchianChannel(high=df['High'],
                         low=df['Low'],
                         close=df['Close'],
                         window=args[0])
    adx = ADXIndicator(high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       window=args[0])
    df['DC_H'] = dc.donchian_channel_hband()
    df['DC_L'] = dc.donchian_channel_lband()
    df['DC_W'] = dc.donchian_channel_wband()
    df['ADX_P'] = adx.adx_pos()
    df['ADX_N'] = adx.adx_neg()
    df.loc[(df.ADX_P < df.ADX_N), 'ADX_T'] = 1
    df.loc[(df.ADX_P > df.ADX_N), 'ADX_T'] = -1
    df.loc[(df.DC_W.shift(1) < args[1]), 'DC_Z'] = 1
    df.loc[(df.DC_W.shift(1) > args[1]), 'DC_Z'] = -1
    df.loc[(df.High >= df.DC_H) & (df.ADX_T > 0), 'DC_T'] = 1
    df.loc[(df.Low <= df.DC_L) & (df.ADX_T < 0), 'DC_T'] = -1
    df['DC_T'] = df['DC_T'].fillna(method='ffill')
    df.loc[(df.Close >= df.DC_H), 'DC_A'] = 1
    df.loc[(df.Close <= df.DC_L), 'DC_A'] = -1
    df.drop(columns=['DC_H', 'DC_L', 'DC_W', 'ADX_P', 'ADX_N', 'ADX_T'],
            inplace=True)
    return df


def ui_s(df, args):  # ui_w	ui_b ui_s
    ui = UlcerIndex(close=df['Close'], window=args[0])
    df['UI'] = ui.ulcer_index()
    df.loc[(df.UI < args[2]), 'UI_Z'] = 1
    df.loc[(df.UI > args[1]), 'UI_Z'] = -1
    df.drop(columns=['UI'], inplace=True)
    return df


def adx_s(df, args):  # adx_w	adx_l
    adx = ADXIndicator(high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       window=args[0])
    df['ADX'] = adx.adx()
    df['ADX_P'] = adx.adx_pos()
    df['ADX_N'] = adx.adx_neg()
    df.loc[(df.ADX_P < df.ADX_N), 'ADX_T'] = 1
    df.loc[(df.ADX_P > df.ADX_N), 'ADX_T'] = -1
    df.loc[(df.ADX_T > 0) & df.ADX > args[1], 'ADX_Z'] = 1
    df.loc[(df.ADX_T < 0) & df.ADX > args[1], 'ADX_Z'] = -1
    df.drop(columns=['ADX', 'ADX_P', 'ADX_N'], inplace=True)
    return df


def ai_s(df, args):  # ai_w
    ai = AroonIndicator(close=df['Close'], window=args[0])
    df['AI_U'] = ai.aroon_up()
    df['AI_D'] = ai.aroon_down()
    df.loc[(df.AI_U.shift(1) > df.AI_D.shift(1)) & (df.AI_U < df.AI_D),
           'AI_A'] = 1
    df.loc[(df.AI_U.shift(1) < df.AI_D.shift(1)) & (df.AI_U > df.AI_D),
           'AI_A'] = -1
    df.drop(columns=['AI_U', 'AI_D'], inplace=True)
    return df


def macd_s(df, args):  # macd_fw macd_sw macd_sg
    macd = MACD(close=df['Close'],
                window_fast=args[0],
                window_slow=args[1],
                window_sign=args[2])
    df['MACD'] = macd.macd()
    df['MACD_S'] = macd.macd_signal()
    df['MACD_H'] = macd.macd_diff()
    df.loc[(df.MACD_H < 0), 'MACD_T'] = 1
    df.loc[(df.MACD_H >= 0), 'MACD_T'] = -1
    df.loc[(df.MACD_H < 0), 'MACD2_T'] = 1
    df.loc[(df.MACD_H >= 0), 'MACD2_T'] = -1
    df.loc[(df.MACD.shift(1) > df.MACD_S.shift(1)) & (df.MACD < df.MACD_S),
           'MACD_A'] = 1
    df.loc[(df.MACD.shift(1) < df.MACD_S.shift(1)) & (df.MACD > df.MACD_S),
           'MACD_A'] = -1
    df.drop(columns=['MACD', 'MACD_S', 'MACD_H'], inplace=True)
    return df


def cci_s(df, args):  # cci_w cci_b cci_s
    cci = CCIIndicator(high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       window=args[0],
                       constant=0.015)
    df['CCI'] = cci.cci()
    df.loc[df.CCI > args[2], 'CCI_Z'] = 1
    df.loc[df.CCI < args[1], 'CCI_Z'] = -1
    df.loc[(df.CCI.shift(1) > args[2]) & (df.CCI < args[2]), 'CCI_A'] = 1
    df.loc[(df.CCI.shift(1) < args[1]) & (df.CCI > args[1]), 'CCI_A'] = -1
    df.drop(columns=['CCI'], inplace=True)
    return df


def dpo_s(df, args):  # dpo_w, dpo_s
    dpo = DPOIndicator(close=df['Close'], window=args[0])
    df['DPO'] = dpo.dpo()
    df.loc[(df.DPO > args[1]), 'DPO_Z'] = 1
    df.loc[(df.DPO < -args[1]), 'DPO_Z'] = -1
    df.drop(columns=['DPO'], inplace=True)
    return df


def mi_s(df, args):  # mi_fw mi_sw std_m
    mi = MassIndex(high=df['High'],
                   low=df['Low'],
                   window_fast=args[0],
                   window_slow=args[1])
    df['MI'] = mi.mass_index()
    mi_s = df.MI.mean() + df.MI.std() * args[2]
    df.loc[df.MI > mi_s, 'MI_Z'] = 1
    df.loc[df.MI < mi_s, 'MI_Z'] = -1
    df.drop(columns=['MI'], inplace=True)
    return df


def ii_s(df, args):  # ii_1 ii_2 ii_3
    ii = IchimokuIndicator(high=df['High'],
                           low=df['Low'],
                           window1=args[0],
                           window2=args[1],
                           window3=args[2])
    df['II'] = ii.ichimoku_conversion_line()
    df['II_S'] = ii.ichimoku_base_line()
    df['II_LA'] = ii.ichimoku_a()
    df['II_LB'] = ii.ichimoku_b()
    df.loc[(df.II > df.II_S), 'II_Z'] = 1
    df.loc[(df.II < df.II_S), 'II_Z'] = -1
    df.loc[(df.Close < df.II_S), 'II_T'] = 1
    df.loc[(df.Close >= df.II_S), 'II_T'] = -1
    df.loc[(df.II.shift(1) > df.II_S.shift(1)) & (df.II < df.II_S) &
           (df.High > df.II_LA), 'II_A'] = 1
    df.loc[(df.II.shift(1) < df.II_S.shift(1)) & (df.II > df.II_S) &
           (df.Low < df.II_LB), 'II_A'] = -1
    df.drop(columns=['II', 'II_S', 'II_LA', 'II_LB'], inplace=True)
    return df


def trix_s(df, args):  # trix_w	trix_sw
    trix = TRIXIndicator(close=df['Close'], window=args[0])
    df['TRIX'] = trix.trix()
    df['TRIX_S'] = df['TRIX'].ewm(span=args[1], adjust=False).mean()
    df.loc[df.TRIX < 0, 'TRIX_T'] = 1
    df.loc[df.TRIX >= 0, 'TRIX_T'] = -1
    df.loc[(df.TRIX.shift(1) > df.TRIX_S.shift(1)) & (df.TRIX < df.TRIX_S),
           'TRIX_A'] = 1
    df.loc[(df.TRIX.shift(1) < df.TRIX_S.shift(1)) & (df.TRIX > df.TRIX_S),
           'TRIX_A'] = -1
    df.drop(columns=['TRIX', 'TRIX_S'], inplace=True)
    return df


def vi_s(df, args):  # vi_w
    vi = VortexIndicator(high=df['High'],
                         low=df['Low'],
                         close=df['Close'],
                         window=args[0])
    df['VI_P'] = vi.vortex_indicator_pos()
    df['VI_N'] = vi.vortex_indicator_neg()
    df['VI_H'] = vi.vortex_indicator_diff()
    df.loc[(df.VI_P < df.VI_N), 'VI_T'] = 1
    df.loc[(df.VI_P > df.VI_N), 'VI_T'] = -1
    df.loc[(df.VI_P < df.VI_N), 'VI2_T'] = 1
    df.loc[(df.VI_P > df.VI_N), 'VI2_T'] = -1
    df.loc[(df.VI_P.shift(1) > df.VI_N.shift(1)) & (df.VI_P < df.VI_N),
           'VI_A'] = 1
    df.loc[(df.VI_P.shift(1) < df.VI_N.shift(1)) & (df.VI_P > df.VI_N),
           'VI_A'] = -1
    df.drop(columns=['VI_H', 'VI_P', 'VI_N'], inplace=True)
    return df


def psar_s(df, args):  # psar_st	psar_ms
    psar = PSARIndicator(high=df['High'], low=df['Low'], close=df['Close'], step=args[0], max_step=args[1])
    df['PSAR'] = psar.psar()
    df.loc[df.PSAR > df.Close, 'PSAR_T'] = 1
    df.loc[df.PSAR < df.Close, 'PSAR_T'] = -1
    df.loc[df.PSAR > df.Close, 'PSAR2_T'] = 1
    df.loc[df.PSAR < df.Close, 'PSAR2_T'] = -1
    df = df.fillna(0)
    df.loc[(df.PSAR_T.shift(1) < 0) & (df.PSAR_T > 0), 'PSAR_A'] = 1
    df.loc[(df.PSAR_T.shift(1) > 0) & (df.PSAR_T < 0), 'PSAR_A'] = -1
    df.drop(columns=['PSAR'], inplace=True)
    return df


def adi_s(df, args):  # adi_fw	adi_sw	adi_sg
    adi = AccDistIndexIndicator(high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                volume=df['Volume'])
    df['ADI'] = adi.acc_dist_index()
    df['ADI_MACD'] = df.ADI.ewm(span=args[0], adjust=False).mean() - df.ADI.ewm(span=args[1], adjust=False).mean()
    df['ADI_MACD_S'] = df.ADI_MACD.rolling(args[2]).mean()
    df.loc[df.ADI_MACD < df.ADI_MACD_S, 'ADI_T'] = 1
    df.loc[df.ADI_MACD > df.ADI_MACD_S, 'ADI_T'] = -1
    df.loc[df.ADI_MACD < df.ADI_MACD_S, 'ADI2_T'] = 1
    df.loc[df.ADI_MACD > df.ADI_MACD_S, 'ADI2_T'] = -1
    df.loc[(df.ADI_MACD.shift(1) > df.ADI_MACD_S.shift(1)) &
           (df.ADI_MACD < df.ADI_MACD_S), 'ADI_A'] = 1
    df.loc[(df.ADI_MACD.shift(1) < df.ADI_MACD_S.shift(1)) &
           (df.ADI_MACD > df.ADI_MACD_S), 'ADI_A'] = -1
    df.drop(columns=['ADI', 'ADI_MACD', 'ADI_MACD_S'], inplace=True)
    return df


def obv_s(df, args):  # obv_fw obv_sw obv_sg
    obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
    df['OBV'] = obv.on_balance_volume()
    df['OBV_MACD'] = df.OBV.ewm(
        span=args[0], adjust=False).mean() - df.OBV.ewm(span=args[1],
                                                        adjust=False).mean()
    df['OBV_MACD_S'] = df.OBV_MACD.rolling(args[2]).mean()
    df.loc[df.OBV_MACD < df.OBV_MACD_S, 'OBV_T'] = 1
    df.loc[df.OBV_MACD > df.OBV_MACD_S, 'OBV_T'] = -1
    df.loc[df.OBV_MACD < df.OBV_MACD_S, 'OBV2_T'] = 1
    df.loc[df.OBV_MACD > df.OBV_MACD_S, 'OBV2_T'] = -1
    df.loc[(df.OBV_MACD.shift(1) > df.OBV_MACD_S.shift(1)) &
           (df.OBV_MACD < df.OBV_MACD_S), 'OBV_A'] = 1
    df.loc[(df.OBV_MACD.shift(1) < df.OBV_MACD_S.shift(1)) &
           (df.OBV_MACD > df.OBV_MACD_S), 'OBV_A'] = -1
    df.drop(columns=[
        'OBV',
        'OBV_MACD',
        'OBV_MACD_S',
    ], inplace=True)
    return df


def eom_s(df, args):  # eom_w eom_sma
    eom = EaseOfMovementIndicator(high=df['High'],
                                  low=df['Low'],
                                  volume=df['Volume'],
                                  window=args[0])
    df['EOM'] = eom.ease_of_movement()
    df['EOM_S'] = df['EOM'].rolling(args[1]).mean()
    df.loc[(df.EOM.shift(1) > df.EOM_S) & (df.EOM < df.EOM_S), 'EOM_A'] = 1
    df.loc[(df.EOM.shift(1) < df.EOM_S) & (df.EOM > df.EOM_S), 'EOM_A'] = -1
    df.drop(columns=['EOM', 'EOM_S'], inplace=True)
    return df


def vpt_s(df, args):  # vpt_sma
    vpt = VolumePriceTrendIndicator(close=df['Close'], volume=df['Volume'])
    df['VPT'] = vpt.volume_price_trend()
    df['VPT_S'] = df['VPT'].rolling(args[0]).mean()
    df.loc[(df.VPT.shift(1) > df.VPT_S) & (df.VPT < df.VPT_S), 'VPT_A'] = 1
    df.loc[(df.VPT.shift(1) < df.VPT_S) & (df.VPT > df.VPT_S), 'VPT_A'] = -1
    df.drop(columns=['VPT', 'VPT_S'], inplace=True)
    return df


def sma_s(df, args):  # sma_w sma_fw
    df['SMA'] = df['Close'].rolling(args[0]).mean()
    df['SMA_S'] = df['Close'].rolling(args[1]).mean()
    df.loc[(df.SMA_S > df.SMA), 'SMA_T'] = 1
    df.loc[(df.SMA_S < df.SMA), 'SMA_T'] = -1
    df.loc[(df.SMA_S > df.SMA), 'SMA2_T'] = 1
    df.loc[(df.SMA_S < df.SMA), 'SMA2_T'] = -1
    df.loc[(df.SMA_S.shift(1) > df.SMA.shift(1)) & (df.SMA_S < df.SMA), 'SMA_A'] = 1
    df.loc[(df.SMA_S.shift(1) < df.SMA.shift(1)) & (df.SMA_S > df.SMA), 'SMA_A'] = -1
    df.drop(columns=['SMA', 'SMA_S'], inplace=True)
    return df


def ema_s(df, args):  # ema_w, ema_fw
    df['EMA'] = df['Close'].ewm(span=args[0],
                                min_periods=0,
                                adjust=False,
                                ignore_na=False).mean()
    df['EMA_S'] = df['Close'].ewm(span=args[1], min_periods=0, adjust=False, ignore_na=False).mean()
    df.loc[(df.EMA_S > df.EMA), 'EMA_T'] = 1
    df.loc[(df.EMA_S < df.EMA), 'EMA_T'] = -1
    df.loc[(df.EMA_S.shift(1) > df.EMA.shift(1)) & (df.EMA_S < df.EMA), 'EMA_A'] = 1
    df.loc[(df.EMA_S.shift(1) < df.EMA.shift(1)) & (df.EMA_S > df.EMA), 'EMA_A'] = -1
    df.loc[(df.EMA_S > df.EMA), 'EMA2_T'] = 1
    df.loc[(df.EMA_S < df.EMA), 'EMA2_T'] = -1
    df.drop(columns=['EMA', 'EMA_S'], inplace=True)
    return df


def set_indicators(df, name, str_dic):
    # Diccionario de funciones:

    f_dic = {
        'RSI_Z': rsi_s,
        'TSI_Z': tsi_s,
        'TSI_T': tsi_s,
        'TSI_A': tsi_s,
        'KST_T': kst_s,
        'KST_A': kst_s,
        'STC_T': stc_s,
        'STC_A': stc_s,
        'CMF_Z': cmf_s,
        'CMF_A': cmf_s,
        'FI_Z': fi_s,
        'FI_A': fi_s,
        'MFI_Z': mfi_s,
        'UO_Z': uo_s,
        'SO_Z': so_s,
        'SO_A': so_s,
        'KI_Z': ki_s,
        'KI_T': ki_s,
        'KI_A': ki_s,
        'ROC_Z': roc_s,
        'ROC_T': roc_s,
        'AO_T': ao_s,
        'AO_A': ao_s,
        'WI_Z': wi_s,
        'SRSI_Z': srsi_s,
        'SRSI_A': srsi_s,
        'PO_T': po_s,
        'PO_A': po_s,
        'PVO_T': pvo_s,
        'PVO_A': pvo_s,
        'ATR_Z': atr_s,
        'BB_Z': bb_s,
        'BB_T': bb_s,
        'BB_A': bb_s,
        'KC_T': kc_s,
        'DC_Z': dc_s,
        'DC_T': dc_s,
        'DC_A': dc_s,
        'UI_Z': ui_s,
        'MACD_T': macd_s,
        'MACD_A': macd_s,
        'ADX_T': adx_s,
        'ADX_Z': adx_s,
        'AI_A': ai_s,
        'CCI_Z': cci_s,
        'CCI_A': cci_s,
        'DPO_Z': dpo_s,
        'MI_Z': mi_s,
        'II_Z': ii_s,
        'II_T': ii_s,
        'II_A': ii_s,
        'TRIX_T': trix_s,
        'TRIX_A': trix_s,
        'VI_T': vi_s,
        'PSAR_T': psar_s,
        'PSAR_A': psar_s,
        'ADI_T': adi_s,
        'ADI_A': adi_s,
        'OBV_T': obv_s,
        'OBV_A': obv_s,
        'EOM_Z': eom_s,
        'EOM_A': eom_s,
        'VPT_T': vpt_s,
        'VPT_A': vpt_s,
        'SMA_T': sma_s,
        'SMA_A': sma_s,
        'EMA_T': ema_s,
        'EMA_A': ema_s
    }

    idx = []

    for x in range(len(str_dic[name])):

        if str_dic[name][x][0] is None:
            continue
        else:

            df = f_dic[str_dic[name][x][0]](df, str_dic[name][x][
                1])  # Aplica los argumentos a una función extraída del diccionario de funciones (f_dic)
            idx.append(str_dic[name][x][0])  # Agrega a la lista idx los indicadores de a 1 (Si hay 1 agrega 1 si hay 3)

    df = df[df.columns.intersection(
        idx)]  # Selecciona las columnas que figuran en la lista
    df = df.fillna(0)

    return df

def tester(df, start_eq, dfl, i_cryptos, desc, str_dic, stop_loss):
    eq_m = df['Close'][-1] / df['Close'][1] * start_eq
    dfr_1 = set_indicators(df, desc, str_dic)
    dfr = pd.concat([df.iloc[:, :5], dfr_1], axis=1)
    dfr = dfr.fillna(method='ffill')
    dfr['Close'] = dfr['Close'].shift(-1)
    col_l = len(dfr.columns) - 5
    ind_list = [x + 5 for x in range(col_l)]
    while len(ind_list) < 2:
        ind_list.insert(0, -1)
    lim = ind_list.count(-1)

    get_ret, stop = stop_fun(stop_loss)
    ret = get_ret(dfr, ind_list, lim, stop)

    eq_b = start_eq
    equi = (1 + ret).cumprod() * start_eq
    if len(equi) > 0:
        eq_b = equi[-1]
    alpha = (eq_b / eq_m - 1) * 100
    result_list = [round(sum(ret), 4), round(eq_b, 2), round(eq_m, 2), round(alpha, 2)]
    # print(f'{coin} Month {t} Ret = {sum(ret):.4f} Eq_Bot = {eq_b:.2f} Eq_Market = {eq_m:.2f} Alpha = {alpha:.2f}%')
    dfl = pd.concat([dfl, pd.DataFrame([result_list], columns=i_cryptos, index=[len(dfl)])])
    #dfl.iloc[t, x * 4:x * 4 + 4] = result_list
    return ret, dfl