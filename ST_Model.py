from tqdm import tqdm
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from Mfun import *

'''
----------------------------------  Technical Analysis Modeling Tool  -----------------------------------

This algorithm will try all technical indicators against an OHLCV time series of an asset. 
Every w loop will define a random list for each parameter of every indicator (windows, spans, smooths, etc) 
and create a dataset where each row is a random list of arguments.
Y loop will take each group of arguments and use them as parameters to setup the indicators and signals, each set of arguments is 1 row of the "dfa" DataFrame.
z loop will try different combinations of indicators for each set of arguments. 
The algorithm will construct a dataframe with the best results (dfr) and stores it every time it gets a better result according to the defined criteria.

'''

pd.set_option('chained_assignment', None)  # To avoid warnings
days = (datetime.strptime(md_end_date, "%Y-%m-%d %H:%M:%S") - datetime.strptime(md_start_date, "%Y-%m-%d %H:%M:%S")).days
score_filename = f'Score Limits {sample} - L{cv_loops} - {d_set} - {days} days.pkl'
coin_list = [cryptos[x].replace("/", "") for x in range(len(cryptos))]
import_name = f'{import_pair.replace("/", "")}-{import_sample}-Price Data-{import_desc}.csv'

try:

    dfl = pd.read_pickle(f'{loc_folder}{score_filename}')

except:

    # This Dataframe will store the scores limits of each coin:
    cv_index = [f'L{x + 1}' for x in range(cv_loops)]
    dfl = pd.DataFrame(columns=coin_list, index=[f'Cum Ret Score', 'C.Ret & Acc Score', 'Equity Score'] + cv_index)
    dfl = dfl.fillna(0)


# Generates combination of indicators to train:
a = shuffle([16, 5, 6, 8, 9, 12])  # Column index for Zone indicators
b = shuffle([16, 7, 10, 11, 13, 14, 15])  # Column index for Signal indicators
combos = list(itertools.product(a, b))  # List of all possible combinations
combos = pd.DataFrame(combos)

if cv_loops > 1:
    train_p = 1 - (1 / (cv_loops + 1))

div = groups

# ---- Stop loss setup and ret function selection ---

get_ret, stop = stop_fun(stop_loss)

while True:

    for c in range(len(cryptos)):  # The range here will define which assets datasets are being selected from the list)

        # DataFrame Loading

        coin = coin_list[c]
        print(f'\nStarting {coin} analysis...')
        s_lim = dfl[coin][0]

        resample_dict = {}
        dfx = pd.read_csv(f'{import_name}', index_col='Datetime', parse_dates=True, encoding='utf7', low_memory=False)
        dfx = df = dfx.resample(sample).agg(
            {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).fillna(method='ffill')
        dfx = dfx.fillna(method='ffill')
        dfx = dfx[md_start_date:md_end_date]

        result_filename = f'TA Results {coin} - {sample} L{cv_loops} T{round(len(dfx) / 1440)} Days - {d_set}'
        if sample != '1min':
            df = dfx.resample(sample).agg(
                {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).fillna(method='ffill')
        else:
            df = dfx.copy()

        if len(df) < 14:
            print('Dataset is to short to be analyzed at the defined sampling')
            exit()

        # The following command creates Result Dataframe to store results and erases previous one (To avoid erasing current dfr simply envelop dfr in quotes ''' )
        print('Resampling Complete')
        try:
            dfr = pd.read_pickle(f'{loc_folder}/{result_filename}.pkl')
            print(f'Result Dataframe found')

        except:
            print(f'Creating Result Dataframe')
            dfr = pd.DataFrame(
                columns=c_index,
                index=pd.to_datetime([]))
            dfr.index.name = 'Datetime'

        dfa = get_dfa(groups, 1)

        # The following line will replace the first random element of each list with the standard parameters of the used indicators so that the first iteration will always be the standard.

        for y in range(groups):

            df = df.iloc[:, :5]  # Cleans signal columns from previous loop
            print(f'\nCurrent Time: {datetime.now().strftime("%H:%M:%S")}')
            print(f'Group {y} - Instantiating Objects...')

            #  Set Indicator

            # Momentum
            uo = UltimateOscillator(high=df['High'], low=df['Low'], close=df['Close'], window1=dfa.uo_1[y],
                                    window2=dfa.uo_2[y], window3=dfa.uo_3[y])
            so = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=dfa.so_w[y],smooth_window=dfa.so_sw[y])
            wi = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=dfa.wi_w[y])
            srsi = StochRSIIndicator(close=df['Close'], window=dfa.srsi_w[y], smooth1=dfa.srsi_kw[y],smooth2=dfa.srsi_dw[y])
            pvo = PercentageVolumeOscillator(volume=df['Volume'], window_slow=dfa.pvo_sw[y], window_fast=dfa.pvo_fw[y],
                                             window_sign=dfa.pvo_sg[y])

            # Trend
            cci = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=dfa.cci_w[y], constant=0.015)

            # Volume
            adi = AccDistIndexIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
            eom = EaseOfMovementIndicator(high=df['High'], low=df['Low'], volume=df['Volume'], window=dfa.eom_w[y])
            vpt = VolumePriceTrendIndicator(close=df['Close'], volume=df['Volume'])

            # Add indicators to Datafram
            df['UO'] = uo.ultimate_oscillator()
            df['SO'] = so.stoch()
            df['SOS'] = so.stoch_signal()
            df['WI'] = wi.williams_r()
            df['SRSI'] = srsi.stochrsi()
            df['SRSI_K'] = srsi.stochrsi_k()
            df['SRSI_D'] = srsi.stochrsi_d()
            df['PVO'] = pvo.pvo()
            df['PVO_S'] = pvo.pvo_signal()
            df['PVO_H'] = pvo.pvo_hist()
            df['CCI'] = cci.cci()
            df['ADI'] = adi.acc_dist_index()
            df['ADI_MACD'] = df.ADI.ewm(span=dfa.adi_fw[y], adjust=False).mean() - df.ADI.ewm(span=dfa.adi_sw[y],
                                                                                              adjust=False).mean()
            df['ADI_MACD_S'] = df.ADI_MACD.rolling(dfa.adi_sg[y]).mean()
            df['EOM'] = eom.ease_of_movement()
            df['EOM_S'] = df['EOM'].rolling(dfa.eom_sma[y]).mean()
            df['VPT'] = vpt.volume_price_trend()
            df['VPT_S'] = df['VPT'].rolling(dfa.vpt_sma[y]).mean()

            print(f'Group {y} - Applying Signals...')

            # Add Trade signal
            # Df Col 5
            df.loc[(df.UO > dfa.uo_s[y]), 'UO_Z'] = 1
            df.loc[(df.UO < dfa.uo_b[y]), 'UO_Z'] = -1
            df.drop(columns=['UO'], inplace=True)

            # Df Col 6
            df.loc[(df.SO > dfa.so_s[y]), 'SO_Z'] = 1
            df.loc[(df.SO < dfa.so_b[y]), 'SO_Z'] = -1

            # Df Col 7
            df.loc[(df.SO.shift(1) > df.SOS.shift(1)) & (df.SO < df.SOS), 'SO_A'] = 1
            df.loc[(df.SO.shift(1) < df.SOS.shift(1)) & (df.SO > df.SOS), 'SO_A'] = -1
            df.drop(columns=['SO', 'SOS'], inplace=True)

            # Df Col 8
            df.loc[(df.WI > dfa.wi_s[y]), 'WI_Z'] = 1
            df.loc[(df.WI < dfa.wi_b[y]), 'WI_Z'] = -1
            df.drop(columns=['WI'], inplace=True)

            # Df Col 9
            df.loc[(df.SRSI > dfa.srsi_s[y]), 'SRSI_Z'] = 1
            df.loc[(df.SRSI < dfa.srsi_b[y]), 'SRSI_Z'] = -1

            # Df Col 10
            df.loc[(df.SRSI_K.shift(1) > df.SRSI_D.shift(1)) & (df.SRSI_K < df.SRSI_D), 'SRSI_A'] = 1
            df.loc[(df.SRSI_K.shift(1) < df.SRSI_D.shift(1)) & (df.SRSI_K > df.SRSI_D), 'SRSI_A'] = -1
            df.drop(columns=['SRSI', 'SRSI_K', 'SRSI_D'], inplace=True)

            # Df Col 11
            df.loc[(df.PVO.shift(1) > df.PVO_S.shift(1)) & (df.PVO < df.PVO_S), 'PVO_A'] = 1
            df.loc[(df.PVO.shift(1) < df.PVO_S.shift(1)) & (df.PVO > df.PVO_S), 'PVO_A'] = -1
            df.drop(columns=['PVO', 'PVO_S', 'PVO_H'], inplace=True)

            # Df Col 12
            df.loc[df.CCI > dfa.cci_s[y], 'CCI_Z'] = 1
            df.loc[df.CCI < dfa.cci_b[y], 'CCI_Z'] = -1
            df.drop(columns=['CCI'], inplace=True)

            # Df Col 13
            df.loc[(df.ADI_MACD.shift(1) > df.ADI_MACD_S.shift(1)) & (df.ADI_MACD < df.ADI_MACD_S), 'ADI_A'] = 1
            df.loc[(df.ADI_MACD.shift(1) < df.ADI_MACD_S.shift(1)) & (df.ADI_MACD > df.ADI_MACD_S), 'ADI_A'] = -1
            df.drop(columns=['ADI', 'ADI_MACD', 'ADI_MACD_S', ], inplace=True)

            # Df Col 14
            df.loc[(df.EOM.shift(1) > df.EOM_S) & (df.EOM < df.EOM_S), 'EOM_A'] = 1
            df.loc[(df.EOM.shift(1) < df.EOM_S) & (df.EOM > df.EOM_S), 'EOM_A'] = -1
            df.drop(columns=['EOM', 'EOM_S'], inplace=True)

            # Df Col 15
            df.loc[(df.VPT.shift(1) > df.VPT_S) & (df.VPT < df.VPT_S), 'VPT_A'] = 1
            df.loc[(df.VPT.shift(1) < df.VPT_S) & (df.VPT > df.VPT_S), 'VPT_A'] = -1
            df.drop(columns=['VPT', 'VPT_S'], inplace=True)

            # Df Col 16
            df['Null'] = df['Close'] * 0
            df.iloc[:, 5:] = df.iloc[:, 5:].shift(1)  # Fixes looking ahead problem
            df = df.fillna(0)

            print(f'Group {y} - Testing Combinations...')

            ''' 
            A partir de acá el algoritmo va a filtrar las estrategias y extraer diferentes valores de acuerdo a su desempeño.
            El programa va a dividir el data set en una cantidad de partes de acuerdo con la cantidad de "cross validation loop"train a test y 
            (cv_loops) y luego utilizara cantidad de porciones de acuerdo al correspondiente loop, el primer loop utilizara el 100%
            y luego se ira restando una porcion por cada loop. Para ejemplificar si cv_loops = 3, la parte es 0.33 del dataset por lo que
            la primera vuelta se trabaja con el 100%, la segunda con el 67% y la tercera con el 33%.
            Luego de esta primera division el data set se vuelva a dividir entre train/test en cada vuelta de acuerdo a la fracción determinada
            por la variable train_p. Si train_p = 0.7 entonces el 70% se usa para training y el 30% para testing.
            El programa seleccionara estrategias cada vez mejores utilizando como valor de optimización el retorno y teniendo en cuenta
            criterios generales como la cantidad mínima de transacciones que queremos que tenga (t_min) y la precisión minima (acc_min)
            En la planilla de resultados quedarán además reflejadas otros datos correspondientes a la estrategia como el dinero generado
            el desempeño del mercado en ese periodo, la acc promedio y su error, así como el número de transacciones.
            Si la estrategia es mejor que la anterior y los resultados se mantienen consistes en las diferentes porciones del data set y base,
            se reemplazá la estrategia vieja por la nueva y el programa seguirá buscando algo mejor para repetir el proceso.
            '''

            for z in tqdm(range(len(combos))):
                end_loop = 0
                df['Trade'] = df['Close'] * 0

                ind_list = [combos.iloc[z][0], combos.iloc[z][1]]
                if ind_list[:-1].count(16) > 1:
                    continue

                idx_num = df.columns.get_loc("Null")
                lim = ind_list.count(idx_num)
                alpha_r = [0 for x in range(cv_loops)]
                acc = pd.Series([0.00 for x in range(cv_loops)])
                cum_ret_r = pd.Series([0.00000 for x in range(cv_loops)])
                sum_cv = 0
                avg_alpha_r = 0
                avg_alpha_r_e = 0
                avg_cum_rate = 0
                avg_acc = 0
                st_e = 0
                score1 = 0
                score2 = 0
                score3 = 0
                n = 0
                comparison_list = [0 for x in range(4*cv_loops)]
                por = int(len(df) / cv_loops)  # Dataset portion

                '''  
                A data set will be train/tested at different intervals depending on the number of cross validation loops (cv_loops) 
                using a Nested Cross validation procedure.
                For example if the dataset has a length of 900 and there are 3 loops. The first loop will select the first 300 rows
                and split it into train/test portions using train_p coefficient (0.7 = 70% Train).The second loop will select the first 600 and the third 
                the whole dataset. So the first loop will train on the first 210 rows and test in the following 90, the second
                will test on the first 420 and test in the following 180 and the third will train in the first 630 and test in the 
                following 270.
                '''

                for w in range(cv_loops):
                    eq = equity  # Each subset resets the starting equity
                    trl = int((w + 1) * por * train_p)  # Last row of the current train portion
                    tel = por * (w + 1)  # Last row of the test portion
                    dft1 = df.iloc[:trl, :]  # Train dataset for this loop
                    ret_1 = get_ret(dft1, ind_list, lim, stop, atr)  # Returns a series with the return % of every transaction
                    ret_1 = (ret_1 - fee)  # If there is a transaction fee it accounts for that

                    if len(ret_1) > 1:

                        # Adjust the minimum transaction limit to the portion of the dataset.
                        delta_tr = df.index[-1] - df.index[0] # Dataframe time
                        t_mi = round(t_min * int(delta_tr.days) * len(dft1) / len(df)) # Calculates minimum amount of transactions according to the daily minimum setup (t_min)
                        t_ma = round(t_max * int(delta_tr.days) * len(dft1) / len(df)) # Calculates minimum amount of transactions according to the daily maximum setup (t_max)

                        ret_1 = pd.DataFrame(ret_1) # Converts the series to Dataframe
                        ret_1['Cum_Ret'] = (1 + ret_1).cumprod() - 1  # Calculates cumulative return
                        ret_1['Equity'] = (1 + ret_1['Cum_Ret']) * equity  # Calculates equity
                        ret_1['Close'] = dft1[dft1.index.isin(ret_1.index)]['Close']  # Adds Close price info from original df
                        ret_1['Cum_Market_Ret'] = ret_1['Close'] / ret_1['Close'][0] - 1  # Calculates Market Return starting from the moment it begins trading
                        ret_1['Cum_Market_Ret'][0] = ret_1['Trade'][0]  # The first trade is always long so the market ret will be equal
                        ret_1['Cum_Alpha'] = round(ret_1['Cum_Ret'] - ret_1['Cum_Market_Ret'], 5)  # Calculates difference between the market and the bot returns to check performance over market.
                        ret_1['Market_Ret'] = (ret_1['Cum_Market_Ret']+1) / (ret_1['Cum_Market_Ret'].shift(1)+1) - 1
                        ret_1['Market_Ret'][0] = ret_1['Cum_Market_Ret'][0]
                        ret_1['Alpha'] = (ret_1['Cum_Alpha']+1) / (ret_1['Cum_Alpha'].shift(1)+1) - 1
                        ret_1['Alpha'][0] = ret_1['Cum_Alpha'][0]
                        max_idx1 = ret_1['Alpha'].idxmax() # Retrieves the index where bot gets peak performance
                        max_idx1 = ret_1.index.get_loc(max_idx1) # Retrieves the max alpha
                        alpha_r[w] = ret_1['Alpha'].mean()  # Average Alpha per trade

                        # Checks if the portions has the min. req. of transactions and the score for that loop
                        if t_ma >= len(ret_1) >= t_mi:
                            acc[w] = round(len(ret_1[ret_1['Trade'] > 0]) / len(ret_1) * 100, 2)  # Calculates Accuracy
                            dft2 = df.iloc[trl:tel, :]  # Selects test portion
                            ret_2 = get_ret(dft2, ind_list, lim, stop, atr) # Calculates return on the test portion

                            if len(ret_2) > 1:
                                st_e = ret_1['Alpha'].std() / (len(ret_1) ** (1 / 2)) # Calculates Standard Error of the bot performance
                                ret_2 = (ret_2 - fee) # Accounts for the fee on the test
                                ret_2 = pd.DataFrame(ret_2)
                                ret_2['Cum_Ret'] = (1 + ret_2).cumprod() - 1  # Calculates cumulative return
                                ret_2['Equity'] = (1 + ret_2['Cum_Ret']) * equity  # Calculates equity
                                ret_2['Close'] = dft2[dft2.index.isin(ret_2.index)][
                                    'Close']  # Adds Close price info from original df
                                ret_2['Cum_Market_Ret'] = ret_2['Close'] / ret_2['Close'][
                                    0] - 1  # Calculates Market Return starting from the moment it begins trading
                                ret_2['Cum_Market_Ret'][0] = ret_2['Trade'][
                                    0]  # The first trade is always long so the market ret will be equal
                                ret_2['Cum_Alpha'] = round(ret_2['Cum_Ret'] - ret_2['Cum_Market_Ret'], 5)  # Calculates difference between the market and the bot returns to check performance over market.
                                ret_2['Market_Ret'] = (ret_2['Cum_Market_Ret'] + 1) / (
                                            ret_2['Cum_Market_Ret'].shift(1) + 1) - 1
                                ret_2['Market_Ret'][0] = ret_2['Cum_Market_Ret'][0]
                                ret_2['Alpha'] = (ret_2['Cum_Alpha'] + 1) / (ret_2['Cum_Alpha'].shift(1) + 1) - 1
                                ret_2['Alpha'][0] = ret_2['Cum_Alpha'][0]
                                max_idx2 = ret_2['Alpha'].idxmax()
                                max_idx2 = ret_2.index.get_loc(max_idx2)
                                alpha2_mean = ret_2['Alpha'].mean()  # Alpha Trade average
                                ret = pd.concat([ret_1, ret_2])  # Joins Train and Test Returns
                                df_f = df[:tel]  # Defines Complete dataset of this loop
                                ret['Cum_Ret'] = (1 + ret['Trade']).cumprod() - 1  # Redefines Cumulative Return for the whole portion

                                # The longest test/train set will be treat it differently to calculate final equity.
                                # The following condition will check for minimum accuracy and consistency by using the standard error
                                if w < cv_loops - 1 and alpha_r[w] > alpha2_mean - st_e and acc[
                                    w] > acc_min:
                                    cum_ret_r[w] = ((ret['Cum_Ret'][-1] + 1) ** (1 / len(ret))) - 1  # Average Return % per trade
                                    avg_cum_rate += cum_ret_r[w] / cv_loops  # Calculates Average Rate
                                    comparison_list[4*w] = round(alpha_r[w]*100, 2)
                                    comparison_list[4*w+1] = round(alpha2_mean*100, 2)
                                    comparison_list[4*w+2] = round(((alpha2_mean/alpha_r[w]-1)*100), 2)
                                    comparison_list[4*w+3] = round(st_e*100, 2)

                                # The following is the last loop or the whole dataset checks for consistency and accuracy.
                                elif alpha_r[w] > alpha2_mean - st_e and acc[w] > acc_min:
                                    cum_ret_r[w] = ((ret['Cum_Ret'][-1] + 1) ** (1 / len(ret))) - 1
                                    avg_cum_rate += cum_ret_r[w] / cv_loops
                                    comparison_list[4*w] = round(alpha_r[w] * 100, 2)
                                    comparison_list[4*w+1] = round(alpha2_mean * 100, 2)
                                    comparison_list[4*w+2] = round((alpha2_mean/alpha_r[w]-1)*100, 2)
                                    comparison_list[4*w+3] = round(st_e*100, 2)

                                    ret = get_ret(df, ind_list, lim, stop, atr)
                                    ret = (ret - fee)  # Accounts for the fee on the test
                                    ret = pd.DataFrame(ret)
                                    ret['Cum_Ret'] = (1 + ret).cumprod() - 1
                                    ret['Equity'] = (1 + ret['Cum_Ret']) * equity  # Calculates equity
                                    ret['Close'] = df[df.index.isin(ret.index)]['Close']  # Adds Close price info from original df
                                    ret['Cum_Market_Ret'] = ret['Close'] / ret['Close'][
                                        0] - 1  # Calculates Market Return starting from the moment it begins trading
                                    ret['Cum_Market_Ret'][0] = ret['Trade'][
                                        0]  # The first trade is always long so the market ret will be equal
                                    ret['Cum_Alpha'] = round(ret['Cum_Ret'] - ret['Cum_Market_Ret'], 5)  # Calculates difference between the market and the bot returns to check performance over market.
                                    ret['Market_Ret'] = (ret['Cum_Market_Ret'] + 1) / (
                                            ret['Cum_Market_Ret'].shift(1) + 1) - 1
                                    ret['Market_Ret'][0] = ret['Cum_Market_Ret'][0]
                                    ret['Alpha'] = (ret['Cum_Alpha'] + 1) / (ret['Cum_Alpha'].shift(1) + 1) - 1
                                    ret['Alpha'][0] = ret['Cum_Alpha'][0]
                                    delta_d = ret.index - df.index[0]

                                    ret['R_Alpha'] = ret['Cum_Alpha'] / (delta_d.seconds / 86400 + delta_d.days)

                                    max_idx = ret['R_Alpha'].idxmax()
                                    min_idx = ret['R_Alpha'].idxmin()
                                    delta = max_idx - df.index[0]
                                    minmax = max_idx - min_idx
                                    minmax = minmax.seconds / 86400 + minmax.days
                                    delta_t = ret.index[-1] - ret.index[0]
                                    delta_t = delta_t.seconds / 86400 + delta_t.days
                                    delta = delta.seconds / 86400 + delta.days

                                    rr_len = len(ret)
                                    max_idx = ret.index.get_loc(max_idx)
                                    alpha_rate = ret['Alpha'].mean() / delta_t * 100  # Alpha Transaction Mean
                                    f_eq = ret['Equity'][-1]
                                    f_acc = round(len(ret[ret['Trade'] > 0]) / len(ret) * 100, 2)
                                    n = len(ret)
                                    cov = ret['Trade'].cov(ret['Market_Ret'])  # Covariance
                                    var = ret['Market_Ret'].var()  # Variance
                                    beta = cov / var  # Beta
                                    risk_free_ret = ((real_rf_ret+1) ** delta_t) - 1
                                    alpha = ret['Cum_Ret'][-1] - risk_free_ret + beta * (ret['Cum_Market_Ret'][-1] - risk_free_ret) # Real Alpha
                                    market_ret = round(ret['Cum_Market_Ret'][-1], 2)
                                    continue

                                else:
                                    end_loop = 1
                                    break
                            else:
                                end_loop = 1
                                break
                        else:
                            end_loop = 1
                            break
                    else:
                        end_loop = 1
                        break
                if end_loop == 1:  # The tested strategy has failed so it moves to the next one
                    continue

                avg_acc = round(acc.mean(), 2)
                if cv_loops < 2:
                    avg_acc_error = 0
                else:
                    avg_acc_error = round(acc.std(), 2)

                avg_cum_re = cum_ret_r.std() / (cv_loops**1/2)  # Standard error of the average return between loops

                score = {'Alpha': alpha, 'Accuracy': avg_acc - avg_acc_error,
                'Return Rate': avg_cum_rate - avg_cum_re, 'Alpha Rate': alpha_rate,
                    'Standard Error': 1/comparison_list[-1], 'Equity': f_eq}

                criteria = score[model_criteria]

                #Checks if testing strategy has better score

                if criteria > s_lim:
                    # Saves strategy if it is better than the previous one
                    dfr = save(loc_folder, result_filename, dfr, dfa, df, n, f_acc, f_eq, criteria, alpha, alpha_rate, delta, minmax,
                               model_criteria, market_ret, beta, avg_acc, avg_acc_error, ind_list, y)
                    for x in range(cv_loops):
                        dfl[coin].iloc[3 + x] = alpha_r[x]
                    s_lim = criteria
                    dfl[coin].iloc[0] = s_lim
                    dfl.to_pickle(f'{loc_folder}/{score_filename}')
                    ret.to_csv(f'{loc_folder}/ret.csv')
                    print('Score Limits Saved')