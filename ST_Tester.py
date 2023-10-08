from datetime import datetime
import warnings
from Mfun import *
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)  # To avoid warnings

#                   -------- Tester -------

t_list = [sample]
cn = len(cryptos)
start_eq = equity
date = td_start_date[:-9]
import_name = f'{import_pair.replace("/", "")}-{import_sample}-Price Data-{import_desc}.csv'
str_dic = {}
str_dic2 = {}
i_cryptos = []
days = (datetime.strptime(md_end_date, "%Y-%m-%d %H:%M:%S") - datetime.strptime(md_start_date, "%Y-%m-%d %H:%M:%S")).days

for x in range(cn):
    for y in range(len(t_list)):
        file_n = f'TA Results {cryptos[x].replace("/", "")} - {t_list[y]} L{cv_loops} T{days} Days - {d_set}.csv'
        dt = pd.read_csv(f'{loc_folder}{file_n}', encoding='utf7')
        dt = dt_etl(dt)
        dt = dt.loc[dt[' Criteria'] == model_criteria]  # Filters dataset by criteria
        idx_max = dt[' Score'].idxmax()  # Gets the index of the result that has the highest accuracy
        zone = dt[' Zone'].iloc[idx_max]  # Selects the zone indicator of the best result
        signal = dt[' Signal'].iloc[idx_max]  # Selects the signal indicator of the best result
        beta_tr = round(dt[' Beta'].iloc[idx_max], 3)
        str_dic[f'{cryptos[x]}{t_list[y]}'] = [[None, None], [None, None]]  # Create a dictionary to store the indicator's names and parameters
        str_dic = get_dic(dt, zone, cryptos[x], t_list[y], str_dic, 0, idx_max)
        str_dic = get_dic(dt, signal, cryptos[x], t_list[y], str_dic, 1, idx_max)

for w in range(len(cryptos)):
    i_cryptos += [f'{cryptos[w]} Ret %', f'{cryptos[w]} Bot Eq.', f'{cryptos[w]} Market Eq.', f'{cryptos[w]} Alpha']

dfl = pd.DataFrame(columns=i_cryptos)
dfl_1 = pd.DataFrame(columns=i_cryptos)

for x in range(cn):
    for t in range(seg_n):
        ticker = f'{cryptos[x].replace("/", "")}'
        current_datetime = datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
        n = -1
        cdf = [0 for x in range(len(t_list))]
        coin = cryptos[x][:-5]
        dfx = pd.read_csv(f'{loc_folder}{import_name}', index_col='Datetime', encoding='utf7', parse_dates=True, low_memory=False)
        dfx = dfx.resample('1min').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).fillna(method='ffill')
        dfx = dfx.fillna(method='ffill')
        dfx = dfx[td_start_date:]
        dfx_1 = dfx.copy()
        t_p1 = t * seg_len
        t_p2 = t_p1 + seg_len
        dfx = dfx.iloc[t_p1: t_p2, :]
        ret, dfl = tester(dfx, start_eq, dfl, i_cryptos, f'{cryptos[x]}{t_list[y]}', str_dic, stop_loss)
        print(dfl)

dfx_1 = dfx_1.iloc[:t_p2, :]
ret_f, dfl_2 = tester(dfx_1, start_eq, dfl_1, i_cryptos, f'{cryptos[x]}{t_list[y]}', str_dic, stop_loss)
dfl = pd.concat([dfl, pd.DataFrame([dfl_2.iloc[0, :]], columns=i_cryptos, index=['Total'])])
print(dfl)
ret_f = pd.DataFrame(ret_f)
ret_f['Cum_Ret'] = (1 + ret_f).cumprod() - 1
ret_f['Equity'] = (1 + ret_f['Cum_Ret']) * start_eq
ret_f['Close'] = dfx_1[dfx_1.index.isin(ret_f.index)]['Close']
ret_f['Cum_Market_Ret'] = ret_f['Close'] / ret_f['Close'][0] - 1
ret_f['Cum_Market_Ret'][0] = ret_f['Trade'][0]
ret_f['Cum_Alpha'] = round(ret_f['Cum_Ret'] - ret_f['Cum_Market_Ret'], 5)
ret_f['Market_Ret'] = (ret_f['Cum_Market_Ret'] + 1) / (
                                            ret_f['Cum_Market_Ret'].shift(1) + 1) - 1
ret_f['Market_Ret'][0] = ret_f['Cum_Market_Ret'][0]
ret_f['Alpha'] = (ret_f['Cum_Alpha'] + 1) / (ret_f['Cum_Alpha'].shift(1) + 1) - 1
ret_f['Alpha'][0] = ret_f['Cum_Alpha'][0]
delta_t = ret_f.index[-1] - ret_f.index[0]
delta_t = delta_t.seconds / 86400 + delta_t.days
delta_d = ret_f.index - dfx_1.index[0]
market_ret = round(ret_f['Cum_Market_Ret'][-1], 2)
f_eq = ret_f['Equity'][-1]
f_acc = round(len(ret_f[ret_f['Trade'] > 0]) / len(ret_f) * 100, 2)
n = len(ret_f)
cov = ret_f['Trade'].cov(ret_f['Market_Ret'])  # Covariance
var = ret_f['Market_Ret'].var()  # Variance
beta = cov / var  # Beta
risk_free_ret = ((real_rf_ret+1) ** delta_t) - 1
alpha = ret_f['Cum_Ret'][-1] - risk_free_ret + beta * (ret_f['Cum_Market_Ret'][-1] - risk_free_ret) # Real Alpha
ret_f = ret_f.reset_index()
ret_f['R_Alpha'] = ret_f['Cum_Alpha'] / (delta_d.seconds / 86400 + delta_d.days)
max_idx = ret_f['R_Alpha'].idxmax()
min_idx = ret_f['R_Alpha'].idxmin()
up_idx = ret_f['Alpha'].idxmax()
dw_idx = ret_f['Alpha'].idxmin()
max_eq = round(ret_f['Equity'].max(), 2)
min_eq = round(ret_f['Equity'].min(), 2)
max_value = round(ret_f['R_Alpha'].max()*100, 2)
min_value = round(ret_f['R_Alpha'].min()*100, 2)
alpha_mean = round(ret_f['Alpha'][1:].mean() * 100, 4)
max_time = ret_f['Datetime'][max_idx]-dfx_1.index[0]
min_time = ret_f['Datetime'][min_idx]-dfx_1.index[0]
max_time = max_time.seconds/86400 + max_time.days
min_time = min_time.seconds/86400 + min_time.days
alpha_daily_rate = max_value
market_eq = (ret_f['Market_Ret'][up_idx]+1)*100
score = dt[' Score'].max()
m_max = round((ret_f['Market_Ret'].max() + 1) * 100, 2)
m_min = round((ret_f['Market_Ret'].min() + 1) * 100, 2)
super_score = round((((max_eq+min_eq)/2) / ((m_max+m_min)/2)-1)*100, 2)
dic_name = f'{cryptos[x]}{t_list[0]}'

try:
    df_res = pd.read_csv(f'{loc_folder}Final.csv')
    df_res = pd.concat([df_res, pd.DataFrame([[cv_loops, days, date, score, n, max_eq, min_eq, m_max, m_min, super_score, max_value, alpha_mean, beta_tr, zone, signal, str_dic[dic_name][0][1], str_dic[dic_name][1][1]]], columns=['Loops', 'Days', 'Date', 'Score', 'Transactions', 'Max Equity', 'Min Equity', 'Max Market', 'Min Market', 'Score %', 'Max Alpha R.', 'Alpha Mean', 'Beta', 'Zone', 'Signal', 'Zone Parameters', 'Signal Parameters'])], ignore_index=True)
    df_res = df_res.iloc[:, 1:]
except:
    df_res = pd.DataFrame([], columns=['Loops', 'Days', 'Date', 'Score', 'Transactions', 'Max Equity', 'Min Equity', 'Max Market', 'Min Market', 'Score %', 'Max Alpha R.', 'Alpha Mean', 'Beta', 'Zone', 'Signal', 'Zone Parameters', 'Signal Parameters'])

    df_res = pd.concat([df_res, pd.DataFrame([[cv_loops, days, date,  score, n, max_eq, min_eq, m_max, m_min, super_score, max_value, alpha_mean, beta_tr, zone, signal, str_dic[dic_name][0][1], str_dic[dic_name][1][1]]],
                                        columns=['Loops', 'Days', 'Date', 'Score', 'Transactions', 'Max Equity', 'Min Equity', 'Max Market', 'Min Market', 'Score %', 'Max Alpha R.', 'Alpha Mean', 'Beta', 'Zone', 'Signal', 'Zone Parameters', 'Signal Parameters'])], ignore_index=True)


dfl.to_csv(f'{loc_folder}Results {d_set}.csv')
ret_f.to_csv(f'{loc_folder}Return {d_set}.csv')
df_res.to_csv(f'{loc_folder}Final.csv')
ret_f = ret_f.sort_values(by=['Alpha'], ascending=False)


print(f'Transactions = {n} \n'
      f'Final Accuracy = {f_acc} \n'
      f'Alpha = {round(alpha*100,2)} \n'
      f'Beta = {round(beta,2)} \n'
      f'Max Equity = {round(max_eq,2)} \n'
      f'Min Equity = {round(min_eq,2)} \n'
      f'Max Market = {round(m_max,2)} \n'
      f'Min Market = {round(m_min,2)} \n'
      f'Market Ret = {round(market_ret*100,2)}% \n'

      f'Alpha R. Max Time = {round(max_time,2)} days\n'
      f'Alpha R. Min Time = {round(min_time,2)} days\n'
      f'Alpha Daily Rate = {round(alpha_daily_rate,2)}\n'
      f'Super Score = {round(super_score,2)}%')
