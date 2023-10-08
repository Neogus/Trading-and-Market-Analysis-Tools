from Mfun import *
pd.set_option('display.max_columns', None)


# Main Config

api_key = B_API_KEY
api_secret = B_API_SECRETKEY
client = Client(api_key, api_secret)
bot = telebot.TeleBot(T_API_KEY)
t_list = [sample]
str_dic = {}
str_dic2 = {}
cn = len(cryptos)
start_eq = equity
sl_sleep = 60
cdf_s = 0
status_s = 0
days = (datetime.strptime(md_end_date, "%Y-%m-%d %H:%M:%S") - datetime.strptime(md_start_date, "%Y-%m-%d %H:%M:%S")).days

# This function will send messages to a telegram chat. These messages can be configured directly inside the function.
def send_alarm(coin, action, strength, current_price, chat_id):
    try:

        if action == 'comprar' and strength == 'Stop Loss':
            message = f'Para evitar mayores perdidas deberías {action} {coin} y saldar el prestamo. El precio actual es de {current_price} USD.'
            bot.send_message(chat_id, message)
        elif action == 'comprar' and strength != 'Stop Loss':
            message = f'Creo que es un momento {strength} para {action} {coin}. El precio actual es de {current_price} USD.'
            bot.send_message(chat_id, message)
        elif action == 'vender' and strength == 'Stop Loss':
            message = f'Para evitar mayores perdidas deberías {action} {coin}. El precio actual es de {current_price} USD.'
            bot.send_message(chat_id, message)
        elif action == 'vender' and strength != 'Stop Loss':
            message = f'Creo que es un momento {strength} para {action} {coin}. El precio actual es de {current_price} USD.'
            bot.send_message(chat_id, message)
    except:
        logger.info('Something went wrong while trying to send messages!')

def place_order(ticker, asset, asset_m, side, e_type, price_point, o_type="LIMIT"):
    exit_code = 0
    order_loop = 0
    order_name = f'{side}'
    while True:
        try:
            exit_loop = 0
            info = client.get_margin_account()
            logger.info('Info Retrieved')
            for x in range(len(info['userAssets'])):
                if info['userAssets'][x]['asset'] == asset:
                    amount = float(info['userAssets'][x][asset_m]) + float(info['userAssets'][x]['interest'])
                    break

            logger.info('Info Loop Ok')
            ticker_price = client.get_orderbook_ticker(symbol=ticker)
            btc_price = client.get_orderbook_ticker(symbol='BTCBUSD')
            logger.info('Info Ticker Retrieved')
            balance = float(btc_price['askPrice']) * float(info['totalNetAssetOfBtc'])
            logger.info(f'Balance:{balance}')
            bid_price = float(ticker_price['bidPrice'])
            ask_price = float(ticker_price['askPrice'])
            logger.info('Info Ticker Retrieved')
            if e_type == 'AUTO_REPAY':
                price = round_down((bid_price + ask_price) / 2, 2)
                amount = round_up(amount, asset_precision)
                order_name = 'pago'
            elif e_type == 'NO_SIDE_EFFECT' and side == 'BUY':
                price = round_down((bid_price + ask_price) / 2, 2)
                amount = round_down(amount / price, asset_precision)
                order_name = 'compra'
            elif e_type == 'NO_SIDE_EFFECT' and side == 'SELL':
                price = round_up((bid_price + ask_price) / 2, 2)
                amount = round_down(amount, asset_precision)
                order_name = 'venta'
            elif e_type == 'MARGIN_BUY':
                price = round_up((bid_price + ask_price) / 2, 2)
                amount = round_down(amount / price, asset_precision)
                order_name = 'prestamo'
            logger.info('Rounded Finished')
            logger.info(f'{price},{amount},{ticker},{side},{asset},{asset_m},{e_type}')
            if price * amount > 10.5:
                if o_type == "MARKET":
                    order = client.create_margin_order(symbol=ticker, side=side, type=o_type,
                                                   quantity=amount, sideEffectType=e_type)
                elif o_type == "LIMIT":
                    order = client.create_margin_order(symbol=ticker, side=side, type=o_type, timeInForce='GTC',
                                           quantity=amount, price=price, sideEffectType=e_type)
                logger.info('Order Placed Ok')
            else:
                message = f'Balance insuficiente para efectuar la orden de {order_name}.'
                logger.info(message)
                # telegram_send.send(messages=[message])
                bot.send_message(chat_id, message)
                exit_code = 1
                return exit_code, price_point
        except Exception as e:
            print(e)
            time.sleep(5)
            order_loop += 1
            if order_loop > 3:
                logger.info(f'No se ha podido colocar la orden de {order_name} debido a un error de conexión.')
                exit_code = 2
                return exit_code, price_point
            else:
                continue

        check_loop = 0
        error_loop = 0
        time.sleep(5)
        order = client.get_margin_order(symbol=ticker, orderId=order['orderId'])
        while order['status'] != 'FILLED':
            try:
                if check_loop > 3 and order['status'] != 'PARTIALLY_FILLED':
                    logger.info(f'Se ha acabado el tiempo para completar la orden de {order_name}.')
                    client.cancel_margin_order(symbol=ticker, orderId=order['orderId'])
                    while order['status'] != 'CANCELED':
                        logger.info(f'Esperando cancelación')
                        time.sleep(5)
                        order = client.get_margin_order(symbol=ticker, orderId=order['orderId'])
                    logger.info(f'La orden fue cancelada.')
                    exit_loop = 1
                    break
                else:

                    logger.info(f'La orden no fue completada todavía...')
                    time.sleep(5)
                    check_loop += 1
                    order = client.get_margin_order(symbol=ticker, orderId=order['orderId'])
                    if order['status'] != 'PARTIALLY_FILLED':
                        logger.info(f'La orden fue completada parcialmente, no puede ser cancelada hasta que termine de completarse.')

                    continue
            except:
                error_loop += 1
                if error_loop > 3:
                    message = f'No se ha podido verificar la ejecución de la orden de {order_name} debido a un error de conexión. Se requiere intervención manual.'
                    logger.info(message)
                    # telegram_send.send(messages=[message])
                    bot.send_message(chat_id, message)
                    exit_code = 3  # Check failed
                    return exit_code
                else:
                    time.sleep(5)
                    continue
        if exit_loop == 0:
            break

    if e_type == 'AUTO_REPAY':
        equivalent = price_point * amount
        t_return = (price/price_point-1) * -100
        message = f'He pagado el préstamo de {amount} {ticker[:-4]} (${equivalent:.2f}) a un precio de ${price:.2f}, obteniendo un retorno de {t_return:.2f}%.'
    elif e_type == 'NO_SIDE_EFFECT' and side == 'BUY':
        equivalent = amount * price
        message = f'He comprado {amount} {ticker[:-4]} (${equivalent:.2f}) a un precio de ${price} \n' \
        f'Balance actual = ${balance:.2f}'
    elif e_type == 'NO_SIDE_EFFECT' and side == 'SELL':
        equivalent = amount * price
        t_return = (price / price_point - 1) * 100
        message = f'He vendido {amount} {ticker[:-4]} (${equivalent:.2f}) a un precio de ${price}, obteniendo un retorno de {t_return:.2f}%. \n' \
                  f'Balance actual = ${balance:.2f}'
    elif e_type == 'MARGIN_BUY':
        equivalent = amount * price
        message = f'He tomado un préstamo de {amount} {ticker[:-4]} (${equivalent:.2f}).'

    logger.info(message)
    # telegram_send.send(messages=[message])
    bot.send_message(chat_id, message)

    return exit_code, price



#                   -------- Trading Bot -------

logger.info(f'\n--------------START-----------------')

for x in range(cn):
    for y in range(len(t_list)):
        file_n = f'TA Results {cryptos[x].replace("/", "")} - {t_list[y]} L{cv_loops} T{days} Days - {d_set}.csv'
        dt = pd.read_csv(f'{loc_folder}{file_n}', encoding='utf7')
        dt = dt_etl(dt)
        dt = dt.loc[dt[' Criteria'] == model_criteria]  # Filters dataset by criteria
        idx_max = dt[' Score'].idxmax()  # Gets the index of the result that has the highest accuracy
        zone = dt[' Zone'].iloc[idx_max]  # Selects the zone indicator of the best result
        signal = dt[' Signal'].iloc[idx_max]  # Selects the signal indicator of the best result
        str_dic[f'{cryptos[x]}{t_list[y]}'] = [[None, None], [None,
                                                              None]]  # Create a dictionary to store the indicator's names and parameters

        str_dic = get_dic(dt, zone, cryptos[x], t_list[y], str_dic, 0, idx_max)
        str_dic = get_dic(dt, signal, cryptos[x], t_list[y], str_dic, 1, idx_max)

try:
    dfs = pd.read_pickle(f'{loc_folder}Status.pkl')
except:
    dfs = pd.DataFrame(
        columns=['Status', 'Balance', 'Crypto Bal.', 'Price Point', 'Curr. Price', 'Div', 'Curr. Eq', 'Rebalance'],
        index=cryptos)
    dfs.fillna(0, inplace=True)
    dfs['Status'] = [[0] for x in range(cn)]
    dfs['Balance'] = [round(start_eq, 2) for x in range(cn)]
    dfs['Div'] = [round(x, 2) for x in dfs['Balance']]
    dfs['Curr. Eq'] = [round(start_eq, 2) for x in range(cn)]
    dfs['Rebalance'] = [1 for x in range(cn)]
    dfs['Price Point'] = [[1] for x in range(cn)]
    dfs['Sell Price'] = [[9999999999] for x in range(cn)]
    dfs['Last Cdf'] = [[0 for y in range(len(t_list))] for x in range(cn)]
    dfs['Countdown'] = [0 for x in range(cn)]
    dfs['Sleep'] = [0 for x in range(cn)]
    dfs['Drawback'] = [1 for x in range(cn)]
    dfs['Drawback T'] = [0 for x in range(cn)]
    dfs['Borrow Amount'] = [0 for x in range(cn)]
    dfs['Borrow Price'] = [0 for x in range(cn)]
    dfs['Stop Range'] = [0 for x in range(cn)]
    equity_1 = start_eq * cn
    equity_2 = start_eq * cn
    # telegram_send.send(
    #    messages=[f'---------------REINICIANDO-----------------'])
    bot.send_message(chat_id,
                     f'---------------REINICIANDO-----------------')

try:
    dfl = pd.read_pickle(f'{loc_folder}Transactions.pkl')
except:
    mux = pd.MultiIndex.from_product([cryptos, ['Datetime', 'Price', 'Str', 'Ret']])
    dfl = pd.DataFrame(columns=mux)

while True:
    # try:
    for x in range(cn):
        if time.time() - dfs['Countdown'][x] > dfs['Sleep'][x]:
            ticker = f'{cryptos[x].replace("/", "")}'
            current_datetime = datetime.now().strftime("%d-%m at %H:%M:%S")
            n = -1
            cdf = [0 for x in range(len(t_list))]
            coin = cryptos[x][:-5]
            dfx = fetch_data(exchange='binance',
                             cryptos=[cryptos[x]],
                             sample_freq=sample_freq,
                             since_hours=since,
                             page_limit=1000)

            current_datetime2 = datetime.now().strftime("%H:%M:%S")
            print(f'{ticker} Dataset download started the {current_datetime} and finished at {current_datetime2}')
            # logger.info(f'{cryptos[x]} Dasaset Span: {int(len(dfx) / 60)} hours')
            # logger.info(f'Current Time: {current_datetime}')
            for y in range(len(t_list)):
                if t_list[y] != '1min':
                    df = dfx.resample(t_list[y]).agg(
                        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).fillna(
                        method='ffill')
                else:
                    df = dfx
                if len(df) < 14:
                    # print(f'Dataset {t_list[y]} is to short to be analyzed at the defined sampling')
                    logger.info(f'Dataset {t_list[y]} is to short to be analyzed at the defined sampling')
                    exit()

                dfr_1 = set_indicators(df, f'{cryptos[x]}{t_list[y]}', str_dic)
                c_length = len(dfr_1.columns)

                # The following lines define which indicators are taken into consideration, the first 3 are for the short window the last is for the long trend window if it has one.

                if dfr_1.iloc[-2].sum() == c_length:
                    cdf[y] = 1
                elif dfr_1.iloc[-2].sum() == -c_length:
                    cdf[y] = -1

            current_price = df['Close'][-1]
            dfs.loc[cryptos[x], 'Curr. Price'] = current_price
            if cdf_s != cdf[0] or status_s != dfs['Status'][x][0]:
                cdf_s = cdf[0]
                status_s = dfs['Status'][x][0]
                price_p = dfs['Price Point'][x][0]
                stop_r = dfs['Stop Range'][x]
                if status_s in [-1, 0, 3]:
                    logger.info(
                        f'Status: {status_s} - Signals: {cdf[0]} - Price Point: {price_p} - Stop Loss: {round(price_p + stop_r, 2)} - Current Price: {current_price}')
                elif status_s in [1, 2, -2]:
                    logger.info(
                        f'Status: {status_s} - Signals: {cdf[0]} - Price Point: {price_p} - Stop Loss: {round(price_p - stop_r, 2)} - Current Price: {current_price}')


            if cdf != dfs['Last Cdf'][x]:
                #df_m_list = [current_datetime] + [current_price] + cdf
                #df_m = append_to_col(df_m, (cryptos[x]), df_m_list)
                # dfs['Last Cdf'][x] = cdf
                dfs.loc[cryptos[x], 'Last Cdf'] = [[cdf[x]] for x in range(len(cdf))]



            if dfs['Status'][x][0] in [-1, 0, 3] and cdf[0] == -1:
                logger.info(f'{coin} Buy Alarm Triggered')
                action = 'comprar'
                strength = 'bueno'
                ret = round((current_price / dfs['Price Point'][x][0] - 1) * -100, 2)
                send_alarm(coin, action, strength, current_price, chat_id)

                # Comprar
                if dfs['Status'][x][0] in [0,3]:
                    exit_code, dfs['Price Point'][x][0] = place_order(ticker, ticker[-4:], 'free', 'BUY',
                                                                      'NO_SIDE_EFFECT', dfs['Price Point'][x][0], o_type)
                else:
                    exit_code, dfs['Price Point'][x][0] = place_order(ticker, ticker[:-4], 'borrowed', 'BUY', 'AUTO_REPAY', dfs['Price Point'][x][0], o_type)
                    dfs['Status'][x][0] = 0
                    exit_code, dfs['Price Point'][x][0] = place_order(ticker, ticker[-4:], 'free', 'BUY', 'NO_SIDE_EFFECT', dfs['Price Point'][x][0], o_type)

                if exit_code == 3:
                    message = f'Por seguridad el programa se detendrá. Por favor revisar el estado de las ordenes en el exchange y reiniciar desde el servidor.'
                    logger.info(message)
                    # telegram_send.send(messages=[message])
                    bot.send_message(chat_id, message)
                    exit()
                elif exit_code in [1, 2]:
                    break


                #dfs.loc[cryptos[x], 'Drawback T'] = current_datetime
                df['TR'] = atr(df, 500)
                dfs.loc[cryptos[x], 'Stop Range'] = df['TR'][-1]

                dfs['Status'][x][0] = 1
                dfl = append_to_col(dfl, (cryptos[x]), [current_datetime, current_price, -1, ret])
                log_status(dfs, cdf, dfs['Price Point'][x][0], dfs['Stop Range'][x], current_price, logger)

            if dfs['Status'][x][0] in [1, 2, -2] and cdf[0] == 1:
                logger.info(f'{coin} Sell Alarm Triggered')
                action = 'vender'
                strength = 'bueno'
                ret = round((current_price / dfs['Price Point'][x][0] - 1) * 100, 2)
                send_alarm(coin, action, strength, current_price, chat_id)

                # Vender
                if dfs['Status'][x][0] in [2, -2]:
                    exit_code, dfs['Price Point'][x][0] = place_order(ticker, ticker[-4:], 'free', 'SELL', 'MARGIN_BUY',
                                                                      dfs['Price Point'][x][0], o_type)
                else:
                    exit_code, dfs['Price Point'][x][0] = place_order(ticker, ticker[:-4], 'free', 'SELL',
                                                                      'NO_SIDE_EFFECT', dfs['Price Point'][x][0], o_type)
                    dfs['Status'][x][0] = 2
                    exit_code, dfs['Price Point'][x][0] = place_order(ticker, ticker[-4:], 'free', 'SELL', 'MARGIN_BUY',
                                                                      dfs['Price Point'][x][0], o_type)


                if exit_code == 3:
                    message = f'Por seguridad el programa se detendrá. Por favor revisar el estado de las ordenes en el exchange y reiniciar desde el servidor.'
                    logger.info(message)
                    # telegram_send.send(messages=[message])
                    bot.send_message(chat_id, message)
                    exit()
                elif exit_code in [1, 2]:
                    break

                df['TR'] = tr(df, 500)
                dfs.loc[cryptos[x], 'Stop Range'] = df['TR'][-1]

                dfs['Status'][x][0] = -1
                dfl = append_to_col(dfl, (cryptos[x]), [current_datetime, current_price, 1, ret])
                log_status(dfs, cdf, dfs['Price Point'][x][0], dfs['Stop Range'][x], current_price, logger)

            if dfs['Status'][x][0] in [1, 2, -2] and current_price < dfs['Price Point'][x][0] - dfs['Stop Range'][x]:
                action = 'vender'
                strength = 'Stop Loss'
                logger.info(f'{coin} Sell Alarm Triggered')
                ret = round((current_price / dfs['Price Point'][x][0] - 1) * 100, 2)
                send_alarm(coin, action, strength, current_price, chat_id)

                # Vender
                if dfs['Status'][x][0] in [2, -2]:
                    exit_code, dfs['Price Point'][x][0] = place_order(ticker, ticker[-4:], 'free', 'SELL', 'MARGIN_BUY',
                                                                      dfs['Price Point'][x][0], o_type)
                else:
                    exit_code, dfs['Price Point'][x][0] = place_order(ticker, ticker[:-4], 'free', 'SELL',
                                                                      'NO_SIDE_EFFECT', dfs['Price Point'][x][0], o_type)
                    dfs['Status'][x][0] = 2
                    exit_code, dfs['Price Point'][x][0] = place_order(ticker, ticker[-4:], 'free', 'SELL', 'MARGIN_BUY',
                                                                      dfs['Price Point'][x][0], o_type)

                if exit_code == 3:
                    message = f'Por seguridad el programa se detendrá. Por favor revisar el estado de las ordenes en el exchange y reiniciar desde el servidor.'
                    logger.info(message)
                    # telegram_send.send(messages=[message])
                    bot.send_message(chat_id, message)
                    exit()
                elif exit_code in [1, 2]:
                    break
                df['TR'] = atr(df, 500)
                dfs.loc[cryptos[x], 'Stop Range'] = df['TR'][-1]
                dfs['Status'][x][0] = -1
                dfl = append_to_col(dfl, (cryptos[x]), [current_datetime, current_price, 1, ret])
                log_status(dfs, cdf, dfs['Price Point'][x][0], dfs['Stop Range'][x], current_price, logger)

            if dfs['Status'][x][0] in [-1, 0, 3] and current_price > dfs['Price Point'][x][0] + dfs['Stop Range'][x]:
                action = 'comprar'
                strength = 'Stop Loss'
                logger.info(f'{coin} Buy Alarm Triggered')
                ret = round((current_price / dfs['Price Point'][x][0] - 1) * -100, 2)
                send_alarm(coin, action, strength, current_price, chat_id)

                # Comprar
                if dfs['Status'][x][0] in [0, 3]:
                    exit_code, dfs['Price Point'][x][0] = place_order(ticker, ticker[-4:], 'free', 'BUY',
                                                                      'NO_SIDE_EFFECT', dfs['Price Point'][x][0], o_type)
                else:
                    exit_code, dfs['Price Point'][x][0] = place_order(ticker, ticker[:-4], 'borrowed', 'BUY',
                                                                      'AUTO_REPAY', dfs['Price Point'][x][0], o_type)
                    dfs['Status'][x][0] = 0
                    exit_code, dfs['Price Point'][x][0] = place_order(ticker, ticker[-4:], 'free', 'BUY',
                                                                      'NO_SIDE_EFFECT', dfs['Price Point'][x][0], o_type)
                if exit_code == 3:
                    message = f'Por seguridad el programa se detendrá. Por favor revisar el estado de las ordenes en el exchange y reiniciar desde el servidor.'
                    logger.info(message)
                    # telegram_send.send(messages=[message])
                    bot.send_message(chat_id, message)
                    exit()
                elif exit_code in [1, 2]:
                    break

                df['TR'] = atr(df, 500)
                dfs.loc[cryptos[x], 'Stop Range'] = df['TR'][-1]
                dfs['Status'][x][0] = 1
                dfl = append_to_col(dfl, (cryptos[x]), [current_datetime, current_price, -1, ret])
                log_status(dfs, cdf, dfs['Price Point'][x][0], dfs['Stop Range'][x], current_price, logger)

            dfs.loc[cryptos[x], 'Countdown'] = time.time()
            if dfs['Status'][x][0] in [-2, 3]:  # Set sleep time after a stop loss
                dfs.loc[cryptos[x], 'Sleep'] = sl_sleep
            else:  # Sets sleep time to the next window when there isn't a stop loss
                dfs.loc[cryptos[x], 'Sleep'] = int(t_list[0][:-3]) * 60

            if current_price < dfs['Drawback'][x] and dfs['Status'][x][0] in [3, 2, 1]:
                dfs.loc[cryptos[x], 'Drawback'] = current_price
                dfs.loc[cryptos[x], 'Drawback T'] = current_datetime

    dfs.to_pickle(f'{loc_folder}Status.pkl')
    dfl.to_csv(f'{loc_folder}Transactions.csv')
    dfl.to_pickle(f'{loc_folder}Transactions.pkl')
    time.sleep(60)