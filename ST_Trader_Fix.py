import time
import math
from binance.client import Client
from config import B_API_KEY, B_API_SECRETKEY, T3_API_KEY
from Mfun import *

api_key = B_API_KEY
api_secret = B_API_SECRETKEY
client = Client(api_key, api_secret)
action = 'repay' # Set the action, usually repay or sell
asset_precision = 5
ticker = 'BTCTUSD'
o_type = "MARKET"

def place_order(ticker, asset, asset_m, side, e_type, price_point):
    exit_code = 0
    order_loop = 0
    order_name = f'{side}'
    while True:
        try:
            exit_loop = 0
            info = client.get_margin_account()
            print('Info Retrieved')
            for x in range(len(info['userAssets'])):
                if info['userAssets'][x]['asset'] == asset:
                    amount = float(info['userAssets'][x][asset_m]) + float(info['userAssets'][x]['interest'])
                    break

            print('Info Loop Ok')
            ticker_price = client.get_orderbook_ticker(symbol=ticker)
            btc_price = client.get_orderbook_ticker(symbol=f'BTC{asset}')
            print('Info Ticker Retrieved')
            balance = float(btc_price['askPrice']) * float(info['totalNetAssetOfBtc'])
            print(f'Balance:{balance}')
            bid_price = float(ticker_price['bidPrice'])
            ask_price = float(ticker_price['askPrice'])
            print('Info Ticker Retrieved')
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
            print('Rounded Finished')
            print(f'{price},{amount},{ticker},{side},{asset},{asset_m},{e_type}')
            if price * amount > 10.5:

                if o_type == "MARKET":
                    order = client.create_margin_order(symbol=ticker, side=side, type=o_type,
                                                   quantity=amount, sideEffectType=e_type)
                elif o_type == "LIMIT":
                    order = client.create_margin_order(symbol=ticker, side=side, type=o_type, timeInForce='GTC',
                                           quantity=amount, price=price, sideEffectType=e_type)
                print('Order Placed Ok')

            else:
                message = f'Balance insuficiente para efectuar la orden de {order_name}.'
                print(message)
                exit_code = 1
                return exit_code, price_point
        except:
            time.sleep(5)
            order_loop += 1
            if order_loop > 3:
                print(f'No se ha podido colocar la orden de {order_name} debido a un error de conexión.')
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
                    print(f'Se ha acabado el tiempo para completar la orden de {order_name}.')
                    client.cancel_margin_order(symbol=ticker, orderId=order['orderId'])
                    while order['status'] != 'CANCELED':
                        print(f'Esperando cancelación')
                        time.sleep(5)
                        order = client.get_margin_order(symbol=ticker, orderId=order['orderId'])
                    print(f'La orden fue cancelada.')
                    exit_loop = 1
                    break
                else:

                    print(f'La orden no fue completada todavía...')
                    time.sleep(5)
                    check_loop += 1
                    order = client.get_margin_order(symbol=ticker, orderId=order['orderId'])
                    if order['status'] != 'PARTIALLY_FILLED':
                        print(f'La orden fue completada parcialmente, no puede ser cancelada hasta que termine de completarse.')

                    continue
            except:
                error_loop += 1
                if error_loop > 3:
                    message = f'No se ha podido verificar la ejecución de la orden de {order_name} debido a un error de conexión. Se requiere intervención manual.'
                    print(message)
                    exit_code = 3  # Check failed
                    return exit_code
                else:
                    time.sleep(5)
                    continue
        if exit_loop == 0:
            break


    return exit_code, price

info = client.get_margin_account()
ticker_price = client.get_orderbook_ticker(symbol=ticker)
bid_price = float(ticker_price['bidPrice'])
ask_price = float(ticker_price['askPrice'])



if action in ['repay', 'sell']:
    asset = ticker[:3] #Sell / Repay
else:
    asset = ticker[-4:] #Buy / Margin Buy

if action in ['buy', 'sell', 'margin_buy']:
    asset_m = 'free' # Buy, Sell, Margin Buy
else:
    asset_m = 'borrowed' # Repay

# Sell / Auto repay
for x in range(len(info['userAssets'])):
    if info['userAssets'][x]['asset'] == asset:
        amount = float(info['userAssets'][x][asset_m]) + float(info['userAssets'][x]['interest'])
        break

if action in ['buy', 'repay']:
    price = round_down((bid_price + ask_price) / 2, 2)  # Buy / Repay
else:
    price = round_up((bid_price + ask_price) / 2, 2)  # Sell / Margin Buy

if action == 'sell':
    amount = round_down(amount, asset_precision)  # Sell
    order = client.create_margin_order(symbol=ticker, side='SELL', type="LIMIT", timeInForce='GTC', quantity = amount , price=price, sideEffectType='NO_SIDE_EFFECT')
elif action == 'repay':
    amount = round_up(amount, asset_precision) # Repay
    order = client.create_margin_order(symbol=ticker, side='BUY', type="LIMIT", timeInForce='GTC', quantity=amount,
                                       price=price, sideEffectType='AUTO_REPAY')
elif action == 'buy':
    amount = round_down(amount / price, asset_precision)  # Margin Buy / Buy
    order = client.create_margin_order(symbol=ticker, side='BUY', type="LIMIT", timeInForce='GTC', quantity = amount , price=price, sideEffectType='NO_SIDE_EFFECT')
elif action == 'margin_buy':
    amount = round_down(amount / price, asset_precision)  # Margin Buy / Buy
    order = client.create_margin_order(symbol="ETHBUSD", side='SELL', type="LIMIT", timeInForce='GTC', quantity=0.009,price=price, sideEffectType="MARGIN_BUY")

print(amount, price)
print(info)

time.sleep(15)
info = client.get_margin_account()
print(info)
exit()






