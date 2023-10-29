import time
import math
from binance.client import Client
from Mfun import *

api_key = B_API_KEY
api_secret = B_API_SECRETKEY
client = Client(api_key, api_secret)
action = 'repay'  # Set the action among the following: 'margin_buy', 'buy', 'sell', 'repay'. (Usually 'repay' or 'sell' to fix)
asset_precision = 5
ticker = 'BTCTUSD'
o_type = "MARKET"  # 'LIMIT' or 'MARKET'

info = client.get_margin_account()
ticker_price = client.get_orderbook_ticker(symbol=ticker)
bid_price = float(ticker_price['bidPrice'])
ask_price = float(ticker_price['askPrice'])

if action in ['repay', 'sell']:
    asset = ticker[:3] #Sell / Repay
else:
    asset = ticker[-4:] #Buy / Margin Buy

if action in ['buy', 'sell', 'margin_buy']:
    asset_m = 'free'  # Buy, Sell, Margin Buy
else:
    asset_m = 'borrowed'  # Repay

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
    order = place_o(client, amount, action, ticker, o_type, price)
elif action == 'repay':
    amount = round_up(amount, asset_precision)  # Repay
    order = place_o(client, amount, action, ticker, o_type, price)
elif action == 'buy':
    amount = round_down(amount / price, asset_precision)  # Buy
    order = place_o(client, amount, action, ticker, o_type, price)
elif action == 'margin_buy':
    amount = round_down(amount / price, asset_precision)  # Margin Buy
    order = place_o(client, amount, action, ticker, o_type, price)

print(amount, price)
print(info)

time.sleep(15)
info = client.get_margin_account()
print(info)
exit()

