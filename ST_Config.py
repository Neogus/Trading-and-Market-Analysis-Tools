from datetime import datetime
from c_config import B_API_KEY, B_API_SECRETKEY, T_API_KEY, chat_id  # Fill-up the c_config file with your keys
from binance.client import Client
import telebot
import logging
import os

#                   ---------------   MAIN SETUP  ----------------
# General configuration
loc_folder = ''  # I/O folder can be set here (Ex:loc_folder = 'C:/Users/USER/PycharmProjects/ProjectA'//). An empty string will look for resources in the root folder.

# Importer files configuration
# WARNING: When using importer the importer will name the file with these values, when using the modeler/tester they will look for the file using these values. If the imported testing dataset name is different from the imported testing dataset name these should be changed accordingly to match the file name that is being used.

import_pair = 'BTC/BUSD'  # Dataset pair
import_sample = '1m'  # Dataset candle
import_since = 8760  # Timefranme of the dataset expressed in hours until the present.
import_desc = 'max'  # Added description to File

# Modeler, Tester & Trader configuration
# Warning: To avoid errors Modeler and Tester configuration will share common variables. If the modeler is used first and the tester is used second to test there should be no errors'''
# Warning: Start/end Date, seg_len, and seg_n values should be configured according to the datasets that are being imported and used to avoid errors by going beyond their datetime indexes or timeframes

cryptos = ['BTC/BUSD']  # The program will look for this dataset so there should be a corresponding file created by the importer first.
sample = '1min'  # The Dataset sample defines the granularity if the dataset this value will be used to resample the dataset to a candle of a higher value
t_min = 1  # Filters strategies based on an average daily minimum number of transactions.
t_max = 25  # Filters strategies based on an average daily maximum number of transactions.
fee = 0  # Transaction fee (5% = 0.05)
equity = 100  # Initial amount of investment.
groups = 19  # Parameter spectrum resolution (It defines the number of divisions made to the range of each parameter, each division adds 1 more group of random parameters to try in the y loop)
cv_loops = 3  # Time Series Cross-Validation loops.
train_p = 0.7  # Training portion of Dataset (Ex: 0.7 = 70% train/30% test), rest is for testing.
stop_loss = 0  # A Stop Loss will be placed at - the % value defined here every time a buy order is placed. True Range (TR) indicator will be used instead of percentage by setting stop_loss = 'tr'
tr = 500  # When setting stop_loss to true range mode this value represents the rolling window of the TR indicator
acc_min = 0  # Discards strategy that does not meet a minimum accuracy
d_set = 'A'  # Added Tag or description to model file
md_start_date = '2022-06-01 0:00:00'  # By setting a start date the program will select a portion of the dataset from that date to train the model. (Date should exist in the dataset to avoid errors). This value is also used to calculate the number of days of the model imprinted in the file
md_end_date = '2023-01-01 0:00:00' # By setting an end date the program will select a portion of the dataset until that date to train the model. (Date should exist in the dataset to avoid errors).  This value is also used to calculate the number of days of the model imprinted in the file
real_rf_ret = 0.0001428  # Based on the current 3-month T-bill that yields 5.35% per year (0.0001428 daily)
model_criteria = 'Alpha'  # The modeler will evaluate strategies based on the selected criteria ('Alpha' (Real Alpha) , 'Accuracy', 'Return Rate' (Average daily return rate), 'Alpha Rate' (Average daily performance rate) , 'Standard Error', 'Equity')
mul = 1  # When creating the lists of the indicator's random parameters this variable can be used to scale them (2x, 3x, etc)
td_start_date = '2023-01-01 0:00:00'  # Starting point for testing in the dataset.
seg_len = 10080  # Tester will use this value to divide the dataset into portions for independent testing of each. Segment length in minutes (1 week = 10080 , 1 month = 43800)
seg_n = 8  # The number of time segments that are going to be tested. If the segment length equals a month in time and this number is 6 it will test 6 months starting from the specified date (md_start_date). Make sure not to go beyond the last date of the imported testing dataset to avoid errors.
asset_precision = 5
o_type = "MARKET"  #Order type ("LIMIT" or "MARKET"). Sets to trade on the market (Market fee) or on a limit (Maker fee) for lower fees.
sample_freq = '1m'
since = 10

# Logger Configuration

file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Logger.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M', filename=file, filemode='a')
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)
logger = logging.getLogger('Log')



