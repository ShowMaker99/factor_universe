# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.extend(['./', '../', '../..'])
from plotting import *
from tools.tools import *
                                                                
from bt_config import Config

tickers = get_ticker_list()
dates = get_trade_dates(Config.start_date, Config.end_date)

# Load factor data from dataserver
# initialize dataserver 
# from PqiDataSdk import * 
# import getpass
# USER = getpass.getuser()
# ds = PqiDataSdk(user=USER, size=1, pool_type='mp')
# # Load data
# eod_history = ds.get_eod_history(tickers=tickers, start_date=dates[0], end_date=dates[-1], fields=['ClosePrice', 'OpenPrice'],
#                             price_mode='AFTER', source='stock')
# # Compute factor
# openPrice = eod_history['OpenPrice']
# closePrice = eod_history['ClosePrice']
# factor = -(openPrice/closePrice.shift(1, axis=1) - 1).rolling(20, axis=1, min_periods=10).mean()
# name = '日内动量roll20mean'

# Load local factor data
path = '/home/shared/Data/data/shared/low_fre_alpha/ap_monster_zoo/factor_day_new1_neu'
name = 'rtn_actTrdValRtoS_roll_20day_top_0.0_to_0.2_mean_neu_momentum_liquidity_size_growth_non_linear_size'
factor =  -read_eod_data(name, tickers, dates, path)
# path1 = '/home/shared/Data/data/shared/low_fre_alpha/ap_monster_zoo/factor_min'
# name1 = 'mins_mf_rtn_actTrdValRtoS_0.0to0.2_mean'
# factor1 = read_eod_data(name1, tickers, dates, path1).rolling(20, axis=1, min_periods=10).mean()
# path2 = '/home/shared/Data/data/shared/low_fre_alpha/ap_monster_zoo/factor_min'
# name2 = 'mins_mf_rtn_actTrdValRtoS_0.6000000000000001to0.8_mean'
# factor2 = read_eod_data(name2, tickers, dates, path2).rolling(20, axis=1, min_periods=10).mean()
# factor = -(factor1 - factor2)


# Backtest
cdb = CrossDayBacktest(factor, Config)

# Results
result_dict = {}
result_dict['factor_name'] = name
result_dict['long_short'] = cdb.long_short_pnl_summary()
result_dict['group'] = cdb.group_pnl_summary()
result_dict['ic'] = cdb.ic_summary()
result_dict['summary'] = cdb.float_item_summary()
result_dict['year'] = cdb.year_stat_summary()

# Plot Results
plot(result_dict)
