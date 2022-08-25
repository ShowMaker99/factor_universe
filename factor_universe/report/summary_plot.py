import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from CrossdayAlphaTest.interface import CrossDayBacktest
from tools.tools import *
from backtest.single.bt_config import Config

# Font config
font1 = {
    'family':'DejaVu Sans',
    'weight':'normal',
    'size':17
        }

# Data-loading parameters
tickers = get_ticker_list()
dates = get_trade_dates(Config.start_date, Config.end_date)

# # Load data
# tickers = get_ticker_list()
# dates = get_trade_dates(Config.start_date, Config.end_date)
# actTrdValRtoS = read_eod_data('mins_mf_actTrdValRtoS_day_mean',
#                 tickers, dates, path)
# actTrdValRtoM = read_eod_data('mins_mf_actTrdValRtoM_day_mean',
#                 tickers, dates, path)
# actTrdValRtoL = read_eod_data('mins_mf_actTrdValRtoL_day_mean',
#                 tickers, dates, path)
# actTrdValRtoXL = read_eod_data('mins_mf_actTrdValRtoXL_day_mean',
#                 tickers, dates, path)

# # Plot 
# plt.figure(figsize=(15, 4), dpi=255)
# plt.plot(pd.to_datetime(dates),actTrdValRtoS.apply(np.mean).values, color='darkblue', label='S')
# plt.plot(pd.to_datetime(dates),actTrdValRtoM.apply(np.mean).values, color='darkred', label='M')
# plt.plot(pd.to_datetime(dates),actTrdValRtoL.apply(np.mean).values, color='darkorange', label='L')
# plt.plot(pd.to_datetime(dates),actTrdValRtoXL.apply(np.mean).values, color='grey', label='XL')
# plt.legend(loc='upper left')
# plt.title('Percentage of Four Types of Traders')

# Load data
longCumRtnNC = {}  # long cummulative no-fee return (each 5 clips of 3 freqs)
stats = {} # 3 summary statistics (SharpeNC, RankIC, ALphaReturnNC) for each 5 clips of 3 freqs
names = {
    'day': " f'rtn_actTrdValRtoS_roll_20day_top_{cutQtlL}_to_{cutQtlR}_mean' ",
    'hh': " f'rtn_actTrdValRtoS_roll_160hh_top_{cutQtlL}_to_{cutQtlR}_mean' ",
    'min': " f'mins_mf_rtn_actTrdValRtoS_{cutQtlL}to{cutQtlR}_mean' "
}
cutQtl = 0.2

for freq in ['day','hh','min']:
    path = f'/home/shared/Data/data/shared/low_fre_alpha/ap_monster_zoo/factor_{freq}_new1'
    longCumRtnNC[freq] = {}
    stats[freq] = {}
    for cutQtlL, cutQtlR in zip(np.round(np.arange(0,1,cutQtl),1), np.round(np.arange(0,1.00001,cutQtl)[1:],1)):
        name = eval(names[freq])
        factor =  -read_eod_data(name, tickers, dates, path)  # 加负号，反转
        if freq == 'min':
            factor = factor.rolling(20, axis=1, min_periods=10).mean()
        cdb = CrossDayBacktest(factor, Config)
        result_dict = {}
        result_dict['long_short'] = cdb.long_short_pnl_summary()
        result_dict['ic'] = cdb.ic_summary()
        result_dict['summary'] = cdb.float_item_summary()
        stats[freq][str(cutQtlL)+'_to_'+str(cutQtlR)] = {}
        longCumRtnNC[freq][str(cutQtlL)+'_to_'+str(cutQtlR)] = result_dict['long_short']['long_short_alpha_cum_pnl']['long_rtn_no_fee']
        stats[freq][str(cutQtlL)+'_to_'+str(cutQtlR)]['SharpeNC'] = result_dict['summary']['AlphaSharpeNC']
        stats[freq][str(cutQtlL)+'_to_'+str(cutQtlR)]['RankIC'] = result_dict['ic']['rank_ic']
        stats[freq][str(cutQtlL)+'_to_'+str(cutQtlR)]['AlphaRtnNC'] = result_dict['summary']['AlphaRtnNC']

# 净值曲线图
fig = plt.figure(figsize=(16,15), dpi=255)
for i, freq in enumerate(['day','hh','min']):
    fig.add_subplot(3,1,i+1)
    dates = pd.to_datetime(dates)
    plt.plot(dates, longCumRtnNC[freq]['0.0_to_0.2'], color='darkred', label='Top 0%~20%')
    plt.plot(dates, longCumRtnNC[freq]['0.2_to_0.4'], color='darkorange', label='20%~40%')
    plt.plot(dates, longCumRtnNC[freq]['0.4_to_0.6'], color='limegreen', label='40%~60%')
    plt.plot(dates, longCumRtnNC[freq]['0.6_to_0.8'], color='royalblue', label='60%~80%')
    plt.plot(dates, longCumRtnNC[freq]['0.8_to_1.0'], color='pink', label='80%~100%')
    plt.title(f'Long No-cost Excess Return for {freq}-cut', fontdict=font1)
    plt.legend(loc='upper left')
    plt.tick_params(labelsize=12)
    plt.grid(linestyle = ':', linewidth = 0.5)

# 指标统计表
stats_unfolded2d = pd.DataFrame.from_dict(stats, orient='index').stack()
stats_df = pd.concat(stats_unfolded2d.map(lambda x: pd.DataFrame([x])).values, axis=0).set_index(stats_unfolded2d.index)
plt.figure(figsize=(16,5), dpi=255)
plt.table(cellText=np.round(stats_df.values,3), rowLabels=stats_df.index, colLabels=stats_df.columns, loc='center', cellLoc='center', rowLoc='center')
plt.axis('off')

# 指标柱状图
