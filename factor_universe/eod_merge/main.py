from PqiDataSdk import *
import multiprocessing as mp
import pandas as pd
import tqdm
import time
import copy
import numpy as np
import os
import pickle
import warnings
import merge_config as mc
import tools
import pickle
import getpass


# 数据准备
USER = getpass.getuser()
ds = PqiDataSdk(user=USER, size=1, pool_type="mp", log=False, offline=True)

stock_pool = tools.get_ticker_list()
date_list = tools.get_trade_dates()

factor_list = [factor for factor in os.listdir(mc.path + 'eod_feature/') if mc.freq in factor] # 支持因子文件名列表
feature_name_day = list(set([name.split('_'+mc.freq+'_')[0][4:] + '_'+mc.freq+'_' + name.split('_'+mc.freq+'_')[1][:-3] for name in factor_list])) # 特征名列表(n份统一名字)
feature_name_hh = list(set(["f'" + name.split('_'+mc.freq+'_')[0][4:] + '_'+mc.freq+'_{n}' + name.split('_'+mc.freq+'_')[1][1:-3] + "'" for name in factor_list])) # 特征名列表(n份统一名字)

# 计算脚本
def get_feature_ser_day(name):
    multiIndexDf = tools.read_eod_data(name, stock_pool, date_list, mc.path).stack(dropna=False).sort_index()
    feature = name.split('_'+mc.freq+'_')[0] + '_' + name.split('_'+mc.freq+'_')[1]
    return (multiIndexDf, feature)

def get_feature_ser_hh(name):
    df_list = []
    for n in range(1, 9):
        tmp_df = tools.read_eod_data(eval(name), stock_pool, date_list, mc.path)
        tmp_df.columns = [date+'_'+mc.freq+'_'+f'{n}'for date in list(tmp_df.columns)]
        df_list.append(tmp_df)
    multiIndexDf = pd.concat(df_list, axis=1).stack(dropna=False).sort_index()
    feature = name.split('_'+mc.freq+'_{n}_')[0][2:] + '_' + name.split('_'+mc.freq+'_{n}_')[1][:-1]
    return (multiIndexDf, feature)

# 多进程处理数据
with mp.Pool(processes=mc.processor_num) as pool:
    data_list = pool.map(eval(f'get_feature_ser_{mc.freq}'), eval(f'feature_name_{mc.freq}'))

# 落地df
print('开始落地数据...')
array = np.vstack([data[0].values for data in data_list])
idx = data_list[0][0].index
columns = [data[1] for data in data_list]
merged = pd.DataFrame(array.T, index=idx, columns=columns)
merged.to_pickle(mc.save_path + 'merged_'+mc.freq+'.pkl')
