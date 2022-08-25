# load package 
import numpy as np 
import pandas as pd 
import time 
import os
# load files 
import sys
import logging 
from collections import defaultdict

# initialize dataserver 
from PqiDataSdk import * 
import getpass
USER = getpass.getuser()
ds = PqiDataSdk(user=USER, size=1, pool_type='mp')


def get_ticker_list():
    "获取票池"
    stock_pool = ds.get_ticker_list(date = 'all')
    for ticker in ['000043', '000022', '601313']: # 删去历史上标的发生过变更的三只票
        stock_pool.remove(ticker)
    return stock_pool


def get_trade_dates(start_date, end_date):
    "获取交易日列表"
    return ds.get_trade_dates(start_date=start_date, end_date=end_date)


def read_eod_data(name, stock_pool, date_list, path):
    "读取因子"
    read_dict = ds.get_eod_feature(fields=['eod_' + name],
                                   where=path,
                                   tickers=stock_pool,
                                   dates=date_list)
    return read_dict['eod_' + name].to_dataframe()


def save_eod_feature(factor_name, factor_df, path):
    " 储存因子到指定路径 "
    ds.save_eod_feature(data={f'eod_{factor_name}': factor_df},
                        where=path,
                        feature_type='eod',
                        encrypt=False,
                        save_method='update'
                        )


def get_feature_ser_day(name, stock_pool, date_list, path):
    "读取日频级别支持因子，不必合并"
    df = read_eod_data(name, stock_pool, date_list, path)
    return df


def get_feature_ser_hh(name, stock_pool, date_list, path):
    "读取半小时级别支持因子，合并后按时间排序"
    df_list = []
    for n in range(1, 9):
        tmp_df = read_eod_data(eval(name), stock_pool, date_list, path)
        tmp_df.columns = [date+'_hh_'+f'{n}'for date in list(tmp_df.columns)]
        df_list.append(tmp_df)
    df = pd.concat(df_list, axis=1).T.sort_index().T#.stack(dropna=False).sort_index()
    return df

