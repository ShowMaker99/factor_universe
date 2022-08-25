# Load external packages
import numpy as np
import pandas as pd
import os
from backtest import BackTestMachine
import multiprocessing as mp

# Initialize dataserver
import getpass
user = getpass.getuser()
from PqiDataSdk import PqiDataSdk
ds = PqiDataSdk(user=user, size=1, pool_type='mp')

# my_factor_path = '/home/shared/Data/data/shared/low_fre_alpha/alpha_zoo/minsV2/'
# my_factor_path = '/home/shared/Data/data/shared/low_fre_alpha/alpha_zoo/mins/'
# my_factor_path = '/home/shared/Data/data/shared/low_fre_alpha/alpha_zoo/demo/factor/'
# my_factor_path = '/home/shared/Data/data/shared/low_fre_alpha/factor_garden_v2/'
my_factor_path = '/home/shared/Data/data/shared/low_fre_alpha/ap_monster_zoo/factor_hh_new/'
# my_factor_path = '/home/shared/Data/data/shared/low_fre_alpha/00 factor_commit/'


def gen_plot(name):
    df = ds.get_eod_feature(fields=[f'eod_{name}'],
                        where=my_factor_path)[f'eod_{name}'].to_dataframe()
    df = df.rank()
    btm = BackTestMachine(start_date='20180101', end_date='20210831')
    btm.DataPrepare()
    btm.backtest(df, path='/home/shared/Data/data/shared/low_fre_alpha/ap_monster_zoo/factor_hh_new/plot/', name=name)
    
    
if __name__ == '__main__':
    # factor_name = 'depth_TradeCountDiff_hh_8_to_mid_180_std_rolling_10_std'
    # factor_name = 'mins_TtlVolDiff_hh_8_to_mid_180_quantile_0.8_rolling_10_std'
    name_lst = [i[4:-3] for i in os.listdir(my_factor_path + '/eod_feature/')]
    with mp.Pool(processes=64) as pool:
        pool.map(gen_plot, name_lst)


