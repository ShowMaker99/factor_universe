"""
辅助函数
"""

# load package 
import numpy as np 
import pandas as pd 
import time 

# load files 
import sys 
sys.path.append('../producer')
import frame.config as config

# initialize dataserver 
from PqiDataSdk import * 
import getpass
USER = getpass.getuser()
ds = PqiDataSdk(user=USER, size=1, pool_type='mp')


# ------------------------ 存取因子 ---------------------------------
def read_eod_feature(factor_name, source):
    """ 读取储存好的因子，转换成dataframe输出 """
    factor_df = ds.get_eod_feature(fields=[f'eod_{factor_name}'], 
                                   where=config.my_support_factor_paths[source])[f'eod_{factor_name}'].to_dataframe()
    return factor_df


def save_eod_feature(factor_name, factor_df, source):
    """ 储存因子到指定路径 """
    ds.save_eod_feature(data={f'eod_{factor_name}': factor_df},
                        where=config.my_support_factor_paths[source],
                        feature_type='eod',
                        encrypt=False,
                        save_method='update'
                        )
