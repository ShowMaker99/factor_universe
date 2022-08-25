"""
生成因子主函数
"""
# load packages
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import warnings
import time
from scipy.stats import norm
warnings.filterwarnings('ignore')

# 控制进程数
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# load files
import config
from tools.utils import save_eod_feature
from backtest.mybacktest import BackTestMachine

# initialize dataserver
import getpass
user = getpass.getuser()
from PqiDataSdk import *
ds = PqiDataSdk(user=user, size=1, pool_type='mp')


class FactorTuner:
    def __init__(self, source):
        """
        初始化（转移设置）
        """
        # 转移设置
        self.source = source
        self.support_factor_path = config.my_support_factor_paths[source]
        self.factor_path = config.my_factor_paths[source]
        self.start_date = config.start_date
        self.end_date = config.end_date
        self.dates = ds.get_trade_calendar(start_date=self.start_date, end_date=self.end_date)
        self.tickers = config.tickers

        # 选择因子列表
        self.eod_agg_lists = config.eod_factor_aggs
        file_name_list = os.listdir(self.support_factor_path + 'eod_feature/')
        self.factor_name_list = [file_name[4:-3] for file_name in file_name_list]

    def save(self, df, name):
        ds.save_eod_feature(data={f'eod_{name}': df},
                            where=self.factor_path,
                            feature_type='eod',
                            encrypt=False,
                            save_method='append'
                            )

    @ staticmethod
    def check(factor_df, btm):
        btm.update_factor_df(factor_df)
        indicator_dict = btm.backtest()
        # define your criterion
        return True

    def process_each_factor(self, factor_list):
        btm = BackTestMachine(start_date='20160701', end_date='20200630')
        btm.DataPrepare()
        # 获取eod_factor数据
        for factor in factor_list:
            df = ds.get_eod_feature(fields=[f'eod_{factor}'],
                                    where=self.support_factor_path,
                                    dates=self.dates)[f'eod_{factor}'].to_dataframe()
            df = df.iloc[:, (df.columns >= self.start_date) & (df.columns <= self.end_date)]
            if self.check(df, btm):
                self.save(df, factor)

            for agg in self.eod_agg_lists:
                # calculate factor_df here
                factor_df = None

                factor_name = '_'.join([factor] + [str(a) for a in agg])
                if self.check(factor_df,  btm):
                    self.save(factor_df, factor_name)

    def run_eod(self):
        """ 运行eod部分 """
        print(f'running eod tuner of {self.source}...')
        # 分组multiprocessing
        num_factors = len(self.factor_name_list)
        num_factors_per_group = 50
        batch_factors = []
        for g in range(num_factors // num_factors_per_group + 1):
            curr_factors = self.factor_name_list[g * num_factors_per_group: (g + 1) * num_factors_per_group]
            if len(curr_factors) == 0:
                continue
            batch_factors.append(curr_factors)

        with mp.Pool(processes=150) as pool:
            pool.map(self.process_each_factor, batch_factors)


if __name__ == '__main__':
    from datetime import datetime
    print(f'Factor Tuner started at {datetime.now().strftime("%H:%M:%S")}-----------------------------------')
    start_time = time.time()
    ft = FactorTuner('mins')
    ft.run_eod()
    print(f'Factor Tuner finished in {round(time.time() - start_time, 2)}s at {datetime.now().strftime("%H:%M:%S")}')
