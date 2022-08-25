"""
生成因子主函数
"""
# load packages
import os, sys
import numpy as np
import pandas as pd
import multiprocessing as mp
import warnings
import time
from datetime import datetime
import pickle
warnings.filterwarnings('ignore')

# 控制进程数
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import config
import MinsSeriesGenerator as msg
sys.path.append('../')
from tools.utils import save_eod_feature, read_eod_feature

# initialize dataserver
import getpass
user = getpass.getuser()
from PqiDataSdk import *
ds = PqiDataSdk(user=user, size=1, pool_type='mp')


class FactorGenerator:
    
    
    def __init__(self):
        """
        初始化（转移设置）
        """
        # 转移设置
        self.support_factor_paths = config.my_support_factor_paths
        self.start_date = config.start_date
        self.end_date = config.end_date
        self.dates = config.dates
        self.tickers = config.tickers

        # 选择因子列表
        self.name_list = config.support_factor_names
        self.cut_lists = config.support_factor_cuts
        self.agg_lists = config.support_factor_aggs
        self.dr_dict = config.data_range_dict

        # 复权因子
        # self.adj_factor = pd.read_csv('adj_factor.csv', index_col=0)
        self.adj_factor = ds.get_eod_history(tickers=self.tickers, start_date=self.start_date, end_date=self.end_date, fields=['AdjFactor'])['AdjFactor']
        self.adj_factor.index = [str(t).zfill(6) for t in self.adj_factor.index]
        self.adj_factor.index.name = 'ticker'


    def unpack_each_agg(self, param_dict):
        source = param_dict['source']
        name = param_dict['name']
        batch_dates = param_dict['batch_dates']
        agg = param_dict['agg']
        factor_cut_list = self.cut_lists[source][name]
        
        for cut in factor_cut_list:
            if isinstance(cut, str):
                factor_name = '_'.join(([name] if isinstance(name, str) else [str(n) for n in name]) +
                                        [cut] +
                                        ([agg] if isinstance(agg, str) else [str(a) for a in agg]))
                factor_dfs = []
                for dates in batch_dates:
                    filename = self.support_factor_paths[source] + f'tmp/{source}_{factor_name}_{dates[0]}.pkl'
                    factor_dfs.append(pd.read_pickle(filename))
                factor_df = pd.concat(factor_dfs, axis=1)
                save_eod_feature(f'{source}_{factor_name}', factor_df, source)
            elif isinstance(cut, tuple):
                cutVar = cut[0]
                cutQtls = cut[1]
                for cutQtl in cutQtls:
                    for cutPointL, cutPointR in zip(np.round(np.arange(0,1,cutQtl),1), np.round(np.arange(0,1.00001,cutQtl)[1:],1)):
                        factor_name = '_'.join(([name] if isinstance(name, str) else [str(f) for f in name]) +
                                                [cutVar] + [str(cutPointL)+'to'+str(cutPointR)] +    
                                                ([agg] if isinstance(agg, str) else [str(a) for a in agg]))
                        factor_dfs = []
                        for dates in batch_dates:
                            filename = self.support_factor_paths[source] + f'tmp/{source}_{factor_name}_{dates[0]}.pkl'
                            factor_dfs.append(pd.read_pickle(filename))
                        factor_df = pd.concat(factor_dfs, axis=1)
                        save_eod_feature(f'{source}_{factor_name}', factor_df, source)
            else:
                pass


    def unpack_process(self, source, name, batch_dates):
        print('concatenating pickles...')
        factor_agg_list = self.agg_lists[source][name]
        with mp.Pool(processes=32) as pool:
            pool.map(self.unpack_each_agg, [dict(source=source, name=name, batch_dates=batch_dates, agg=agg)
                                            for agg in factor_agg_list])


    def pre_process_mins(self, name, date, df):
        adj_factor = self.adj_factor[date]
        if isinstance(name, str):
            df[name] = eval(f'msg.{name}')(df, adj_factor)
        else:
            for n in name:
                df[n] = eval(f'msg.{n}')(df, adj_factor)
        return df
    
    
    def pre_process_mins_mf(self, name, date, df):
        adj_factor = self.adj_factor[date]
        if isinstance(name, str):
            df[name] = eval(f'msg.{name}')(df, adj_factor)
        else:
            for n in name:
                df[n] = eval(f'msg.{n}')(df, adj_factor)
        return df


    def load_data(self, source, name, dates):
        if source == 'mins':
            ds_data_dict = ds.get_mins_history(tickers=self.tickers,
                                               start_date=dates[0], end_date=dates[-1], source='depth')
        elif source == 'depth':
            ds_data_dict = ds.get_depth_history(tickers=self.tickers,
                                                start_date=dates[0], end_date=dates[-1], source='depth')
        elif source == 'mins_mf':
            mins_data_dict = ds.get_mins_history(tickers=self.tickers,
                                               start_date=dates[0], end_date=dates[-1], source='depth')
            mf_data_dict = ds.get_mins_history(tickers=self.tickers,
                                                start_date=dates[0], end_date=dates[-1], source='money_flow')
            
        data_dict = dict()
        for date in dates:
            if source != 'mins_mf':
                df = pd.concat([ds_data_dict[ticker][date] for ticker in self.tickers], keys=self.tickers)
            else:
                df1 = pd.concat([mins_data_dict[ticker][date] for ticker in self.tickers], keys=self.tickers)
                df2 = pd.concat([mf_data_dict[ticker][date] for ticker in self.tickers], keys=self.tickers)
                df = pd.concat([df1, df2], axis=1) if (not df1.empty) and (not df2.empty) else pd.DataFrame()
            if not df.empty:
                df.index.names = ['ticker', source]
                for cut, value in self.dr_dict[source].items():
                    if isinstance(cut, str):
                        (prm1, prm2) = value
                        df[cut] = (df.index.get_level_values(source) >= prm1) & (df.index.get_level_values(source) <= prm2)
                    elif isinstance(cut, tuple):
                        cutVar = cut[0]
                        cutQtls = cut[1]
                        cutVarSeries = eval(f'self.pre_process_{source}')(cutVar, date, df)[cutVar]
                        for cutQtl in cutQtls:
                            for cutPointL, cutPointR in zip(np.round(np.arange(0,1,cutQtl),1), np.round(np.arange(0,1.00001,cutQtl)[1:],1)):
                                df['_'.join(([cutVar] + [str(cutPointL)+'to'+str(cutPointR)]))] = cutVarSeries.groupby(['ticker']).rank(ascending=False, pct=True).between(cutPointL, cutPointR)
                            # df['_'.join(([cutVar] + ['Top'] + [str(cutQtl)]))] = (cutVarSeries.groupby(['ticker']).rank(ascending=False, pct=True) <= cutQtl)
                            # df['_'.join(([cutVar] + ['Btm'] + [str(cutQtl)]))] = (cutVarSeries.groupby(['ticker']).rank(ascending=False, pct=True) >= (1-cutQtl))
                    else:
                        pass
                df = eval(f'self.pre_process_{source}')(name, date, df)
            data_dict[date] = df
        return data_dict


    def process_each_date(self, param_dict):
        source = param_dict['source']
        name = param_dict['name']
        dates = param_dict['dates']
        data_dict = self.load_data(source, name, dates)
        # 计算因子
        factor_cut_list = self.cut_lists[source][name]
        factor_agg_list = self.agg_lists[source][name]
        
        for cut in factor_cut_list:
            if isinstance(cut, str):
                for agg in factor_agg_list:
                    factor_name = '_'.join(([name] if isinstance(name, str) else [str(f) for f in name]) +
                                        [cut] +
                                        ([agg] if isinstance(agg, str) else [str(a) for a in agg]))
                    ret_srs = []
                    for date, df in data_dict.items():
                        if df.empty:
                            ser = pd.Series(dtype='float64')
                        else:
                            df.index.names = ['ticker', source]
                            df = df.loc[df[cut]]

                            # do your operation here
                            ser = df.groupby(['ticker'])[name].apply(eval(f'pd.DataFrame.{agg}'))

                        ret_srs.append(ser)
                    filename = self.support_factor_paths[source] + f'tmp/{source}_{factor_name}_{dates[0]}.pkl'
                    pd.concat(ret_srs, axis=1, keys=dates).to_pickle(filename)
            elif isinstance(cut, tuple):
                cutVar = cut[0]
                cutQtls = cut[1]
                for cutQtl in cutQtls:
                    for cutPointL, cutPointR in zip(np.round(np.arange(0,1,cutQtl),1), np.round(np.arange(0,1.00001,cutQtl)[1:],1)):
                        for agg in factor_agg_list:
                            factor_name = '_'.join(([name] if isinstance(name, str) else [str(f) for f in name]) +
                                                    [cutVar] + [str(cutPointL)+'to'+str(cutPointR)] +
                                                    ([agg] if isinstance(agg, str) else [str(a) for a in agg]))
                            ret_srs = []
                            for date, df in data_dict.items():
                                if df.empty:
                                    ser = pd.Series(dtype='float64')
                                else:
                                    df.index.names = ['ticker', source]
                                    df = df.loc[df['_'.join(([cutVar] + [str(cutPointL)+'to'+str(cutPointR)]))]]

                                    # do your operation here
                                    ser = df.groupby(['ticker'])[name].apply(eval(f'pd.DataFrame.{agg}'))

                                ret_srs.append(ser)
                            filename = self.support_factor_paths[source] + f'tmp/{source}_{factor_name}_{dates[0]}.pkl'
                            pd.concat(ret_srs, axis=1, keys=dates).to_pickle(filename) 
            else:
                pass


    def run(self, source, name):
        """ 运行mins/depth部分 """
        print(f'{source} - {name} generation start at {datetime.now().strftime("%H:%M:%S")}------------------------')

        # 分组multiprocessing
        cur_time = time.time()
        num_dates = len(self.dates)
        num_dates_per_group = 10
        batch_dates = []
        for g in range(num_dates // num_dates_per_group + 1):
            curr_dates = self.dates[g * num_dates_per_group: (g + 1) * num_dates_per_group]
            if len(curr_dates) == 0:
                break
            batch_dates.append(curr_dates)

        with mp.Pool(processes=32) as pool:
            pool.map(self.process_each_date, [dict(source=source, name=name, dates=dates) for dates in batch_dates])

        self.unpack_process(source, name, batch_dates)

        # delete cache data
        for filename in os.listdir(self.support_factor_paths[source] + 'tmp'):
            os.remove(self.support_factor_paths[source] + f'tmp/{filename}')

        print(f'{source} - {name} generation finished in '
              f'{round(time.time() - cur_time, 2)}s at {datetime.now().strftime("%H:%M:%S")}')


    def run_total(self, source):
        for name in self.name_list[source]:
            self.run(source, name)


if __name__ == '__main__':
    src = 'mins_mf'
    if not os.path.exists('./tmp/'):
        os.mkdir('./tmp/')
    fg = FactorGenerator()
    fg.run_total(src)


