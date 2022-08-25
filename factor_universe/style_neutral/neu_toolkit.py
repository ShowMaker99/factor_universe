from PqiDataSdk import *
from functools import partial
from itertools import product
from matplotlib import pyplot as plt
from loguru import logger
from tabulate import tabulate
from matplotlib.gridspec import GridSpec
from numba import jit, njit, objmode
from CrossdayAlphaTest.interface import CrossDayBacktest
from scipy.stats import norm
import seaborn as sns
import multiprocessing as mp
import pandas as pd
import tqdm
import time
import copy
import getpass
import numpy as np
import os
import pickle
import warnings

user = getpass.getuser()

class NeuKit():
    
    def __init__(self, start_date='20180101', end_date='20211231'):
        self.ds = PqiDataSdk(user=user, size=1, pool_type="mt", log=False, offline=True)
        self.start_date = start_date
        self.end_date = end_date
        self.date_list = self.ds.get_trade_dates(start_date=self.start_date, end_date=self.end_date)
        self.stock_pool = self.ds.get_ticker_list(date = 'all')
        self.eod_data_dict = self.ds.get_eod_history(tickers=self.stock_pool, start_date=self.start_date, end_date=self.end_date, price_mode='AFTER')
        for ticker in ['000043', '000022', '601313']:
            self.stock_pool.remove(ticker)
        self.risk_fac_list = ['beta', 'book_to_price', 'earnings_yield', 'growth', 'leverage',
                            'liquidity', 'momentum', 'non_linear_size', 'residual_volatility', 'size'] # Barra 10 style factors
        # self.risk_fac_list = ['beta', 'book_to_price', 'earnings_yield', 'growth', 'leverage',
        #                       'liquidity', 'momentum', 'non_linear_size', 'residual_volatility', 'size', 'comovement', 
        #                       'agriculture', 'steel','nonferrous_metals', 'electronics', 
        #                       'household_appliance', 'food_n_beverage', 'textiles_n_apparel', 'light_mfg', 
        #                       'biomedicine', 'utility', 'transportation', 'real_estate',
        #                       'comprehensive', 'arch_mat', 'arch_deco', 'electrical_eqpt', 'military',
        #                       'computer', 'media', 'telecom', 'bank', 'non_bank_finance', 'automobile', 'machinery_n_eqpt'] # 风格因子和行业因子
        self.expo_dict = self.ds.get_factor_exposure(tickers=self.stock_pool, start_date=self.start_date,
                                      end_date=self.end_date, factors=self.risk_fac_list)
        self.attr_df = self.ds.get_factor_return(start_date=self.start_date, end_date=self.end_date, factors=self.risk_fac_list)
        self.hs300_mask = (self.ds.get_index_weight(ticker='000300', format='eod', 
                                     start_date=self.start_date, end_date=self.end_date) > 0).astype('int').replace(0, np.nan)
        self.zz500_mask = (self.ds.get_index_weight(ticker='000905', format='eod', 
                                     start_date=self.start_date, end_date=self.end_date) > 0).astype('int').replace(0, np.nan)
        self.zz1000_mask = (self.ds.get_index_weight(ticker='000852', format='eod', 
                                     start_date=self.start_date, end_date=self.end_date) > 0).astype('int').replace(0, np.nan)
        self.OA_mask = (self.ds.get_index_weight(ticker='diy_OA', format='eod', 
                                     start_date=self.start_date, end_date=self.end_date) > 0).astype('int').replace(0, np.nan)
        self.OB_mask = (self.ds.get_index_weight(ticker='diy_OB', format='eod', 
                                     start_date=self.start_date, end_date=self.end_date) > 0).astype('int').replace(0, np.nan)
        self.universe = self.ds.get_file('universe', tickers=self.stock_pool, start_date=self.start_date, 
                                         end_date=self.end_date, format='ticker_date_real')
        for period_name in ['TwapOpen60', 'TwapClose60', 'TwapMorLast15', 'TwapAftBegin15']:
            self.eod_data_dict[period_name] = self.ds.get_eod_feature(fields=['eod_' + period_name],
                                                   where='/data/shared/low_fre_alpha/yh_zp_factor_base/data/twap_prices',
                                                   tickers=self.stock_pool,
                                                   dates=self.date_list)['eod_' + period_name].to_dataframe()
        
        self.ret_dict = {
            1: self.eod_data_dict['OpenPrice'].shift(-1,axis=1) / self.eod_data_dict['ClosePrice'] - 1,
            2: self.eod_data_dict['TwapOpen60'].shift(-1,axis=1) / self.eod_data_dict['OpenPrice'].shift(-1,axis=1) - 1,
            3: self.eod_data_dict['TwapMorLast15'].shift(-1,axis=1) / self.eod_data_dict['TwapOpen60'].shift(-1,axis=1) - 1,
            4: self.eod_data_dict['TwapAftBegin15'].shift(-1,axis=1) / self.eod_data_dict['TwapMorLast15'].shift(-1,axis=1) - 1,
            5: self.eod_data_dict['TwapClose60'].shift(-1,axis=1) / self.eod_data_dict['TwapAftBegin15'].shift(-1,axis=1) - 1,
            6: self.eod_data_dict['ClosePrice'].shift(-1,axis=1) / self.eod_data_dict['TwapClose60'].shift(-1,axis=1) - 1,
            7: self.eod_data_dict['OpenPrice'].shift(-2,axis=1) / self.eod_data_dict['ClosePrice'].shift(-1,axis=1) - 1,
            8: self.eod_data_dict['TwapOpen60'].shift(-2,axis=1) / self.eod_data_dict['OpenPrice'].shift(-2,axis=1) - 1,
        } 
        
        self.name_dict = {
            1: 'YesterdayClose to Open',
            2: 'Open to TwapOpen60',
            3: 'TwapOpen60 to TwapMorningLast15',
            4: 'TwapMorningLast15 to TwapAfternoonBegin15',
            5: 'TwapAfternoonBegin15 to TwapClose60',
            6: 'TwapClose60 to Close',
            7: 'Close to TomorrowOpen',
            8: 'TomorrowOpen to TomorrowTwapOpen60'
        }
        
        
        def get_sw_ind_df(level=1):
            sw1_df = (self.ds.get_sw_members(level=1)) * 1
            sw1_df = sw1_df[sw1_df['out_date'].isna()][['index_code', 'con_code']]
            ind_df = pd.DataFrame(index=['100000'] + list(np.sort(list(set(sw1_df['index_code'].values)))), columns=self.stock_pool, dtype='float')
            for i in range(sw1_df.shape[0]):
                ind_df.loc[sw1_df.iloc[i, 0], sw1_df.iloc[i, 1]] = 1
            ind_df = ind_df.fillna(0)
            ind_df.loc['100000'] = 1 - ind_df.iloc[1:].sum()
            return ind_df
        
        self.ind_df = get_sw_ind_df(level=1)
        
    def fac_value_check(self, fac_df):
        fac_df = fac_df.loc[:, self.date_list]
        filled_nums = np.sum(~np.isnan(fac_df))
        all_nums = np.sum(~np.isnan(self.eod_data_dict['ClosePrice'].loc[:, self.date_list]))
        inf_nums = np.sum(abs(fac_df) == np.inf)
        unique_nums = fac_df.nunique()
        max_facs = fac_df.replace(np.inf, np.nan).max()
        min_facs = fac_df.replace(-np.inf, np.nan).min()
        filled_ext = list(filled_nums[(abs(filled_nums.diff(1)) > 200) & (abs(filled_nums.diff(-1)) > 200)].index)
        inf_ext = list(inf_nums[inf_nums > 0].index)

        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        l11 = ax1.plot(list(all_nums.index), list(all_nums), label='all stocks')
        l12 = ax1.plot(list(filled_nums.index), list(filled_nums), label='filled stocks')
        l13 = ax1.plot(list(unique_nums.index), list(unique_nums), label='unique values')
        ax1.set_xticks(list(filled_nums.index)[::int(len(list(filled_nums.index))/6)])
        ax1.set_ylabel('stock nums')
        ax1.grid()

        ax12 = ax1.twinx()
        ax12.bar(range(len(inf_nums)), inf_nums, width=1, color='magenta', label='inf stocks')
        ax12.set_ylabel('inf nums')
        l1 = l11 + l12 + l13
        labs1 = [l.get_label() for l in l1]
        ax1.legend(l1, labs1, loc='lower right')
        plt.title('all/filled/inf nums')

        ax2 = fig.add_subplot(gs[0, 1])
        l21 = ax2.plot(list(max_facs.index), list(max_facs), color='magenta', label='max factor values')
        ax2.set_ylabel('factor max values')
        ax22 = ax2.twinx()
        l22 = ax22.plot(list(min_facs.index), list(min_facs), color='deepskyblue', label='min factor values')
        ax22.set_ylabel('factor min values')
        ax2.set_xticks(list(max_facs.index)[::int(len(list(max_facs.index))/6)])
        ax2.grid()
        l2 = l21 + l22
        labs2 = [l.get_label() for l in l2]
        ax2.legend(l2, labs2)
        plt.title('daily max/min values')

        ax3 = fig.add_subplot(gs[1, :])
        fac_temp = fac_df.replace(np.inf, np.nan)
        fac_temp = fac_temp.replace(-np.inf, np.nan)
        sns.violinplot(data=fac_temp.iloc[:, ::int(len(fac_temp.T) / 30)])
        plt.title('factor value distribution')

        plt.xticks(rotation=45)

        print('filled extreme days: ', filled_ext)
        print('inf extreme days: ', inf_ext)

    def ind_neu(self, factor_df):
        factor_df_list = []
        for ind in self.ind_df.index:
            ind_list = self.ind_df.loc[ind].replace(0, np.nan).dropna().index
            factor_df_list.append((factor_df.loc[ind_list] - factor_df.loc[ind_list].mean()) / factor_df.loc[ind_list].std())
            factor_ind_neu_df = pd.concat(factor_df_list)
        factor_ind_neu_df.index = list(np.sort(list(factor_ind_neu_df.index)))
        return factor_ind_neu_df

    def style_neu(self, factor_df, neu_list):
        factor_df = factor_df.loc[:, self.date_list]
        @njit()
        def groupOLS(i):
            y = Y[i]
            x = X[i]

            res = np.empty(shape=y.shape)
            mask = np.isnan(y)
            res[mask] = np.NAN
            if len(res[~mask]) > 0:
                # regress and fetch residual
                coef, _, _, _ = np.linalg.lstsq(x[~mask], y[~mask], rcond=-1)
                res[~mask] = y[~mask] - x[~mask]@coef
            else:
                res[~mask] = 0
            return res
        
        # process_X
        X_list = [pd.DataFrame(np.ones(shape=factor_df.shape[0] * factor_df.shape[1]).reshape(factor_df.shape[0], factor_df.shape[1]))]
        if 'all' in neu_list:
            X_list = X_list + [self.expo_dict[fac].fillna(self.expo_dict[fac].mean()) for fac in self.risk_fac_list]
        else:
            X_list = X_list + [self.expo_dict[fac].fillna(self.expo_dict[fac].mean()) for fac in neu_list]
        X = np.dstack(X_list)        
        X = np.ascontiguousarray(np.swapaxes(X, 0, 1))

        # process_Y
        Y = factor_df.values
        Y = np.ascontiguousarray(Y.T)

        idx = list(range(factor_df.shape[1]))
        res_df = factor_df.copy()
        res_df.loc[:, :] = np.array(list(map(groupOLS, idx))).T

        return res_df
    
    def neutralize(self, factor_df, neu_list):
        factor_df = factor_df.loc[:, self.date_list]
        @njit()
        def groupOLS(i):
            y = Y[i]
            x = X[i]

            res = np.empty(shape=y.shape)
            mask = np.isnan(y)
            res[mask] = np.NAN
            if len(res[~mask]) > 0:
                # regress and fetch residual
                coef, _, _, _ = np.linalg.lstsq(x[~mask], y[~mask], rcond=-1)
                res[~mask] = y[~mask] - x[~mask]@coef
            else:
                res[~mask] = 0
            return res

        # process_X
        X_list = [pd.DataFrame(np.ones(shape=factor_df.shape[0] * factor_df.shape[1]).reshape(factor_df.shape[0], factor_df.shape[1]))]
        for df in neu_list:
            df = df.loc[:, self.date_list]
            df = df * 0 + df
            X_list = X_list + [df.fillna(df.mean())]
        X = np.dstack(X_list)        
        X = np.ascontiguousarray(np.swapaxes(X, 0, 1))

        # process_Y
        Y = factor_df.values
        Y = np.ascontiguousarray(Y.T)

        idx = list(range(factor_df.shape[1]))
        res_df = factor_df.copy()
        res_df.loc[:, :] = np.array(list(map(groupOLS, idx))).T

        return res_df
     
    def origin_process(self, factor_df):
        return (factor_df - factor_df.mean()) / ((factor_df - factor_df.mean()).abs().sum() / 2)
    
    def linear_process(self, factor_df):
        linear_factor_df = factor_df.rank(pct=True) * 2 - 1
        linear_factor_df = linear_factor_df * 2 / linear_factor_df.abs().sum()
        return linear_factor_df
    
    def normal_process(self, factor_df):
        normal_factor_df = factor_df.copy()
        normal_factor_df.loc[:, :] = norm.ppf(factor_df.rank(pct=True) - 1 / (2 * factor_df.rank().max()))
        normal_factor_df = normal_factor_df * 2 / normal_factor_df.abs().sum()
        return normal_factor_df
    
    def sftmax_process(self, factor_df):
        sftmax_factor_df = factor_df.copy()
        sftmax_factor_df = np.exp(factor_df) / (1 + np.exp(factor_df))
        sftmax_factor_df = sftmax_factor_df - sftmax_factor_df.mean()
        sftmax_factor_df = sftmax_factor_df * 2 / sftmax_factor_df.abs().sum()
        return sftmax_factor_df
        
    def long_short(self, factor_df, ret_df):
        factor_df = factor_df.loc[:, self.date_list]
        ret_df = ret_df.loc[:, self.date_list]

        def judge_pos(ret_series):
            return ret_series * np.sign(ret_series.sum())

        posneg_factor_df = self.origin_process(factor_df)
        posneg_ret_df = judge_pos((posneg_factor_df * ret_df).sum())
        linear_factor_df = self.linear_process(factor_df)
        linear_ret_df = judge_pos((linear_factor_df * ret_df).sum())
        normal_factor_df = self.normal_process(factor_df)
        normal_ret_df = judge_pos((normal_factor_df * ret_df).sum())
        sftmax_factor_df = self.sftmax_process(factor_df)
        sftmax_ret_df = judge_pos((sftmax_factor_df * ret_df).sum())
        
        if np.sign((posneg_factor_df * ret_df).sum().sum()) > 0:
            flag = 'pos'
        elif np.sign((posneg_factor_df * ret_df).sum().sum()) < 0:
            flag = 'neg'
        else:
            flag = 'zero'

        plt.figure(figsize=(20, 4), dpi=200)
        plt.plot(self.date_list, list(posneg_ret_df.cumsum().values), color='lightcoral')
        plt.plot(self.date_list, list(linear_ret_df.cumsum().values), color='darkorange')
        plt.plot(self.date_list, list(normal_ret_df.cumsum().values), color='limegreen')
        plt.plot(self.date_list, list(sftmax_ret_df.cumsum().values), color='cyan')
        plt.xticks(self.date_list[::int(len(self.date_list) / 10)])
        plt.grid(b=True, axis='y')
        plt.title('Factor Direction: {}'.format(flag))
        plt.legend(labels=['origin-IR: {}'.format(round(posneg_ret_df.mean() / posneg_ret_df.std() * np.sqrt(245), 3)),
                     'linear-IR: {}'.format(round(linear_ret_df.mean() / linear_ret_df.std() * np.sqrt(245), 3)),
                     'normal-IR: {}'.format(round(normal_ret_df.mean() / normal_ret_df.std() * np.sqrt(245), 3)),
                     'sftmax-IR: {}'.format(round(sftmax_ret_df.mean() / sftmax_ret_df.std() * np.sqrt(245), 3))])
        
        
    def style_anal(self, factor_df, style_list='all', processed=True):
        '''因子风格分析'''
        if processed:
            signal_df = self.origin_process(factor_df)
        else:
            signal_df = factor_df.copy()
            
        '''算/画暴露'''
        plt.figure(figsize=(20, 5), dpi=200)
        if 'all' in style_list:
            expo_df = pd.DataFrame(index=self.risk_fac_list, columns=self.date_list, dtype='float')
            for fac in self.risk_fac_list:
                expo_df.loc[fac] = (self.expo_dict[fac].shift(-1, axis=1) * signal_df).sum()
                plt.plot(self.date_list, list(expo_df.loc[fac].values))
            plt.legend(labels=self.risk_fac_list)
        else:
            expo_df = pd.DataFrame(index=style_list, columns=self.date_list, dtype='float')
            for fac in style_list:
                expo_df.loc[fac] = (self.expo_dict[fac].shift(-1, axis=1) * signal_df).sum()
                plt.plot(self.date_list, list(expo_df.loc[fac].values))
            plt.legend(labels=style_list)

        plt.xticks(self.date_list[::int(len(self.date_list) / 10)])
        plt.grid(b=True, axis='y')
        plt.title('Style Exposure')
        
        '''算/画收益'''
        plt.figure(figsize=(20, 5), dpi=200)
        if 'all' in style_list:
            attr_df = (expo_df * self.attr_df).cumsum(axis=1)
            for fac in self.risk_fac_list:
                plt.plot(self.date_list, list(attr_df.loc[fac].values))
            plt.legend(labels=self.risk_fac_list)
        else:
            attr_df = (expo_df * self.attr_df.loc[style_list]).cumsum(axis=1)
            for fac in style_list:
                plt.plot(self.date_list, list(attr_df.loc[fac].values))
            plt.legend(labels=style_list)
        plt.xticks(self.date_list[::int(len(self.date_list) / 10)])
        plt.grid(b=True, axis='y')
        plt.title('Style Attribution')
        
        return expo_df, attr_df
    
    def group_TS_plot(self, factor_df, ret_df, group_num, period='3m'):
        def gen_colors(color_num):
            values = [int(i * 250 / color_num) for i in range(color_num)]
            colors=["#%02x%02x%02x"%(200, int(g), 40) for g in values]
            return colors
        clrs = gen_colors(color_num=group_num)
        
        factor_df = factor_df.loc[:, self.date_list]
        ret_df = ret_df.loc[:, self.date_list]
        uprank_df = factor_df.rank(axis=0, ascending=False)
        # 分组
        group_signal_df_list = list(np.arange(group_num))
        stock_num_base = factor_df.rank(axis=0).max() / group_num
        group_ret_series_list_no_cost = list(np.arange(group_num))

        plt.figure(figsize = (20, 5), dpi = 200)
        for i in range(group_num):
            group_signal_df_list[i] = (uprank_df <= (i+1) * stock_num_base) * (uprank_df > i * stock_num_base)
            group_ret_series_list_no_cost[i] = (group_signal_df_list[i] * ret_df).sum(axis=0) / group_signal_df_list[i].sum(axis=0)
            group_ret_series_list_no_cost[i] = group_ret_series_list_no_cost[i].fillna(0).replace([-np.inf, np.inf], np.nan)
            group_ret_series_list_no_cost[i].index = pd.to_datetime(group_ret_series_list_no_cost[i].index)
            group_ret_series_list_no_cost[i] = group_ret_series_list_no_cost[i].resample(period).sum()

        group_mean = sum([group_ret_series_list_no_cost[i] for i in range(group_num)]) / group_num
        for i in range(group_num):
            if i == 0:
                plt.plot(group_ret_series_list_no_cost[i] - group_mean, marker='o', markersize=3, color=clrs[i], linewidth=1, linestyle='--')
            else:
                plt.plot(group_ret_series_list_no_cost[i] - group_mean, marker='o', markersize=3, color=clrs[i], linewidth=1)

        plt.grid(b=True, axis='y')
        plt.legend(labels=[i for i in range(group_num)])
        plt.title('Group Return Periodical')
        plt.show()
    
    def group_plot(self, factor_df, ret_df, group_num):
        factor_df = factor_df.loc[:, self.date_list]
        uprank_df = factor_df.rank(axis=0, ascending=False)
        # 分组
        group_signal_df_list = list(np.arange(group_num))
        stock_num_base = factor_df.rank(axis=0).max() / group_num
        group_ret_series_list_no_cost = list(np.arange(group_num))
        for i in range(group_num):
            group_signal_df_list[i] = (uprank_df <= (i+1) * stock_num_base) * (uprank_df > i * stock_num_base)
            group_ret_series_list_no_cost[i] = (group_signal_df_list[i] * ret_df).sum(axis=0) / group_signal_df_list[i].sum(axis=0)
            group_ret_series_list_no_cost[i] = group_ret_series_list_no_cost[i].fillna(0).replace([-np.inf, np.inf], np.nan)
        
        total_ret = [np.nansum(ret) for ret in group_ret_series_list_no_cost]
        plt.figure(figsize = (20, 5), dpi = 200)
        plt.bar(range(len(total_ret)), [i- np.mean(total_ret) for i in total_ret], color='r')
        plt.hlines(np.mean(total_ret), xmin=0, xmax=len(total_ret)-1, color='grey')
        plt.xticks(range(group_num))
        plt.grid(b=True, axis='y')
        plt.title('Group Return No Cost Bar')
        plt.show()
        
    def quantile_cut(self, factor_df, down, up):
        '''因子分位数切割'''
        factor_df[(factor_df > factor_df.quantile(up)) | (factor_df < factor_df.quantile(down))] = np.nan
        return factor_df
        
        
        
    def factor_corr(self, factor_df, path):
        factor_list = [i[4:-3] for i in os.listdir(path + '/eod_feature/')]
        for fac_name in factor_list:
            target_df = self.ds.get_eod_feature(fields=['eod_' + fac_name],
                                    where=path,
                                    tickers=self.stock_pool,
                                    dates=self.date_list)['eod_' + fac_name].to_dataframe()
            corr = factor_df.corrwith(target_df).mean()
            print('Corr with {} is: {}'.format(fac_name, round(corr, 4)))
    
    def IC_ts_stat(self, factor_df, ret_df):
        '''时序IC统计'''
        factor_df = factor_df.loc[:, self.date_list]
        ret_df = ret_df.loc[:, self.date_list]
        
        IC_srs = factor_df.corrwith(ret_df)
        IC_srs.index = [pd.to_datetime(i) for i in IC_srs.index]
        week_IC = IC_srs.resample('w').mean()

        rankIC_srs = factor_df.corrwith(ret_df, method='spearman')
        rankIC_srs.index = [pd.to_datetime(i) for i in rankIC_srs.index]
        week_rankIC = rankIC_srs.resample('w').mean()
        
        fig = plt.figure(figsize = (20, 8), dpi = 200)

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.bar(list(week_IC.index), list(week_IC.values), width=5, color='orangered')
        ax1.set_xticks(list(week_IC.index)[::int(len(week_IC) / 10)])
        ax1.grid(b=True, axis='y')
        ax1.legend(labels=['IC_pos_ratio: {}%'.format(round(100 * len(week_IC[week_IC>=0]) / len(week_IC), 2))])
        ax1.set_title('IC: {}, absIC: {}'.format(round(IC_srs.mean(), 4), round(IC_srs.abs().mean(), 4)))

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.bar(list(week_rankIC.index), list(week_rankIC.values), width=5, color='orange')
        ax2.set_xticks(list(week_rankIC.index)[::int(len(week_rankIC) / 10)])
        ax2.grid(b=True, axis='y')
        ax2.legend(labels=['rankIC_pos_ratio: {}%'.format(round(100 * len(week_rankIC[week_rankIC>=0]) / len(week_rankIC), 2))])
        ax2.set_title('rankIC: {}, absrankIC: {}'.format(round(rankIC_srs.mean(), 4), round(rankIC_srs.abs().mean(), 4)))
        plt.show()
          
    
    def backtest_by_interval(self, fac_df, group_num, interval='all'):
        '''时段interval：all表示所有时段'''
        
        if interval == 'all':
            interval = list(range(1, 9))
        
        # 多空分组,按因子值均值分两组
        fac_df = (fac_df * self.universe).loc[:, self.date_list]
        standard_df = self.origin_process(factor_df = fac_df)
        # 生成排序
        uprank_df = standard_df.rank(axis=0, ascending=False)
        long_signal_df = ((standard_df > 0).astype('int') * standard_df).fillna(0)
        short_signal_df = -1 * ((standard_df <= 0).astype('int') * standard_df).fillna(0)
        
        # 分组
        group_signal_df_list = list(np.arange(group_num))
        stock_num_base = standard_df.rank(axis=0).max() / group_num
        for i in range(group_num):
            group_signal_df_list[i] = (uprank_df <= (i+1) * stock_num_base) * (uprank_df > i * stock_num_base)
            
        for num in interval:  # 某时段
            ret_df = (self.ret_dict[num] * self.universe).loc[:, self.start_date: self.end_date]  # 该时段选定票池上的收益率
            index_ret = (ret_df * self.universe).mean(axis=0)

            ic_sep = fac_df.corrwith(ret_df)
            ic = ic_sep.cumsum()  # ic值
            rank_ic_sep = fac_df.corrwith(ret_df, method='spearman')
            rank_ic = rank_ic_sep.cumsum()# rank ic值

            long_group_cum = ((long_signal_df * ret_df).sum() - index_ret).cumsum()  # 多组累计收益率
            short_group_cum = ((short_signal_df * ret_df).sum() - index_ret).cumsum()  # 空组累计收益率

            # 初始化分组总收益列表
            group_ret_series_list_no_cost = list(np.arange(group_num))
            for i in range(group_num):
                group_ret_series_list_no_cost[i] = (group_signal_df_list[i] * ret_df).sum(axis=0) / group_signal_df_list[i].sum(axis=0)
                group_ret_series_list_no_cost[i] = group_ret_series_list_no_cost[i].fillna(0).replace([-np.inf, np.inf], np.nan)
            
            fig = plt.figure(figsize = (20, 8), dpi = 200)
            ax = fig.add_subplot(2, 1, 1)
            # 费前多空曲线
            ax.plot(list(long_group_cum.index), list(long_group_cum.values), color='darkorange')
            ax.plot(list(short_group_cum.index), list(short_group_cum.values), color='limegreen')
            ax.set_xticks(list(long_group_cum.index)[::int(len(list(long_group_cum.index))/6)])
            ax.legend(['long_no_fee', 'short_no_fee'])
            ax.grid(b=True, axis='y')
            ax.set_title(f'Long Short Excess Return of interval {self.name_dict[num]}')

            ax1 = fig.add_subplot(2, 2, 3)
            total_ret = [np.nansum(ret) for ret in group_ret_series_list_no_cost]
            ax1.bar(range(len(total_ret)), total_ret)
            ax1.hlines(np.mean(total_ret), xmin=0, xmax=len(total_ret)-1, color='r')
            ax1.set_xticks(range(group_num))
            ax1.grid(b=True, axis='y')
            ax1.set_title('Group Return No Cost Bar')

            ax2 = fig.add_subplot(2, 2, 4)
            ax2.plot(list(ic.index), list(ic.values), color='orangered')
            ax2.plot(list(rank_ic.index), list(rank_ic.values), color='orange')
            ax2.set_xticks(list(ic.index)[::int(len(list(ic.index))/6)])
            ax2.legend(labels=['Cumulated IC_IR: {}'.format(round(ic.diff().mean() / ic.diff().std(), 3)),
                         'Cumulated rankIC_IR: {}'.format(round(rank_ic.diff().mean() / rank_ic.diff().std(), 3))])
            ax2.grid(b=True, axis='y')
            ax2.set_title('IC: {} & rank IC: {}'.format(round(ic_sep.mean(), 3), round(rank_ic_sep.mean(), 3)))    
            
            plt.show()
        
    def modify_fac(self, ret_df, factor_df, group_num, head=True, tail=False, period=20):  
        # 正相关因子的调整：序贯剔除头尾组因子值
        factor_df = factor_df.loc[:, self.date_list]
        ret_df = ret_df.loc[:, self.date_list]
        interval = len(self.date_list)
        start = period
        while start < interval:
            fac_earlier = factor_df.iloc[:,(start-period):start].copy()  # 前一个period的因子值，用作判断条件
            first = (fac_earlier > fac_earlier.quantile(1-1/group_num))  # 头组
            second = ((fac_earlier > fac_earlier.quantile(1-2/group_num)) & (fac_earlier <= fac_earlier.quantile(1-1/group_num)))  # 次组

            end = (fac_earlier < fac_earlier.quantile(1/group_num))  # 尾组
            last = ((fac_earlier < fac_earlier.quantile(2/group_num)) & (fac_earlier >= fac_earlier.quantile(1/group_num)))  # 次尾组

            fac_copy = factor_df.iloc[:,start:].copy()  # 当前及后续因子值
            if head:
                if (first.astype('int') * ret_df).mean().mean() < (second.astype('int') * ret_df).mean().mean():
                    # 前一个period第一个多组收益均值比第二个小，则永久剔除头组因子值
                    factor_df.iloc[:,start:][fac_copy > fac_copy.quantile(1-1/group_num)] = np.nan
            if tail:
                if (last.astype('int') * ret_df).mean().mean() < (end.astype('int') * ret_df).mean().mean():   
                    # 前一个period次尾组收益均值比尾组小，则永久剔除尾组因子值
                    factor_df.iloc[:,start:][fac_copy < fac_copy.quantile(1/group_num)] = np.nan
            start += period
        return factor_df
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        