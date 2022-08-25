from PqiDataSdk import *
from functools import partial
from matplotlib import pyplot as plt
import multiprocessing as mp
import pandas as pd
import tqdm
import time
import copy
import numpy as np
import os
import pickle
import getpass
import warnings
warnings.filterwarnings("ignore")
# import multiprocessing as mp
import seaborn as sns
user = getpass.getuser()

# 控制线程数(单进程情况下，只占用一个核;多进程会各自独立占用每个核)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

class BackTestMachine():


    def __init__(self, start_date, end_date, name=None):
        self.ds = PqiDataSdk(user=user, size=1, pool_type="mp", log=False, offline=True)
        self.start_date = start_date
        self.end_date = end_date
        self.extend_end_date = self.ds.get_next_trade_date(self.ds.get_next_trade_date(end_date))


    def DataPrepare(self):
        
        stock_pool = self.ds.get_ticker_list(date='all')
        self.stock_pool = stock_pool
        date_list = self.ds.get_trade_calendar(start_date=self.start_date, end_date=self.extend_end_date)
        adj_factor = self.ds.get_eod_history(tickers=stock_pool, start_date=self.start_date, end_date=self.extend_end_date, fields=['AdjFactor'])['AdjFactor']
        twap_open_price = self.ds.get_eod_history(tickers=stock_pool, start_date=self.start_date, end_date=self.extend_end_date, 
                                                  price_mode='AFTER', source="ext_stock", fields=['TwapBegin60'])['TwapBegin60']
        # adj_factor = self.ds.get_eod_feature(fields=['eod_AdjFactor'], 
        #                            where='/data/shared/low_fre_alpha/factor_zoo/future_worrior/')['eod_AdjFactor'].to_dataframe()[date_list]
        # twap_open_price = self.ds.get_eod_feature(fields=['eod_TwapBegin60'], 
        #                            where='/data/shared/low_fre_alpha/factor_zoo/future_worrior/')['eod_TwapBegin60'].to_dataframe()[date_list]
        open_price = self.ds.get_eod_history(tickers=stock_pool, start_date=self.start_date, end_date=self.extend_end_date, source="stock", fields=['OpenPrice'])['OpenPrice']
        close_price = self.ds.get_eod_history(tickers=stock_pool, start_date=self.start_date, end_date=self.extend_end_date, source="stock", fields=['ClosePrice'])['ClosePrice']
        date_list = list(adj_factor.columns)

        universe = self.ds.get_file('universe', tickers=stock_pool,
                                    start_date=self.start_date,
                                    end_date=self.extend_end_date,
                                    format='ticker_date_real')

        up_feasible_stock = self.ds.get_file('up_feasible', tickers=stock_pool,
                                             start_date=self.start_date,
                                             end_date=self.extend_end_date,
                                             format='ticker_date_real')

        universe.index = [str(i).zfill(6) for i in universe.index]
        up_feasible_stock.index = [str(i).zfill(6) for i in up_feasible_stock.index]

        self.local_universe = universe[date_list]
        self.local_up_feasible_stock = up_feasible_stock[date_list]

        self.price_dict = {}
        self.price_dict['TwapOpen'] = twap_open_price * adj_factor
        self.price_dict['Open'] = open_price * adj_factor
        self.price_dict['Close'] = close_price * adj_factor
        self.index_data = {}
        self.index_data['TwapOpenPrice'] = self.ds.get_eod_history(tickers=['000905', '000016', '000300', '000852', '000985'],
    
                                                                   start_date=self.start_date, end_date=self.extend_end_date, source='ext_index', fields=['TwapBegin30'])['TwapBegin30']

        mask300 = self.ds.get_index_weight(ticker = '000300',start_date = self.start_date, end_date=self.extend_end_date,format = 'eod') 
        mask300.loc[:,:] = np.where(np.isnan(mask300),np.nan,1)
        
        mask500 = self.ds.get_index_weight(ticker = '000905',start_date = self.start_date, end_date=self.extend_end_date,format = 'eod') 
        mask500.loc[:,:] = np.where(np.isnan(mask500),np.nan,1)

        mask1000 = self.ds.get_index_weight(ticker = '000852',start_date = self.start_date, end_date=self.extend_end_date,format = 'eod') 
        mask1000.loc[:,:] = np.where(np.isnan(mask1000),np.nan,1)

        mask1800 = mask1000.copy()
        mask1800.loc[:,:] = np.where(np.isnan(mask300)&np.isnan(mask500)&np.isnan(mask1000),np.nan,1)
        self.mask1800 = mask1800


    def backtest(self, factor_df,mask1800 = False, name = 'test', head=400, method='factor', cost=0.0015, group_num=10, benchmark = 'mean', index='000852', return_type='TwapOpen_to_TwapOpen', plot=True, risk_plot=False,path = None):
        '''
        :Input:
        factor_df, dataframe, 被测试的因子
        head, integer, 测试头组数目
        method, str, 'factor / equal', factor指因子值加权, equal指头组指定数目等权
        cost, float, 手续费
        group_num, integer, 分组个数
        benchmark, str, 'mean / index / weighted'
        index, str, 对标指数benchmark,
        return_type, str, 'Open_to_Open/TwapOpen_to_TwapOpen/TwapClose_to_TwapClose/Vwap_to_Vwap/Close_to_Close', 回测收益率
        start_date/end_date, str, 起始日期
        plot, bool, 是否画回测收益图
        risk_plot, bool, 是否画风格分析图

        :Output:
        data_dict, pd.DataFrame, 回测统计指标
        factor_ret_no_cost, pd.Series, 多空收益pnl
        '''
        # try:
        # 读取因子数据
        factor_df = factor_df.iloc[:, (factor_df.columns >= self.start_date) & (factor_df.columns <= self.end_date)]
        dates = self.ds.get_trade_dates(start_date = self.start_date,end_date=self.end_date)
        factor_df = factor_df[dates]
        factor_df = factor_df * self.local_universe
        self.factor_df = factor_df

        factor_mask = factor_df.copy()
        factor_mask.loc[:,:] = np.where(np.isnan(factor_df),np.nan,1)
        
        self.name = name
        # 生成数据
        date_list_in_use = self.ds.get_trade_dates(start_date=self.start_date, end_date=self.end_date)

        if mask1800 == False:
            backtest_df = self.factor_df * self.local_universe * self.local_up_feasible_stock
        else:
            backtest_df = self.factor_df * self.local_universe * self.local_up_feasible_stock * self.mask1800
        
        backtest_df = backtest_df[date_list_in_use]
        demean_backtest_df = backtest_df - backtest_df.mean()
        std_backtest_df = demean_backtest_df / (demean_backtest_df.abs().sum() / 2)

        # 生成收益率矩阵
        price_type = return_type.split('_')[0]
        price_df = self.price_dict[price_type]
        if 'TwapClose' in price_type:
            theoretical_rtn_df = price_df.shift(-1, axis=1) / price_df - 1
            theoretical_rtn_df = theoretical_rtn_df[date_list_in_use]
        else:
            theoretical_rtn_df = price_df.shift(-2, axis=1) / price_df.shift(-1, axis=1) - 1
            theoretical_rtn_df = theoretical_rtn_df[date_list_in_use]

        theoretical_rtn_df = theoretical_rtn_df.replace([np.inf, -np.inf], 0)

        downfeasible_df = self.ds.get_file('down_feasible', tickers=self.stock_pool, 
                                            start_date=self.start_date, end_date=self.extend_end_date, 
                                            format='ticker_date_real')
        downfeasible_price_df = price_df * downfeasible_df
        real_price_df = downfeasible_price_df.bfill(axis=1)
        real_rtn_metric = real_price_df.shift(-2, axis=1) / real_price_df.shift(-1, axis=1) - 1
        real_rtn_metric = real_rtn_metric.replace([np.inf, -np.inf], 0)

        # 算ic和rankic
        ic_list = self.factor_df[date_list_in_use].corrwith(theoretical_rtn_df[date_list_in_use])
        ic = ic_list.mean()
        rankic_list = self.factor_df[date_list_in_use].corrwith(theoretical_rtn_df[date_list_in_use], method='spearman')
        rankic = rankic_list.mean()
        
        # 算ic decay
        ic_decay_list = []
        rankic_decay_list = []
        
        # args = []
        # for i in range(10):
        #     args.append((i,self.factor_df[date_list_in_use],ret_df[date_list_in_use].shift(-i, axis=1)))
        
        # with mp.Pool(processes=10) as pool:
        #     res = pool.map(self.ic_decay_i,args)
        
        # for item in res:
        #     i,ic_decay,rank_ic_decay = item
        #     ic_decay_list[i] = ic_decay
        #     rankic_decay_list[i] = rank_ic_decay

            
        for i in range(10):
            ic_decay_list.append(self.factor_df[date_list_in_use].corrwith(theoretical_rtn_df[date_list_in_use].shift(-i, axis=1)).mean())
            rankic_decay_list.append(self.factor_df[date_list_in_use].corrwith(theoretical_rtn_df[date_list_in_use].shift(-i, axis=1), method='spearman').mean())

        # 生成回测收益率矩阵
        # limit_price_df = price_df * self.local_down_feasible_stock
        # bfill_df = (price_df - price_df) + limit_price_df.bfill(axis=1)
        # if 'TwapClose' in price_type:
        #     bfill_ret_df = bfill_df.shift(-1, axis=1) / bfill_df - 1
        #     bfill_ret_df = bfill_ret_df[date_list_in_use]
        # else:
        #     bfill_ret_df = bfill_df.shift(-2, axis=1) / bfill_df.shift(-1, axis=1) - 1
        #     bfill_ret_df = bfill_ret_df[date_list_in_use]

        bfill_ret_df = real_rtn_metric

        # 生成排序
        uprank_df = backtest_df.rank(axis=0, ascending=False)
        downrank_df = backtest_df.rank(axis=0, ascending=True)

        # 生成多头信号
        if 'factor' in method:
            long_signal_df = backtest_df.copy()
            long_signal_df.iloc[:, :] = np.where(std_backtest_df >= 0, std_backtest_df, 0)
            long_cost_df = np.abs(cost * (long_signal_df.shift(1, axis=1) - long_signal_df)) / 2
            short_signal_df = backtest_df.copy()
            short_signal_df.iloc[:, :] = np.where(std_backtest_df <= 0, -1 * std_backtest_df, 0)
            short_cost_df = np.abs(cost * (short_signal_df.shift(1, axis=1) - short_signal_df)) / 2
        else:
            long_signal_df = backtest_df.copy()
            long_signal_df.iloc[:, :] = np.where(uprank_df <= head, 1, 0)
            long_cost_df = np.abs(cost * (long_signal_df.shift(1, axis=1) - long_signal_df)) / 2
            short_signal_df = backtest_df.copy()
            short_signal_df.iloc[:, :] = np.where(downrank_df <= head, 1, 0)
            short_cost_df = np.abs(cost * (short_signal_df.shift(1, axis=1) - short_signal_df)) / 2

        # 生成换手序列
        weight_df = (long_signal_df / long_signal_df.sum(axis=0)).fillna(0).replace(np.inf, 0)
        turnover = np.abs(weight_df - weight_df.shift(1, axis=1)).sum(axis=0)
        turnover_series = turnover.fillna(0).replace(np.infty, 0)

        # 生成按因子值加权的多空pnl
        factor_neutral = (backtest_df - backtest_df.mean()) / backtest_df.std()
        factor_neutral = factor_neutral / (factor_neutral.abs().sum())
        factor_ret_no_cost = (factor_neutral * bfill_ret_df).sum(axis=0)

        # 生成分组信号
        group_signal_df_list = list(np.arange(group_num))
        stock_num_base = backtest_df.rank(axis=0).max() / group_num
        # print('股票分组个数：{}'.format(stock_num_base))
        for i in range(group_num):
            group_signal_df_list[i] = backtest_df.copy()
            group_signal_df_list[i].iloc[:, :] = np.where((uprank_df <= (i + 1) * stock_num_base) * (uprank_df > i * stock_num_base), 1, 0)

        # 生成多空头pnl
        long_ret_no_cost = (long_signal_df * bfill_ret_df).sum(axis=0) / long_signal_df.sum(axis=0)
        short_ret_no_cost = (short_signal_df * bfill_ret_df).sum(axis=0) / short_signal_df.sum(axis=0)
        long_ret_after_cost = long_ret_no_cost - long_cost_df.sum(axis=0) / long_signal_df.sum(axis=0)
        short_ret_after_cost = short_ret_no_cost - short_cost_df.sum(axis=0) / short_signal_df.sum(axis=0)
        # long_ret_no_cost = long_ret_no_cost.fillna(0)
        # long_ret_after_cost = long_ret_after_cost.fillna(0)
        # short_ret_no_cost = short_ret_no_cost.fillna(0)
        # short_ret_after_cost = short_ret_after_cost.fillna(0)

        # 判断Long/Short的方向
        if long_ret_no_cost.sum() < short_ret_no_cost.sum():
            long_ret_no_cost, short_ret_no_cost = short_ret_no_cost, long_ret_no_cost
            long_ret_after_cost, short_ret_after_cost = short_ret_after_cost, long_ret_after_cost
            long_signal_df, short_signal_df = short_signal_df, long_signal_df

        # 生成换手序列
        weight_df = long_signal_df / (long_signal_df.sum(axis=0)).fillna(0).replace(np.inf, 0)
        turnover = np.abs(weight_df - weight_df.shift(1, axis=1)).sum(axis=0)
        turnover_series = turnover.fillna(0).replace(np.infty, 0)

        # 生成指数
        if 'mean' in benchmark:
            index_ret = (theoretical_rtn_df * self.local_universe[theoretical_rtn_df.columns]).mean(axis=0)
        elif 'index' in benchmark:
            index_ret = self.index_data[price_type + 'Price'].shift(-2, axis=1) / self.index_data[price_type + 'Price'].shift(-1,
                                                                                                                    axis=1) - 1
            index_ret = index_ret.loc[index, date_list_in_use]
        elif 'weight' in benchmark:
            fm_mask = np.where(theoretical_rtn_df.isna(), np.nan, 1)
            fm_df = fm_mask * self.eod_data_dict['FloatMarketValue'][theoretical_rtn_df.columns]
            index_ret = (theoretical_rtn_df * self.local_universe[theoretical_rtn_df.columns] * fm_df).sum(axis=0) / fm_df.sum(axis=0)
        elif 'universe' in benchmark:
            index_ret = (theoretical_rtn_df * self.local_universe[theoretical_rtn_df.columns] * factor_mask[theoretical_rtn_df.columns]).mean(axis=0)
        
        else:
            index_ret = (theoretical_rtn_df * self.local_universe[theoretical_rtn_df.columns]).mean(axis=0)

        # 生成分组pnl
        group_ret_series_list_no_cost = list(np.arange(group_num))
        group_ret_series_list_after_cost = list(np.arange(group_num))
        group_ret_list_no_cost = list(np.arange(group_num))
        group_ret_list_after_cost = list(np.arange(group_num))
        group_cost_df_list = list(np.arange(group_num))
        group_tov_list = list(np.arange(group_num))
        for i in range(group_num):
            group_ret_series_list_no_cost[i] = (group_signal_df_list[i] * bfill_ret_df).sum(axis=0) / \
                                            group_signal_df_list[i].sum(axis=0)
            
            group_cost_df_list[i] = np.abs(
                cost * (group_signal_df_list[i].shift(1, axis=1) - group_signal_df_list[i])) / 2
            group_ret_series_list_after_cost[i] = group_ret_series_list_no_cost[i] - group_cost_df_list[i].sum(axis=0) / \
                                                group_signal_df_list[i].sum(axis=0)

            group_ret_list_no_cost[i] = group_ret_series_list_no_cost[i].cumsum().values[-1]
            group_ret_list_after_cost[i] = group_ret_series_list_after_cost[i].cumsum().values[-1]

            weight_df = group_signal_df_list[i] / (group_signal_df_list[i].sum(axis=0)).fillna(0).replace(np.inf, 0)
            turnover = np.abs(weight_df - weight_df.shift(1, axis=1)).sum(axis=0)
            group_tov_list[i] = turnover.fillna(0).replace(np.infty, 0)
        
        annual_coef = 252 / len(long_ret_after_cost)
        data_dict = {}
        data_dict['IC'] = ic
        data_dict['rankIC'] = rankic
        data_dict['GroupIC'] = np.corrcoef(group_num - np.arange(group_num), np.array(group_ret_list_after_cost))[0, 1]
        data_dict['GroupICNC'] = np.corrcoef(group_num - np.arange(group_num), np.array(group_ret_list_no_cost))[0, 1]
        data_dict['IR'] = ic_list.mean() / ic_list.std()
        data_dict['TurnOver'] = turnover_series.mean()
        data_dict['AlphaRet'] = (long_ret_after_cost - index_ret).cumsum().dropna().values[-1] * annual_coef
        data_dict['AlphaRetNC'] = (long_ret_no_cost - index_ret).cumsum().dropna().values[-1] * annual_coef
        data_dict['AlphaSharpe'] = (long_ret_after_cost - index_ret).mean() / (long_ret_after_cost - index_ret).std() * np.sqrt(252)
        data_dict['AlphaSharpeNC'] = (long_ret_no_cost - index_ret).mean() / (long_ret_no_cost - index_ret).std() * np.sqrt(252)
        data_dict['AlphaDrawdown'] = self.cal_maxdd((long_ret_after_cost - index_ret).cumsum().dropna().values)
        data_dict['AlphaDrawdownNC'] = self.cal_maxdd((long_ret_no_cost - index_ret).cumsum().dropna().values)
        data_dict['Score'] = data_dict['AlphaRetNC'] ** 2 * data_dict['AlphaSharpeNC'] / (data_dict['AlphaDrawdownNC'] * data_dict['TurnOver'])

        
        table = {}
        table["DDRatioNC"] = data_dict['AlphaDrawdownNC']
        table["RetNC"] = data_dict['AlphaRetNC'] 
        table['SharpeNC'] = data_dict['AlphaSharpeNC']
        table['Turnover'] = data_dict['TurnOver']
        table["GroupICNC"] = data_dict['GroupICNC']
        table["Ret"] = data_dict['AlphaRet'] 
        table = pd.DataFrame([table])
        # ============================ 画图区域 ============================ #
        if plot:
            fig = plt.figure(figsize=(20, 30), dpi=200)
            fig.suptitle('{}_{}_{}_{}_Performance'.format(self.name, benchmark, method, return_type.split('_')[0]))

            ax0 = fig.add_subplot(8,1,1)
            table_plot = ax0.table(cellText=np.round(table.values,4), rowLabels=['Stats'], colLabels=table.columns , loc='center', cellLoc='center',rowLoc='center')
            table_plot.auto_set_font_size(False)
            table_plot.set_fontsize(10) #字体大小
            table_plot.scale(1.1, 2.3)

            ax1 = fig.add_subplot(8, 2, 3)
            ax1.plot(list(long_ret_no_cost.index), list(long_ret_no_cost.cumsum().values), color='darkorange')
            ax1.plot(list(short_ret_no_cost.index), list(short_ret_no_cost.cumsum().values), color='limegreen')
            ax1.plot(list(index_ret.index), list(index_ret.cumsum().values), color='indianred')
            ax1.set_xticks(list(long_ret_no_cost.index)[::int(len(list(long_ret_no_cost.index)) / 6)])
            ax1.legend(['long', 'short', 'index'])
            ax1.grid(axis='y')
            ax1.set_title('Long Short Absolute No Cost Return')

            ax2 = fig.add_subplot(8, 2, 4)
            ax2.plot(list(long_ret_no_cost.index), list((long_ret_no_cost - index_ret).cumsum().values),
                    color='darkorange')
            ax2.plot(list(short_ret_no_cost.index), list((short_ret_no_cost - index_ret).cumsum().values),
                    color='limegreen')
            ax2.set_xticks(list(long_ret_no_cost.index)[::int(len(list(long_ret_no_cost.index)) / 6)])
            ax2.legend(['long', 'short'])
            ax2.grid(axis='y')
            ax2.set_title('Long Short Excess No Cost Return')

            ax3 = fig.add_subplot(8, 2, 5)
            ax3.plot(list(long_ret_after_cost.index), list(long_ret_after_cost.cumsum().values), color='darkorange')
            ax3.plot(list(short_ret_after_cost.index), list(short_ret_after_cost.cumsum().values), color='limegreen')
            ax3.plot(list(index_ret.index), list(index_ret.cumsum().values), color='indianred')
            ax3.set_xticks(list(long_ret_after_cost.index)[::int(len(list(long_ret_after_cost.index)) / 6)])
            ax3.legend(['long', 'short', 'index'])
            ax3.grid(axis='y')
            ax3.set_title('Long Short Absolute After Cost Return')

            ax4 = fig.add_subplot(8, 2, 6)
            ax4.plot(list(long_ret_after_cost.index), list((long_ret_after_cost - index_ret).cumsum().values),
                    color='darkorange')
            ax4.plot(list(short_ret_after_cost.index), list((short_ret_after_cost - index_ret).cumsum().values),
                    color='limegreen')
            ax4.set_xticks(list(long_ret_after_cost.index)[::int(len(list(long_ret_after_cost.index)) / 6)])
            ax4.legend(['long', 'short'])
            ax4.grid(axis='y')
            ax4.set_title('Long Short Excess After Cost Return')

            ax5 = fig.add_subplot(8, 2, 7)
            for i in range(group_num):
                ax5.plot(list(group_ret_series_list_no_cost[i].index),
                        list((group_ret_series_list_no_cost[i]).cumsum().values))
            ax5.set_xticks(list(group_ret_series_list_no_cost[0].index)[
                        ::int(len(list(group_ret_series_list_no_cost[0].index)) / 6)])
            ax5.legend(np.arange(group_num), fontsize=8)
            ax5.grid(b=True, axis='y')
            ax5.set_title('Group Absolute Return No Cost')

            ax6 = fig.add_subplot(8, 2, 8)
            total_ret = [np.nansum(ret) for ret in group_ret_series_list_no_cost]
            ax6.bar(range(len(total_ret)), total_ret)
            ax6.hlines(np.mean(total_ret), xmin=0, xmax=len(total_ret) - 1, color='r')
            ax6.set_xticks(range(group_num))
            ax6.set_title('Group Return No Cost Bar')

            ax7 = fig.add_subplot(8, 2, 9)
            for i in range(group_num):
                ax7.plot(list(group_ret_series_list_after_cost[i].index),
                        list((group_ret_series_list_after_cost[i]).cumsum().values))
            ax7.set_xticks(list(group_ret_series_list_after_cost[0].index)[
                        ::int(len(list(group_ret_series_list_after_cost[0].index)) / 6)])
            ax7.legend(np.arange(group_num), fontsize=8)
            ax7.grid(b=True, axis='y')
            ax7.set_title('Group Absolute Return After Cost')

            ax8 = fig.add_subplot(8, 2, 10)
            total_ret = [np.nansum(ret) for ret in group_ret_series_list_after_cost]
            ax8.bar(range(len(total_ret)), total_ret)
            ax8.hlines(np.mean(total_ret), xmin=0, xmax=len(total_ret) - 1, color='r')
            ax8.set_xticks(range(group_num))
            ax8.set_title('Group Return After Cost Bar')

            ax9 = fig.add_subplot(8, 2, 11)
            width = 0.4
            ax9.bar(np.arange(len(ic_decay_list)) - width / 2, ic_decay_list, width)
            ax9.bar(np.arange(len(rankic_decay_list)) + width / 2, rankic_decay_list, width)
            ax9.set_xticks(range(10))
            ax9.legend(labels=['IC', 'rankIC'])
            ax9.grid(b=True, axis='y')
            ax9.set_title('IC and rankIC decay')

            ax10 = fig.add_subplot(8, 2, 12)
            ax10.plot(list(ic_list.index), list(ic_list.cumsum().values))
            ax10.set_xticks(list(ic_list.index)[::int(len(list(ic_list.index)) / 6)])
            ax10.grid(b=True, axis='y')
            ax10.set_title('Cumulated IC_IR: {}'.format(round(ic_list.mean() / ic_list.std(), 3)))

            ax11 = fig.add_subplot(8,2,13)

            fac_df = backtest_df
            filled_nums = np.sum(~np.isnan(fac_df))
            all_nums = np.sum(~np.isnan(self.price_dict['Close'][date_list_in_use]))
            inf_nums = np.sum(abs(fac_df) == np.inf)
            unique_nums = fac_df.nunique()
            max_facs = fac_df.replace(np.inf, np.nan).max()
            min_facs = fac_df.replace(-np.inf, np.nan).min()
            filled_ext =  list(filled_nums[(abs(filled_nums.diff(1)) > 200) & (abs(filled_nums.diff(-1)) > 200)].index)
            inf_ext = list(inf_nums[inf_nums > 0].index)
            l11 = ax11.plot(list(all_nums.index), list(all_nums), label='all stocks')
            l12 = ax11.plot(list(filled_nums.index), list(filled_nums), label='filled stocks')
            l13 = ax11.plot(list(unique_nums.index), list(unique_nums), label='unique values')
            ax11.set_xticks(list(filled_nums.index)[::int(len(list(filled_nums.index))/6)])
            ax11.set_ylabel('stock nums')
            ax11.grid()
            ax12 = ax11.twinx()
            ax12.bar(range(len(inf_nums)), inf_nums, width=1, color='magenta', label='inf stocks')
            ax12.set_ylabel('inf nums')
            l1 = l11 + l12 + l13
            labs1 = [l.get_label() for l in l1]
            ax12.legend(l1, labs1, loc='lower right')
            ax11.set_title('all/filled/inf nums')

            ax13 = fig.add_subplot(8,2,14)

            l21 = ax13.plot(list(max_facs.index), list(max_facs), color='magenta', label='max factor values')
            ax13.set_ylabel('factor max values')
            ax14 = ax13.twinx()
            l22 = ax14.plot(list(min_facs.index), list(min_facs), color='deepskyblue', label='min factor values')
            ax14.set_ylabel('factor min values')
            ax13.set_xticks(list(max_facs.index)[::int(len(list(max_facs.index))/6)])
            ax13.grid()
            l2 = l21 + l22
            labs2 = [l.get_label() for l in l2]
            ax13.legend(l2, labs2)
            ax13.set_title('daily max/min values')

            ax15 = fig.add_subplot(8,1,8)
            fac_temp = fac_df.replace(np.inf, np.nan)
            fac_temp = fac_temp.replace(-np.inf, np.nan)
            sns.violinplot(data=fac_temp.iloc[:, ::int(len(fac_temp.T) / 30)])
            ax15.set_title('factor value distribution')
            plt.xticks(rotation=45)
            if path is not None:
                 plt.savefig(os.path.join(path, f'{self.name}.png'), dpi=224, bbox_inches='tight')
            plt.show()

            

            # # 最近一年数据统计
            # recent_annual_data_dict = {}
            # recent_annual_data_dict['IC'] = ic_list[-252:].mean()
            # recent_annual_data_dict['rankIC'] = rankic_list[-252:].mean()
            # recent_annual_data_dict['IR'] = ic_list[-252:].mean() / ic_list[-252:].std()
            # recent_annual_data_dict['TurnOver'] = turnover_series.iloc[-252:].mean()
            # recent_annual_data_dict['AlphaRet'] = (long_ret_after_cost.iloc[-252:] - index_ret.iloc[-252:]).cumsum().dropna().values[-1]
            # recent_annual_data_dict['AlphaRetNC'] = (long_ret_no_cost.iloc[-252:] - index_ret.iloc[-252:]).cumsum().dropna().values[-1]
            # recent_annual_data_dict['AlphaSharpe'] = (long_ret_after_cost.iloc[-252:] - index_ret.iloc[-252:]).mean() / (long_ret_after_cost.iloc[-252:] - index_ret.iloc[-252:]).std() * np.sqrt(252)
            # recent_annual_data_dict['AlphaSharpeNC'] = (long_ret_no_cost.iloc[-252:] - index_ret.iloc[-252:]).mean() / (long_ret_no_cost.iloc[-252:] - index_ret.iloc[-252:]).std() * np.sqrt(252)
            # recent_annual_data_dict['AlphaDrawdown'] = self.cal_maxdd((long_ret_after_cost.iloc[-252:] - index_ret.iloc[-252:]).cumsum().dropna().values)
            # recent_annual_data_dict['AlphaDrawdownNC'] = self.cal_maxdd((long_ret_no_cost.iloc[-252:] - index_ret.iloc[-252:]).cumsum().dropna().values)
            # recent_annual_data_dict['DrawdownRatio'] = recent_annual_data_dict['AlphaDrawdownNC'] / recent_annual_data_dict['AlphaRetNC']
            # recent_annual_data_dict['Score'] = recent_annual_data_dict['AlphaRetNC'] ** 2 * recent_annual_data_dict['AlphaSharpeNC'] / (recent_annual_data_dict['AlphaDrawdownNC'] * recent_annual_data_dict['TurnOver'])

            # # 分组收益率数据统计
            # group_stat_df = pd.DataFrame(index=np.arange(group_num), columns=list(data_dict.keys())[5:], dtype='float')
            # for num in range(group_num):
            #     group_stat_df.loc[num, 'TurnOver'] = group_tov_list[num].mean()
            #     group_stat_df.loc[num, 'AlphaRet'] = (group_ret_series_list_after_cost[num] - index_ret).cumsum().dropna().values[-1] * annual_coef
            #     group_stat_df.loc[num, 'AlphaRetNC'] = (group_ret_series_list_no_cost[num] - index_ret).cumsum().dropna().values[-1] * annual_coef
            #     group_stat_df.loc[num, 'AlphaSharpe'] = (group_ret_series_list_after_cost[num] - index_ret).mean() / (group_ret_series_list_after_cost[num] - index_ret).std() * np.sqrt(len(group_ret_series_list_after_cost[num]))
            #     group_stat_df.loc[num, 'AlphaSharpeNC'] = (group_ret_series_list_no_cost[num] - index_ret).mean() / (group_ret_series_list_no_cost[num] - index_ret).std() * np.sqrt(len(group_ret_series_list_no_cost[num]))
            #     group_stat_df.loc[num, 'AlphaDrawdown'] = self.cal_maxdd((group_ret_series_list_after_cost[num] - index_ret).cumsum().dropna().values)
            #     group_stat_df.loc[num, 'AlphaDrawdownNC'] = self.cal_maxdd((group_ret_series_list_no_cost[num] - index_ret).cumsum().dropna().values)

            # if risk_plot:
            #     # rp = risk_plotter(benchmark_index=index)
            #     # rp.plot(fig_name='style_analysis',
            #     #         long_signal_df=long_signal_df,
            #     #         is_long=True)
            #     pass

            # return pd.DataFrame([data_dict]).T, pd.DataFrame([recent_annual_data_dict]).T, (long_ret_no_cost - index_ret, long_ret_after_cost - index_ret), group_stat_df.T,theoretical_rtn_df
        # except:
        #     return None, None, None, None, None
        
        
    @staticmethod
    def cal_maxdd(array):
        drawdowns = []
        max_so_far = array[0]
        for i in range(len(array)):
            if array[i] > max_so_far:
                drawdown = 0
                drawdowns.append(drawdown)
                max_so_far = array[i]
            else:
                drawdown = max_so_far - array[i]
                drawdowns.append(drawdown)
        return max(drawdowns)


    @staticmethod
    def ic_decay_i(arg):
        i,array1,array2 = arg
        ic_decay = array1.corrwith(array2.shift(-i, axis=1)).mean()
        rank_ic_decay = array1.corrwith(array2.shift(-i, axis=1), method='spearman').mean()
        return (i,ic_decay,rank_ic_decay)


    def cal_corr(self, test_df):
        '''
        Input:
        test_df, pd.DataFrame, 被测试的因子

        Output:
        corr_df.T, pd.DataFrame, 与因子库中的相关系数
        '''
        # 获取因子库中因子列表
        file_list = [i[:-3] for i in os.listdir('/data/shared/low_fre_alpha/factor_base_v2/eod_feature/')]

        # 读取因子库中因子
        self.factor_dict = self.ds.get_eod_feature(fields=file_list,
                                                   where='/data/shared/low_fre_alpha/factor_base_v2',
                                                   tickers=self.ds.get_ticker_list(date='all'),
                                                   dates=self.ds.get_trade_dates(start_date='20160101', end_date='20200630'))

        corr_dict = {}
        for file in file_list:
            factor_df = self.factor_dict[file].to_dataframe()
            ic_list = factor_df.corrwith(test_df)
            corr_dict[file] = np.nanmean(ic_list)
        corr_df = pd.DataFrame.from_dict(corr_dict, orient='index')
        corr_df.columns = ['corr']

        return corr_df.T


    def factor_analysis(self, new_df, threshold):
        '''
        Input:
        new_df, pd.DataFrame, 被测试的因子
        threshold, float, 相关系数的阈值

        Output:
        None
        有print的Output
        '''
        # 计时器
        start_time = time.time()
        with open('/home/yhzhou/simplifed_alphatest/factor_name_dict_v2.pkl', 'rb') as f:
            self.fac_name_dict = pickle.load(f)

        # 计算相关性
        result_df = self.cal_corr(test_df=new_df)

        # 输出预先结果
        print('\n================================ Testing Corr ================================')

        # 输出有问题因子
        bad_fac_list = result_df[result_df.abs() > threshold].dropna(axis=1).columns.to_list()
        if len(bad_fac_list) == 0:
            print('All factors in factor base are lowly correlated with the new factor.')
        else:
            for fac in bad_fac_list:
                print(
                    f'FACTOR{fac[8:]} {self.fac_name_dict[fac[4:]]} IS HIGHLY CORRELATED, correlation is {round(result_df[fac].values[0], 4)}.')

        # 画相关系数图
        plt.figure(figsize=(12, 3), dpi=100)
        xticks = [i[8:] for i in list(result_df.columns)]
        plt.bar(x=xticks, height=list(result_df.values[0]), color='grey')
        plt.plot(xticks, [threshold] * len(xticks), linestyle='--', color='red')
        plt.plot(xticks, [-threshold] * len(xticks), linestyle='--', color='red')
        plt.xticks(xticks[::5])
        plt.xlabel('fac_num')
        plt.ylabel('correlation')
        plt.title('corr with fac base')
        plt.grid(b=True, axis='y')
        plt.show()

        # 输出时间
        print('================================Used Time: {}s================================\n'.format(round(time.time() - start_time), 3))
