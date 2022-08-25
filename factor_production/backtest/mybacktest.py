import pandas as pd
import numpy as np
import time
import getpass
import warnings
warnings.filterwarnings("ignore")

from PqiDataSdk import *
user = getpass.getuser()


class BackTestMachine:
    def __init__(self, start_date, end_date, name=None):
        self.ds = PqiDataSdk(user=user, size=1, pool_type="mp", log=False, offline=True)
        self.start_date = start_date
        self.end_date = end_date
        self.date_list = self.ds.get_trade_dates(start_date=self.start_date, end_date=self.end_date)
        self.extend_end_date = self.ds.get_next_trade_date(self.ds.get_next_trade_date(end_date))
        self.name = name

    def DataPrepare(self):
        stock_pool = self.ds.get_ticker_list(date='all')
        adj_factor = self.ds.get_eod_history(tickers=stock_pool, start_date=self.start_date, end_date=self.extend_end_date, fields=['AdjFactor'])['AdjFactor']
        twap_open_price = self.ds.get_eod_history(tickers=stock_pool, start_date=self.start_date, end_date=self.extend_end_date, source="ext_stock", fields=['TwapBegin30'])['TwapBegin30']

        self.local_universe = self.ds.get_file('universe', tickers=stock_pool,
                                               start_date=self.start_date,
                                               end_date=self.end_date,
                                               format='ticker_date_real')

        self.local_up_feasible_stock = self.ds.get_file('up_feasible', tickers=stock_pool,
                                                        start_date=self.start_date,
                                                        end_date=self.end_date,
                                                        format='ticker_date_real')

        price_df = twap_open_price * adj_factor
        ret_df = np.log(price_df.shift(-2, axis=1) / price_df.shift(-1, axis=1))
        self.ret_df = ret_df.iloc[:, :-2]

    def update_factor_df(self, factor_df, name=None):
        factor_df = factor_df.iloc[:, (factor_df.columns >= self.start_date) &
                                      (factor_df.columns <= self.end_date)]
        quantiles = factor_df.where((factor_df != -np.inf) & (factor_df != np.inf)).quantile([0.01, 0.99])
        factor_df = pd.DataFrame(np.where(factor_df < quantiles.loc[0.01], quantiles.loc[0.01], factor_df),
                                 index=factor_df.index, columns=factor_df.columns)
        factor_df = pd.DataFrame(np.where(factor_df > quantiles.loc[0.99], quantiles.loc[0.99], factor_df),
                                 index=factor_df.index, columns=factor_df.columns)
        self.factor_df = factor_df
        self.name = name

    @ staticmethod
    def ir(ser):
        ser_mean = ser.mean()
        ser_std = ser.std() if ser.std() else np.nan
        return ser_mean / ser_std * np.sqrt(252)

    def backtest(self):
        ret_dict = {}

        # 生成因子矩阵
        backtest_df = self.factor_df * self.local_universe * self.local_up_feasible_stock
        demean_backtest_df = backtest_df - backtest_df.mean()
        std_backtest_df = demean_backtest_df / (demean_backtest_df.abs().sum() / 2)

        # 算ic和rankic
        ic_list = backtest_df.corrwith(self.ret_df)
        ret_dict['ic'] = ic_list.mean()
        rankic_list = backtest_df.corrwith(self.ret_df, method='spearman')
        ret_dict['rankic'] = rankic_list.mean()

        # 生成按因子值加权的多空pnl
        factor_neutral = (backtest_df - backtest_df.mean()) / backtest_df.std()
        factor_neutral = factor_neutral / (factor_neutral.abs().sum())
        factor_ret_no_cost = (factor_neutral * self.ret_df).sum(axis=0)

        # 回撤
        factor_cum_ret_no_cost = factor_ret_no_cost.cumsum()
        if factor_cum_ret_no_cost.iloc[-1] > 0:
            ret_dict['drawdown'] = (factor_cum_ret_no_cost.cummax() - factor_cum_ret_no_cost).max()
        else:
            ret_dict['drawdown'] = (factor_cum_ret_no_cost - factor_cum_ret_no_cost.cummin()).max()

        # Sharpe
        ret_dict['sharpe'] = self.ir(factor_ret_no_cost)

        long_signal_df = backtest_df.copy()
        long_signal_df.iloc[:, :] = np.where(std_backtest_df >= 0, std_backtest_df, 0)
        short_signal_df = backtest_df.copy()
        short_signal_df.iloc[:, :] = np.where(std_backtest_df <= 0, -1 * std_backtest_df, 0)
        long_ret_no_cost = (long_signal_df * self.ret_df).sum(axis=0) / long_signal_df.sum(axis=0)
        short_ret_no_cost = (short_signal_df * self.ret_df).sum(axis=0) / short_signal_df.sum(axis=0)
        index_ret = (self.ret_df * self.local_universe).mean(axis=0)
        long_excess_ret_no_cost = long_ret_no_cost - index_ret
        short_excess_ret_no_cost = short_ret_no_cost - index_ret

        ret_dict['long_sharpe'] = self.ir(long_excess_ret_no_cost)
        ret_dict['short_sharpe'] = self.ir(short_excess_ret_no_cost)

        step = long_excess_ret_no_cost.shape[0] // 4
        ret_dict['no_cost_long_sharpe1'] = self.ir(long_excess_ret_no_cost[:step])
        ret_dict['no_cost_short_sharpe1'] = self.ir(short_excess_ret_no_cost[:step])
        ret_dict['no_cost_long_sharpe2'] = self.ir(long_excess_ret_no_cost[step:2*step])
        ret_dict['no_cost_short_sharpe2'] = self.ir(short_excess_ret_no_cost[step:2*step])
        ret_dict['no_cost_long_sharpe3'] = self.ir(long_excess_ret_no_cost[2*step:3*step])
        ret_dict['no_cost_short_sharpe3'] = self.ir(short_excess_ret_no_cost[2*step:3*step])
        ret_dict['no_cost_long_sharpe4'] = self.ir(long_excess_ret_no_cost[3*step:])
        ret_dict['no_cost_short_sharpe4'] = self.ir(short_excess_ret_no_cost[3*step:])

        return ret_dict
