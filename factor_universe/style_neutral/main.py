# Load packages
import numpy as np
import sys
sys.path.extend(['./', '../'])
from neu_toolkit import NeuKit
from tools.tools import *
import neu_config as nc

# Connect data server
from PqiDataSdk import * 
import getpass
USER = getpass.getuser()
ds = PqiDataSdk(user=USER, size=1, pool_type='mp')



class StyleNeutral():
    
    
    def __init__(self):
        
        self.name = nc.name
        self.start_date = nc.start_date
        self.end_date = nc.end_date
        self.path = nc.path
        self.save_path = nc.save_path
        self.style = nc.style
        
        self.tickers = get_ticker_list()
        self.dates = get_trade_dates(self.start_date, self.end_date)
        
        self.factor = read_eod_data(self.name, self.tickers, self.dates, self.path).rolling(20, axis=1, min_periods=10).mean() # min fators
        # self.factor = read_eod_data(self.name, self.tickers, self.dates, self.path) # day or hh factors
        self.neu_kit = NeuKit(self.start_date, self.end_date)
        self.ret_df = self.neu_kit.eod_data_dict['TwapOpen60'].shift(-2, axis=1) / self.neu_kit.eod_data_dict['TwapOpen60'].shift(-1, axis=1) - 1 # TwapOpen2TwapOpen计算的简单收益率
    
        self.universe = ds.get_file('universe', tickers=self.tickers, start_date=self.start_date, end_date=self.end_date, format='ticker_date_real')
        self.up_feasible_stock = ds.get_file('up_feasible', tickers=self.tickers, start_date=self.start_date, end_date=self.end_date, format='ticker_date_real')
        self.down_feasible_stock = ds.get_file('down_feasible', tickers=self.tickers, start_date=self.start_date, end_date=self.end_date, format='ticker_date_real')
        
    
    def deal_nan(self, series):
        if series.count()==0:
            series.iloc[0] = 0
        return series


    def judge_pos(self, ret_series):
        return ret_series * np.sign(ret_series.sum())


    def cal_LS_SharpeNC(self, factor_df):
        "按因子值分多空，计算多空费前夏普"
        posneg_factor_df = self.neu_kit.origin_process(factor_df)
        posneg_ret_df = self.judge_pos((posneg_factor_df * (self.ret_df * self.up_feasible_stock).loc[:,self.dates]).sum())
        posneg_ret_df = posneg_ret_df.loc[self.dates]
        sharpe = posneg_ret_df.mean() / posneg_ret_df.std() * np.sqrt(245)
        em_sharpe = posneg_ret_df.ewm(span=240).mean().values[-1] / posneg_ret_df.ewm(span=240).std().values[-1] * np.sqrt(245)
        return sharpe, em_sharpe


    def choose_neut(self, factor_df, start_date, end_date):
        "因子中性化"
        fac = factor_df.loc[:, start_date:end_date]
        fac = fac.apply(self.deal_nan)
        
        style_list = self.style
        # style_list = ["residual_volatility","momentum","beta","liquidity","size", "non_linear_size"]
        for style in style_list:
            fac_neu = self.neu_kit.style_neu(factor_df=fac.copy(), neu_list=[style])
            if self.cal_LS_SharpeNC(fac_neu)[0] <= self.cal_LS_SharpeNC(fac)[0] + 0.25: # 如果风格中性化之后，费前夏普下降了超过0.25
                style_list.remove(style)    # 则不剔除该风格因子
        if len(style_list) > 0:
            fac = self.neu_kit.style_neu(factor_df=fac.copy(), neu_list=style_list)
        return fac, style_list

    
    def run(self):
        factor, style = self.choose_neut(self.factor, self.start_date, self.end_date)
        name_neu = self.name + '_'.join(['_rolling20mean', 'neu']+style)
        save_eod_feature(name_neu, factor, self.save_path)


if __name__ == "__main__":
    sn = StyleNeutral()
    sn.run()    
    