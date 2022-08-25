"""
Ryan Patrick
v2 factor universe
"""
# 外部包
import os, sys
import numpy as np
import pandas as pd
import multiprocessing as mp
import warnings
import time
from datetime import datetime
import pickle
sys.path.append('../')

# 配置参数
import fg_config as fgc
import merge.merge_config as mc

# 工具包
from ops.operation import *
from tools.tools import *
warnings.filterwarnings('ignore')

# 控制进程数
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# 连接data server
from PqiDataSdk import * 
import getpass
USER = getpass.getuser()
ds = PqiDataSdk(user=USER, size=1, pool_type='mp')


class FactorGenerator:
    
    
    def __init__(self):
        """
        初始化（转移设置）
        """
        # 因子切割参数
        self.objFactor = fgc.objFactor
        self.cutFactor = fgc.cutFactor
        self.rollWindow = fgc.rollWindow
        self.cutQuantile = fgc.cutQuantile
        self.cutAgg = fgc.cutAgg
        self.agg_mapper = fgc.agg_mapper
        
        # 配置参数
        self.num_processor = mc.processor_num
        self.start_date = mc.start_date
        self.end_date = mc.end_date
        self.date_list = get_trade_dates(self.start_date, self.end_date)
        self.stock_pool = get_ticker_list()
        self.path = mc.path
        self.save_path = mc.save_path
        self.freq = mc.freq
        

    def process_each_comb(self, param_dict):
        
        obj, cut, rollWindow, cutQtls, cutAgg = param_dict.values()
        
        # TODO 这里的mean要改
        obj_sp_name = '_'.join(['mins_mf' , obj , self.freq, 'mean']) if self.freq == 'day' else 'f"' + '_'.join(['mins_mf' , obj , "hh_{n}", 'mean']) + '"'
        cut_sp_name = '_'.join(['mins_mf' , cut , self.freq, 'mean']) if self.freq == 'day' else 'f"' + '_'.join(['mins_mf' , cut , "hh_{n}", 'mean']) + '"'
        obj_df = eval(f'get_feature_ser_{self.freq}')(obj_sp_name, self.stock_pool, self.date_list, self.path)
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 日间动量做obj
        # eod_history = ds.get_eod_history(tickers=self.stock_pool, start_date=self.start_date, end_date=self.end_date, fields=['ClosePrice'], price_mode='AFTER', source='stock')
        # # Compute factor
        # closePrice = eod_history['ClosePrice']
        # obj_df = -(closePrice/closePrice.shift(1, axis=1) - 1)
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
        tool_df = eval(f'get_feature_ser_{self.freq}')(cut_sp_name, self.stock_pool, self.date_list, self.path)
        # paras = list(product(rollWindow, cutQtl, cutAgg, ['top', 'bot'])) # 参数组合列表
        cutQtlLR = []
        for cutQtl in cutQtls:
            cutQtlLR.extend([(cutQtlL, cutQtlR) for cutQtlL, cutQtlR in zip(np.round(np.arange(0,1,cutQtl),1), np.round(np.arange(0,1.00001,cutQtl)[1:],1))])
        paras = list(product(rollWindow, cutAgg, cutQtlLR)) # 参数组合列表

        for para in paras:
            
            w, agg, (cutQtlL, cutQtlR) = para
            name = '_'.join([str(i) for i in [obj, cut, 'roll', (str(w)+self.freq), 'top', cutQtlL, 'to', cutQtlR, self.agg_mapper[agg]]])
            
            arr = np.dstack([tool_df, obj_df]).reshape(obj_df.shape[0], -1) # 将tool和obj沿第三维度拼接，返回tool和obj的矩阵元素值两两相邻的新矩阵
            res = np.apply_along_axis(func1d=lambda x: roll_cut_agg(x, w*2, cutQtlL, cutQtlR, step=2, in_group=agg), axis=1, arr=arr) # 对每只票的tool和obj合并行 应用roll_cut_agg函数
            
            factor_df = pd.DataFrame(index=obj_df.index, columns=obj_df.columns, dtype='float') # 创建空因子df
            factor_df.iloc[:, int(w) - 1:] = res # 将numba加速的结果赋给factor_df对应的行列
            factor_df = factor_df * 0 + factor_df # TODO 这行的作用是啥？？？
            if self.freq == 'hh':
                factor_df = factor_df.iloc[:, 7::8] # 选取"_hh_8"列(即只在每天尾盘半小时的时候回看生成因子)
                factor_df.columns = [date[:-5] for date in factor_df.columns] # 删掉"_hh_8"的后缀    
            else:
                pass
            save_eod_feature(name, factor_df, self.save_path)


    def run(self):

        # 分组multiprocessing
        obj_cut_combinations = [[obj, cut] for obj in self.objFactor for cut in self.cutFactor]
        
        print(f'Generation start at {datetime.now().strftime("%H:%M:%S")}------------------------')
        cur_time = time.time()

        with mp.Pool(processes=self.num_processor) as pool:
            pool.map(self.process_each_comb, 
                     [dict(obj=obj_cut[0], cut=obj_cut[1], rollWindow = self.rollWindow, cutQuantile=self.cutQuantile, cutAgg=self.cutAgg) for obj_cut in obj_cut_combinations])

        print(f'Generation finished in '
              f'{round(time.time() - cur_time, 2)}s at {datetime.now().strftime("%H:%M:%S")}')


if __name__ == '__main__':
    fg = FactorGenerator()
    fg.run()
    

    
