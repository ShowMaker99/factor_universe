"""
因子生成设置
"""
# load packages
import getpass

# initialize dataserver
from PqiDataSdk import *
USER = getpass.getuser()
ds = PqiDataSdk(user=USER, size=1, pool_type='mp')

# 设置因子存取路径
my_support_factor_paths = {
    'mins': '/home/shared/Data/data/shared/low_fre_alpha/ap_monster_zoo/',
    'depth': '',
    'mins_mf': '/home/shared/Data/data/shared/low_fre_alpha/ap_monster_zoo/'
}
my_factor_paths = {
    'mins': '',
    'depth': '',
    'mins_mf': ''
}

# 设置因子生成时间
start_date = '20180101'
end_date = '20210831'
tickers = ds.get_ticker_list(date='all')
dates = ds.get_trade_calendar(start_date=start_date, end_date=end_date)

default_cuts = [
                # 'day',
                # 'hh_1', 'hh_2', 'hh_3', 'hh_4', 'hh_5', 'hh_6', 'hh_7', 'hh_8',
                # ('actBuyValTotRto', [0.05,0.1,0.2,0.5]), ('actSellValTotRto', [0.05,0.1,0.2,0.5]),
                # ('actBuyValTotRtoS', [0.05,0.1,0.2,0.5]), ('actBuyValTotRtoM', [0.05,0.1,0.2,0.5]), ('actBuyValTotRtoL', [0.05,0.1,0.2,0.5]), ('actBuyValTotRtoXL', [0.05,0.1,0.2,0.5]),
                # ('actSellValTotRtoS', [0.05,0.1,0.2,0.5]), ('actSellValTotRtoM', [0.05,0.1,0.2,0.5]), ('actSellValTotRtoL', [0.05,0.1,0.2,0.5]), ('actSellValTotRtoXL', [0.05,0.1,0.2,0.5]),
                # ('actTrdValRtoS', [0.2,0.5]), ('actTrdValRtoM', [0.2,0.5]), ('actTrdValRtoL', [0.2,0.5]), ('actTrdValRtoXL', [0.2,0.5]),
                ('totTrdValRtoS', [0.2,0.5]), ('totTrdValRtoM', [0.2,0.5]), ('totTrdValRtoL', [0.2,0.5]), ('totTrdValRtoXL', [0.2,0.5]),
                # ('actTrdVolRtoS', [0.2,0.5]), ('actTrdVolRtoM', [0.2,0.5]), ('actTrdVolRtoL', [0.2,0.5]), ('actTrdVolRtoXL', [0.2,0.5]),
                # ('totTrdVolRtoS', [0.2,0.5]), ('totTrdVolRtoM', [0.2,0.5]), ('totTrdVolRtoL', [0.2,0.5]), ('totTrdVolRtoXL', [0.2,0.5]),
                # ('actBuyValLclRtoS', [0.05,0.1,0.2,0.5]), ('actBuyValLclRtoM', [0.05,0.1,0.2,0.5]), ('actBuyValLclRtoL', [0.05,0.1,0.2,0.5]), ('actBuyValLclRtoXL', [0.05,0.1,0.2,0.5]),
                # ('actBuyValRtoS', [0.05,0.1,0.2,0.5]), ('actBuyValRtoM', [0.05,0.1,0.2,0.5]), ('actBuyValRtoL', [0.05,0.1,0.2,0.5]), ('actBuyValRtoXL', [0.05,0.1,0.2,0.5]),
                # ('actSellValRtoS', [0.05,0.1,0.2,0.5]), ('actSellValRtoM', [0.05,0.1,0.2,0.5]), ('actSellValRtoL', [0.05,0.1,0.2,0.5]), ('actSellValRtoXL', [0.05,0.1,0.2,0.5])
                ]
default_aggs = ['mean']

support_factor_names = dict()  # 序列
support_factor_cuts = dict()  # 切割
support_factor_aggs = dict()  # 低频化算符
data_range_dict = dict()  # 切割参数

# str or tuple
support_factor_names['mins'] = ['rtn']
support_factor_names['depth'] = []
support_factor_names['mins_mf'] = [
                                    'rtn',
                                    # 'actBuyValTotRto','actSellValTotRto',
                                    # 'actBuyValTotRtoS', 'actBuyValTotRtoM', 'actBuyValTotRtoL', 'actBuyValTotRtoXL',
                                    # 'actSellValTotRtoS', 'actSellValTotRtoM', 'actSellValTotRtoL', 'actSellValTotRtoXL',
                                    # 'actTrdValRtoS', 'actTrdValRtoM', 'actTrdValRtoL', 'actTrdValRtoXL',
                                    # 'totTrdValRtoS', 'totTrdValRtoM', 'totTrdValRtoL', 'totTrdValRtoXL',
                                    # 'actTrdVolRtoS', 'actTrdVolRtoM', 'actTrdVolRtoL', 'actTrdVolRtoXL',
                                    # 'totTrdVolRtoS', 'totTrdVolRtoM', 'totTrdVolRtoL', 'totTrdVolRtoXL',
                                    # 'actBuyValLclRtoS', 'actBuyValLclRtoM', 'actBuyValLclRtoL', 'actBuyValLclRtoXL',
                                    # 'actBuyValRtoS', 'actBuyValRtoM', 'actBuyValRtoL', 'actBuyValRtoXL',
                                    # 'actSellValRtoS', 'actSellValRtoM', 'actSellValRtoL', 'actSellValRtoXL'
                                    ]

support_factor_cuts['mins'] = {factor: default_cuts for factor in support_factor_names['mins']}
support_factor_cuts['depth'] = {factor: default_cuts for factor in support_factor_names['depth']}
support_factor_cuts['mins_mf'] = {factor: default_cuts for factor in support_factor_names['mins_mf']}
support_factor_aggs['mins'] = {factor: default_aggs for factor in support_factor_names['mins'] if isinstance(factor, str)}
support_factor_aggs['depth'] = {factor: default_aggs for factor in support_factor_names['depth'] if isinstance(factor, str)}
support_factor_aggs['mins_mf'] = {factor: default_aggs for factor in support_factor_names['mins_mf'] if isinstance(factor, str)}

data_range_dict['mins'] = {
    'hh_1': (0,29),
    'hh_2': (30,59),
    'hh_3': (60,89),
    'hh_4': (90,119),
    'hh_5': (120,149),
    'hh_6': (150,179),
    'hh_7': (180,209),
    'hh_8': (210,236)
}    
data_range_dict['depth'] = {}
data_range_dict['mins_mf']= {
    # 'day': (0,236),
    # 'hh_1': (0,29),
    # 'hh_2': (30,59),
    # 'hh_3': (60,89),
    # 'hh_4': (90,119),
    # 'hh_5': (120,149),
    # 'hh_6': (150,179),
    # 'hh_7': (180,209),
    # 'hh_8': (210,236),
    # ('actBuyValTotRto', (0.05,0.1,0.2,0.5)): None,
    # ('actSellValTotRto', (0.05,0.1,0.2,0.5)): None,
    # ('actBuyValTotRtoS', (0.05,0.1,0.2,0.5)): None,
    # ('actBuyValTotRtoM', (0.05,0.1,0.2,0.5)): None,
    # ('actBuyValTotRtoL', (0.05,0.1,0.2,0.5)): None,
    # ('actBuyValTotRtoXL', (0.05,0.1,0.2,0.5)): None,
    # ('actSellValTotRtoS', (0.05,0.1,0.2,0.5)): None, 
    # ('actSellValTotRtoM', (0.05,0.1,0.2,0.5)): None,
    # ('actSellValTotRtoL', (0.05,0.1,0.2,0.5)): None, 
    # ('actSellValTotRtoXL', (0.05,0.1,0.2,0.5)): None,
    # ('actTrdValRtoS', (0.2,0.5)): None,
    # ('actTrdValRtoM', (0.2,0.5)): None,
    # ('actTrdValRtoL', (0.2,0.5)): None,
    # ('actTrdValRtoXL', (0.2,0.5)): None,
    ('totTrdValRtoS', (0.2,0.5)): None,
    ('totTrdValRtoM', (0.2,0.5)): None,
    ('totTrdValRtoL', (0.2,0.5)): None,
    ('totTrdValRtoXL', (0.2,0.5)): None,
    # ('actTrdVolRtoS', (0.2,0.5)): None,
    # ('actTrdVolRtoM', (0.2,0.5)): None,
    # ('actTrdVolRtoL', (0.2,0.5)): None,
    # ('actTrdVolRtoXL', (0.2,0.5)): None,
    # ('totTrdVolRtoS', (0.2,0.5)): None,
    # ('totTrdVolRtoM', (0.2,0.5)): None,
    # ('totTrdVolRtoL', (0.2,0.5)): None,
    # ('totTrdVolRtoXL', (0.2,0.5)): None, 
    # ('actBuyValLclRtoS', (0.05,0.1,0.2,0.5)): None,
    # ('actBuyValLclRtoM', (0.05,0.1,0.2,0.5)): None,
    # ('actBuyValLclRtoL', (0.05,0.1,0.2,0.5)): None,
    # ('actBuyValLclRtoXL', (0.05,0.1,0.2,0.5)): None,
    # ('actBuyValRtoS', (0.05,0.1,0.2,0.5)): None,
    # ('actBuyValRtoM', (0.05,0.1,0.2,0.5)): None,
    # ('actBuyValRtoL', (0.05,0.1,0.2,0.5)): None,
    # ('actBuyValRtoXL', (0.05,0.1,0.2,0.5)): None,
    # ('actSellValRtoS', (0.05,0.1,0.2,0.5)): None,
    # ('actSellValRtoM', (0.05,0.1,0.2,0.5)): None,
    # ('actSellValRtoL', (0.05,0.1,0.2,0.5)): None,
    # ('actSellValRtoXL', (0.05,0.1,0.2,0.5)): None
}  
# eod算符
eod_factor_aggs = []
