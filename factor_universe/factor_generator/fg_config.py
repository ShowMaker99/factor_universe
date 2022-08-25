"""
因子切割参数
"""

import sys
sys.path.append('../')
from ops.operation import mean


# 被切割的原始因子
objFactor = [
    'rtn'
]

# 用来切割的工具因子
cutFactor = [
    # 'actBuyValTotRto','actSellValTotRto',
    # 'actBuyValTotRtoS', 'actBuyValTotRtoM', 'actBuyValTotRtoL', 'actBuyValTotRtoXL',
    # 'actSellValTotRtoS', 'actSellValTotRtoM', 'actSellValTotRtoL', 'actSellValTotRtoXL',
    # 'actTrdValRtoS', 'actTrdValRtoM', 'actTrdValRtoL', 'actTrdValRtoXL',
    # 'totTrdValRtoS', 'totTrdValRtoM', 'totTrdValRtoL', 'totTrdValRtoXL',
    'actTrdVolRtoS', 
    'actTrdVolRtoM', 'actTrdVolRtoL', 'actTrdVolRtoXL',
    'totTrdVolRtoS', 'totTrdVolRtoM', 'totTrdVolRtoL', 'totTrdVolRtoXL',
    # 'actBuyValLclRtoS', 'actBuyValLclRtoM', 'actBuyValLclRtoL', 'actBuyValLclRtoXL',
    # 'actBuyValRtoS', 'actBuyValRtoM', 'actBuyValRtoL', 'actBuyValRtoXL',
    # 'actSellValRtoS', 'actSellValRtoM', 'actSellValRtoL', 'actSellValRtoXL'
]

# 回看时长(单位： merge_config.freq)
rollWindow = [160, 320, 480]

# 因子切割的分组参数
cutQuantile = [0.2, 0.5]

# 因子切割的聚合参数
agg_mapper = {
    mean:'mean'
}
cutAgg= [mean]

