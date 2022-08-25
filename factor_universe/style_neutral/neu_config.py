"""
风格中性参数
"""

# 因子名字
name = 'mins_mf_rtn_actTrdValRtoS_0.8to1.0_mean'

# 因子生成起止日期
start_date = '20180101'
end_date = '20210831'

# 因子读写路径
path = '/home/shared/Data/data/shared/low_fre_alpha/ap_monster_zoo/factor_min' # 读入原始因子
save_path = '/home/shared/Data/data/shared/low_fre_alpha/ap_monster_zoo/factor_min_neu' # 写入风格中性后的因子

# 需要中性化的风格因子
# style = [
#             "residual_volatility","momentum","beta","liquidity","size",
#             "earnings_yield","growth","leverage","non_linear_size","book_to_price"
#         ] # Barra 10 style factors

style = ["size"]

# style =  ['beta', 'book_to_price', 'earnings_yield', 'growth', 'leverage',
#                               'liquidity', 'momentum', 'non_linear_size', 'residual_volatility', 'size', 'comovement', 
#                               'agriculture', 'steel','nonferrous_metals', 'electronics', 
#                               'household_appliance', 'food_n_beverage', 'textiles_n_apparel', 'light_mfg', 
#                               'biomedicine', 'utility', 'transportation', 'real_estate',
#                               'comprehensive', 'arch_mat', 'arch_deco', 'electrical_eqpt', 'military',
#                               'computer', 'media', 'telecom', 'bank', 'non_bank_finance', 'automobile', 'machinery_n_eqpt']