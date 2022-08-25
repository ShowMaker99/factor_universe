import numpy as np


# Simple Return
def rtn(df, adj_factor):
    return np.log(df['MpClose']/df['MpOpen'])

# 主买/主卖占比->买卖意愿
def actBuyValTotRto(df, adj_factor):
    return df.eval('(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL)/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')

def actSellValTotRto(df, adj_factor):
    return df.eval('(ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')

# 四类交易者主买/主卖占总交易值比例
def actBuyValTotRtoS(df, adj_factor):
    return df.eval('ActBuyValueS/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')

def actBuyValTotRtoM(df, adj_factor):
    return df.eval('ActBuyValueM/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')

def actBuyValTotRtoL(df, adj_factor):
    return df.eval('ActBuyValueL/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')

def actBuyValTotRtoXL(df, adj_factor):
    return df.eval('ActBuyValueXL/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')

def actSellValTotRtoS(df, adj_factor):
    return df.eval('ActSellValueS/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')

def actSellValTotRtoM(df, adj_factor):
    return df.eval('ActSellValueM/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')

def actSellValTotRtoL(df, adj_factor):
    return df.eval('ActSellValueL/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')

def actSellValTotRtoXL(df, adj_factor):
    return df.eval('ActSellValueXL/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')

# 四类交易者的交易额占比（主动交易）
def actTrdValRtoS(df, adj_factor):
    return df.eval('(ActBuyValueS+ActSellValueS)/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')

def actTrdValRtoM(df, adj_factor):
    return df.eval('(ActBuyValueM+ActSellValueM)/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')

def actTrdValRtoL(df, adj_factor):
    return df.eval('(ActBuyValueL+ActSellValueL)/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')

def actTrdValRtoXL(df, adj_factor):
    return df.eval('(ActBuyValueXL+ActSellValueXL)/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')

# 四类交易者的交易额占比（全部交易）
def totTrdValRtoS(df, adj_factor):
    return df.eval('(ActBuyValueS+ActSellValueS+PasBuyValueS+PasSellValueS)/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL+PasBuyValueS+PasBuyValueM+PasBuyValueL+PasBuyValueXL+PasSellValueS+PasSellValueM+PasSellValueL+PasSellValueXL)')

def totTrdValRtoM(df, adj_factor):
    return df.eval('(ActBuyValueM+ActSellValueM+PasBuyValueM+PasSellValueM)/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL+PasBuyValueS+PasBuyValueM+PasBuyValueL+PasBuyValueXL+PasSellValueS+PasSellValueM+PasSellValueL+PasSellValueXL)')

def totTrdValRtoL(df, adj_factor):
    return df.eval('(ActBuyValueL+ActSellValueL+PasBuyValueL+PasSellValueL)/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL+PasBuyValueS+PasBuyValueM+PasBuyValueL+PasBuyValueXL+PasSellValueS+PasSellValueM+PasSellValueL+PasSellValueXL)')

def totTrdValRtoXL(df, adj_factor):
    return df.eval('(ActBuyValueXL+ActSellValueXL+PasBuyValueXL+PasSellValueXL)/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL+ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL+PasBuyValueS+PasBuyValueM+PasBuyValueL+PasBuyValueXL+PasSellValueS+PasSellValueM+PasSellValueL+PasSellValueXL)')

# 四类交易者的交易量占比（主动交易）
def actTrdVolRtoS(df, adj_factor):
    return df.eval('(ActBuyVolumeS+ActSellVolumeS)/(ActBuyVolumeS+ActBuyVolumeM+ActBuyVolumeL+ActBuyVolumeXL+ActSellVolumeS+ActSellVolumeM+ActSellVolumeL+ActSellVolumeXL)')

def actTrdVolRtoM(df, adj_factor):
    return df.eval('(ActBuyVolumeM+ActSellVolumeM)/(ActBuyVolumeS+ActBuyVolumeM+ActBuyVolumeL+ActBuyVolumeXL+ActSellVolumeS+ActSellVolumeM+ActSellVolumeL+ActSellVolumeXL)')

def actTrdVolRtoL(df, adj_factor):
    return df.eval('(ActBuyVolumeL+ActSellVolumeL)/(ActBuyVolumeS+ActBuyVolumeM+ActBuyVolumeL+ActBuyVolumeXL+ActSellVolumeS+ActSellVolumeM+ActSellVolumeL+ActSellVolumeXL)')

def actTrdVolRtoXL(df, adj_factor):
    return df.eval('(ActBuyVolumeXL+ActSellVolumeXL)/(ActBuyVolumeS+ActBuyVolumeM+ActBuyVolumeL+ActBuyVolumeXL+ActSellVolumeS+ActSellVolumeM+ActSellVolumeL+ActSellVolumeXL)')

# 四类交易者的交易量占比（全部交易）
def totTrdVolRtoS(df, adj_factor):
    return df.eval('(ActBuyVolumeS+ActSellVolumeS+PasBuyVolumeS+PasSellVolumeS)/(ActBuyVolumeS+ActBuyVolumeM+ActBuyVolumeL+ActBuyVolumeXL+ActSellVolumeS+ActSellVolumeM+ActSellVolumeL+ActSellVolumeXL+PasBuyVolumeS+PasBuyVolumeM+PasBuyVolumeL+PasBuyVolumeXL+PasSellVolumeS+PasSellVolumeM+PasSellVolumeL+PasSellVolumeXL)')

def totTrdVolRtoM(df, adj_factor):
    return df.eval('(ActBuyVolumeM+ActSellVolumeM+PasBuyVolumeM+PasSellVolumeM)/(ActBuyVolumeS+ActBuyVolumeM+ActBuyVolumeL+ActBuyVolumeXL+ActSellVolumeS+ActSellVolumeM+ActSellVolumeL+ActSellVolumeXL+PasBuyVolumeS+PasBuyVolumeM+PasBuyVolumeL+PasBuyVolumeXL+PasSellVolumeS+PasSellVolumeM+PasSellVolumeL+PasSellVolumeXL)')

def totTrdVolRtoL(df, adj_factor):
    return df.eval('(ActBuyVolumeL+ActSellVolumeL+PasBuyVolumeL+PasSellVolumeL)/(ActBuyVolumeS+ActBuyVolumeM+ActBuyVolumeL+ActBuyVolumeXL+ActSellVolumeS+ActSellVolumeM+ActSellVolumeL+ActSellVolumeXL+PasBuyVolumeS+PasBuyVolumeM+PasBuyVolumeL+PasBuyVolumeXL+PasSellVolumeS+PasSellVolumeM+PasSellVolumeL+PasSellVolumeXL)')

def totTrdVolRtoXL(df, adj_factor):
    return df.eval('(ActBuyVolumeXL+ActSellVolumeXL+PasBuyVolumeXL+PasSellVolumeXL)/(ActBuyVolumeS+ActBuyVolumeM+ActBuyVolumeL+ActBuyVolumeXL+ActSellVolumeS+ActSellVolumeM+ActSellVolumeL+ActSellVolumeXL+PasBuyVolumeS+PasBuyVolumeM+PasBuyVolumeL+PasBuyVolumeXL+PasSellVolumeS+PasSellVolumeM+PasSellVolumeL+PasSellVolumeXL)')

# 四类交易者主买占其买卖之和的比例->四类交易者的买卖意愿
def actBuyValLclRtoS(df, adj_factor):
    return df.eval('ActBuyValueS/(ActBuyValueS+ActSellValueS)')

def actBuyValLclRtoM(df, adj_factor):
    return df.eval('ActBuyValueM/(ActBuyValueM+ActSellValueM)')

def actBuyValLclRtoL(df, adj_factor):
    return df.eval('ActBuyValueL/(ActBuyValueL+ActSellValueL)')

def actBuyValLclRtoXL(df, adj_factor):
    return df.eval('ActBuyValueXL/(ActBuyValueXL+ActSellValueXL)')

# 四类交易者主买/主卖占总主买/主卖比例
def actBuyValRtoS(df, adj_factor):
    return df.eval('ActBuyValueS/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL)')

def actBuyValRtoM(df, adj_factor):
    return df.eval('ActBuyValueM/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL)')

def actBuyValRtoL(df, adj_factor):
    return df.eval('ActBuyValueL/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL)')

def actBuyValRtoXL(df, adj_factor):
    return df.eval('ActBuyValueXL/(ActBuyValueS+ActBuyValueM+ActBuyValueL+ActBuyValueXL)')

def actSellValRtoS(df, adj_factor):
    return df.eval('ActSellValueS/(ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')

def actSellValRtoM(df, adj_factor):
    return df.eval('ActSellValueM/(ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')

def actSellValRtoL(df, adj_factor):
    return df.eval('ActSellValueL/(ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')

def actSellValRtoXL(df, adj_factor):
    return df.eval('ActSellValueXL/(ActSellValueS+ActSellValueM+ActSellValueL+ActSellValueXL)')


