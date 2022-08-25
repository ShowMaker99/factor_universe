import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Font config
font1 = {
    'family':'DejaVu Sans',
    'weight':'normal',
    'size':17
        }
    

# Plotting Results
def plot(result_dict, font1=font1):

    # Plot overall setting
    fig = plt.figure(figsize=(15, 20), dpi=255)
    fig.suptitle(result_dict['factor_name'])
    
    # Plot Long Short Excess Return
    fig.add_subplot(5,1,1)
    dates = pd.to_datetime(result_dict['long_short']['long_short_alpha_cum_pnl']['long_rtn_no_fee'].index)
    plt.plot(dates,result_dict['long_short']['long_short_alpha_cum_pnl']['long_rtn_no_fee'], color='darkorange', label='long_no_fee')
    plt.plot(dates,result_dict['long_short']['long_short_alpha_cum_pnl']['long_rtn_after_fee'], color='darkorange', linestyle='dashed', label='long_after_fee')
    plt.plot(dates,result_dict['long_short']['long_short_alpha_cum_pnl']['short_rtn_no_fee'], color='limegreen', label='short_no_fee')
    plt.plot(dates,result_dict['long_short']['long_short_alpha_cum_pnl']['short_rtn_after_fee'], color='limegreen', linestyle='dashed', label='short_after_fee')
    plt.title('Long Short Excess Return', fontdict=font1)
    plt.legend(loc='upper left')
    plt.tick_params(labelsize=12)
    plt.grid(linestyle = ':', linewidth = 0.5)

    # Plot Group Return

    fig.add_subplot(5,2,3)
    plt.bar(result_dict['group']['group_pnl_cumfloat'].keys(),result_dict['group']['group_pnl_cumfloat'].values())
    plt.hlines(np.mean(list(result_dict['group']['group_pnl_cumfloat'].values())), xmin=0, xmax=19, lw=0.5, colors='red')
    plt.title('Group Return No Cost Bar')

    fig.add_subplot(5,2,4)
    plt.bar(result_dict['group']['group_pnl_after_fee_cumfloat'].keys(),result_dict['group']['group_pnl_after_fee_cumfloat'].values())
    plt.hlines(np.mean(list(result_dict['group']['group_pnl_after_fee_cumfloat'].values())), xmin=0, xmax=19, lw=0.5, colors='red')
    plt.title('Group Return After Cost Bar')

    # Plot IC

    fig.add_subplot(5,2,5)
    labels = list(map(str,list(result_dict['ic']['ic_decay'].keys())))
    ic_decay = result_dict['ic']['ic_decay'].values()
    rankic_decay = result_dict['ic']['rankic_decay'].values()

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    plt.bar(x - width/2, ic_decay, width, label='ic_decay')
    plt.bar(x + width/2, rankic_decay, width, label='rankic_decay')
    plt.title('IC and rankIC decay')
    plt.xticks(x,labels)
    plt.legend()

    fig.add_subplot(5,2,7)
    labels = list(map(str,list(result_dict['ic']['ic_cum_decay'].keys())))
    ic_decay = result_dict['ic']['ic_cum_decay'].values()
    rankic_decay = result_dict['ic']['rankic_cum_decay'].values()

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    plt.bar(x - width/2, ic_decay, width, label='ic_cum_decay')
    plt.bar(x + width/2, rankic_decay, width, label='rankic_cum_decay')
    plt.title('IC and rankIC decay')
    plt.xticks(x,labels)
    plt.legend()

    fig.add_subplot(5,2,6)
    IC = round(result_dict['ic']['ic'],3)
    dates = pd.to_datetime(result_dict['ic']['ic_cumsum'].index)
    plt.plot(dates, result_dict['ic']['ic_cumsum'])
    plt.title(f'Cumulated IC_IR:{IC}')

    fig.add_subplot(5,2,8)
    rankIC = round(result_dict['ic']['rank_ic'],3)
    dates = pd.to_datetime(result_dict['ic']['rankic_cumsum'].index)
    plt.plot(dates, result_dict['ic']['rankic_cumsum'])
    plt.title(f'Cumulated rankIC_IR:{rankIC}')

    # Plot summary metrics table
    fig.add_subplot(20,1,17)
    table = {}
    table['IC'] = result_dict['ic']['ic']
    table['Rank_IC'] = result_dict['ic']['rank_ic']
    table['IR'] = result_dict['ic']['ir']
    table.update(result_dict['summary'])
    table.pop('GroupIC_NC')
    table.pop('GroupIC')
    table = pd.DataFrame([table])
    plt.table(cellText=np.round(table.values,4), rowLabels=['Stats'], colLabels=table.columns , loc='center', cellLoc='center',rowLoc='center')
    plt.axis('off')
    
    # Plot yearly ic table and return table
    # fig.add_subplot(20,1,18)
    # ic_stat = pd.DataFrame(result_dict['year']['ic_stat'])
    # plt.table(cellText=np.round(ic_stat.values,4), rowLabels=ic_stat.index, colLabels=ic_stat.columns, loc='center', cellLoc='center',rowLoc='center')
    # plt.axis('off')
    # fig.add_subplot(10,1,10)
    # long_stat = pd.DataFrame(result_dict['year']['long_stat'])
    # plt.table(cellText=np.round(long_stat.values,4), rowLabels=long_stat.index, colLabels=long_stat.columns, loc='center', cellLoc='center',rowLoc='center')
    # plt.axis('off')
    
    fig.add_subplot(7,1,7)
    ic_stat = pd.DataFrame(result_dict['year']['ic_stat'])
    long_stat = pd.DataFrame(result_dict['year']['long_stat'])
    stat = {}
    for year in ic_stat.keys():
        stat[year] = {}
        stat[year].update(ic_stat[year])
        stat[year].update(long_stat[year])
    stat = pd.DataFrame(stat)
    plt.table(cellText=np.round(stat.values,4), rowLabels=stat.index, colLabels=stat.columns, loc='center', cellLoc='center',rowLoc='center')
    plt.axis('off')
    
    
def plot_long_short(result_dict, font1=font1):
    
    # Plot overall setting
    fig = plt.figure(figsize=(15, 4), dpi=255)
    fig.suptitle(result_dict['factor_name'])
    
    # Plot Long-Short Hedging Excess Return
    fig.add_subplot(1,1,1)
    dates = pd.to_datetime(result_dict['long_short']['long_short_alpha_cum_pnl']['long_rtn_no_fee'].index)
    plt.plot(dates,result_dict['long_short']['long_short_alpha_cum_pnl']['long_rtn_no_fee'] - 
             result_dict['long_short']['long_short_alpha_cum_pnl']['short_rtn_no_fee'], color='darkorange', label='no_fee')
    plt.plot(dates,result_dict['long_short']['long_short_alpha_cum_pnl']['long_rtn_after_fee'] - 
             result_dict['long_short']['long_short_alpha_cum_pnl']['short_rtn_after_fee'], color='darkorange', linestyle='dashed', label='after_fee')
    plt.title('Long Short Hedging Excess Return', fontdict=font1)
    plt.legend(loc='upper left')
    plt.tick_params(labelsize=12)
    plt.grid(linestyle = ':', linewidth = 0.5)