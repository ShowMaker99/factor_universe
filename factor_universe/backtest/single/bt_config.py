class Config: # 回测接口的配置类

        # 用户名 - str
        user = 'ydma'
        # 票池 - 可以填list['000001', '000002'] / 指数代号'000852' / 自定义票池'xxx.csv'
        # 注意使用自定义票池的时候start_date和end_date要与配置里一致。
        # 自定义票池的数据形式为：index为ticker，column为date，值为1或nan的2d dataframe
        tickers = 'all'
        # 拉取数据的进程数，如果要使用接口批量测因子，这里要写为1，如果不填默认是12
        size = 12
        # 拉取数据最早日期，可以早于start_date
        earliest_date = '20180101'
        # 回测起止日期 - str
        start_date = '20180101'
        end_date = '20210831'
        # 全部参考指数 - 可以把想看的指数全部填在indexs的list中
        indexs = ['000016','000852', '000905', '000300']
        # 看超额基准指数 - summary图中alpha_rtn的基准指数，如果填'mean'则benchmark为票池平均return，填指定指数则benchmark为指数收益
        benchmark_index = 'mean'
        # 费率 - 换手手续费
        fee_rate = 0.0015
        # return的计算方式 - 'Open2Open' / 'Close2Close' / 'TwapOpen2TwapOpen' / 'TwapClose2TwapClose' 注意大小写以及中间的2
        # 选择 TwapOpen2TwapOpen / TwapClose2TwapClose 的 rtn_type 回测start_date不能早于20150101
        rtn_type = 'TwapOpen2TwapOpen'
        # 分组数
        group_num = 20
        # 正常回测写normal，如果有从文件中读dataframe的需求这里写“exist_df”
        mode = 'exist_df'
        # 添加按头组数目分多空分组，填'mean'则是按因子值均值分两组，填整数则为指定多空组票数，如果填一个小于1的浮点数则是按比例入选多空分组
        long_short_type = 0.2
        # 多空组内持仓比例方式，'factor_value'-因子值加权；'equal'-组内等权
        weight_type = 'factor_value'


