# factor_universe
# File Digestion 模块解析
- [factor_universe](factor_universe): 多重周期因子生成&测试工具
  - [eod_merge](factor_universe/merge) 合并eod工具，合并day和hh级别的支持因子
    - [merge_config](factor_universe/merge/merge_config): 参数设置文件
    - [main](factor_universe/merge/main): 主程序
  - [facotr_generator](factor_universe/factor_generator) eod因子批量生成工具，调用了eod_merge
    - [fg_config](factor_universe/factor_generator/fg_config): 参数设置文件
    - [main](factor_universe/factor_generator/main): 主程序
  - [ops](factor_universe/ops): 计算加速工具
    - [operation](factor_universe/ops/operation): numba加速函数
  - [style_neutral](factor_universe/style_nuetral): 风格因子中性工具
    - [neu_config](factor_universe/style_nuetral/neu_config): 参数设置文件
    - [neu_toolkit](factor_universe/style_nuetral/neu_toolkit): 风格因子中性工具箱
    - [main](factor_universe/style_nuetral/main): 主函数
  - [backtest](factor_universe/backtest): 回测工具
    - [mp](factor_universe/backtest/mp): 多进程回测工具
      - [backtest](factor_universe/backtest/backtest): 多进程回测工具箱（自定义计算方法，不使用PqiDataSdk接口）
      - [factor_plot](factor_universe/backtest/factor_plot): 主函数
    - [single](factor_universe/backtest/single): 单进程回测工具
      - [bt_config](factor_universe/backtest/single/bt_config): 参数设置文件
      - [plotting](factor_universe/backtest/single/plotting): 作图函数
      - [main](factor_universe/backtest/single/main): 交互式主函数
    - [tools](factor_universe/tools): 读写工具
      - [tools](factor_universe/tools/tools): 读写工具
    

