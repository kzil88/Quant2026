# TODO

## 数据层 (Data)
- [x] 完善 AkShareProvider.get_financial_data (财务报表数据)
- [x] 完善 AkShareProvider.get_index_quotes (指数行情, 沪深300/中证500)
- [x] 完善 AkShareProvider.get_industry_classification (申万行业分类)
- [x] 添加本地数据缓存 (SQLite/Parquet), 避免重复请求
- [ ] 支持增量更新 (只拉取新数据)
- [ ] 添加复权价格验证

## 因子层 (Factor)
- [x] 添加价值因子: PE/PB/股息率 (EP, BP, DividendYield)
- [x] 添加质量因子: ROE/毛利率/资产负债率
- [x] 添加成长因子: 营收增速/净利增速
- [x] 添加技术因子: RSI/MACD/布林带/动量/波动率/换手率
- [x] 因子 IC/IR 检验工具 (评估因子有效性)
- [x] 因子衰减分析 (不同持仓周期下的因子表现)
- [x] 多重共线性检测 (因子间相关性)

## 策略层 (Strategy)
- [x] 实现多因子策略 (multi_factor): 加权打分选股
- [x] 实现动量策略 (momentum): 双均线交叉 + 动量打分
- [x] 实现均值回归策略 (mean_reversion): Z-score + 布林带 + RSI
- [x] 实现统计套利/配对交易策略 (stat_arb): 协整配对 + 价差回归
- [x] 实现事件驱动策略 (event_driven): 财报超预期/股东增减持/大宗交易
- [x] 实现 ML 策略 (ml_model): LightGBM/XGBoost 选股
- [x] 策略参数优化框架 (网格搜索/随机搜索/贝叶斯优化)
- [x] 策略绩效对比报告生成

## ML 策略改进 (待做)
- [ ] 标签改为超额收益 (个股收益 - 沪深300/行业指数收益), 目前用的是绝对收益, 模型学的是 beta 不是 alpha
- [ ] 添加 regime 特征 (大盘20日/60日均线关系、市场波动率分位数、涨跌家数比等), 让模型感知市场环境
- [ ] 分 regime 训练 (牛市/震荡/熊市使用不同模型或加 regime 标签)
- [ ] 降低动量类因子权重或添加反转特征, 防止模型退化为追涨杀跌
- [ ] 添加过拟合检测指标 (训练集 vs 测试集 IC 差异、Walk-Forward 稳定性)

## 择时与信号驱动调仓 (待做)
- [ ] 回测引擎接入策略 signals (目前 BUY/SELL/HOLD 信号算了但没用)
- [ ] 支持信号触发调仓 (不只是定期调仓, 策略发出 BUY/SELL 时也能触发)
- [ ] 信号强度映射仓位 (分数高多买, 分数低少买, 而非固定权重)
- [ ] 多策略信号融合 (多个策略同时说买 → 强信号, 单个说买 → 弱信号)
- [ ] 止损信号联动 (止损触发时强制 SELL, 不等下次调仓)
- [ ] 为多因子和事件驱动策略补充 BUY/SELL 信号输出

## 组合层 (Portfolio)
- [x] 实现 Markowitz 均值方差优化
- [x] 实现 Risk Parity (风险平价) 分配
- [ ] 实现 Black-Litterman 模型
- [ ] 策略动态权重调整 (基于近期表现)
- [x] 换手率约束 (控制调仓成本)

## 风控层 (Risk)
- [ ] 实现实时风险监控 (仓位漂移检测)
- [x] 添加 VaR / CVaR 计算
- [ ] 因子风险暴露监控 (Barra 风格)
- [x] 止损策略: 个股止损 / 组合止损 / 移动止损
- [x] 黑名单机制 (手动排除特定股票/行业)

## 回测层 (Backtest)
- [x] 添加基准对比
- [x] 绩效归因分析 (Brinson 行业归因 + 因子归因)
- [x] 分年度/分月度绩效统计
- [x] 滚动回测 (Walk-Forward analysis)
- [x] 回测报告自动生成 (HTML)

## 执行层 (Execution)
- [x] 完善 T+1 约束模拟
- [x] 添加成交量约束 (不超过当日成交量的 N%)
- [x] 市价单/限价单模拟
- [ ] 预留实盘接口 (券商 API 对接)

## 基础设施
- [x] 完善单元测试 (214个测试)
- [x] 添加 CI (GitHub Actions)
- [x] 添加配置文件 (YAML), 支持不同策略组合配置
- [x] 写一个完整的 demo pipeline (数据获取→因子→策略→组合→回测)
- [x] 日志规范化 (统一 loguru 配置)
- [x] 添加 CLI 入口 (quant2026 backtest/optimize/walkforward/validate/init)

## 未来规划
- [ ] 实盘接口 (券商 API 对接)
- [ ] Web Dashboard
- [ ] 实现 Black-Litterman 模型
- [ ] 策略动态权重调整
- [ ] 实时风险监控 / Barra 因子暴露监控
- [ ] 增量数据更新
- [ ] 复权价格验证
