# TODO

## 数据层 (Data)
- [ ] 完善 AkShareProvider.get_financial_data (财务报表数据)
- [ ] 完善 AkShareProvider.get_index_quotes (指数行情, 沪深300/中证500)
- [ ] 完善 AkShareProvider.get_industry_classification (申万行业分类)
- [ ] 添加本地数据缓存 (SQLite/Parquet), 避免重复请求
- [ ] 支持增量更新 (只拉取新数据)
- [ ] 添加复权价格验证

## 因子层 (Factor)
- [ ] 添加价值因子: PE/PB/PS/PCF/股息率
- [ ] 添加质量因子: ROE/ROA/毛利率/资产负债率
- [ ] 添加成长因子: 营收增速/净利增速/ROE变化
- [ ] 添加技术因子: RSI/MACD/布林带
- [ ] 因子 IC/IR 检验工具 (评估因子有效性)
- [ ] 因子衰减分析 (不同持仓周期下的因子表现)
- [ ] 多重共线性检测 (因子间相关性)

## 策略层 (Strategy)
- [ ] 实现均值回归策略 (mean_reversion)
- [ ] 实现统计套利/配对交易策略 (stat_arb)
- [ ] 实现事件驱动策略 (event_driven): 财报超预期/股东增减持/大宗交易
- [ ] 实现 ML 策略 (ml_model): LightGBM/XGBoost 选股
- [ ] 策略参数优化框架 (网格搜索/贝叶斯优化)
- [ ] 策略绩效对比报告生成

## 组合层 (Portfolio)
- [ ] 实现 Markowitz 均值方差优化
- [ ] 实现 Risk Parity (风险平价) 分配
- [ ] 实现 Black-Litterman 模型
- [ ] 策略动态权重调整 (基于近期表现)
- [ ] 换手率约束 (控制调仓成本)

## 风控层 (Risk)
- [ ] 实现实时风险监控 (仓位漂移检测)
- [ ] 添加 VaR / CVaR 计算
- [ ] 因子风险暴露监控 (Barra 风格)
- [ ] 止损策略: 个股止损 / 组合止损 / 移动止损
- [ ] 黑名单机制 (手动排除特定股票/行业)

## 回测层 (Backtest)
- [ ] 添加基准对比 (沪深300/中证500)
- [ ] 绩效归因分析 (行业/因子贡献拆解)
- [ ] 分年度/分月度绩效统计
- [ ] 滚动回测 (walk-forward analysis)
- [ ] 回测报告自动生成 (HTML/PDF)

## 执行层 (Execution)
- [ ] 完善 T+1 约束模拟
- [ ] 添加成交量约束 (不超过当日成交量的 N%)
- [ ] 市价单/限价单模拟
- [ ] 预留实盘接口 (券商 API 对接)

## 基础设施
- [ ] 完善单元测试 (pytest)
- [ ] 添加 CI (GitHub Actions)
- [ ] 添加配置文件 (YAML), 支持不同策略组合配置
- [ ] 写一个完整的 demo pipeline (数据获取→因子→策略→组合→回测)
- [ ] 日志规范化 (统一 loguru 配置)
- [ ] 添加 CLI 入口 (python -m quant2026 run/backtest/report)

## 优先级建议
1. **本地数据缓存** — 没有缓存每次跑都要拉数据, 太慢
2. **价值/质量/成长因子** — 多因子策略的核心
3. **demo pipeline** — 端到端跑通一次
4. **基准对比 + 回测报告** — 看得到结果才能迭代
5. **ML 策略** — 最有潜力的方向
