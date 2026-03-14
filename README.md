# Quant2026

[![Tests](https://github.com/kzil88/Quant2026/actions/workflows/test.yml/badge.svg)](https://github.com/kzil88/Quant2026/actions/workflows/test.yml)

A股多策略量化投资框架 | Multi-strategy quantitative investment framework for China A-shares

## ✨ 特性

- **6 种策略**：多因子、动量、均值回归、统计套利、ML 机器学习、事件驱动
- **14 个因子**：价值（PE/PB/股息率）、质量（ROE/毛利率/负债率）、成长（营收/利润增速）、技术（RSI/MACD/布林带/动量/波动率/换手率）
- **3 种组合优化**：等权、Markowitz 均值方差、Risk Parity 风险平价
- **完整风控**：VaR/CVaR、个股/组合/移动止损、黑名单、换手率约束
- **A 股特色**：T+1、涨跌停、印花税、成交量约束、限价单模拟
- **参数优化**：网格搜索、随机搜索、贝叶斯优化（GP+EI）
- **Walk-Forward**：滚动回测，过拟合检测
- **绩效归因**：Brinson 行业归因、因子归因、月度归因
- **配置驱动**：YAML 配置文件，不改代码切换策略
- **CLI 工具**：`quant2026 backtest/optimize/walkforward/validate/init`
- **214 个测试**，GitHub Actions CI

## 📦 安装

```bash
git clone https://github.com/kzil88/Quant2026.git
cd Quant2026
pip install uv
uv pip install -e ".[dev]"
```

## 🚀 快速开始

### 方式一：CLI 命令行

```bash
# 用默认配置跑回测
quant2026 backtest

# 用自定义配置
quant2026 backtest -c config/aggressive.yaml

# 覆盖参数
quant2026 backtest -o backtest.initial_capital=2000000 -o portfolio.method=risk_parity

# 参数优化
quant2026 optimize -s mean_reversion --method bayesian --n-iter 30

# Walk-Forward 滚动回测
quant2026 walkforward --train-months 6 --test-months 2

# 验证配置文件
quant2026 validate -c config/default.yaml

# 生成默认配置
quant2026 init -o config/my_strategy.yaml
```

### 方式二：配置文件驱动

```bash
python examples/demo_config_pipeline.py --config config/default.yaml
```

### 方式三：Python API

```python
from datetime import date
from quant2026.data.akshare_provider import AkShareProvider
from quant2026.data.cache import CachedProvider
from quant2026.factor.library import MomentumFactor, RSIFactor, EPFactor
from quant2026.factor.preprocessing import FactorPreprocessor
from quant2026.strategy.multi_factor.strategy import MultiFactorStrategy
from quant2026.strategy.mean_reversion.strategy import MeanReversionStrategy
from quant2026.portfolio.optimizer import PortfolioOptimizer
from quant2026.portfolio.markowitz import MarkowitzOptimizer
from quant2026.portfolio.turnover import TurnoverConstraint
from quant2026.backtest.engine import BacktestEngine, BacktestConfig
from quant2026.backtest.report import BacktestReporter

# 1. 数据（带缓存）
provider = CachedProvider(AkShareProvider())
stocks = ["600519", "601318", "000858", "000333"]
data = provider.get_daily_quotes(stocks, date(2024,1,1), date(2024,12,31))

# 2. 因子
factors = [MomentumFactor(20), RSIFactor(14), EPFactor()]
preprocessor = FactorPreprocessor()

# 3. 策略
strategy1 = MultiFactorStrategy(top_n=10)
strategy2 = MeanReversionStrategy(window=20, zscore_threshold=-1.0)

# 4. 组合优化（Markowitz + 换手率约束）
optimizer = PortfolioOptimizer(
    method="markowitz",
    turnover_constraint=TurnoverConstraint(max_turnover=0.3)
)

# 5. 回测
config = BacktestConfig(
    initial_capital=1_000_000,
    commission=0.0003,
    stamp_tax=0.001,
    t_plus_one=True,
)
engine = BacktestEngine(config)
result = engine.run(data, targets)

# 6. 报告
reporter = BacktestReporter()
reporter.generate(result, "output/report.html")
```

## 🏗️ 架构

```
┌─────────────────────────────────────────────┐
│  Data Layer 数据层                            │
│  AkShareProvider → CachedProvider (SQLite)    │
│  日线/财务/行业分类/事件数据                      │
├─────────────────────────────────────────────┤
│  Factor Layer 因子层                          │
│  14个因子 → 预处理(去极值/标准化/中性化)           │
│  IC/IR评估 → 因子衰减分析 → 相关性检测            │
├─────────────────────────────────────────────┤
│  Strategy Layer 策略层                        │
│  多因子 / 动量 / 均值回归 / 统计套利              │
│  ML机器学习 / 事件驱动                          │
├─────────────────────────────────────────────┤
│  Portfolio Layer 组合层                       │
│  等权 / Markowitz / Risk Parity              │
│  换手率约束 + 成本估算                          │
├─────────────────────────────────────────────┤
│  Risk Layer 风控层                            │
│  VaR/CVaR / 止损(个股/组合/移动)                │
│  黑名单 / 仓位限制                             │
├─────────────────────────────────────────────┤
│  Execution Layer 执行层                       │
│  T+1 / 成交量约束 / 限价单模拟 / VWAP           │
├─────────────────────────────────────────────┤
│  Backtest Layer 回测层                        │
│  历史回测 / Walk-Forward / 绩效归因             │
│  HTML报告 / 参数优化                           │
├─────────────────────────────────────────────┤
│  Infrastructure 基础设施                      │
│  YAML配置 / CLI / CI / 日志规范化              │
└─────────────────────────────────────────────┘
```

## 📁 项目结构

```
Quant2026/
├── config/                          # YAML 配置文件
│   ├── default.yaml                 # 默认配置
│   ├── aggressive.yaml              # 激进配置
│   └── conservative.yaml            # 保守配置
├── quant2026/                       # 核心代码
│   ├── data/                        # 数据层
│   │   ├── akshare_provider.py      #   AkShare 数据源
│   │   ├── cache.py                 #   SQLite+Parquet 缓存
│   │   ├── base.py                  #   DataProvider 接口
│   │   └── cleaner.py               #   数据清洗
│   ├── factor/                      # 因子层
│   │   ├── library.py               #   14个因子实现
│   │   ├── evaluation.py            #   IC/IR 评估
│   │   ├── preprocessing.py         #   去极值/标准化/中性化
│   │   └── base.py                  #   Factor 接口
│   ├── strategy/                    # 策略层
│   │   ├── multi_factor/            #   多因子策略
│   │   ├── momentum/                #   动量策略
│   │   ├── mean_reversion/          #   均值回归策略
│   │   ├── stat_arb/                #   统计套利/配对交易
│   │   ├── ml_model/                #   ML 机器学习策略
│   │   ├── event_driven/            #   事件驱动策略
│   │   └── base.py                  #   Strategy 接口
│   ├── portfolio/                   # 组合层
│   │   ├── optimizer.py             #   组合优化器
│   │   ├── markowitz.py             #   Markowitz 均值方差
│   │   ├── risk_parity.py           #   风险平价
│   │   └── turnover.py              #   换手率约束
│   ├── risk/                        # 风控层
│   │   ├── manager.py               #   风控管理器
│   │   ├── var.py                   #   VaR/CVaR
│   │   └── stop_loss.py             #   止损策略
│   ├── execution/                   # 执行层
│   │   ├── t_plus_one.py            #   T+1 约束
│   │   ├── volume_constraint.py     #   成交量约束
│   │   └── order_simulator.py       #   订单模拟器
│   ├── backtest/                    # 回测层
│   │   ├── engine.py                #   回测引擎
│   │   ├── report.py                #   HTML 报告
│   │   ├── walk_forward.py          #   Walk-Forward 分析
│   │   └── attribution.py           #   绩效归因
│   ├── optimization/                # 参数优化
│   │   └── param_optimizer.py       #   网格/随机/贝叶斯搜索
│   ├── config.py                    # 配置加载器
│   ├── factory.py                   # 组件工厂
│   ├── logging.py                   # 日志配置
│   ├── cli.py                       # CLI 入口
│   └── types.py                     # 类型定义
├── examples/                        # 示例脚本
│   ├── demo_pipeline.py             #   多策略融合回测
│   ├── demo_config_pipeline.py      #   配置驱动回测
│   ├── demo_ml_pipeline.py          #   ML 策略回测
│   ├── demo_stat_arb.py             #   统计套利回测
│   ├── demo_factor_evaluation.py    #   因子 IC/IR 评估
│   ├── demo_walk_forward.py         #   Walk-Forward 分析
│   ├── demo_attribution.py          #   绩效归因
│   ├── demo_optimization.py         #   参数优化
│   ├── demo_risk.py                 #   风控展示
│   └── demo_event_driven.py         #   事件驱动
├── tests/                           # 测试 (214个)
│   ├── conftest.py                  #   共享 fixture
│   ├── test_integration.py          #   集成测试
│   └── test_*.py                    #   单元测试
└── .github/workflows/test.yml       # CI
```

## 📊 策略说明

| 策略 | 类 | 核心逻辑 |
|------|------|----------|
| 多因子 | `MultiFactorStrategy` | 综合多个因子打分排序选股 |
| 动量 | `MomentumStrategy` | 买入近期涨幅最大的股票 |
| 均值回归 | `MeanReversionStrategy` | 买入偏离均值最大的超卖股票 |
| 统计套利 | `StatArbStrategy` | 协整配对交易，价差回归 |
| ML 机器学习 | `MLStrategy` | 随机森林/梯度提升预测收益 |
| 事件驱动 | `EventDrivenStrategy` | 财报超预期/大宗交易/股东增减持 |

## ⚙️ 配置示例

```yaml
# config/default.yaml
strategies:
  - name: multi_factor
    type: MultiFactorStrategy
    weight: 0.6
    params:
      top_n: 10
  - name: mean_reversion
    type: MeanReversionStrategy
    weight: 0.4
    params:
      window: 20
      zscore_threshold: -1.0

portfolio:
  method: markowitz
  turnover:
    max_turnover: 0.3

risk:
  stop_loss:
    stock_stop_loss: -0.10
    portfolio_stop_loss: -0.15
    trailing_stop: -0.08

execution:
  t_plus_one: true
  order_type: market
```

## 🧪 测试

```bash
# 运行全部测试
pytest -v

# 跳过网络测试
pytest -m "not network"

# 带覆盖率
pytest --cov=quant2026 --cov-report=term

# 只跑集成测试
pytest -m integration
```

## 📈 回测结果示例

2024 年回测（30 只蓝筹，月度调仓）：

| 配置 | 年化收益 | 夏普比率 | 最大回撤 |
|------|---------|---------|---------|
| 多因子（等权） | 18.33% | 0.80 | -14.38% |
| 多因子+均值回归（等权） | 11.68% | 0.53 | -13.99% |
| 多因子+均值回归（Markowitz） | 14.78% | 0.71 | -14.01% |
| 统计套利 | 6.74% | 0.32 | -16.95% |

## 🛣️ Roadmap

- [x] 数据层（AkShare + 缓存 + 财务 + 行业）
- [x] 因子层（14 因子 + IC/IR 评估）
- [x] 策略层（6 种策略）
- [x] 组合层（Markowitz + Risk Parity + 换手率约束）
- [x] 风控层（VaR/CVaR + 止损 + 黑名单）
- [x] 回测层（引擎 + 报告 + Walk-Forward + 归因）
- [x] 参数优化（网格/随机/贝叶斯）
- [x] 基础设施（YAML 配置 + CLI + CI + 日志）
- [x] 执行层（T+1 + 成交量约束 + 限价单模拟）
- [ ] 实盘接口（券商 API 对接）
- [ ] Web Dashboard

## License

MIT
