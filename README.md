# Quant2026

Multi-strategy quantitative investment framework for China A-shares (A股多策略量化投资框架).

## Architecture

```
┌────────────────────────────────────────┐
│  1. Data Layer (数据层)                 │
│     Market data, financials, cleaning   │
├────────────────────────────────────────┤
│  2. Factor Layer (因子层)               │
│     Factor compute, standardize,        │
│     neutralize                          │
├────────────────────────────────────────┤
│  3. Strategy Layer (策略层)             │
│     Multi-factor / Momentum / MR / ...  │
├────────────────────────────────────────┤
│  4. Portfolio Layer (组合层)            │
│     Signal aggregation, optimization    │
├────────────────────────────────────────┤
│  5. Risk Layer (风控层)                 │
│     Position limits, drawdown control   │
├────────────────────────────────────────┤
│  6. Backtest Layer (回测层)             │
│     Historical simulation, metrics      │
├────────────────────────────────────────┤
│  7. Execution Layer (执行层)            │
│     Order generation, cost simulation   │
└────────────────────────────────────────┘
```

## A-Share Specific Features

- T+1 settlement constraint
- Limit up/down (涨跌停) handling
- ST stock filtering
- New stock (次新股) exclusion
- Industry neutralization (行业中性化)
- Stamp tax + commission cost modeling

## Quick Start

```bash
pip install -e .
```

```python
from quant2026.data.akshare_provider import AkShareProvider
from quant2026.factor.library import MomentumFactor, VolatilityFactor
from quant2026.factor.registry import FactorRegistry
from quant2026.strategy.multi_factor.strategy import MultiFactorStrategy
from quant2026.portfolio.optimizer import PortfolioOptimizer

# 1. Get data
provider = AkShareProvider()
data = provider.get_daily_quotes(["000001", "600519"], start=date(2024,1,1), end=date(2024,12,31))

# 2. Compute factors
registry = FactorRegistry()
registry.register(MomentumFactor(20))
registry.register(VolatilityFactor(20))
factors = registry.compute_all(data, target_date=date(2024,12,31))

# 3. Run strategy
strategy = MultiFactorStrategy()
result = strategy.generate(data, factors, target_date=date(2024,12,31))

# 4. Build portfolio
optimizer = PortfolioOptimizer(max_stocks=30)
portfolio = optimizer.combine([result], target_date=date(2024,12,31))
```

## License

MIT
