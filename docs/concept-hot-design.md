# 概念热度分析设计文档

## 1. 目标

基于已有的本地缓存数据，计算每个概念板块的"热度"得分，支持：
- 今日概念热度排名
- 过去 N 天热度趋势
- 概念内龙头股识别
- 概念发酵阶段判断（启动 / 加速 / 高潮 / 退潮）

## 2. 数据全景

| 数据源 | 存储位置 | 时间深度 | 可用于概念热度的关键字段 |
|--------|---------|---------|----------------------|
| concept_cons | concept_cons/{板块代码}.parquet | 当日快照 | 概念→个股映射（桥梁） |
| concept_board | concept_board.parquet | 多日快照 | 概念当日涨跌幅、上涨/下跌家数、领涨股 |
| daily | daily/{股票代码}.parquet | ~4年日K | 涨跌幅、成交量、成交额、换手率 |
| fund_flow | fund_flow/{股票代码}.parquet | ~128天 | 主力/超大单/大单/中单/小单 净流入 |
| chips | chips/{股票代码}.parquet | ~91天 | 获利比例、平均成本、集中度 |
| intraday | intraday/{股票代码}.parquet | 分时 | 5分钟级别成交量、涨跌 |
| zt_pool | zt_pool.parquet | 多日快照 | 涨停股代码、封板资金、连板数、首次封板时间 |
| zt_pool_strong | zt_pool_strong.parquet | 多日快照 | 强势股、入选理由(60日新高等)、量比 |
| lhb_detail | lhb_detail.parquet | 30天 | 龙虎榜净买额、上榜原因、后续涨跌 |
| big_deal | big_deal.parquet | 当日 | 大单成交时间、金额、买/卖盘性质 |
| hsgt_hold | hsgt_hold/{股票代码}.parquet | 当日 | 北向持股数、增持/减持估计 |
| sector_flow | sector_flow.parquet | 多日快照 | 行业主力净流入（不同分类体系，辅助参考） |
| financial_report | financial_report.parquet | 4季度 | 营收增长、净利润增长、ROE |
| financial_balance | financial_balance.parquet | 最新季度 | 总资产、负债率 |

## 3. 核心思路

以 `concept_cons` 为桥梁，通过股票代码 JOIN 各维度个股数据，按概念聚合得到多维热度指标。

```
concept_cons (概念→个股映射)
         │
         │  通过股票代码 JOIN
         ▼
   ┌─────┼─────┬──────────┬──────────┬──────────┐
   │     │     │          │          │          │
 daily  fund_flow  chips   zt_pool  lhb_detail  hsgt_hold
   │     │     │          │          │          │
   ▼     ▼     ▼          ▼          ▼          ▼
 按概念聚合 → 得到每个概念的多维热度指标
```

## 4. 热度五维模型

### 4.1 维度一：价格动量（daily K线）

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| avg_change_1d | 成分股今日涨跌幅均值 | 短期动量 |
| avg_change_3d / 5d / 10d | 成分股近 N 日累计涨跌幅均值 | 中期趋势 |
| up_ratio | 上涨家数 / 总家数 | 板块普涨程度 |
| new_high_ratio | 创 20 日新高股数 / 总家数 | 强势扩散度 |

**数据来源**：`daily` × `concept_cons`，可回溯 4 年。

### 4.2 维度二：资金动向（fund_flow）

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| sum_main_inflow | 成分股主力净流入合计（元） | 主力资金态度 |
| avg_main_pct | 成分股主力净流入占比均值 | 资金介入强度 |
| big_order_pct | 超大单净流入占比均值 | 区分主力 vs 游资 |
| inflow_streak_ratio | 连续 N 日主力净流入的成分股比例 | 资金持续性 |

**数据来源**：`fund_flow` × `concept_cons`，可回溯 ~128 天。

### 4.3 维度三：极端信号（zt_pool + lhb_detail + big_deal）

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| zt_count | 概念内今日涨停股数量 | 爆发强度 |
| max_lianban | 概念内最大连板数 | 龙头高度 |
| lhb_net_buy | 概念内龙虎榜净买入合计 | 机构态度 |
| big_buy_amount | 概念内大单买入金额合计 | 盘中资金抢筹 |

**数据来源**：`zt_pool` / `lhb_detail` / `big_deal` × `concept_cons`。

### 4.4 维度四：筹码结构（chips）

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| avg_profit_ratio | 成分股平均获利比例 | 浮盈压力（过高=有抛压） |
| concentration_change | 70%集中度近5日变化 | 集中度上升=主力收集 |

**数据来源**：`chips` × `concept_cons`。

### 4.5 维度五：聪明钱态度（hsgt_hold）

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| north_hold_value | 北向持有该概念成分股的总市值 | 外资持仓规模 |
| north_add_count | 今日北向净增持的成分股数量 | 外资看好广度 |
| north_add_value | 今日北向净增持市值合计 | 外资加仓力度 |

**数据来源**：`hsgt_hold` × `concept_cons`。

## 5. 综合评分公式

```
概念热度 = w1 × 价格动量得分
         + w2 × 资金动向得分
         + w3 × 极端信号得分
         + w4 × 筹码结构得分
         + w5 × 聪明钱得分
```

各维度先标准化到 0~100 分，权重建议初始值：

| 维度 | 权重 | 理由 |
|------|------|------|
| 价格动量 | 30% | 最直观的热度表征 |
| 资金动向 | 30% | 主力真金白银的态度 |
| 极端信号 | 20% | 涨停/龙虎榜是市场聚焦的放大器 |
| 筹码结构 | 10% | 辅助判断持续性，但滞后性较强 |
| 聪明钱 | 10% | 北向数据有参考价值但覆盖面有限（仅2767只） |

## 6. 可行性评估

| 分析目标 | 是否可行 | 说明 |
|---------|---------|------|
| 今日概念热度排名 | ✅ 可以 | 所有数据源都有当日快照 |
| 过去 N 天热度趋势 | ✅ 可以（部分） | daily(4年) + fund_flow(128天) 可按日回溯聚合 |
| 概念板块自身的历史K线 | ❌ 不行 | 需新增 `stock_board_concept_hist_em` 接口 |
| 概念内龙头股识别 | ✅ 可以 | 涨停次数 + 资金流 + 龙虎榜交叉分析 |
| 概念发酵阶段判断 | ✅ 可以 | 涨停数趋势 + 筹码集中度变化 + 资金持续性 |
| 概念间的联动/轮动 | ✅ 可以 | 多概念的时序热度对比 |

## 7. 实现优先级

### Phase 1：今日概念热度快照

用 `concept_cons` + `daily`(当日涨跌) + `fund_flow`(当日主力净流入) + `zt_pool`(涨停数) 一次聚合出排名。

**输入**：内存中的 extra['concept_cons'] + extra['zt_pool'] + stocks_data + extra['fund_flow']

**输出**：DataFrame，每行一个概念，包含热度得分和各维度子分。

### Phase 2：N日热度趋势

用 `daily` 和 `fund_flow` 的历史数据按日回溯聚合，生成每个概念过去 N 天的热度时间序列。

**依赖**：Phase 1 的计算逻辑 + 按日期循环聚合。

### Phase 3：极端信号加成

加入 `lhb_detail` + `big_deal` + `hsgt_hold` 作为额外权重维度，完善综合评分。

### Phase 4（可选）：概念历史K线

新增 `stock_board_concept_hist_em` 数据拉取和缓存，用概念板块自身的 OHLCV 做趋势分析，与自下而上的聚合结果互相验证。
