# AkShare 新闻相关接口整理

本文整理 AkShare 中和“新闻、资讯、公告、研报、舆情”相关的常用接口，方便后续接入到 `Quantify`。

## 1. 个股新闻

### `ak.stock_news_em(symbol="603777")`

- 来源：东方财富
- 用途：按股票代码获取最近 100 条个股新闻
- 适合场景：
  - 查询某只股票最近新闻
  - 做个股事件回溯
  - 为选股结果补充新闻摘要

示例：

```python
import akshare as ak

df = ak.stock_news_em(symbol="601789")
print(df.head())
```

建议：

- 这是最适合做“个股新闻搜索”的接口
- 可以和选股结果联动，比如对 `BottomDojiFlow` 输出股票补抓最近 20 条新闻

## 2. 公告

### `ak.stock_notice_report(symbol="全部", date="20220511")`

- 来源：东方财富公告大全
- 用途：获取指定日期的 A 股公告
- 支持类型：
  - `全部`
  - `重大事项`
  - `财务报告`
  - `融资公告`
  - `风险提示`
  - `资产重组`
  - `信息变更`
  - `持股变动`

示例：

```python
import akshare as ak

df = ak.stock_notice_report(symbol="重大事项", date="20260310")
print(df.head())
```

适合场景：

- 做“今日公告扫描”
- 查异动股是否有公告催化
- 给 Doji / panic bottom 候选股补充事件解释

## 3. 个股研报

### `ak.stock_research_report_em(symbol="000001")`

- 来源：东方财富研报中心
- 用途：获取单只股票对应的研究报告

示例：

```python
import akshare as ak

df = ak.stock_research_report_em(symbol="601789")
print(df.head())
```

适合场景：

- 看机构最近是否集中覆盖
- 做“研报热度”辅助因子
- 判断个股是否处于机构关注阶段

## 4. 新闻联播

### `ak.news_cctv(date="20240305")`

- 来源：央视新闻联播
- 用途：获取某天新闻联播文字稿

示例：

```python
import akshare as ak

df = ak.news_cctv(date="20260305")
print(df.head())
```

适合场景：

- 做政策关键词提取
- 识别宏观方向、产业政策线索

## 5. 财新新闻流

### `ak.stock_news_main_cx()`

- 来源：财新数据通
- 用途：抓取财新新闻流

示例：

```python
import akshare as ak

df = ak.stock_news_main_cx()
print(df.head())
```

适合场景：

- 补充宏观和产业新闻
- 辅助主题热点分析

## 6. 微博舆情

### `ak.stock_js_weibo_report(time_period="CNHOUR12")`

- 来源：金十数据中心
- 用途：获取微博舆情报告
- 可选周期：
  - `CNHOUR2`
  - `CNHOUR6`
  - `CNHOUR12`
  - `CNHOUR24`
  - `CNDAY7`
  - `CNDAY30`

示例：

```python
import akshare as ak

df = ak.stock_js_weibo_report(time_period="CNHOUR12")
print(df.head())
```

适合场景：

- 看短期舆情热度
- 监控热点股情绪升温

## 7. 新闻情绪指数

### `ak.index_news_sentiment_scope()`

- 来源：数库
- 用途：A 股新闻情绪指数

示例：

```python
import akshare as ak

df = ak.index_news_sentiment_scope()
print(df.tail())
```

适合场景：

- 作为市场情绪辅助指标
- 给 `market_temperature` 增加新闻维度

## 8. 百度股市通日历型资讯

### `ak.news_economic_baidu(date="20260310", cookie=None)`

- 经济数据日历

### `ak.news_report_time_baidu(date="20260310", cookie=None)`

- 财报发行日历

### `ak.news_trade_notify_dividend_baidu(date="20260310", cookie=None)`

- 分红派息提醒

### `ak.news_trade_notify_suspend_baidu(date="20260310", cookie=None)`

- 停复牌提醒

示例：

```python
import akshare as ak

df = ak.news_trade_notify_suspend_baidu(date="20260310")
print(df.head())
```

适合场景：

- 盘前日历提醒
- 停复牌、分红等事件同步

## 9. 期货/商品新闻

### `ak.futures_news_shmet()`

- 来源：上海金属网
- 用途：期货、金属产业链相关新闻

示例：

```python
import akshare as ak

df = ak.futures_news_shmet()
print(df.head())
```

## 10. 推荐优先级

如果是做 `Quantify` 的落地接入，建议优先顺序：

1. `stock_news_em`
2. `stock_notice_report`
3. `stock_research_report_em`
4. `stock_js_weibo_report`
5. `index_news_sentiment_scope`

原因：

- `stock_news_em`：最适合个股新闻补充
- `stock_notice_report`：最适合事件催化确认
- `stock_research_report_em`：最适合机构关注度
- `stock_js_weibo_report`：最适合短线热度
- `index_news_sentiment_scope`：最适合全市场情绪

## 11. 对 `Quantify` 的接入建议

### 方案 A：个股新闻增强

在 `BottomDojiFlow`、`TrendSurgeScreener` 等结果出来后，对候选股补抓：

- 最近新闻数
- 最近公告数
- 最近研报数

输出成：

- `最近3日新闻条数`
- `最近3日公告条数`
- `最近30日研报条数`

### 方案 B：热点验证

在概念热度分析里引入：

- 概念成分股近 3 日新闻数
- 概念成分股近 3 日公告数
- 微博舆情热度

### 方案 C：市场情绪补充

给 `market_temperature` 增加新闻维度：

- `index_news_sentiment_scope`
- `stock_js_weibo_report`

## 12. 注意事项

- 这些接口大多是抓公开网页，稳定性会受源站变动影响
- 个别百度接口可能需要 `cookie`
- 新闻类数据天然噪声大，更适合做“辅助解释”和“热度增强”，不建议单独作为买卖信号

