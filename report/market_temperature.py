# -*- encoding: UTF-8 -*-

"""市场情绪温度计（短线量化模型）

基于五大核心数据维度，计算 0-100 分的情绪分值，并给出情绪阶段判断和游资策略建议。

数据来源：
    zt_pool          — 今日涨停池（连板数、炸板次数）         来自 DataManager.extra
    zt_pool_previous — 昨日涨停池（用于计算今日溢价率、晋级率） 来自 DataManager.extra
    全市场行情        — 优先级：
                         1. extra['realtime']（当日有效快照，最新价>0 行数 >= 1000）
                         2. realtime_quotes.parquet（按快照日期过滤）
                         3. daily K线目录并发重建（覆盖任意历史交易日，无需实时数据）

维度定义：
    U   — 今日实际涨停家数（剔除 ST、未开板新股）
    D   — 今日跌停家数（剔除 ST）
    M   — 大面数量（当日高点→收盘回撤 > 10%）
    B   — 盘中触及涨停但最终炸板的家数
    FR  — 封板率 = U / (U + B)
    R_yu— 昨日涨停今日平均收益率
    PR  — 连板晋级率 = 今日连板家数 / 昨日首板及以上家数
    H   — 今日市场最高连板数（剔除新股）
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Set

import pandas as pd

try:
    from data_manager import DataManager
except ModuleNotFoundError:
    from ..data_manager import DataManager  # type: ignore[no-redef]

# 交易日历本地缓存文件名（存放于 DataManager 的 cache_dir 下）
_TRADE_CALENDAR_FILE = 'trade_calendar.parquet'
# 日历缓存最长有效期（天）——含节假日调整的日历一年更新一次即可
_TRADE_CALENDAR_TTL_DAYS = 90


# ══════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class TemperatureMetrics:
    """五大维度的原始指标，中间计算产物。"""

    # ── 维度1：涨跌停数据 ─────────────────────────────────────
    U: int = 0          # 有效涨停家数
    D: int = 0          # 跌停家数
    M: int = 0          # 大面数量（高点→收盘回撤 > 10%）
    B: int = 0          # 炸板家数

    # ── 维度2：封板成功率 ─────────────────────────────────────
    FR: float = 0.0     # 封板率 = U / (U + B)

    # ── 维度3：昨日涨停今日表现 ───────────────────────────────
    R_yu: float = 0.0   # 昨日涨停今日平均收益率（%）

    # ── 维度4：连板晋级率 ─────────────────────────────────────
    PR: float = 0.0     # 连板晋级率 = 今日连板数 / 昨日首板及以上数

    # ── 维度5：市场最高空间 ───────────────────────────────────
    H: int = 0          # 今日最高连板数（剔除新股）


@dataclass
class TemperatureResult:
    """情绪温度计最终输出。"""

    # 原始指标
    metrics: TemperatureMetrics = field(default_factory=TemperatureMetrics)

    # 分项得分
    score_earn: float = 0.0     # 赚钱效应得分（0-40）
    score_safety: float = 0.0   # 打板安全度得分（0-20）
    score_relay: float = 0.0    # 接力意愿得分（0-20）
    score_panic: float = 0.0    # 恐慌扣分（0 ~ -20）
    score_space: float = 0.0    # 空间溢价加分（0-20）

    # 综合得分 0-100
    score: float = 0.0

    # 情绪阶段及策略建议
    phase: str = ''             # 冰点期 / 混沌/修复期 / 主升/发酵期 / 高潮期
    strategy: str = ''          # 对应阶段的明日操作策略

    # 情绪方向与策略适配
    trend: str = ''             # 上升 / 震荡 / 下降（基于原始指标今昨对比）
    strategy_hint: str = ''     # 策略排序，如 "低吸 > 半路 > 接力"
    strategy_reason: str = ''   # 一句话理由

    # 基准日期（通常为上一个交易日）
    trade_date: Optional[str] = None

    # 热门行业板块分析（由 IndustryTemperature 填充）
    hot_sectors: list = field(default_factory=list)

    # 龙头股追踪（由 LeaderStockFinder 填充）
    leader_stocks: list = field(default_factory=list)

    # 概念板块热度（由 ConceptHotAnalyzer 填充）
    concept_hot_result: Optional[object] = None


# ══════════════════════════════════════════════════════════════
#  主类
# ══════════════════════════════════════════════════════════════

class MarketTemperature:
    """市场情绪温度计。

    用法::

        dm = DataManager(config)
        dm.refresh()
        mt = MarketTemperature(dm)
        result = mt.run()
        print(result.score, result.phase)
    """

    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
        # 行情快照缓存：key = 'YYYY-MM-DD'，避免同一次 run() 重复读磁盘
        self._market_df_cache: Dict[str, Optional[pd.DataFrame]] = {}

    # ──────────────────────────────────────────────────────────
    #  公开入口
    # ──────────────────────────────────────────────────────────

    def run(self) -> TemperatureResult:
        """计算情绪温度，返回完整结果。

        流程：
          1. 获取基准交易日
          2. 提取五大维度指标
          3. 计算各分项得分
          4. 合成综合得分
          5. 判断情绪阶段与策略建议
        """
        result = TemperatureResult()

        # ── Step 1：基准交易日 ────────────────────────────────
        result.trade_date = self._get_trade_date()

        # ── Step 2：提取五大维度 ──────────────────────────────
        U, B, H             = self._extract_zt_data()
        D, M                = self._extract_dt_and_mian()
        FR                  = self._calc_seal_rate(U, B)
        R_yu                = self._calc_prev_zt_return()
        PR                  = self._calc_promotion_rate()

        result.metrics = TemperatureMetrics(
            U=U, D=D, M=M, B=B, FR=FR, R_yu=R_yu, PR=PR, H=H
        )

        # ── Step 3：各分项得分 ────────────────────────────────
        result.score_earn   = self._score_earn(R_yu)
        result.score_safety = self._score_safety(FR)
        result.score_relay  = self._score_relay(PR)
        result.score_panic  = self._score_panic(D, M)
        result.score_space  = self._score_space(H, U)

        # ── Step 4：综合得分 ──────────────────────────────────
        result.score = self._calc_total_score(
            result.score_earn,
            result.score_safety,
            result.score_relay,
            result.score_panic,
            result.score_space,
        )

        # ── Step 5：情绪阶段与策略建议 ────────────────────────
        result.phase, result.strategy = self._classify_phase(result.score)
        
        
        # ── Step 6：热门行业板块分析 ────────────────────────────
        from report.industry_temperature import IndustryTemperature
        try:
            it = IndustryTemperature(self.dm, self)
            result.hot_sectors = it.run(top_n=10)
        except Exception as e:
            logging.warning('IndustryTemperature 计算失败: %s', e)
            result.hot_sectors = []

        # ── Step 7：龙头股识别 ────────────────────────────────
        from report.leader_stock import LeaderStockFinder
        try:
            finder = LeaderStockFinder(self.dm, self)
            result.leader_stocks = finder.run(
                result.hot_sectors,
                market_score=result.score,
                market_phase=result.phase,
            )
        except Exception as e:
            logging.warning('LeaderStockFinder 计算失败: %s', e)
            result.leader_stocks = []

        # ── Step 7.5：概念板块热度分析 ──────────────────────────
        from report.concept_hot import ConceptHotAnalyzer
        try:
            ch = ConceptHotAnalyzer(self.dm)
            result.concept_hot_result = ch.run(top_n=20)
        except Exception as e:
            logging.warning('ConceptHotAnalyzer 计算失败: %s', e)

        # ── Step 8：情绪方向 + 策略适配提示 ──────────────────────
        U_prev, H_prev = self._derive_prev_metrics()
        result.trend = self._classify_trend(result.metrics, U_prev, H_prev)
        result.strategy_hint, result.strategy_reason = self._suggest_strategy_mix(
            result.metrics, result.trend, result.phase, result.hot_sectors,
        )
        logging.debug(
            'trend=%s, strategy_hint=%s', result.trend, result.strategy_hint,
        )

        return result

    # ──────────────────────────────────────────────────────────
    #  Step 1：维度提取
    # ──────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────
    #  全市场行情（realtime 优先，降级为 daily K线重建）
    # ──────────────────────────────────────────────────────────

    def _load_code_name_map(self) -> Dict[str, str]:
        """从 realtime_quotes.parquet 构建 代码→名称 映射，供 daily 数据的 ST 过滤使用。"""
        rt_path = os.path.join(self.dm._cache.cache_dir, 'realtime_quotes.parquet')
        if os.path.exists(rt_path):
            try:
                df = pd.read_parquet(rt_path, columns=['代码', '名称'])
                # 去重，取最后一次出现的名称（最新）
                df = df.drop_duplicates(subset=['代码'], keep='last')
                return dict(zip(df['代码'].astype(str), df['名称'].astype(str)))
            except Exception as e:
                logging.debug('_load_code_name_map: %s', e)
        return {}

    def _build_market_df_from_daily(self, trade_date_str: str) -> Optional[pd.DataFrame]:
        """从 daily K线目录并发重建指定交易日的全市场行情快照。

        返回列：代码, 名称, 最新价, 最高, 最低, 昨收, 涨跌幅
        （列名与 realtime_quotes 对齐，可直接传入现有计算逻辑）
        """
        daily_dir = os.path.join(self.dm._cache.cache_dir, 'daily')
        if not os.path.exists(daily_dir):
            logging.debug('_build_market_df_from_daily: daily 目录不存在')
            return None

        name_map = self._load_code_name_map()
        files = [f for f in os.listdir(daily_dir) if f.endswith('.parquet')]

        def _read_one(fname: str):
            code = fname[:-8]   # strip '.parquet'
            try:
                df = pd.read_parquet(os.path.join(daily_dir, fname))
                df['日期'] = df['日期'].astype(str)
                pos_list = df.index[df['日期'] == trade_date_str].tolist()
                if not pos_list:
                    return None
                pos = df.index.get_loc(pos_list[0])
                row = df.iloc[pos]
                prev_close = float(df.iloc[pos - 1]['收盘']) if pos > 0 else None
                close  = float(row['收盘']) if pd.notna(row['收盘']) else None
                high   = float(row['最高']) if pd.notna(row['最高']) else None
                low    = float(row['最低']) if pd.notna(row['最低']) else None
                pct    = float(row['涨跌幅']) if pd.notna(row.get('涨跌幅', float('nan'))) else None
                # 如果 daily 没有涨跌幅，则用收盘/昨收自行计算
                if pct is None and close is not None and prev_close and prev_close > 0:
                    pct = (close - prev_close) / prev_close * 100.0
                return {
                    '代码':   code,
                    '名称':   name_map.get(code, ''),
                    '最新价': close,
                    '最高':   high,
                    '最低':   low,
                    '昨收':   prev_close,
                    '涨跌幅': pct,
                }
            except Exception:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(_read_one, files))

        rows = [r for r in results if r is not None]
        if not rows:
            return None
        result = pd.DataFrame(rows)
        logging.debug('_build_market_df_from_daily: %s 共 %d 只', trade_date_str, len(result))
        return result

    def _get_market_df(self, trade_date: str) -> Optional[pd.DataFrame]:
        """获取指定交易日的全市场行情 DataFrame，带内存缓存。

        优先级：
          1. dm.extra['realtime']（过滤指定日期，有效最新价 >= 1000 行）
          2. realtime_quotes.parquet 文件（按快照日期过滤）
          3. daily K线目录并发重建（最慢，但覆盖任意历史交易日）
        """
        trade_date_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"

        if trade_date_str in self._market_df_cache:
            return self._market_df_cache[trade_date_str]

        def _valid_count(df: pd.DataFrame) -> int:
            if df is None or df.empty or '最新价' not in df.columns:
                return 0
            return int((pd.to_numeric(df['最新价'], errors='coerce') > 0).sum())

        # ── 1. dm.extra['realtime'] ───────────────────────────
        rt = self.dm.extra.get('realtime')
        if rt is not None and not rt.empty:
            rt_day = rt[rt['快照日期'] == trade_date_str].copy() \
                if '快照日期' in rt.columns else rt.copy()
            if _valid_count(rt_day) >= 1000:
                logging.debug('_get_market_df[%s]: realtime extra（%d只）', trade_date_str, len(rt_day))
                self._market_df_cache[trade_date_str] = rt_day
                return rt_day

        # ── 2. realtime_quotes.parquet 文件 ──────────────────
        rt_path = os.path.join(self.dm._cache.cache_dir, 'realtime_quotes.parquet')
        if os.path.exists(rt_path):
            try:
                rt_all = pd.read_parquet(rt_path)
                if '快照日期' in rt_all.columns:
                    rt_day = rt_all[rt_all['快照日期'] == trade_date_str].copy()
                    if _valid_count(rt_day) >= 1000:
                        logging.debug('_get_market_df[%s]: realtime_quotes.parquet（%d只）',
                                      trade_date_str, len(rt_day))
                        self._market_df_cache[trade_date_str] = rt_day
                        return rt_day
            except Exception as e:
                logging.debug('_get_market_df: 读取 realtime_quotes.parquet 失败: %s', e)

        # ── 3. daily K线重建 ──────────────────────────────────
        logging.debug('_get_market_df[%s]: fallback → daily K线重建', trade_date_str)
        df = self._build_market_df_from_daily(trade_date_str)
        self._market_df_cache[trade_date_str] = df
        return df

    # ──────────────────────────────────────────────────────────
    #  交易日历（含节假日）
    # ──────────────────────────────────────────────────────────

    def _trade_calendar_path(self) -> str:
        """交易日历 parquet 本地缓存路径。"""
        return os.path.join(self.dm._cache.cache_dir, _TRADE_CALENDAR_FILE)

    def _load_trade_calendar(self) -> Set[date]:
        """加载 A 股交易日历，优先读本地缓存；缓存过期或不存在时调 akshare 刷新。

        返回：所有交易日的 date 集合。
        网络不通时降级为「排除周末」的近似集合（忽略节假日）。
        """
        path = self._trade_calendar_path()

        # ── 尝试读本地缓存 ────────────────────────────────────
        if os.path.exists(path):
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(path))
                if (datetime.now() - mtime).days < _TRADE_CALENDAR_TTL_DAYS:
                    df = pd.read_parquet(path)
                    return set(pd.to_datetime(df['trade_date']).dt.date)
            except Exception as e:
                logging.debug("读取交易日历缓存失败: %s", e)

        # ── 从 akshare 获取完整日历 ───────────────────────────
        try:
            import akshare as ak
            df = ak.tool_trade_date_hist_sina()
            # 只保留 trade_date 列，存为 parquet
            df = df[['trade_date']].copy()
            df.to_parquet(path, index=False)
            logging.info("交易日历已刷新，共 %d 个交易日", len(df))
            return set(pd.to_datetime(df['trade_date']).dt.date)
        except Exception as e:
            logging.warning("获取交易日历失败，降级为仅排除周末: %s", e)

        # ── 降级：用周末过滤近 5 年日期（不含节假日）────────────
        today = date.today()
        start = today - timedelta(days=365 * 5)
        return {
            start + timedelta(days=i)
            for i in range((today - start).days + 1)
            if (start + timedelta(days=i)).weekday() < 5
        }

    @staticmethod
    def _zt_limit_ratio(code_series: pd.Series) -> pd.Series:
        """根据股票代码判断涨跌停幅度。

        主板 (60xxxx/00xxxx) → 0.10
        创业板 (300xxx)       → 0.20
        科创板 (688xxx)       → 0.20
        北交所 (8xxxxx/4xxxxx) → 0.30
        ST 的 5% 限制已在上游剔除 ST 后不需考虑。
        """
        code = code_series.astype(str).str[:3]
        ratio = pd.Series(0.10, index=code_series.index)
        ratio[code.isin(['300', '301'])] = 0.20
        ratio[code == '688'] = 0.20
        ratio[code.isin(['430', '830', '870'])] = 0.30
        return ratio

    def _get_trade_date(self) -> str:
        """获取基准交易日字符串（格式 YYYYMMDD），含节假日处理。

        规则：
        - 今天是交易日 且 当前时间 >= 15:00（已收盘）→ 返回今天
        - 否则（今天非交易日，或尚未收盘）→ 返回最近一个已过去的交易日

        Returns:
            形如 '20260228' 的字符串
        """
        calendar = self._load_trade_calendar()
        today = date.today()
        now_hour = datetime.now().hour

        # 今天已收盘，直接用今天
        if today in calendar and now_hour >= 15:
            return today.strftime('%Y%m%d')

        # 往前找最近一个交易日（最多回溯 10 天，覆盖最长假期）
        d = today - timedelta(days=1)
        for _ in range(10):
            if d in calendar:
                return d.strftime('%Y%m%d')
            d -= timedelta(days=1)

        # 兜底：从日历集合中取最大值（理论上不会走到这里）
        if calendar:
            return max(calendar).strftime('%Y%m%d')
        return today.strftime('%Y%m%d')

    def _extract_zt_data(self) -> tuple[int, int, int]:
        """提取有效涨停数 U、炸板数 B、今日最高连板数 H。

        数据来源：
            U / H  — extra['zt_pool']（已封板股票，含连板数）
            B      — _get_market_df()（盘中触及涨停价但收盘未封板的股票）
                     优先 realtime_quotes 快照，降级为 daily K线重建

        剔除：股票名含 'ST' 的股票。
        注：新股（上市不足 30 日的连续一字板）可通过 连板数 >= 10 做近似过滤，
            但无准确上市日期数据时默认不过滤，保留在统计中。

        Returns:
            (U, B, H)
        """
        zt_df = self.dm.extra.get('zt_pool')
        rt_df = self._get_market_df(self._get_trade_date())

        # ── U 与 H：来自涨停池 ────────────────────────────────
        U, H = 0, 0
        if zt_df is not None and not zt_df.empty:
            # 剔除 ST
            mask_st = zt_df['名称'].str.upper().str.contains('ST', na=False)
            valid = zt_df[~mask_st]

            U = len(valid)

            # 最高连板数（剔除连板数异常大的疑似新股一字板，阈值 20）
            if '连板数' in valid.columns:
                col = pd.to_numeric(valid['连板数'], errors='coerce').dropna()
                col = col[col < 20]   # 连板 ≥ 20 视为新股一字板，排除
                H = int(col.max()) if len(col) > 0 else 0

        # ── B：今日触及涨停但未封板的股票（炸板） ────────────
        # 判定逻辑：最高价 ≥ 涨停价 且 最新价 < 涨停价（未以涨停收盘）
        # 涨停幅度按板块区分：主板10%、创业板/科创板20%、北交所30%
        B = 0
        if rt_df is not None and not rt_df.empty:
            rt = rt_df.copy()
            # 过滤 ST
            mask_st_rt = rt['名称'].str.upper().str.contains('ST', na=False)
            rt = rt[~mask_st_rt]

            # 只保留必要列且数值有效的行
            needed = {'昨收', '最高', '最新价'}
            if needed.issubset(rt.columns):
                rt = rt.dropna(subset=list(needed))
                rt['昨收']  = pd.to_numeric(rt['昨收'],  errors='coerce')
                rt['最高']  = pd.to_numeric(rt['最高'],  errors='coerce')
                rt['最新价'] = pd.to_numeric(rt['最新价'], errors='coerce')
                rt = rt.dropna(subset=['昨收', '最高', '最新价'])

                rt['_zt_ratio'] = self._zt_limit_ratio(rt['代码'])
                rt['_zt_price'] = (rt['昨收'] * (1 + rt['_zt_ratio'])).round(2)

                # 炸板：盘中最高 ≥ 涨停价 且 最新价 < 涨停价
                zhaban_mask = (rt['最高'] >= rt['_zt_price']) & \
                              (rt['最新价'] < rt['_zt_price'])

                # 排除已经在 zt_pool 中的代码（防双计）
                if zt_df is not None and not zt_df.empty and '代码' in zt_df.columns:
                    zt_codes = set(zt_df['代码'].astype(str))
                    zhaban_mask = zhaban_mask & ~rt['代码'].astype(str).isin(zt_codes)

                B = int(zhaban_mask.sum())

        logging.debug("_extract_zt_data: U=%d, B=%d, H=%d", U, B, H)
        return U, B, H

    def _extract_dt_and_mian(self) -> tuple[int, int]:
        """提取跌停数 D 与大面数 M。

        数据来源：_get_market_df()（优先 realtime_quotes，降级为 daily K线重建）
        - 跌停 D：涨跌幅 <= -9.9%，剔除 ST。
          （用 -9.9% 而非 -10% 是因为浮点精度：实际跌停涨跌幅可能为 -9.93% 等）
        - 大面 M：(最高价 - 收盘价) / 最高价 >= 10%，剔除 ST。
          即当日高点到收盘回撤超过 10%，俗称"吃大面"。

        Returns:
            (D, M)
        """
        rt_df = self._get_market_df(self._get_trade_date())
        D, M = 0, 0

        if rt_df is None or rt_df.empty:
            logging.debug("_extract_dt_and_mian: 行情数据为空")
            return D, M

        rt = rt_df.copy()

        # 剔除 ST
        mask_st = rt['名称'].str.upper().str.contains('ST', na=False)
        rt = rt[~mask_st]

        # 转为数值
        rt['涨跌幅'] = pd.to_numeric(rt['涨跌幅'], errors='coerce')
        rt['最高']   = pd.to_numeric(rt['最高'],   errors='coerce')
        rt['最新价'] = pd.to_numeric(rt['最新价'],  errors='coerce')

        rt = rt.dropna(subset=['涨跌幅', '最高', '最新价'])

        # ── D：跌停（按板块区分涨跌停幅度） ─────────────────
        if '代码' in rt.columns and '昨收' in rt.columns:
            rt['昨收'] = pd.to_numeric(rt['昨收'], errors='coerce')
            limit = self._zt_limit_ratio(rt['代码'])
            dt_price = (rt['昨收'] * (1 - limit)).round(2)
            D = int((rt['最新价'] <= dt_price).sum())
        else:
            D = int((rt['涨跌幅'] <= -9.9).sum())

        # ── M：大面（高点→收盘回撤 >= 10%） ──────────────────
        # 只对最高价 > 0 的行计算，避免除零
        valid_high = rt[rt['最高'] > 0].copy()
        drawdown = (valid_high['最高'] - valid_high['最新价']) / valid_high['最高']
        M = int((drawdown >= 0.10).sum())

        logging.debug("_extract_dt_and_mian: D=%d, M=%d", D, M)
        return D, M

    def _calc_prev_zt_return(self) -> float:
        """计算昨日涨停今日平均收益率 R_yu（%）。

        实现逻辑：
          1. 取 zt_pool_previous 中 快照日期 == 当前基准交易日 的行
             （这些行即"前一交易日涨停、当日有表现数据"的股票）
          2. 若 zt_pool_previous 快照日期与基准日不符，则用
             extra['realtime'] 对 zt_pool_previous 的代码做 join，
             取 realtime.涨跌幅 补充。
          3. 剔除 ST，对 涨跌幅 取平均。

        "昨日" 定义：当前基准交易日（_get_trade_date()）的前一个交易日，
        由 DataManager 在 _fetch_zt_pool_previous 时已隐式处理。

        Returns:
            R_yu（%），无数据时返回 0.0
        """
        prev_df = self.dm.extra.get('zt_pool_previous')
        rt_df   = None   # 延迟加载，仅在 zt_pool_previous 没有涨跌幅时才用

        if prev_df is None or prev_df.empty:
            logging.debug("_calc_prev_zt_return: zt_pool_previous 为空")
            return 0.0

        trade_date = self._get_trade_date()           # 'YYYYMMDD'
        trade_date_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"  # 'YYYY-MM-DD'

        # ── 取当前基准日对应的快照 ────────────────────────────
        if '快照日期' in prev_df.columns:
            snap = prev_df[prev_df['快照日期'].astype(str) == trade_date_str].copy()
            if snap.empty:
                # 退而求其次：取最新快照
                snap = prev_df[prev_df['快照日期'] == prev_df['快照日期'].max()].copy()
                logging.debug(
                    "_calc_prev_zt_return: 未找到 %s 快照，使用最新快照 %s",
                    trade_date_str, snap['快照日期'].iloc[0] if not snap.empty else 'N/A'
                )
        else:
            snap = prev_df.copy()

        # 剔除 ST
        mask_st = snap['名称'].str.upper().str.contains('ST', na=False)
        snap = snap[~mask_st]

        if snap.empty:
            return 0.0

        # ── 优先用 zt_pool_previous 自带的涨跌幅 ─────────────
        if '涨跌幅' in snap.columns:
            returns = pd.to_numeric(snap['涨跌幅'], errors='coerce').dropna()
            if len(returns) > 0:
                R_yu = float(returns.mean())
                logging.debug(
                    "_calc_prev_zt_return: 昨日涨停%d只，今日均收益=%.2f%%",
                    len(returns), R_yu
                )
                return R_yu

        # ── 备用：从全市场行情查最新涨跌幅 ──────────────────────
        if '代码' in snap.columns:
            market_df = self._get_market_df(self._get_trade_date())
            if market_df is not None and not market_df.empty:
                codes = set(snap['代码'].astype(str))
                matched = market_df[market_df['代码'].astype(str).isin(codes)].copy()
                matched['涨跌幅'] = pd.to_numeric(matched['涨跌幅'], errors='coerce')
                returns = matched['涨跌幅'].dropna()
                if len(returns) > 0:
                    R_yu = float(returns.mean())
                    logging.debug(
                        "_calc_prev_zt_return(market fallback): %d只，均=%.2f%%",
                        len(returns), R_yu
                    )
                    return R_yu

        logging.debug("_calc_prev_zt_return: 无有效数据，返回 0.0")
        return 0.0

    def _calc_seal_rate(self, U: int, B: int) -> float:
        """计算封板率 FR = U / (U + B)。

        Args:
            U: 有效涨停数
            B: 炸板数

        Returns:
            FR，范围 [0, 1]；分母为 0 时返回 0.0
        """
        total = U + B
        return U / total if total > 0 else 0.0

    def _calc_promotion_rate(self) -> float:
        """计算连板晋级率 PR = 今日连板数 / 昨日首板及以上数。

        - 今日连板数：zt_pool 中 连板数 >= 2 的股票数。
        - 昨日首板及以上数：zt_pool_previous 中 连板数 >= 1 的股票数。

        Returns:
            PR，范围 [0, 1]；分母为 0 时返回 0.0
        """
        zt_df   = self.dm.extra.get('zt_pool')
        prev_df = self.dm.extra.get('zt_pool_previous')

        # ── 今日连板数（zt_pool 中 连板数 >= 2）──
        today_multi = 0
        if zt_df is not None and not zt_df.empty and '连板数' in zt_df.columns:
            mask_st = zt_df['名称'].str.upper().str.contains('ST', na=False)
            valid   = zt_df[~mask_st].copy()
            col     = pd.to_numeric(valid['连板数'], errors='coerce')
            today_multi = int((col >= 2).sum())

        # ── 昨日首板及以上数（zt_pool_previous 最新快照中 昨日连板数 >= 1）──
        # zt_pool_previous 中连板字段名为「昨日连板数」
        prev_any = 0
        _prev_col = '昨日连板数' if (prev_df is not None and '昨日连板数' in prev_df.columns) else '连板数'
        if prev_df is not None and not prev_df.empty and _prev_col in prev_df.columns:
            # 取最新快照（同 _calc_prev_zt_return 的逻辑）
            if '快照日期' in prev_df.columns:
                snap = prev_df[prev_df['快照日期'] == prev_df['快照日期'].max()].copy()
            else:
                snap = prev_df.copy()
            mask_st = snap['名称'].str.upper().str.contains('ST', na=False)
            snap    = snap[~mask_st]
            col     = pd.to_numeric(snap[_prev_col], errors='coerce')
            prev_any = int((col >= 1).sum())

        pr = today_multi / prev_any if prev_any > 0 else 0.0
        logging.debug(
            '_calc_promotion_rate: 今日连板(>=2)=%d, 昨日首板及以上(>=1)=%d, PR=%.3f',
            today_multi, prev_any, pr,
        )
        return pr

    # ──────────────────────────────────────────────────────────
    #  Step 2：分项得分计算
    # ──────────────────────────────────────────────────────────

    def _score_earn(self, R_yu: float) -> float:
        """赚钱效应得分（0-40 分）。

        R_yu >= 3%  → 40 分
        R_yu <= -2% → 0 分
        中间线性插值。
        """
        if R_yu >= 3.0:
            return 40.0
        if R_yu <= -2.0:
            return 0.0
        return (R_yu - (-2.0)) / (3.0 - (-2.0)) * 40.0

    def _score_safety(self, FR: float) -> float:
        """打板安全度得分（0-20 分）。

        FR >= 80% → 20 分
        FR <= 50% → 0 分
        中间线性插值。
        """
        if FR >= 0.80:
            return 20.0
        if FR <= 0.50:
            return 0.0
        return (FR - 0.50) / (0.80 - 0.50) * 20.0

    def _score_relay(self, PR: float) -> float:
        """接力意愿得分（0-20 分）。

        PR >= 30% → 20 分
        PR <= 10% → 0 分
        中间线性插值。
        """
        if PR >= 0.30:
            return 20.0
        if PR <= 0.10:
            return 0.0
        return (PR - 0.10) / (0.30 - 0.10) * 20.0

    def _score_panic(self, D: int, M: int) -> float:
        """恐慌情绪扣分项（-20 ~ 0 分）。

        Panic = D + 0.5 * M
        Panic >= 30 → -20 分
        Panic <= 5  → 0 分
        中间线性插值。
        注意：返回值为负数（扣分），合成总分时直接相加。
        """
        panic = D + 0.5 * M
        if panic >= 30:
            return -20.0
        if panic <= 5:
            return 0.0
        return -((panic - 5) / (30 - 5)) * 20.0

    def _score_space(self, H: int, U: int) -> float:
        """空间溢价加分项（0 ~ 20 分），阶梯式平滑过渡。

        H >= 6 且 U >= 60 → 20 分
        H == 5 且 U >= 50 → 10 分
        H == 4 且 U >= 40 →  5 分
        否则               →  0 分
        """
        if H >= 6 and U >= 60:
            return 20.0
        if H >= 5 and U >= 50:
            return 10.0
        if H >= 4 and U >= 40:
            return 5.0
        return 0.0

    # ──────────────────────────────────────────────────────────
    #  Step 3：综合评估
    # ──────────────────────────────────────────────────────────

    def _calc_total_score(
        self,
        score_earn: float,
        score_safety: float,
        score_relay: float,
        score_panic: float,
        score_space: float,
    ) -> float:
        """合成最终情绪得分，clamp 到 [0, 100]。"""
        raw = score_earn + score_safety + score_relay + score_panic + score_space
        return max(0.0, min(100.0, raw))

    def _classify_phase(self, score: float) -> tuple[str, str]:
        """根据综合得分判断情绪阶段与操作策略。

        分段：
            0-20   → 冰点期：空仓，连续低于15分考虑"冰点试错"。
            20-50  → 混沌/修复期：半仓，只做前排核心。
            50-80  → 主升/发酵期：重仓出击，积极参与接力。
            80-100 → 高潮期：持筹享溢价，停止买入，准备兑现。

        Returns:
            (phase_name, strategy_text)
        """
        if score < 20:
            return (
                '冰点期',
                '管住手，空仓观望。连续两天低于15分可小仓位试错新题材首板（冰点试错）。',
            )
        if score < 50:
            return (
                '混沌/修复期',
                '市场试探新方向，只做前排核心，不做后排跟风，半仓操作。',
            )
        if score < 80:
            return (
                '主升/发酵期',
                '赚钱效应爆棚，重仓出击！积极参与龙头接力和核心题材的2进3。',
            )
        return (
            '高潮期',
            '情绪过热，盛极必衰。持筹者享受溢价，持币者停止买入，准备冲高兑现利润，防范高潮后的分歧下杀。',
        )

    # ──────────────────────────────────────────────────────────
    #  Step 8：情绪方向 + 策略适配
    # ──────────────────────────────────────────────────────────

    def _derive_prev_metrics(self) -> tuple[int, int]:
        """从 zt_pool_previous 提取昨日原始指标 (U_prev, H_prev)。

        不依赖任何缓存的派生分数，纯粹基于原始数据。
        """
        prev_df = self.dm.extra.get('zt_pool_previous')
        if prev_df is None or prev_df.empty:
            return 0, 0

        if '快照日期' in prev_df.columns:
            snap = prev_df[prev_df['快照日期'] == prev_df['快照日期'].max()].copy()
        else:
            snap = prev_df.copy()

        mask_st = snap['名称'].str.upper().str.contains('ST', na=False)
        snap = snap[~mask_st]

        if snap.empty:
            return 0, 0

        U_prev = len(snap)

        _col = '昨日连板数' if '昨日连板数' in snap.columns else '连板数'
        if _col in snap.columns:
            h_series = pd.to_numeric(snap[_col], errors='coerce').dropna()
            H_prev = int(h_series.max()) if len(h_series) > 0 else 0
        else:
            H_prev = 0

        logging.debug(
            '_derive_prev_metrics: U_prev=%d, H_prev=%d', U_prev, H_prev,
        )
        return U_prev, H_prev

    @staticmethod
    def _classify_trend(
        m: TemperatureMetrics,
        U_prev: int,
        H_prev: int,
    ) -> str:
        """基于今昨原始指标对比，判断情绪方向。

        规则按优先级：
          1. 空间龙断板 (H_prev>=5, H 下降 >1)  → 下降
          2. R_yu<0 且 U 缩量                     → 下降
          3. R_yu>2 且 U 不缩                     → 上升
          4. 新高度打开 (H>H_prev 且 H>=5)        → 上升
          5. 其余                                 → 震荡
        """
        if H_prev >= 5 and m.H < H_prev - 1:
            return '下降'
        if m.R_yu < 0 and m.U < U_prev:
            return '下降'
        if m.R_yu > 2 and U_prev > 0 and m.U >= U_prev:
            return '上升'
        if m.H > H_prev and m.H >= 5:
            return '上升'
        return '震荡'

    @staticmethod
    def _suggest_strategy_mix(
        m: TemperatureMetrics,
        trend: str,
        phase: str,
        hot_sectors: list,
    ) -> tuple[str, str]:
        """根据原始指标 + 情绪方向 + 板块状态，输出策略排序和理由。

        Returns:
            (strategy_hint, strategy_reason)
        """
        if phase == '冰点期':
            return (
                '观望 > 冰点试错',
                '市场冰点，空仓为主。连续冰点可小仓位试错新题材首板。',
            )

        if trend == '下降':
            return (
                '观望 > 低吸（极轻仓）',
                '情绪退化中，等企稳再动手。若参与仅限核心票分歧低吸。',
            )

        has_launch = any(
            getattr(s, 'phase', '') == '启动' for s in hot_sectors
        )

        if trend == '上升':
            if m.PR >= 0.20:
                return (
                    '接力 > 半路 > 低吸',
                    f'接力环境回暖（PR={m.PR:.0%}），龙头有溢价，优先做连板接力。',
                )
            return (
                '半路 > 低吸 > 接力',
                f'赚钱效应好（R_yu={m.R_yu:+.1f}%）但晋级率低（PR={m.PR:.0%}），'
                '首板机会优于高位接力。',
            )

        # trend == '震荡'
        if m.PR >= 0.20:
            return (
                '接力 > 趋势 > 轮动',
                f'有连板基础（PR={m.PR:.0%}），可做接力和趋势跟踪。',
            )
        if m.FR >= 0.70:
            return (
                '半路 > 低吸 > 轮动',
                f'封板率尚可（FR={m.FR:.0%}），适合首板半路和核心票低吸。',
            )
        if has_launch:
            return (
                '轮动 > 半路 > 趋势',
                '有新板块处于启动期，跟踪板块轮动节奏。',
            )
        return (
            '趋势 > 轮动 > 观望',
            '无明确方向，轻仓跟踪趋势股为主。',
        )

    # ──────────────────────────────────────────────────────────
    #  输出/展示
    # ──────────────────────────────────────────────────────────

    def to_markdown(self, result: TemperatureResult) -> str:
        """将 TemperatureResult 格式化为 Markdown 报告字符串。"""
        m = result.metrics
        phase, strategy = result.phase, result.strategy
        lines = [
            f'## 打板复盘报告（交易日：{result.trade_date}）',
            '',
            '### 市场快照',
            f'| 指标 | 值 | 说明 |',
            f'|------|-----|------|',
            f'| 涨停数 U | {m.U} | 非ST有效涨停 |',
            f'| 跌停数 D | {m.D} | 非ST跌停 |',
            f'| 大面数 M | {m.M} | 高点到收盘回撤>10% |',
            f'| 炸板数 B | {m.B} | 盘中触板但收盘未封 |',
            f'| 封板率 FR | {m.FR:.1%} | U/(U+B) |',
            f'| 昨日涨停溢价 R_yu | {m.R_yu:+.2f}% | 昨日涨停今日均收益 |',
            f'| 连板晋级率 PR | {m.PR:.1%} | 今日连板/昨日首板及以上 |',
            f'| 最高连板 H | {m.H} | 板 |',
            '',
            '### 环境评分',
            f'| 维度 | 得分 | 满分 |',
            f'|------|------|------|',
            f'| 赚钱效应（R_yu） | {result.score_earn:.1f} | 40 |',
            f'| 打板安全度（FR） | {result.score_safety:.1f} | 20 |',
            f'| 接力意愿（PR） | {result.score_relay:.1f} | 20 |',
            f'| 恐慌扣分（D+M） | {result.score_panic:.1f} | 0（扣分项）|',
            f'| 空间溢价（H+U） | {result.score_space:.1f} | 20 |',
            '',
            f'### 连板接力环境：**{result.score:.1f} 分**',
            '',
            f'### 明日操作策略：**{phase}**',
            f'> {strategy}',
        ]

        if result.strategy_hint:
            lines.append('')
            lines.append(
                f'### 明日策略倾向：**{result.strategy_hint}**'
                f'（情绪{result.trend}）'
            )
            lines.append(f'> {result.strategy_reason}')

        # 行业板块热度
        if result.hot_sectors:
            from report.industry_temperature import IndustryTemperature
            lines.append('')
            lines.append(IndustryTemperature.to_markdown(result.hot_sectors))

        # 龙头股追踪（连接 leader_stock 模块输出）
        if result.leader_stocks:
            from report.leader_stock import LeaderStockFinder
            lines.append('')
            lines.append(LeaderStockFinder.to_markdown(
                result.leader_stocks,
                market_score=result.score,
                market_phase=result.phase,
            ))

        # 概念板块热度
        if result.concept_hot_result is not None:
            from report.concept_hot import ConceptHotAnalyzer
            lines.append('')
            lines.append(ConceptHotAnalyzer.to_markdown(result.concept_hot_result))

        return '\n'.join(lines)
