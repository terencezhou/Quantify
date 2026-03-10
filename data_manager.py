# -*- encoding: UTF-8 -*-

"""数据管理器 — 统一管理数据拉取、缓存刷新与数据访问

完全独立实现所有数据拉取逻辑，不依赖 data_fetcher.py。

使用方式：
    dm = DataManager(config)
    dm.refresh()                     # 拉取 & 更新所有数据
    dm.stocks_data                   # {(code, name): daily_df}
    dm.extra                         # {fund_flow: ..., chips: ..., ...}
    dm.all_data                      # 全市场实时行情 DataFrame
    dm.report_cache_status()         # 打印缓存摘要
"""

import os
import re
import logging
import time
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any

import akshare as ak
import pandas as pd
import talib as tl

from data_cache import DataCache


_MARKET_PREFIXES = {
    '沪主板': ['60'],
    '深主板': ['00'],
    '创业板': ['30'],
    '科创板': ['68'],
    '北交所': ['43', '83', '87', '92'],
}

_EXCHANGE_MAP = {
    'sh': ['60', '68', '90'],
    'sz': ['00', '30', '12', '92'],
    'bj': ['43', '83', '87'],
}


def _warmup_v8():
    try:
        from py_mini_racer import MiniRacer
        ctx = MiniRacer()
        ctx.eval("1+1")
        del ctx
    except Exception:
        pass


_warmup_v8()


@dataclass
class CacheStatus:
    """缓存状态快照"""
    daily_count: int = 0
    fund_flow_count: int = 0
    intraday_count: int = 0
    chips_count: int = 0
    financial_count: int = 0
    hsgt_hold_count: int = 0
    stock_info_count: int = 0
    concept_cons_count: int = 0
    snapshot_names: List[str] = field(default_factory=list)


class DataManager:
    """统一数据管理器，独立实现全部数据拉取 + 缓存逻辑。

    生命周期：
        __init__  → 加载配置，初始化缓存
        refresh() → 并行拉取/更新全部数据
        之后通过属性访问数据
    """

    def __init__(self, config: dict):
        self._config = config
        self._cache = DataCache(config.get('data_dir', 'data'))
        self._timeout = config.get('fetch_timeout', 300)

        self._all_data: Optional[pd.DataFrame] = None
        self._stocks: List[Tuple[str, str]] = []
        self._stocks_data: Dict[Tuple, pd.DataFrame] = {}
        self._extra: Dict[str, Any] = {}
        self._refreshed = False
        self._proxy_pool = None

        self._init_proxy_pool()

    def _init_proxy_pool(self):
        """根据配置初始化隧道代理池并安装 monkey-patch。"""
        proxy_cfg = self._config.get('proxy', {})
        if not proxy_cfg.get('enable', False):
            return
        try:
            from proxy_pool import ProxyPool
            pool = ProxyPool(proxy_cfg)
            pool.install()
            self._proxy_pool = pool
        except Exception as e:
            logging.warning("代理池初始化失败，将直连: %s", e)

    # ══════════════════════════════════════════════
    #  公开属性
    # ══════════════════════════════════════════════

    @property
    def config(self) -> dict:
        return self._config

    @property
    def all_data(self) -> Optional[pd.DataFrame]:
        return self._all_data

    @property
    def stocks(self) -> List[Tuple[str, str]]:
        return self._stocks

    @property
    def stocks_data(self) -> Dict[Tuple, pd.DataFrame]:
        return self._stocks_data

    @property
    def extra(self) -> Dict[str, Any]:
        return self._extra

    @property
    def is_refreshed(self) -> bool:
        return self._refreshed

    # ══════════════════════════════════════════════
    #  核心流程
    # ══════════════════════════════════════════════

    def refresh(self) -> bool:
        """拉取并更新所有数据，返回是否成功。"""
        logging.info("===== DataManager: 开始刷新数据 =====")

        self._all_data = self._fetch_realtime_quotes()
        if self._all_data is None or self._all_data.empty:
            logging.error("无法获取实时行情数据")
            return False

        active = self._all_data[
            self._all_data['最新价'].notna() & (self._all_data['最新价'] > 0)
        ]
        logging.info("实时行情 %d 只, 有效(最新价>0) %d 只, 过滤退市/停牌 %d 只",
                     len(self._all_data), len(active),
                     len(self._all_data) - len(active))
        subset = active[['代码', '名称']]
        self._stocks = [tuple(x) for x in subset.values]
        self._stocks = self._filter_by_market(self._stocks)

        self._stocks_data, self._extra = self._parallel_fetch(self._stocks)

        if not self._stocks_data:
            logging.error("日K数据为空，数据刷新失败")
            return False

        if self._all_data is not None:
            self._extra['realtime'] = self._all_data

        sf = self._extra.get('sector_flow')
        if sf is not None and not (hasattr(sf, 'empty') and sf.empty):
            self._extra['hot_sectors'] = self._fetch_hot_sector_stocks(sf)

        self._refreshed = True
        logging.info("===== DataManager: 数据刷新完成 (%d只股票) =====",
                     len(self._stocks_data))
        return True

    def refresh_for_stocks(self, stock_list: List[Tuple[str, str]]) -> bool:
        """只刷新指定股票的数据（用于持仓分析等场景）。"""
        logging.info("DataManager: 定向刷新 %d 只股票", len(stock_list))

        self._all_data = self._fetch_realtime_quotes()
        self._stocks = stock_list
        self._stocks_data, self._extra = self._parallel_fetch(stock_list)

        if self._all_data is not None:
            self._extra['realtime'] = self._all_data

        self._refreshed = bool(self._stocks_data)
        return self._refreshed

    # ══════════════════════════════════════════════
    #  快速 / 纯缓存 模式
    # ══════════════════════════════════════════════

    def refresh_fast(self) -> bool:
        """快速模式：仅拉取实时行情，其余数据从本地缓存加载。"""
        logging.info("===== DataManager: 快速刷新模式 =====")

        self._all_data = self._fetch_realtime_quotes()
        if self._all_data is None or self._all_data.empty:
            logging.error("无法获取实时行情数据")
            return False

        active = self._all_data[
            self._all_data['最新价'].notna() & (self._all_data['最新价'] > 0)
        ]
        subset = active[['代码', '名称']]
        self._stocks = [tuple(x) for x in subset.values]
        self._stocks = self._filter_by_market(self._stocks)

        self._stocks_data = self._fallback_daily_from_cache(self._stocks)
        if not self._stocks_data:
            logging.error("日K缓存为空，快速模式失败")
            return False

        self._extra = self._load_extra_from_cache(self._stocks)
        self._extra['realtime'] = self._all_data

        self._refreshed = True
        logging.info("===== DataManager: 快速刷新完成 (%d只) =====",
                     len(self._stocks_data))
        return True

    def load_from_cache(self) -> bool:
        """纯缓存模式：不发起任何网络请求，全部从本地缓存加载。"""
        logging.info("===== DataManager: 纯缓存模式 =====")

        self._all_data = self._cache.get_snapshot_latest("realtime_quotes")
        if self._all_data is None or self._all_data.empty:
            logging.error("缓存中无实时行情数据，请至少运行一次完整刷新")
            return False

        active = self._all_data[
            self._all_data['最新价'].notna() & (self._all_data['最新价'] > 0)
        ]
        subset = active[['代码', '名称']]
        self._stocks = [tuple(x) for x in subset.values]
        self._stocks = self._filter_by_market(self._stocks)

        self._stocks_data = self._fallback_daily_from_cache(self._stocks)
        if not self._stocks_data:
            logging.error("日K缓存为空")
            return False

        self._extra = self._load_extra_from_cache(self._stocks)
        self._extra['realtime'] = self._all_data

        self._refreshed = True
        logging.info("===== DataManager: 缓存加载完成 (%d只) =====",
                     len(self._stocks_data))
        return True

    def _load_extra_from_cache(self, stocks) -> dict:
        """从缓存加载 extra 数据（快照 + 逐只数据）。"""
        extra = {}

        snapshot_keys = [
            'sector_flow', 'concept_board', 'lhb_detail', 'zt_pool',
            'zt_pool_previous', 'zt_pool_strong', 'hsgt_flow',
            'financial_report', 'financial_balance', 'big_deal',
        ]
        for key in snapshot_keys:
            cached = self._cache.get_snapshot_latest(key)
            if cached is not None and not cached.empty:
                extra[key] = cached

        per_stock_sources = [
            ('fetch_chips',      'chips',      self._cache.get_chips),
            ('fetch_fund_flow',  'fund_flow',  self._cache.get_fund_flow),
            ('fetch_intraday',   'intraday',   self._cache.get_intraday),
            ('fetch_hsgt_hold',  'hsgt_hold',  self._cache.get_hsgt_hold),
            ('fetch_stock_info', 'stock_info', self._cache.get_stock_info),
        ]
        for cfg_key, data_key, get_fn in per_stock_sources:
            if self._config.get(cfg_key, False):
                extra[data_key] = self._load_per_stock_cache(
                    stocks, get_fn, data_key,
                )

        if self._config.get('fetch_concept_cons', False):
            concept_df = extra.get('concept_board')
            if concept_df is not None and not concept_df.empty:
                cons = {}
                for _, row in concept_df.iterrows():
                    code = row.get('板块代码', '')
                    if code:
                        df = self._cache.get_concept_cons(code)
                        if df is not None and not df.empty:
                            cons[code] = df
                if cons:
                    extra['concept_cons'] = cons
                    logging.info("概念成分股缓存加载: %d个板块", len(cons))

        return extra

    def _load_per_stock_cache(self, stocks, get_fn, label) -> dict:
        """并发从缓存加载逐只数据。"""
        result = {}

        def _load_one(stock):
            df = get_fn(stock[0])
            return stock, df

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            for stock, df in pool.map(_load_one, stocks):
                if df is not None and not df.empty:
                    result[stock] = df
        logging.info("%s缓存加载: %d只", label, len(result))
        return result

    def get_cache_status(self) -> CacheStatus:
        stats = self._cache.get_cache_stats()
        return CacheStatus(
            daily_count=stats.get('daily', 0),
            fund_flow_count=stats.get('fund_flow', 0),
            intraday_count=stats.get('intraday', 0),
            chips_count=stats.get('chips', 0),
            financial_count=stats.get('financial', 0),
            hsgt_hold_count=stats.get('hsgt_hold', 0),
            stock_info_count=stats.get('stock_info', 0),
            concept_cons_count=stats.get('concept_cons', 0),
        )

    def report_cache_status(self):
        s = self.get_cache_status()
        logging.info(
            "本地缓存: 日K=%d, 资金流=%d, 分时=%d, 筹码=%d, 财务=%d, "
            "北向持股=%d, 个股信息=%d, 概念成分=%d",
            s.daily_count, s.fund_flow_count, s.intraday_count,
            s.chips_count, s.financial_count, s.hsgt_hold_count,
            s.stock_info_count, s.concept_cons_count,
        )

    def build_rt_lookup(self) -> Dict[str, Any]:
        rt = {}
        if self._all_data is not None:
            for _, row in self._all_data.iterrows():
                rt[str(row['代码'])] = row
        return rt

    def build_ff_lookup(self) -> Dict[str, Any]:
        ff = {}
        for code_name, ff_df in self._extra.get('fund_flow', {}).items():
            if ff_df is not None and not ff_df.empty:
                ff[code_name[0]] = ff_df.iloc[-1]
        return ff

    # ══════════════════════════════════════════════
    #  并行调度
    # ══════════════════════════════════════════════

    def _filter_by_market(self, stocks):
        # 拉取原始数据时不做市场过滤，覆盖全量 A 股
        return stocks

    def _parallel_fetch(self, stocks):
        """串行调度各数据源，每种数据内部各自控制并发量。"""
        cfg = self._config
        extra = {}
        stocks_data = None

        # ── 1. 日K（逐只并发=2）──
        try:
            stocks_data = self._run_daily(stocks)
        except Exception as e:
            logging.error("日K数据拉取异常: %s", e, exc_info=True)

        if not stocks_data:
            stocks_data = self._fallback_daily_from_cache(stocks)

        # ── 2. 分时K线（逐只并发=2）──
        if cfg.get('fetch_intraday', False):
            try:
                period = cfg.get('intraday_period', '5')
                extra['intraday'] = self._run_intraday(stocks, period)
            except Exception as e:
                logging.warning("数据[intraday] 拉取异常: %s，跳过", e)

        # ── 3. 资金流（逐只并发=2）──
        if cfg.get('fetch_fund_flow', False):
            try:
                extra['fund_flow'] = self._run_fund_flow(stocks)
            except Exception as e:
                logging.warning("数据[fund_flow] 拉取异常: %s，跳过", e)

        # ── 4. 快照类数据（单次请求，串行即可）──
        snapshot_tasks = []
        if cfg.get('fetch_sector_flow', False):
            snapshot_tasks.append(('sector_flow', self._fetch_sector_fund_flow))
        if cfg.get('fetch_lhb_detail', False):
            snapshot_tasks.append(('lhb_detail', self._fetch_lhb_detail))
        if cfg.get('fetch_zt_pool', False):
            snapshot_tasks.append(('zt_pool', self._fetch_zt_pool))
            snapshot_tasks.append(('zt_pool_previous', self._fetch_zt_pool_previous))
            snapshot_tasks.append(('zt_pool_strong', self._fetch_zt_pool_strong))
        if cfg.get('fetch_concept_board', False):
            snapshot_tasks.append(('concept_board', self._fetch_concept_board))
        if cfg.get('fetch_hsgt_flow', False):
            snapshot_tasks.append(('hsgt_flow', self._fetch_hsgt_flow))
        if cfg.get('fetch_financial', False):
            snapshot_tasks.append(('financial_report', self._fetch_financial_report))
            snapshot_tasks.append(('financial_balance', self._fetch_financial_balance))
        if cfg.get('fetch_big_deal', False):
            snapshot_tasks.append(('big_deal', self._fetch_big_deal))

        for key, fn in snapshot_tasks:
            try:
                extra[key] = fn()
            except Exception as e:
                logging.warning("数据[%s] 拉取异常: %s，跳过", key, e)

        # ── 5. 概念板块成分股（依赖 concept_board 列表，须在快照之后）──
        if cfg.get('fetch_concept_cons', False):
            try:
                extra['concept_cons'] = self._run_concept_cons(extra)
            except Exception as e:
                logging.warning("数据[concept_cons] 拉取异常: %s，跳过", e)

        # ── 6. 逐只批量类（北向持股、筹码、个股信息）──
        if cfg.get('fetch_hsgt_hold', False):
            try:
                extra['hsgt_hold'] = self._run_hsgt_hold(stocks)
            except Exception as e:
                logging.warning("数据[hsgt_hold] 拉取异常: %s，跳过", e)

        if cfg.get('fetch_chips', False):
            try:
                extra['chips'] = self._run_chips(stocks)
            except Exception as e:
                logging.warning("数据[chips] 拉取异常: %s，跳过", e)

        if cfg.get('fetch_stock_info', False):
            try:
                extra['stock_info'] = self._run_stock_info(stocks)
            except Exception as e:
                logging.warning("数据[stock_info] 拉取异常: %s，跳过", e)

        return stocks_data or {}, extra

    # ══════════════════════════════════════════════
    #  实时行情
    # ══════════════════════════════════════════════

    def _fetch_realtime_quotes(self) -> Optional[pd.DataFrame]:
        if self._cache.is_snapshot_fresh("realtime_quotes"):
            df = self._cache.get_snapshot_latest("realtime_quotes")
            if df is not None and not df.empty:
                logging.info("实时行情: 使用缓存 (%d条)", len(df))
                return df

        for attempt in range(3):
            try:
                df = ak.stock_zh_a_spot_em()
                if df is not None and not df.empty:
                    self._cache.merge_snapshot("realtime_quotes", df)
                    logging.info("实时行情: 东方财富成功 (%d条)", len(df))
                    return df
            except Exception as e:
                logging.warning("拉取实时行情失败 (东方财富, 第%d次): %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))

        try:
            logging.info("切换到新浪实时行情备用接口…")
            df = ak.stock_zh_a_spot()
            if df is not None and not df.empty:
                df = df.copy()
                df['代码'] = df['代码'].str.replace(r'^(sh|sz|bj)', '', regex=True)
                self._cache.merge_snapshot("realtime_quotes", df)
                logging.info("实时行情: 新浪备用成功 (%d条)", len(df))
                return df
        except Exception as e:
            logging.warning("拉取实时行情失败 (新浪备用): %s", e)

        cached = self._cache.get_snapshot_latest("realtime_quotes")
        if cached is not None and not cached.empty:
            logging.warning("实时行情: 使用本地缓存兜底 (%d条)", len(cached))
            return cached
        return None

    # ══════════════════════════════════════════════
    #  日K线（增量缓存 + 双接口 + 缓存兜底）
    # ══════════════════════════════════════════════

    @staticmethod
    def _is_likely_no_new_data(last_cached_date):
        today = date.today()
        diff = (today - last_cached_date).days
        if diff < 0:
            return True
        now_hour = datetime.now().hour
        if diff == 0:
            return now_hour < 15
        weekday_today = today.weekday()
        weekday_cached = last_cached_date.weekday()
        if weekday_cached == 4 and weekday_today in (5, 6):
            return True
        if weekday_cached == 4 and weekday_today == 0 and diff <= 3:
            if now_hour < 15:
                return True
        if diff == 1 and weekday_today < 5 and now_hour < 15:
            return True
        return False

    @staticmethod
    def _sina_symbol(stock_code):
        if stock_code.startswith(('6', '5')):
            return 'sh' + stock_code
        return 'sz' + stock_code

    def _fetch_daily_sina(self, stock, fetch_start, fetch_end=None):
        end_dt = fetch_end or '21001231'
        df = ak.stock_zh_a_daily(
            symbol=self._sina_symbol(stock),
            start_date=fetch_start, end_date=end_dt, adjust='qfq',
        )
        if df is None or df.empty:
            return None

        df = df.rename(columns={
            'date': '日期', 'open': '开盘', 'close': '收盘',
            'high': '最高', 'low': '最低', 'amount': '成交额', 'turnover': '换手率',
        })
        df['日期'] = pd.to_datetime(df['日期']).dt.date
        df['成交量'] = (df['volume'] / 100).round().astype(int)
        df['股票代码'] = stock

        df = df.sort_values('日期').reset_index(drop=True)
        prev_close = df['收盘'].shift(1)
        df['涨跌额'] = (df['收盘'] - prev_close).round(2)
        df['涨跌幅'] = ((df['涨跌额'] / prev_close) * 100).round(2)
        df['振幅'] = (((df['最高'] - df['最低']) / prev_close) * 100).round(2)

        keep = ['日期', '股票代码', '开盘', '收盘', '最高', '最低',
                '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        return df[[c for c in keep if c in df.columns]]

    def _daily_file_saved_post_close(self, stock):
        path = self._cache._parquet_path("daily", stock)
        if not os.path.exists(path):
            return False
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return mtime.date() == date.today() and mtime.hour >= 15

    def _fetch_daily_one(self, code_name, lookback_days=30):
        stock = code_name[0]
        last_cached_date = self._cache.get_daily_last_date(stock)

        if last_cached_date is not None:
            today = date.today()
            diff = (today - last_cached_date).days

            if self._is_likely_no_new_data(last_cached_date):
                data = self._cache.get_daily(stock)
                if data is not None and not data.empty:
                    if 'p_change' not in data.columns:
                        data['p_change'] = tl.ROC(data['收盘'], 1)
                    return data

            if diff == 0 and datetime.now().hour >= 15 and self._daily_file_saved_post_close(stock):
                data = self._cache.get_daily(stock)
                if data is not None and not data.empty:
                    if 'p_change' not in data.columns:
                        data['p_change'] = tl.ROC(data['收盘'], 1)
                    return data

            fetch_origin = (last_cached_date - timedelta(days=lookback_days))
            fetch_start = fetch_origin.strftime('%Y%m%d')
        else:
            fetch_start = "20220101"

        new_data = None
        try:
            new_data = ak.stock_zh_a_hist(
                symbol=stock, period="daily",
                start_date=fetch_start, adjust="qfq"
            )
            if new_data is not None and not new_data.empty:
                logging.info("日K(东财) %s OK, %d 条", stock, len(new_data))
        except Exception as e:
            logging.warning("日K(东财)失败 %s: %s", stock, e)

        if new_data is None or new_data.empty:
            try:
                new_data = self._fetch_daily_sina(stock, fetch_start)
                if new_data is not None and not new_data.empty:
                    logging.warning("日K备用(新浪)成功 %s", stock)
            except Exception as e:
                logging.warning("日K备用(新浪)失败 %s: %s", stock, e)
                new_data = None

        if new_data is None or new_data.empty:
            cached = self._cache.get_daily(stock)
            if cached is not None and not cached.empty:
                if 'p_change' not in cached.columns:
                    cached['p_change'] = tl.ROC(cached['收盘'].astype(float).values, 1)
                return cached
            logging.debug("股票 %s 没有数据，略过", stock)
            return None

        data = self._cache.merge_daily(stock, new_data)
        if data is not None and not data.empty:
            data['p_change'] = tl.ROC(data['收盘'], 1)
        return data

    def _run_daily(self, stocks, lookback_days=30):
        stocks_data = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
            future_to_stock = {
                executor.submit(self._fetch_daily_one, stock, lookback_days): stock
                for stock in stocks
            }
            for future in concurrent.futures.as_completed(future_to_stock):
                stock = future_to_stock[future]
                try:
                    data = future.result()
                    if data is not None:
                        data = data.astype({'成交量': 'float64'})
                        if 'p_change' not in data.columns:
                            data['p_change'] = tl.ROC(data['收盘'].astype(float).values, 1)
                        stocks_data[stock] = data
                except Exception as exc:
                    logging.warning("%s(%s) 日K异常: %s", stock[1], stock[0], exc)

        stats = self._cache.get_cache_stats()
        logging.info("日K加载完成: %d只, 缓存: %s", len(stocks_data), stats)
        return stocks_data

    def _fallback_daily_from_cache(self, stocks):
        logging.warning("日K在线拉取失败，尝试从本地缓存加载...")

        def _load_one(stock):
            df = self._cache.get_daily(stock[0])
            if df is not None and not df.empty:
                df = df.astype({'成交量': 'float64'})
                if 'p_change' not in df.columns:
                    df['p_change'] = tl.ROC(df['收盘'].astype(float).values, 1)
                return stock, df
            return stock, None

        result = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as fb_pool:
            for stock, df in fb_pool.map(_load_one, stocks):
                if df is not None:
                    result[stock] = df

        if result:
            logging.info("从缓存加载日K数据: %d 只", len(result))
        else:
            logging.error("本地缓存也无日K数据")
        return result

    # ══════════════════════════════════════════════
    #  个股资金流
    # ══════════════════════════════════════════════

    @staticmethod
    def _get_exchange(stock_code):
        for market, prefixes in _EXCHANGE_MAP.items():
            for prefix in prefixes:
                if stock_code.startswith(prefix):
                    return market
        return 'sz'

    # 排行榜列名 → 个股历史格式列名（与 stock_individual_fund_flow 保持一致）
    _RANK_COL_MAP = {
        '最新价':              '收盘价',
        '今日涨跌幅':          '涨跌幅',
        '今日主力净流入-净额':  '主力净流入-净额',
        '今日主力净流入-净占比': '主力净流入-净占比',
        '今日超大单净流入-净额':  '超大单净流入-净额',
        '今日超大单净流入-净占比': '超大单净流入-净占比',
        '今日大单净流入-净额':   '大单净流入-净额',
        '今日大单净流入-净占比':  '大单净流入-净占比',
        '今日中单净流入-净额':   '中单净流入-净额',
        '今日中单净流入-净占比':  '中单净流入-净占比',
        '今日小单净流入-净额':   '小单净流入-净额',
        '今日小单净流入-净占比':  '小单净流入-净占比',
    }

    def _fetch_fund_flow_rank_batch(self, stocks):
        """
        用 stock_individual_fund_flow_rank("今日") 一次性拉全市场资金流，
        按股票代码拆分后写入与 _fetch_fund_flow_one 相同的 per-stock 缓存。
        返回 {(code, name): df} 字典；写入失败的股票不出现在返回值中。
        """
        try:
            rank_df = ak.stock_individual_fund_flow_rank(indicator="今日")
        except Exception as e:
            logging.warning("资金流排行榜拉取失败: %s", e)
            return {}

        if rank_df is None or rank_df.empty:
            return {}

        today = date.today().isoformat()
        # 构建 code → name 映射，便于后续拼 key
        code_to_name = {s[0]: s[1] for s in stocks}

        # 重命名列
        rank_df = rank_df.rename(columns=self._RANK_COL_MAP)
        # 确保代码列是纯字符串
        rank_df['代码'] = rank_df['代码'].astype(str).str.zfill(6)

        result = {}
        saved = 0
        for _, row in rank_df.iterrows():
            code = row['代码']
            if code not in code_to_name:
                continue
            # 构造单行 DataFrame，列顺序与 stock_individual_fund_flow 保持一致
            one_row = pd.DataFrame([{
                '日期':             today,
                '收盘价':           pd.to_numeric(row.get('收盘价'), errors='coerce'),
                '涨跌幅':           pd.to_numeric(row.get('涨跌幅'), errors='coerce'),
                '主力净流入-净额':   pd.to_numeric(row.get('主力净流入-净额'), errors='coerce'),
                '主力净流入-净占比': pd.to_numeric(row.get('主力净流入-净占比'), errors='coerce'),
                '超大单净流入-净额':  pd.to_numeric(row.get('超大单净流入-净额'), errors='coerce'),
                '超大单净流入-净占比': pd.to_numeric(row.get('超大单净流入-净占比'), errors='coerce'),
                '大单净流入-净额':   pd.to_numeric(row.get('大单净流入-净额'), errors='coerce'),
                '大单净流入-净占比':  pd.to_numeric(row.get('大单净流入-净占比'), errors='coerce'),
                '中单净流入-净额':   pd.to_numeric(row.get('中单净流入-净额'), errors='coerce'),
                '中单净流入-净占比':  pd.to_numeric(row.get('中单净流入-净占比'), errors='coerce'),
                '小单净流入-净额':   pd.to_numeric(row.get('小单净流入-净额'), errors='coerce'),
                '小单净流入-净占比':  pd.to_numeric(row.get('小单净流入-净占比'), errors='coerce'),
            }])
            merged = self._cache.merge_fund_flow(code, one_row)
            if merged is not None:
                key = (code, code_to_name[code])
                result[key] = merged
                saved += 1

        logging.info("资金流排行榜批量写入缓存: %d 只", saved)
        return result

    def _fetch_fund_flow_one(self, code_name):
        stock = code_name[0]
        if self._cache.is_fund_flow_fresh(stock):
            return self._cache.get_fund_flow(stock)

        market = self._get_exchange(stock)
        try:
            new_df = ak.stock_individual_fund_flow(stock=stock, market=market)
            if new_df is not None and not new_df.empty:
                logging.debug("资金流(个股) %s 拉取成功", stock)
                return self._cache.merge_fund_flow(stock, new_df)
        except Exception as e:
            logging.warning("拉取资金流失败(个股) %s: %s", stock, e)
        return self._cache.get_fund_flow(stock)

    def _run_fund_flow(self, stocks):
        result = {}
        cached_hit = 0
        fetch_ok = 0
        fetch_fail = 0

        # ── 第一步：从缓存直接读已新鲜的股票 ──────────────────────────────
        stale = []
        for s in stocks:
            if self._cache.is_fund_flow_fresh(s[0]):
                df = self._cache.get_fund_flow(s[0])
                if df is not None:
                    result[s] = df
                    cached_hit += 1
                    continue
            stale.append(s)

        if not stale:
            logging.info("资金流全部命中缓存: %d只", cached_hit)
            return result

        # ── 第二步：批量拉取排行榜（一次请求覆盖全市场）────────────────────
        logging.info("资金流缓存未命中 %d 只，尝试批量排行榜接口…", len(stale))
        batch_result = self._fetch_fund_flow_rank_batch(stale)
        result.update(batch_result)
        fetch_ok += len(batch_result)

        # ── 第三步：排行榜未覆盖的股票，逐只用历史接口补拉 ──────────────────
        covered = {k[0] for k in batch_result}
        fallback = [s for s in stale if s[0] not in covered]
        if fallback:
            logging.info("排行榜未覆盖 %d 只，降级逐只拉取…", len(fallback))
            with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
                fmap = {executor.submit(self._fetch_fund_flow_one, s): s for s in fallback}
                for future in concurrent.futures.as_completed(fmap):
                    stock = fmap[future]
                    try:
                        data = future.result()
                        if data is not None:
                            result[stock] = data
                            fetch_ok += 1
                        else:
                            fetch_fail += 1
                    except Exception as exc:
                        fetch_fail += 1
                        logging.debug("资金流(降级) %s(%s): %s", stock[1], stock[0], exc)

        logging.info(
            "资金流加载完成: %d只 (总数%d, 缓存命中%d, 新拉取%d, 失败%d)",
            len(result), len(stocks), cached_hit, fetch_ok, fetch_fail,
        )
        return result

    # ══════════════════════════════════════════════
    #  分时K线（东财 + 新浪备用）
    # ══════════════════════════════════════════════

    def _fetch_intraday_one(self, code_name, period='3'):
        stock = code_name[0]
        if self._cache.is_intraday_fresh(stock):
            return self._cache.get_intraday(stock)

        cached = self._cache.get_intraday(stock)
        start_dt = None
        if cached is not None and not cached.empty and '时间' in cached.columns:
            try:
                latest = pd.to_datetime(cached['时间'].max())
                start_dt = (latest - timedelta(days=3)).strftime('%Y-%m-%d 09:00:00')
            except Exception:
                pass

        df = self._fetch_intraday_em(stock, period, start_dt=start_dt)
        if df is None or df.empty:
            df = self._fetch_intraday_sina(stock, period)

        if df is not None and not df.empty:
            return self._cache.merge_intraday(stock, df)
        return cached

    def _fetch_intraday_em(self, stock, period='5', start_dt=None):
        """东财源 — stock_zh_a_hist_min_em，有缓存时从缓存最新时间往前3天拉取，无缓存时用默认全量"""
        try:
            kwargs = dict(symbol=stock, period=period, adjust='qfq')
            if start_dt is not None:
                kwargs['start_date'] = start_dt
                kwargs['end_date'] = datetime.now().strftime('%Y-%m-%d 15:30:00')
            df = ak.stock_zh_a_hist_min_em(**kwargs)
            if df is not None and not df.empty:
                logging.info("分时(东财) %s OK, %d bars", stock, len(df))
                return df
        except Exception as e:
            logging.warning("分时(东财)失败 %s: %s", stock, e)
        return None

    def _fetch_intraday_sina(self, stock, period='5'):
        """新浪源 — stock_zh_a_minute（历史更长，需 sh/sz 前缀）"""
        try:
            sina_sym = self._sina_symbol(stock)
            df = ak.stock_zh_a_minute(symbol=sina_sym, period=period, adjust='qfq')
            if df is not None and not df.empty:
                df = df.rename(columns={
                    'day': '时间', 'open': '开盘', 'high': '最高',
                    'low': '最低', 'close': '收盘', 'volume': '成交量',
                })
                for col in ('开盘', '最高', '最低', '收盘'):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                if '成交量' in df.columns:
                    df['成交量'] = pd.to_numeric(df['成交量'], errors='coerce').fillna(0).astype(int)
                logging.info("分时(新浪) %s OK, %d bars", stock, len(df))
                return df
        except Exception as e:
            logging.warning("分时(新浪)失败 %s: %s", stock, e)
        return None

    def _run_intraday(self, stocks, period='5'):
        result = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            fmap = {executor.submit(self._fetch_intraday_one, s, period): s for s in stocks}
            for future in concurrent.futures.as_completed(fmap):
                stock = fmap[future]
                try:
                    data = future.result()
                    if data is not None:
                        result[stock] = data
                except Exception as exc:
                    logging.debug("分时 %s(%s): %s", stock[1], stock[0], exc)
        logging.info("分时数据加载完成: %d只", len(result))
        return result

    # ══════════════════════════════════════════════
    #  筹码分布
    # ══════════════════════════════════════════════

    def _fetch_chips_one(self, code_name):
        """拉取单只筹码，返回 (df, status)，status: 'cached'/'fetched'/'fail'。"""
        stock, name = code_name[0], code_name[1]
        if self._cache.is_chips_fresh(stock):
            return self._cache.get_chips(stock), 'cached'

        try:
            df = ak.stock_cyq_em(symbol=stock, adjust='qfq')
            if df is not None and not df.empty:
                return self._cache.merge_chips(stock, df), 'fetched'
        except Exception:
            pass
        return self._cache.get_chips(stock), 'fail'

    def _run_chips(self, stocks):
        result = {}
        cached, fetched, fail = 0, 0, 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
            fmap = {executor.submit(self._fetch_chips_one, s): s for s in stocks}
            for future in concurrent.futures.as_completed(fmap):
                stock = fmap[future]
                try:
                    data, status = future.result()
                    if data is not None:
                        result[stock] = data
                    if status == 'cached':
                        cached += 1
                    elif status == 'fetched':
                        fetched += 1
                    else:
                        fail += 1
                except Exception:
                    fail += 1
        logging.info(
            "筹码分布加载完成: %d只 (缓存命中%d, 新拉取%d, 失败%d)",
            len(result), cached, fetched, fail,
        )
        return result

    # ══════════════════════════════════════════════
    #  全市场快照类数据
    # ══════════════════════════════════════════════

    def _fetch_sector_fund_flow(self, indicator='今日'):
        if self._cache.is_snapshot_fresh("sector_flow"):
            return self._cache.get_snapshot_latest("sector_flow")

        df = self._fetch_sector_flow_em(indicator)
        if df is None or df.empty:
            df = self._fetch_sector_flow_ths()

        if df is not None and not df.empty:
            self._cache.merge_snapshot("sector_flow", df)
            return df
        return self._cache.get_snapshot_latest("sector_flow")

    def _fetch_sector_flow_em(self, indicator='今日'):
        """东财源 — stock_sector_fund_flow_rank"""
        try:
            df = ak.stock_sector_fund_flow_rank(indicator=indicator, sector_type='行业资金流')
            if df is not None and not df.empty:
                logging.debug("行业资金流(东财) OK, %d rows", len(df))
                return df
        except Exception as e:
            logging.warning("行业资金流(东财)失败: %s", e)
        return None

    def _fetch_sector_flow_ths(self):
        """同花顺源 — stock_board_industry_summary_ths（含净流入、涨跌幅等）"""
        try:
            df = ak.stock_board_industry_summary_ths()
            if df is not None and not df.empty:
                logging.debug("行业资金流(同花顺) OK, %d rows", len(df))
                return df
        except Exception as e:
            logging.warning("行业资金流(同花顺)失败: %s", e)
        return None

    def _fetch_lhb_detail(self, days=30):
        if self._cache.is_snapshot_fresh("lhb_detail"):
            return self._cache.get_snapshot("lhb_detail")
        try:
            end_dt = datetime.now().strftime('%Y%m%d')
            start_dt = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            df = ak.stock_lhb_detail_em(start_date=start_dt, end_date=end_dt)
            if df is not None and not df.empty:
                self._cache.merge_snapshot("lhb_detail", df)
                return self._cache.get_snapshot("lhb_detail")
        except Exception as e:
            logging.warning("拉取龙虎榜详情失败: %s", e)
        return self._cache.get_snapshot("lhb_detail")

    @staticmethod
    def _today_str():
        return datetime.now().strftime('%Y%m%d')

    def _fetch_zt_pool(self):
        if self._cache.is_snapshot_fresh("zt_pool"):
            return self._cache.get_snapshot_latest("zt_pool")
        try:
            df = ak.stock_zt_pool_em(date=self._today_str())
            if df is not None and not df.empty:
                self._cache.merge_snapshot("zt_pool", df)
                return df
        except Exception as e:
            logging.warning("拉取涨停池失败: %s", e)
        return self._cache.get_snapshot_latest("zt_pool")

    def _fetch_zt_pool_previous(self):
        if self._cache.is_snapshot_fresh("zt_pool_previous"):
            return self._cache.get_snapshot_latest("zt_pool_previous")
        try:
            df = ak.stock_zt_pool_previous_em(date=self._today_str())
            if df is not None and not df.empty:
                self._cache.merge_snapshot("zt_pool_previous", df)
                return df
        except Exception as e:
            logging.warning("拉取昨日涨停池失败: %s", e)
        return self._cache.get_snapshot_latest("zt_pool_previous")

    def _fetch_zt_pool_strong(self):
        if self._cache.is_snapshot_fresh("zt_pool_strong"):
            return self._cache.get_snapshot_latest("zt_pool_strong")
        try:
            df = ak.stock_zt_pool_strong_em(date=self._today_str())
            if df is not None and not df.empty:
                self._cache.merge_snapshot("zt_pool_strong", df)
                return df
        except Exception as e:
            logging.warning("拉取强势股池失败: %s", e)
        return self._cache.get_snapshot_latest("zt_pool_strong")

    def _fetch_concept_board(self):
        if self._cache.is_snapshot_fresh("concept_board"):
            return self._cache.get_snapshot_latest("concept_board")
        try:
            df = ak.stock_board_concept_name_em()
            if df is not None and not df.empty:
                self._cache.merge_snapshot("concept_board", df)
                return df
        except Exception as e:
            logging.warning("拉取概念板块失败: %s", e)
        return self._cache.get_snapshot_latest("concept_board")

    def _fetch_hsgt_flow(self):
        if self._cache.is_snapshot_fresh("hsgt_flow"):
            return self._cache.get_snapshot_latest("hsgt_flow")
        try:
            df = ak.stock_hsgt_fund_flow_summary_em()
            if df is not None and not df.empty:
                self._cache.merge_snapshot("hsgt_flow", df)
                return df
        except Exception as e:
            logging.warning("拉取北向资金汇总失败: %s", e)
        return self._cache.get_snapshot_latest("hsgt_flow")

    # ══════════════════════════════════════════════
    #  北向持股明细（一次拉全量，按个股缓存）
    # ══════════════════════════════════════════════

    def _fetch_hsgt_hold_batch(self, stocks):
        """调用 stock_hsgt_hold_stock_em 拉取北向全量，按 '代码' 分组写入 per-stock 缓存。"""
        try:
            df = ak.stock_hsgt_hold_stock_em(market="北向", indicator="今日排行")
        except Exception as e:
            logging.warning("北向持股拉取失败: %s", e)
            return {}

        if df is None or df.empty:
            return {}

        df = df.copy()
        df['代码'] = df['代码'].astype(str).str.zfill(6)
        if '日期' not in df.columns:
            df['日期'] = date.today().isoformat()

        code_to_name = {s[0]: s[1] for s in stocks}
        result = {}
        saved = 0
        for code, group in df.groupby('代码'):
            merged = self._cache.merge_hsgt_hold(code, group.reset_index(drop=True))
            if merged is not None:
                name = code_to_name.get(code, '')
                result[(code, name)] = merged
                saved += 1

        logging.info("北向持股批量写入缓存: %d 只", saved)
        return result

    def _run_hsgt_hold(self, stocks):
        """Layer-1 走缓存；缓存过期时整体批量拉取，剩余股票从缓存兜底。"""
        result = {}
        stale = []
        for s in stocks:
            if self._cache.is_hsgt_hold_fresh(s[0]):
                df = self._cache.get_hsgt_hold(s[0])
                if df is not None:
                    result[s] = df
                    continue
            stale.append(s)

        if stale:
            batch = self._fetch_hsgt_hold_batch(stale)
            result.update(batch)
            for s in stale:
                if s not in result:
                    df = self._cache.get_hsgt_hold(s[0])
                    if df is not None:
                        result[s] = df

        logging.info("北向持股加载完成: %d只", len(result))
        return result

    # ══════════════════════════════════════════════
    #  同花顺大单追踪（全市场快照，按成交时间排序）
    # ══════════════════════════════════════════════

    def _fetch_big_deal(self):
        if self._cache.is_snapshot_fresh("big_deal"):
            cached = self._cache.get_snapshot("big_deal")
            if cached is not None and not cached.empty:
                logging.info("大单追踪: 使用缓存 (%d条)", len(cached))
                return cached

        try:
            df = ak.stock_fund_flow_big_deal()
            if df is not None and not df.empty:
                self._cache.merge_snapshot("big_deal", df)
                logging.info("大单追踪: 拉取成功 (%d条)", len(df))
                return self._cache.get_snapshot("big_deal")
        except Exception as e:
            logging.warning("拉取大单追踪失败: %s", e)

        return self._cache.get_snapshot("big_deal")

    # ══════════════════════════════════════════════
    #  个股基本信息（逐只，7天更新一次）
    # ══════════════════════════════════════════════

    def _fetch_stock_info_one(self, code_name):
        stock = code_name[0]
        if self._cache.is_stock_info_fresh(stock):
            return self._cache.get_stock_info(stock)

        try:
            df = ak.stock_individual_info_em(symbol=stock)
            if df is not None and not df.empty:
                self._cache.save_stock_info(stock, df)
                return df
        except Exception as e:
            logging.debug("拉取个股基本信息失败 %s: %s", stock, e)

        return self._cache.get_stock_info(stock)

    def _run_stock_info(self, stocks):
        result = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
            fmap = {executor.submit(self._fetch_stock_info_one, s): s for s in stocks}
            for future in concurrent.futures.as_completed(fmap):
                stock = fmap[future]
                try:
                    data = future.result()
                    if data is not None:
                        result[stock] = data
                except Exception as exc:
                    logging.debug("个股基本信息 %s(%s): %s", stock[1], stock[0], exc)
        logging.info("个股基本信息加载完成: %d只", len(result))
        return result

    # ══════════════════════════════════════════════
    #  财务数据
    # ══════════════════════════════════════════════

    @staticmethod
    def _recent_quarters(n=4):
        today = date.today()
        year, month = today.year, today.month
        all_q = []
        y = year
        while len(all_q) < n:
            for q in ['0930', '0630', '0331', '1231']:
                qy = y if q != '1231' else y - 1
                qd = f"{qy}{q}"
                qm = int(q[:2])
                if qy > year or (qy == year and qm > month):
                    continue
                all_q.append(qd)
                if len(all_q) >= n:
                    break
            y -= 1
        return all_q

    def _fetch_financial_report(self):
        if self._cache.is_snapshot_fresh('financial_report'):
            cached = self._cache.get_snapshot('financial_report')
            if cached is not None and not cached.empty:
                logging.info("业绩报表: 使用缓存 (%d条)", len(cached))
                return cached

        quarters = self._recent_quarters(4)
        frames = []
        for q in quarters:
            try:
                df = ak.stock_yjbb_em(date=q)
                if df is not None and not df.empty:
                    df['报告期'] = q
                    frames.append(df)
                    logging.info("业绩报表 %s: %d 条", q, len(df))
            except Exception as e:
                logging.warning("拉取业绩报表 %s 失败: %s", q, e)

        if not frames:
            return self._cache.get_snapshot('financial_report')

        combined = pd.concat(frames, ignore_index=True)
        self._cache.save_snapshot('financial_report', combined)
        logging.info("业绩报表合计: %d 条", len(combined))
        return combined

    def _fetch_financial_balance(self):
        if self._cache.is_snapshot_fresh('financial_balance'):
            cached = self._cache.get_snapshot('financial_balance')
            if cached is not None and not cached.empty and len(cached) > 100:
                logging.info("资产负债表: 使用缓存 (%d条)", len(cached))
                return cached

        quarters = self._recent_quarters(4)
        for q in quarters:
            try:
                df = ak.stock_zcfz_em(date=q)
                if df is not None and not df.empty and len(df) > 1000:
                    df['报告期'] = q
                    self._cache.save_snapshot('financial_balance', df)
                    logging.info("资产负债表 %s: %d 条", q, len(df))
                    return df
                elif df is not None:
                    logging.info("资产负债表 %s: 仅 %d 条(不完整)，尝试上一期", q, len(df))
            except Exception as e:
                logging.warning("拉取资产负债表 %s 失败: %s", q, e)

        return self._cache.get_snapshot('financial_balance')

    # ══════════════════════════════════════════════
    #  概念板块成分股（按概念缓存，与日K线同周期更新）
    # ══════════════════════════════════════════════

    def _fetch_concept_cons_one(self, concept_name, concept_code):
        """拉取单个概念板块的成分股，优先走缓存。"""
        if self._cache.is_concept_cons_fresh(concept_code):
            return self._cache.get_concept_cons(concept_code)

        try:
            df = ak.stock_board_concept_cons_em(symbol=concept_name)
            if df is not None and not df.empty:
                df['板块名称'] = concept_name
                df['板块代码'] = concept_code
                return self._cache.merge_concept_cons(concept_code, df)
        except Exception as e:
            logging.debug("概念成分股 %s(%s) 拉取失败: %s", concept_name, concept_code, e)

        return self._cache.get_concept_cons(concept_code)

    def _run_concept_cons(self, extra):
        """拉取全部概念板块的成分股，返回 {concept_code: DataFrame}。

        先用 concept_board 快照获取概念列表，再逐概念并发拉取成分股。
        """
        concept_df = extra.get('concept_board')
        if concept_df is None or concept_df.empty:
            concept_df = self._cache.get_snapshot_latest("concept_board")
        if concept_df is None or concept_df.empty:
            logging.warning("概念成分股: 无概念列表，跳过")
            return {}

        tasks = []
        for _, row in concept_df.iterrows():
            name = row.get('板块名称', '')
            code = row.get('板块代码', '')
            if name and code:
                tasks.append((name, code))

        cached_hit = 0
        fetch_needed = []
        result = {}
        for name, code in tasks:
            if self._cache.is_concept_cons_fresh(code):
                df = self._cache.get_concept_cons(code)
                if df is not None:
                    result[code] = df
                    cached_hit += 1
                    continue
            fetch_needed.append((name, code))

        if not fetch_needed:
            logging.info("概念成分股全部命中缓存: %d个", cached_hit)
            return result

        logging.info("概念成分股: 缓存命中 %d, 需拉取 %d", cached_hit, len(fetch_needed))
        fetch_ok = 0
        fetch_fail = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            fmap = {
                executor.submit(self._fetch_concept_cons_one, name, code): (name, code)
                for name, code in fetch_needed
            }
            for future in concurrent.futures.as_completed(fmap):
                name, code = fmap[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        result[code] = data
                        fetch_ok += 1
                    else:
                        fetch_fail += 1
                except Exception as exc:
                    fetch_fail += 1
                    logging.debug("概念成分股 %s(%s) 异常: %s", name, code, exc)

        logging.info(
            "概念成分股加载完成: %d个 (缓存%d, 新拉取%d, 失败%d)",
            len(result), cached_hit, fetch_ok, fetch_fail,
        )
        return result

    # ══════════════════════════════════════════════
    #  热门板块成分股
    # ══════════════════════════════════════════════

    def _fetch_hot_sector_stocks(self, sector_flow_df, top_n=5):
        if sector_flow_df is None or sector_flow_df.empty:
            return {}

        df = sector_flow_df.copy()
        if '快照日期' in df.columns:
            latest = df['快照日期'].max()
            df = df[df['快照日期'] == latest]

        col_change = '今日涨跌幅' if '今日涨跌幅' in df.columns else '涨跌幅'
        col_inflow = '今日主力净流入-净额' if '今日主力净流入-净额' in df.columns else '净流入'
        col_name = '名称' if '名称' in df.columns else '板块'

        if col_change not in df.columns or col_inflow not in df.columns:
            logging.warning("行业资金流列名不匹配, 列: %s", list(df.columns))
            return {}

        df[col_change] = pd.to_numeric(df[col_change], errors='coerce')
        df[col_inflow] = pd.to_numeric(df[col_inflow], errors='coerce')
        df = df[(df[col_change] >= 1.0) & (df[col_inflow] > 0)]
        if df.empty:
            logging.info("今日无符合条件的热门板块")
            return {}

        df = df.copy()
        df['_base'] = df[col_name].apply(lambda n: re.sub(r'[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+$', '', n))
        df = df.sort_values(col_change, ascending=False)
        df = df.drop_duplicates(subset=['_base'], keep='first').head(top_n)

        hot_stocks = {}
        sectors_ok = []
        for _, row in df.iterrows():
            sector_name = row[col_name]
            sector_change = float(row[col_change])
            try:
                cons = ak.stock_board_industry_cons_em(symbol=sector_name)
                if cons is not None and not cons.empty:
                    sectors_ok.append(sector_name)
                    for _, sr in cons.iterrows():
                        code = str(sr.get('代码', ''))
                        if not code:
                            continue
                        hot_stocks[code] = {
                            'sector': sector_name,
                            'sector_change': sector_change,
                            'stock_change': float(sr.get('涨跌幅', 0) or 0),
                        }
            except Exception as e:
                logging.debug("拉取板块成分股失败 %s: %s", sector_name, e)

        logging.info("热门板块成分股加载完成: %d只, 板块: %s", len(hot_stocks), sectors_ok)
        return hot_stocks
