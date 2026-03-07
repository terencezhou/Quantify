# -*- encoding: UTF-8 -*-

import os
import logging
import pandas as pd
from datetime import datetime, timedelta, date


# ──────────────────────────────────────────────
#  内部工具：基于交易时段判断数据新鲜度
# ──────────────────────────────────────────────

def _has_trading_close_between(last_fetch, now):
    """判断 last_fetch 和 now 之间是否存在某个交易日的 15:00 收盘。

    交易日近似为周一至周五（不含法定节假日）。
    """
    d = last_fetch.date()
    end_date = now.date()
    one_day = timedelta(days=1)
    while d <= end_date:
        if d.weekday() < 5:
            session_close = datetime.combine(d, datetime.min.time().replace(hour=15))
            if last_fetch < session_close <= now:
                return True
        d += one_day
    return False


def _is_fetched_at_fresh(fetched_at_col):
    """
    根据 parquet 中 _fetched_at 列的最新值判断数据是否新鲜。

    规则：
      1. 交易日（工作日）盘中（9:15~15:30）运行 → 总是需要更新
         （盘中数据不完整，需要持续刷新）
      2. 否则，检查「最近拉取时间」到「当前时间」之间是否有交易日 15:00 收盘
         - 无 → 新鲜，不用更新
         - 有 → 过期，需要更新
    """
    if fetched_at_col is None:
        return False
    try:
        dts = pd.to_datetime(fetched_at_col).dropna()
        if dts.empty:
            return False
        latest = dts.max()
        if hasattr(latest, 'to_pydatetime'):
            latest = latest.to_pydatetime()

        now = datetime.now()

        in_trading_hours = (
            now.weekday() < 5
            and (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
            and (now.hour < 15 or (now.hour == 15 and now.minute < 30))
        )
        if in_trading_hours:
            return False

        return not _has_trading_close_between(latest, now)
    except Exception:
        return False


class DataCache:
    """本地数据缓存管理器，使用 parquet 文件存储，支持增量更新。

    新鲜度判断原则：
      - 不依赖文件 mtime，只看数据内容本身的时间
      - 分时数据：检查 '时间' 列最新值（内容时间戳）
      - 资金流/筹码/快照：写入时在新行打 _fetched_at，读时检查该列

    目录结构：
        {cache_dir}/
            daily/{stock_code}.parquet       - 日K线
            fund_flow/{stock_code}.parquet   - 个股资金流
            intraday/{stock_code}.parquet    - 分时数据(5分钟)
            chips/{stock_code}.parquet       - 筹码分布
            financial/{stock_code}.parquet   - 财务指标
            *.parquet                        - 全市场快照
    """

    def __init__(self, cache_dir="data"):
        self.cache_dir = cache_dir
        self._ensure_dirs()
        self._daily_last_date_cache = {}

    def _ensure_dirs(self):
        for sub in ["daily", "fund_flow", "intraday", "chips", "financial",
                    "hsgt_hold", "stock_info", "concept_cons"]:
            path = os.path.join(self.cache_dir, sub)
            os.makedirs(path, exist_ok=True)

    def _parquet_path(self, category, filename):
        return os.path.join(self.cache_dir, category, f"{filename}.parquet")

    # ──────────────────────────────────────────────
    #  日K线缓存（增量更新）
    # ──────────────────────────────────────────────

    def get_daily(self, stock_code):
        path = self._parquet_path("daily", stock_code)
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                if not df.empty:
                    return df
            except Exception as e:
                logging.warning(f"读取日K缓存失败 {stock_code}: {e}")
        return None

    def save_daily(self, stock_code, df):
        if df is None or df.empty:
            return
        path = self._parquet_path("daily", stock_code)
        df.to_parquet(path, index=False)
        last = df['日期'].max()
        if isinstance(last, str):
            self._daily_last_date_cache[stock_code] = datetime.strptime(last, '%Y-%m-%d').date()
        elif isinstance(last, datetime):
            self._daily_last_date_cache[stock_code] = last.date()
        elif isinstance(last, date):
            self._daily_last_date_cache[stock_code] = last

    def get_daily_last_date(self, stock_code):
        if stock_code in self._daily_last_date_cache:
            return self._daily_last_date_cache[stock_code]

        path = self._parquet_path("daily", stock_code)
        if not os.path.exists(path):
            return None
        try:
            tbl = pd.read_parquet(path, columns=['日期'])
            if tbl.empty:
                return None
            last = tbl['日期'].max()
            if isinstance(last, str):
                d = datetime.strptime(last, '%Y-%m-%d').date()
            elif isinstance(last, datetime):
                d = last.date()
            elif isinstance(last, date):
                d = last
            else:
                return None
            self._daily_last_date_cache[stock_code] = d
            return d
        except Exception:
            return None

    def merge_daily(self, stock_code, new_df):
        if new_df is None or new_df.empty:
            return self.get_daily(stock_code)

        cached = self.get_daily(stock_code)
        if cached is not None and not cached.empty:
            combined = pd.concat([cached, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['日期'], keep='last')
            combined = combined.sort_values('日期').reset_index(drop=True)
        else:
            combined = new_df.sort_values('日期').reset_index(drop=True)

        self.save_daily(stock_code, combined)
        return combined

    # ──────────────────────────────────────────────
    #  个股资金流缓存
    #  新鲜度：检查 _fetched_at 列（记录实际拉取时刻）
    # ──────────────────────────────────────────────

    def get_fund_flow(self, stock_code):
        path = self._parquet_path("fund_flow", stock_code)
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception as e:
                logging.warning(f"读取资金流缓存失败 {stock_code}: {e}")
        return None

    def is_fund_flow_fresh(self, stock_code):
        """通过 _fetched_at 列判断资金流是否新鲜，不依赖文件 mtime。"""
        df = self.get_fund_flow(stock_code)
        if df is None or df.empty or '_fetched_at' not in df.columns:
            return False
        return _is_fetched_at_fresh(df['_fetched_at'])

    def save_fund_flow(self, stock_code, df):
        if df is None or df.empty:
            return
        path = self._parquet_path("fund_flow", stock_code)
        df.to_parquet(path, index=False)

    def merge_fund_flow(self, stock_code, new_df):
        if new_df is None or new_df.empty:
            return self.get_fund_flow(stock_code)

        new_df = new_df.copy()
        new_df['_fetched_at'] = datetime.now()
        new_df['日期'] = new_df['日期'].astype(str)

        cached = self.get_fund_flow(stock_code)
        if cached is not None and not cached.empty:
            cached['日期'] = cached['日期'].astype(str)
            combined = pd.concat([cached, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['日期'], keep='last')
            combined = combined.sort_values('日期').reset_index(drop=True)
        else:
            combined = new_df.sort_values('日期').reset_index(drop=True)

        self.save_fund_flow(stock_code, combined)
        return combined

    # ──────────────────────────────────────────────
    #  分时数据缓存（合并累积，保留历史数据）
    #  新鲜度：直接检查 '时间' 列最新值（内容即时间戳）
    # ──────────────────────────────────────────────

    def get_intraday(self, stock_code):
        path = self._parquet_path("intraday", stock_code)
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception as e:
                logging.warning(f"读取分时缓存失败 {stock_code}: {e}")
        return None

    def is_intraday_fresh(self, stock_code):
        """
        通过数据内容的 '时间' 列判断分时数据是否新鲜。

        - 盘中（工作日 9:15~15:05）：要求最新 bar 距现在 < 10 分钟
        - 非盘中：只要最近拉取时间之后没有新的交易日收盘，就算新鲜
        """
        df = self.get_intraday(stock_code)
        if df is None or df.empty or '时间' not in df.columns:
            return False
        try:
            latest_dt = pd.to_datetime(df['时间'].max())
            if hasattr(latest_dt, 'to_pydatetime'):
                latest_dt = latest_dt.to_pydatetime()
        except Exception:
            return False

        now = datetime.now()
        in_trading = (
            now.weekday() < 5
            and (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
            and (now.hour < 15 or (now.hour == 15 and now.minute < 5))
        )

        if in_trading:
            if latest_dt.date() != date.today():
                return False
            return (now - latest_dt).total_seconds() < 600

        bar_minutes = latest_dt.hour * 60 + latest_dt.minute
        has_close_bar = bar_minutes >= 14 * 60 + 55
        if has_close_bar and not _has_trading_close_between(latest_dt, now):
            return True
        return False

    def save_intraday(self, stock_code, df):
        if df is None or df.empty:
            return
        path = self._parquet_path("intraday", stock_code)
        df.to_parquet(path, index=False)

    def merge_intraday(self, stock_code, new_df):
        if new_df is None or new_df.empty:
            return self.get_intraday(stock_code)

        cached = self.get_intraday(stock_code)
        if cached is not None and not cached.empty:
            combined = pd.concat([cached, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['时间'], keep='last')
            combined = combined.sort_values('时间').reset_index(drop=True)
        else:
            combined = new_df.sort_values('时间').reset_index(drop=True)

        self.save_intraday(stock_code, combined)
        return combined

    # ──────────────────────────────────────────────
    #  筹码分布缓存（逐个股，合并累积）
    #  新鲜度：检查 _fetched_at 列
    # ──────────────────────────────────────────────

    def get_chips(self, stock_code):
        path = self._parquet_path("chips", stock_code)
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception as e:
                logging.warning(f"读取筹码缓存失败 {stock_code}: {e}")
        return None

    def is_chips_fresh(self, stock_code):
        """通过 _fetched_at 列判断筹码是否新鲜，不依赖文件 mtime。"""
        df = self.get_chips(stock_code)
        if df is None or df.empty or '_fetched_at' not in df.columns:
            return False
        return _is_fetched_at_fresh(df['_fetched_at'])

    def save_chips(self, stock_code, df):
        if df is None or df.empty:
            return
        path = self._parquet_path("chips", stock_code)
        df.to_parquet(path, index=False)

    def merge_chips(self, stock_code, new_df):
        if new_df is None or new_df.empty:
            return self.get_chips(stock_code)

        # 打上本次拉取时刻
        new_df = new_df.copy()
        new_df['_fetched_at'] = datetime.now()

        cached = self.get_chips(stock_code)
        if cached is not None and not cached.empty:
            combined = pd.concat([cached, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['日期'], keep='last')
            combined = combined.sort_values('日期').reset_index(drop=True)
        else:
            combined = new_df.sort_values('日期').reset_index(drop=True)

        self.save_chips(stock_code, combined)
        return combined

    # ──────────────────────────────────────────────
    #  财务指标缓存（逐个股，季度更新，7天新鲜度）
    # ──────────────────────────────────────────────

    def get_financial(self, stock_code):
        path = self._parquet_path("financial", stock_code)
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception as e:
                logging.warning(f"读取财务缓存失败 {stock_code}: {e}")
        return None

    def is_financial_fresh(self, stock_code, max_age_days=7):
        path = self._parquet_path("financial", stock_code)
        if not os.path.exists(path):
            return False
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return (datetime.now() - mtime).days < max_age_days

    def save_financial(self, stock_code, df):
        if df is None or df.empty:
            return
        path = self._parquet_path("financial", stock_code)
        df.to_parquet(path, index=False)

    def merge_financial(self, stock_code, new_df):
        if new_df is None or new_df.empty:
            return self.get_financial(stock_code)

        cached = self.get_financial(stock_code)
        if cached is not None and not cached.empty:
            combined = pd.concat([cached, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['日期'], keep='last')
            combined = combined.sort_values('日期').reset_index(drop=True)
        else:
            combined = new_df.sort_values('日期').reset_index(drop=True)

        self.save_financial(stock_code, combined)
        return combined

    # ──────────────────────────────────────────────
    #  全市场快照类缓存（合并累积，保留历史数据）
    #  新鲜度：检查 _fetched_at 列（写入时自动添加）
    # ──────────────────────────────────────────────

    _SNAPSHOT_KEYS = {
        'realtime_quotes':  {'dedup': ['代码', '快照日期'], 'sort': ['快照日期', '代码']},
        'sector_flow':      {'dedup': ['名称', '快照日期'], 'sort': ['快照日期', '名称']},
        'lhb_detail':       {'dedup': ['代码', '上榜日', '上榜原因'], 'sort': ['上榜日', '代码']},
        'zt_pool':          {'dedup': ['代码', '快照日期'], 'sort': ['快照日期', '代码']},
        'zt_pool_previous': {'dedup': ['代码', '快照日期'], 'sort': ['快照日期', '代码']},
        'zt_pool_strong':   {'dedup': ['代码', '快照日期'], 'sort': ['快照日期', '代码']},
        'concept_board':    {'dedup': ['板块名称', '快照日期'], 'sort': ['快照日期', '排名']},
        'hsgt_flow':        {'dedup': ['交易日', '类型', '板块'], 'sort': ['交易日', '类型']},
        'financial_report':  {'dedup': ['股票代码', '报告期'], 'sort': ['报告期', '股票代码']},
        'financial_balance': {'dedup': ['股票代码', '报告期'], 'sort': ['报告期', '股票代码']},
        'big_deal':         {'dedup': ['成交时间', '股票代码'], 'sort': ['成交时间']},
    }

    def _snapshot_path(self, name):
        return os.path.join(self.cache_dir, f"{name}.parquet")

    def get_snapshot(self, name):
        path = self._snapshot_path(name)
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception as e:
                logging.warning(f"读取快照缓存失败 {name}: {e}")
        return None

    def is_snapshot_fresh(self, name):
        """通过 _fetched_at 列判断快照是否新鲜，基于交易时段逻辑。"""
        df = self.get_snapshot(name)
        if df is None or df.empty or '_fetched_at' not in df.columns:
            return False
        return _is_fetched_at_fresh(df['_fetched_at'])

    def save_snapshot(self, name, df):
        if df is None or df.empty:
            return
        df = df.copy()
        df['_fetched_at'] = datetime.now()
        path = self._snapshot_path(name)
        df.to_parquet(path, index=False)

    def merge_snapshot(self, name, new_df):
        if new_df is None or new_df.empty:
            return self.get_snapshot(name)

        keys = self._SNAPSHOT_KEYS.get(name)
        today_str = date.today().isoformat()

        new_df = new_df.copy()
        # 添加快照日期（用于去重键）
        if keys and '快照日期' in keys['dedup'] and '快照日期' not in new_df.columns:
            new_df['快照日期'] = today_str
        # 打上本次拉取时刻（内容级时间戳）
        new_df['_fetched_at'] = datetime.now()

        cached = self.get_snapshot(name)
        if cached is not None and not cached.empty:
            if keys and '快照日期' in keys['dedup'] and '快照日期' not in cached.columns:
                cached['快照日期'] = today_str
            combined = pd.concat([cached, new_df], ignore_index=True)
            if keys:
                combined = combined.drop_duplicates(subset=keys['dedup'], keep='last')
                combined = combined.sort_values(keys['sort']).reset_index(drop=True)
            else:
                combined = combined.drop_duplicates(keep='last').reset_index(drop=True)
        else:
            combined = new_df.reset_index(drop=True)

        self.save_snapshot(name, combined)
        return combined

    def get_snapshot_latest(self, name):
        df = self.get_snapshot(name)
        if df is not None and not df.empty and '快照日期' in df.columns:
            latest_date = df['快照日期'].max()
            return df[df['快照日期'] == latest_date].copy()
        return df

    # ──────────────────────────────────────────────
    #  北向持股缓存（按个股，按日期增量）
    #  新鲜度：检查 _fetched_at 列（写入时自动添加）
    # ──────────────────────────────────────────────

    def get_hsgt_hold(self, stock_code):
        path = self._parquet_path("hsgt_hold", stock_code)
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception as e:
                logging.warning(f"读取北向持股缓存失败 {stock_code}: {e}")
        return None

    def is_hsgt_hold_fresh(self, stock_code):
        """通过 _fetched_at 列判断北向持股是否新鲜，基于交易时段逻辑。"""
        df = self.get_hsgt_hold(stock_code)
        if df is None or df.empty or '_fetched_at' not in df.columns:
            return False
        return _is_fetched_at_fresh(df['_fetched_at'])

    def save_hsgt_hold(self, stock_code, df):
        if df is None or df.empty:
            return
        path = self._parquet_path("hsgt_hold", stock_code)
        df.to_parquet(path, index=False)

    def merge_hsgt_hold(self, stock_code, new_df):
        if new_df is None or new_df.empty:
            return self.get_hsgt_hold(stock_code)

        new_df = new_df.copy()
        new_df['_fetched_at'] = datetime.now()

        cached = self.get_hsgt_hold(stock_code)
        if cached is not None and not cached.empty:
            combined = pd.concat([cached, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['日期'], keep='last')
            combined = combined.sort_values('日期').reset_index(drop=True)
        else:
            if '日期' in new_df.columns:
                combined = new_df.sort_values('日期').reset_index(drop=True)
            else:
                combined = new_df.reset_index(drop=True)

        self.save_hsgt_hold(stock_code, combined)
        return combined

    # ──────────────────────────────────────────────
    #  个股基本信息缓存（逐只，7天新鲜度）
    #  数据结构：两列 item / value 的键值对表
    # ──────────────────────────────────────────────

    def get_stock_info(self, stock_code):
        path = self._parquet_path("stock_info", stock_code)
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception as e:
                logging.warning(f"读取个股基本信息缓存失败 {stock_code}: {e}")
        return None

    def is_stock_info_fresh(self, stock_code, max_age_days=7):
        """基本面变化慢，7天内不重复拉取。"""
        path = self._parquet_path("stock_info", stock_code)
        if not os.path.exists(path):
            return False
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return (datetime.now() - mtime).days < max_age_days

    def save_stock_info(self, stock_code, df):
        if df is None or df.empty:
            return
        path = self._parquet_path("stock_info", stock_code)
        df.to_parquet(path, index=False)

    # ──────────────────────────────────────────────
    #  概念板块成分股缓存（按概念板块，与日K线同周期更新）
    #  新鲜度：检查 _fetched_at 列
    # ──────────────────────────────────────────────

    def get_concept_cons(self, concept_code):
        path = self._parquet_path("concept_cons", concept_code)
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception as e:
                logging.warning(f"读取概念成分股缓存失败 {concept_code}: {e}")
        return None

    def is_concept_cons_fresh(self, concept_code):
        """通过 _fetched_at 列判断概念成分股是否新鲜，与日K线同逻辑。"""
        df = self.get_concept_cons(concept_code)
        if df is None or df.empty or '_fetched_at' not in df.columns:
            return False
        return _is_fetched_at_fresh(df['_fetched_at'])

    def save_concept_cons(self, concept_code, df):
        if df is None or df.empty:
            return
        path = self._parquet_path("concept_cons", concept_code)
        df.to_parquet(path, index=False)

    def merge_concept_cons(self, concept_code, new_df):
        if new_df is None or new_df.empty:
            return self.get_concept_cons(concept_code)

        new_df = new_df.copy()
        new_df['_fetched_at'] = datetime.now()

        self.save_concept_cons(concept_code, new_df)
        return new_df

    # ──────────────────────────────────────────────
    #  缓存清理
    # ──────────────────────────────────────────────

    def clear_all(self):
        import shutil
        for sub in ["daily", "fund_flow", "intraday", "chips", "financial",
                    "hsgt_hold", "stock_info", "concept_cons"]:
            path = os.path.join(self.cache_dir, sub)
            if os.path.exists(path):
                shutil.rmtree(path)
        for name in list(self._SNAPSHOT_KEYS.keys()):
            path = self._snapshot_path(name)
            if os.path.exists(path):
                os.remove(path)
        self._ensure_dirs()
        self._daily_last_date_cache.clear()
        logging.info("已清除所有缓存")

    def get_cache_stats(self):
        stats = {}
        for sub in ["daily", "fund_flow", "intraday", "chips", "financial",
                    "hsgt_hold", "stock_info", "concept_cons"]:
            path = os.path.join(self.cache_dir, sub)
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if f.endswith('.parquet')]
                stats[sub] = len(files)
            else:
                stats[sub] = 0
        return stats

    # ──────────────────────────────────────────────
    #  详细缓存状态（用于 status 命令）
    # ──────────────────────────────────────────────

    @staticmethod
    def _calc_last_trade_day():
        """计算最近已收盘的交易日（近似，不含节假日）。"""
        today = date.today()
        now_hour = datetime.now().hour
        if today.weekday() < 5 and now_hour >= 15:
            return today
        d = today - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d

    def _detail_daily(self, last_trade_day):
        """日K线：按最新日期判断有效/未更新。"""
        path = os.path.join(self.cache_dir, 'daily')
        total = valid = stale = 0
        latest_date = None
        if not os.path.exists(path):
            return {'total': 0, 'valid': 0, 'stale': 0, 'latest': None}
        for fname in os.listdir(path):
            if not fname.endswith('.parquet'):
                continue
            total += 1
            try:
                tbl = pd.read_parquet(os.path.join(path, fname), columns=['日期'])
                last = tbl['日期'].max()
                if isinstance(last, str):
                    last_d = datetime.strptime(last, '%Y-%m-%d').date()
                elif hasattr(last, 'date'):
                    last_d = last.date()
                else:
                    last_d = last
                if latest_date is None or last_d > latest_date:
                    latest_date = last_d
                if last_d >= last_trade_day:
                    valid += 1
                else:
                    stale += 1
            except Exception:
                stale += 1
        return {
            'total': total, 'valid': valid, 'stale': stale,
            'latest': latest_date.isoformat() if latest_date else None,
        }

    def _detail_per_stock_fetched_at(self, sub):
        """per-stock 缓存：通过 _fetched_at 列判断新鲜度。"""
        path = os.path.join(self.cache_dir, sub)
        total = valid = stale = 0
        latest_dt = None
        if not os.path.exists(path):
            return {'total': 0, 'valid': 0, 'stale': 0, 'latest': None}
        for fname in os.listdir(path):
            if not fname.endswith('.parquet'):
                continue
            total += 1
            try:
                tbl = pd.read_parquet(os.path.join(path, fname), columns=['_fetched_at'])
                ft = pd.to_datetime(tbl['_fetched_at']).max()
                if hasattr(ft, 'to_pydatetime'):
                    ft = ft.to_pydatetime()
                if latest_dt is None or ft > latest_dt:
                    latest_dt = ft
                if _is_fetched_at_fresh(tbl['_fetched_at']):
                    valid += 1
                else:
                    stale += 1
            except Exception:
                stale += 1
        return {
            'total': total, 'valid': valid, 'stale': stale,
            'latest': latest_dt.strftime('%Y-%m-%d %H:%M') if latest_dt else None,
        }

    def _detail_per_stock_mtime(self, sub, max_age_days=7):
        """per-stock 缓存：通过文件 mtime 判断新鲜度（无需读文件内容）。"""
        path = os.path.join(self.cache_dir, sub)
        total = valid = stale = 0
        latest_dt = None
        if not os.path.exists(path):
            return {'total': 0, 'valid': 0, 'stale': 0, 'latest': None}
        now = datetime.now()
        for fname in os.listdir(path):
            if not fname.endswith('.parquet'):
                continue
            total += 1
            fpath = os.path.join(path, fname)
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
            if latest_dt is None or mtime > latest_dt:
                latest_dt = mtime
            if (now - mtime).days < max_age_days:
                valid += 1
            else:
                stale += 1
        return {
            'total': total, 'valid': valid, 'stale': stale,
            'latest': latest_dt.strftime('%Y-%m-%d %H:%M') if latest_dt else None,
        }

    def _detail_snapshot(self, name):
        """全市场快照：返回行数 / 是否新鲜 / 最新拉取时间。"""
        df = self.get_snapshot(name)
        if df is None or df.empty:
            return {'exists': False, 'fresh': False, 'rows': 0, 'latest': None}
        fresh = self.is_snapshot_fresh(name)
        latest_dt = None
        if '_fetched_at' in df.columns:
            ft = pd.to_datetime(df['_fetched_at']).max()
            if hasattr(ft, 'to_pydatetime'):
                ft = ft.to_pydatetime()
            latest_dt = ft
        return {
            'exists': True, 'fresh': fresh, 'rows': len(df),
            'latest': latest_dt.strftime('%Y-%m-%d %H:%M') if latest_dt else None,
        }

    def get_detailed_stats(self):
        """返回详细的缓存状态，供 status 命令使用。"""
        last_trade_day = self._calc_last_trade_day()
        return {
            'last_trade_day': last_trade_day.isoformat(),
            'daily':         self._detail_daily(last_trade_day),
            'fund_flow':     self._detail_per_stock_fetched_at('fund_flow'),
            'intraday':      self._detail_per_stock_mtime('intraday', max_age_days=1),
            'chips':         self._detail_per_stock_mtime('chips', max_age_days=3),
            'hsgt_hold':     self._detail_per_stock_fetched_at('hsgt_hold'),
            'stock_info':    self._detail_per_stock_mtime('stock_info', max_age_days=7),
            'concept_cons':  self._detail_per_stock_fetched_at('concept_cons'),
            'snapshots': {
                name: self._detail_snapshot(name)
                for name in [
                    'zt_pool', 'zt_pool_strong', 'lhb_detail', 'sector_flow',
                    'big_deal', 'hsgt_flow', 'concept_board',
                    'financial_report', 'financial_balance',
                ]
            },
        }
