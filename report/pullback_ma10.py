# -*- encoding: UTF-8 -*-

"""多头排列回踩MA10/MA20缩量阴线 — 趋势回踩买入策略

对全市场日K数据进行扫描，找出满足「多头排列 + 缩量阴线回踩MA10或MA20」条件的股票。

设计文档: docs/pullback_ma10_strategy.md

四大核心条件:
    1. 均线多头排列 + 斜率向上且温和 (MA5>MA10>MA20>MA30, T/T-1单日斜率0.05%~1.5%, slope_ma5<=7%)
    2. 均线粘合度 — MA10距MA20<=7.5%, 距MA30<=10%
    3. 当日缩量阴线 + 收盘在 MA10+2% ~ MA20-2% 区间内（MA10-MA20之间全部放行）
    4. 过去20日有放量上涨 (涨幅>=3% 且 量>=MA20量×1.5)

附加指标:
    - 20日最大回撤: 过去20个交易日中，最低价距MA20的最大下偏幅度

快速模式 (-f):
    使用实时行情数据替代最新一日K线，适合盘中/盘后快速扫描。

数据依赖: DataManager.stocks_data（日K）+ DataManager.all_data（实时行情，仅-f模式）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from data_manager import DataManager
except ModuleNotFoundError:
    from ..data_manager import DataManager  # type: ignore[no-redef]

try:
    from report.stock_group_calc import build_stock_concept_map
except ModuleNotFoundError:
    from stock_group_calc import build_stock_concept_map  # type: ignore[no-redef]


MA_PERIODS = [5, 10, 20, 30]
MIN_DATA_LEN = 50

# ══════════════════════════════════════════════════════════════
#  阈值常量
# ══════════════════════════════════════════════════════════════

SLOPE_MA10_MIN = 0.05     # MA10 单日斜率下限 (%)
SLOPE_MA10_MAX = 1.5      # MA10 单日斜率上限 (%)
SLOPE_MA5_MAX = 7.0       # MA5 斜率上限 (%)
SPREAD_10_20_MAX = 7.5    # MA10-MA20 粘合度上限 (%)
SPREAD_10_30_MAX = 10.0   # MA10-MA30 粘合度上限 (%)
VOL_RATIO_MAX = 0.8       # 缩量比上限
BODY_PCT_MAX = 7.0        # 阴线实体上限 (%)
DIST_MA10_ABOVE_MAX = 2.0  # 回踩区间上界: 收盘高于MA10的最大距离 (%)
DIST_MA20_BELOW_MAX = 2.0  # 回踩区间下界: 收盘低于MA20的最大距离 (%)
SURGE_CHANGE_MIN = 3.0    # 放量上涨涨幅下限 (%)
SURGE_VOL_MULT = 1.5      # 放量上涨量倍数
SURGE_LOOKBACK = 20       # 放量上涨回看天数
RISE_FROM_LOW_MAX = 0.40  # 距60日低点涨幅上限
ALIGN_PERSIST_MIN = 3     # 多头排列最少持续天数
ALIGN_MA20_30_TOL = 0.3   # MA20与MA30接近时的容差 (%): MA20 >= MA30*(1-tol) 视为通过


# ══════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class PullbackItem:
    """单只股票的回踩信号检测结果。"""
    code: str = ''
    name: str = ''
    trade_date: str = ''

    # 价格
    close: float = 0.0
    open_price: float = 0.0
    high: float = 0.0
    low: float = 0.0
    change_pct: float = 0.0

    # 均线值
    ma5: float = 0.0
    ma10: float = 0.0
    ma20: float = 0.0
    ma30: float = 0.0

    # 斜率
    slope_ma10_pct: float = 0.0
    slope_ma5_pct: float = 0.0
    ma_up_count: int = 0

    # 粘合度
    spread_10_20_pct: float = 0.0
    spread_10_30_pct: float = 0.0

    # 回踩质量
    vol_ratio: float = 0.0
    body_pct: float = 0.0
    dist_to_ma10_pct: float = 0.0
    dist_to_ma20_pct: float = 0.0
    pullback_target: str = 'MA10'
    low_touch_ma10: bool = False
    max_drawdown_ma20_pct: float = 0.0

    # 放量历史
    surge_days: int = 0
    max_surge_pct: float = 0.0
    days_since_surge: int = 999

    # 评分
    score_ma_quality: float = 0.0
    score_cohesion: float = 0.0
    score_pullback: float = 0.0
    score_surge_hist: float = 0.0
    score_env: float = 0.0
    total_score: float = 0.0

    grade: str = 'C'
    rise_from_low_pct: float = 0.0
    is_realtime: bool = False

    # 附加信号
    top_divergence: bool = False
    divergence_detail: str = ''
    triangle_support: bool = False
    triangle_detail: str = ''

    # 概念/板块
    concepts: List[str] = field(default_factory=list)

    @property
    def grade_label(self) -> str:
        return {'S': 'S — 极佳', 'A': 'A — 良好',
                'B': 'B — 可关注', 'C': 'C — 较弱'}.get(self.grade, self.grade)


@dataclass
class PullbackScanResult:
    """全市场扫描结果。"""
    trade_date: str = ''
    total_scanned: int = 0
    items: List[PullbackItem] = field(default_factory=list)
    is_fast_mode: bool = False

    @property
    def s_items(self) -> List[PullbackItem]:
        return [x for x in self.items if x.grade == 'S']

    @property
    def a_items(self) -> List[PullbackItem]:
        return [x for x in self.items if x.grade == 'A']

    @property
    def b_items(self) -> List[PullbackItem]:
        return [x for x in self.items if x.grade == 'B']


# ══════════════════════════════════════════════════════════════
#  均线计算工具
# ══════════════════════════════════════════════════════════════

def _compute_ma(arr: np.ndarray, period: int) -> np.ndarray:
    """使用 cumsum 快速计算简单移动平均。"""
    if len(arr) < period:
        return np.full(len(arr), np.nan)
    cumsum = np.cumsum(arr)
    cumsum = np.insert(cumsum, 0, 0.0)
    ma = (cumsum[period:] - cumsum[:-period]) / period
    out = np.empty(len(arr))
    out[:] = np.nan
    out[period - 1:] = ma
    return out


def _compute_all_ma(closes: np.ndarray) -> Dict[int, np.ndarray]:
    return {p: _compute_ma(closes, p) for p in MA_PERIODS}


# ══════════════════════════════════════════════════════════════
#  MACD 顶背离检测
# ══════════════════════════════════════════════════════════════

DIVERGENCE_LOOKBACK = 40   # 回看窗口 (交易日)
PEAK_NEIGHBOR = 3          # 局部极大值: 需高于前后各 N 日
PEAK_MIN_GAP = 5           # 两个波峰最小间隔 (交易日)
DIFF_PEAK_SEARCH = 5       # 在价格峰附近 ±N 日内找 DIFF 峰


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """指数移动平均 (与 pandas ewm 一致)。"""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _compute_macd(closes: np.ndarray):
    """返回 (diff_line, dea_line, macd_hist)。
    diff = EMA12 - EMA26, dea = EMA9(diff), hist = diff - dea。
    """
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    diff = ema12 - ema26
    dea = _ema(diff, 9)
    hist = diff - dea
    return diff, dea, hist


def _find_local_peaks(
    arr: np.ndarray, pos: int, lookback: int, neighbor: int,
) -> List[int]:
    """在 [pos-lookback, pos] 范围内找局部极大值。"""
    start = max(neighbor, pos - lookback)
    end = min(pos, len(arr) - neighbor - 1)
    peaks = []
    for i in range(start, end + 1):
        window_left = arr[i - neighbor:i]
        window_right = arr[i + 1:i + neighbor + 1]
        if len(window_left) == 0 or len(window_right) == 0:
            continue
        if arr[i] >= np.max(window_left) and arr[i] >= np.max(window_right):
            peaks.append(i)
    return peaks


def _filter_peaks_by_gap(peaks: List[int], min_gap: int) -> List[int]:
    """去除间距不足的波峰，保留靠后的那个。"""
    if len(peaks) <= 1:
        return peaks
    filtered: List[int] = [peaks[0]]
    for p in peaks[1:]:
        if p - filtered[-1] >= min_gap:
            filtered.append(p)
    return filtered


def _find_nearby_diff_peak(
    diff: np.ndarray, center: int, radius: int,
) -> Tuple[int, float]:
    """在 center ± radius 范围内找 DIFF 的最大值位置和值。"""
    lo = max(0, center - radius)
    hi = min(len(diff) - 1, center + radius)
    seg = diff[lo:hi + 1]
    offset = int(np.argmax(seg))
    idx = lo + offset
    return idx, diff[idx]


def detect_top_divergence(
    closes: np.ndarray, pos: int,
    lookback: int = DIVERGENCE_LOOKBACK,
) -> Tuple[bool, str]:
    """检测 pos 之前是否存在 MACD-DIFF 顶背离。

    算法:
        1. 用收盘价找两个近期价格波峰
        2. 在每个价格峰附近 ±DIFF_PEAK_SEARCH 日找 DIFF 线的峰值
        3. 价格更高但 DIFF 峰更低 → 顶背离

    Returns:
        (is_divergence, detail_str)
    """
    if pos < 30:
        return False, ''

    diff, dea, hist = _compute_macd(closes)

    price_peaks = _find_local_peaks(closes, pos, lookback, PEAK_NEIGHBOR)
    price_peaks = _filter_peaks_by_gap(price_peaks, PEAK_MIN_GAP)

    if len(price_peaks) < 2:
        return False, ''

    peak_a, peak_b = price_peaks[-2], price_peaks[-1]

    diff_idx_a, diff_val_a = _find_nearby_diff_peak(diff, peak_a, DIFF_PEAK_SEARCH)
    diff_idx_b, diff_val_b = _find_nearby_diff_peak(diff, peak_b, DIFF_PEAK_SEARCH)

    price_higher = closes[peak_b] >= closes[peak_a] * 0.99
    diff_lower = diff_val_b < diff_val_a * 0.90

    is_div = price_higher and diff_lower
    detail = (
        f"价格峰1(T-{pos - peak_a})={closes[peak_a]:.2f}/DIFF(T-{pos - diff_idx_a})={diff_val_a:.3f}"
        f" vs 价格峰2(T-{pos - peak_b})={closes[peak_b]:.2f}/DIFF(T-{pos - diff_idx_b})={diff_val_b:.3f}"
    )
    return is_div, detail


# ══════════════════════════════════════════════════════════════
#  三角架托检测
# ══════════════════════════════════════════════════════════════

TRIANGLE_LOOKBACK = 40     # 回看窗口 (交易日)
TRIANGLE_NEAR_PCT = 0.5    # MA10 "即将上穿" MA20 的距离阈值 (%)


def _find_cross_up(fast: np.ndarray, slow: np.ndarray,
                   start: int, end: int) -> int:
    """在 [start, end] 中找 fast 上穿 slow 的最早位置，未找到返回 -1。"""
    for i in range(max(start, 1), end + 1):
        if fast[i] > slow[i] and fast[i - 1] <= slow[i - 1]:
            return i
    return -1


def _find_cross_up_last(fast: np.ndarray, slow: np.ndarray,
                        start: int, end: int) -> int:
    """在 [start, end] 中找 fast 上穿 slow 的最晚位置，未找到返回 -1。"""
    for i in range(end, max(start - 1, 0), -1):
        if i < 1:
            break
        if fast[i] > slow[i] and fast[i - 1] <= slow[i - 1]:
            return i
    return -1


def detect_triangle_support(
    ma_dict: Dict[int, np.ndarray], pos: int,
    lookback: int = TRIANGLE_LOOKBACK,
) -> Tuple[bool, str]:
    """检测 pos 之前距今最近的一次三角架托形态。

    从最后一次 MA10×MA20 金叉倒推，找与之匹配的最近 MA5×MA20、MA5×MA10。

    三角架托 = 三次金叉依次出现:
        1) MA5 上穿 MA10
        2) MA5 上穿 MA20 (在 1 之后)
        3) MA10 上穿 MA20 (在 2 之后)，或 MA10 距 MA20 <= TRIANGLE_NEAR_PCT%

    Returns:
        (is_triangle, detail_str)
    """
    ma5 = ma_dict[5]
    ma10 = ma_dict[10]
    ma20 = ma_dict[20]

    start = max(1, pos - lookback)

    # 倒序查找：先找最近的 MA10×MA20 金叉（或即将上穿），再往前匹配
    cross3 = _find_cross_up_last(ma10, ma20, start, pos)

    if cross3 >= 0:
        # 在 cross3 之前找最近的 MA5×MA20
        cross2 = _find_cross_up_last(ma5, ma20, start, cross3)
        if cross2 >= 0:
            # 在 cross2 之前找最近的 MA5×MA10
            cross1 = _find_cross_up_last(ma5, ma10, start, cross2)
            if cross1 >= 0:
                detail = (
                    f"MA5×MA10(T-{pos - cross1})"
                    f" → MA5×MA20(T-{pos - cross2})"
                    f" → MA10×MA20(T-{pos - cross3})"
                )
                return True, detail

    # 未找到已完成的架托，检查"即将上穿"
    ma10_val = ma10[pos]
    ma20_val = ma20[pos]
    if (not np.isnan(ma10_val) and not np.isnan(ma20_val)
            and ma20_val > 0):
        gap_pct = (ma20_val - ma10_val) / ma20_val * 100
        if 0 < gap_pct <= TRIANGLE_NEAR_PCT:
            cross2 = _find_cross_up_last(ma5, ma20, start, pos)
            if cross2 >= 0:
                cross1 = _find_cross_up_last(ma5, ma10, start, cross2)
                if cross1 >= 0:
                    detail = (
                        f"MA5×MA10(T-{pos - cross1})"
                        f" → MA5×MA20(T-{pos - cross2})"
                        f" → MA10即将上穿MA20(差{gap_pct:.2f}%)"
                    )
                    return True, detail

    return False, ''


# ══════════════════════════════════════════════════════════════
#  过滤
# ══════════════════════════════════════════════════════════════

def _is_excluded(code: str, name: str) -> bool:
    if 'ST' in name:
        return True
    if code.startswith(('8', '4')):
        return True
    if code.startswith('68'):
        return True
    if code.startswith('3'):
        return True
    return False


def _is_bullish_aligned(ma5: float, ma10: float, ma20: float, ma30: float) -> bool:
    """多头排列判断: MA5>MA10>MA20>MA30，MA20与MA30接近时允许容差。"""
    if not (ma5 > ma10 > ma20):
        return False
    if ma20 > ma30:
        return True
    return ma20 >= ma30 * (1 - ALIGN_MA20_30_TOL / 100)


# ══════════════════════════════════════════════════════════════
#  评分函数
# ══════════════════════════════════════════════════════════════

def _score_ma_quality(ma_up_count: int, slope_ma10: float, slope_ma5: float) -> float:
    """A. 均线质量分 (0-100)。"""
    if ma_up_count >= 4:
        base = 50.0
    elif ma_up_count == 3:
        base = 35.0
    elif ma_up_count == 2:
        base = 20.0
    else:
        base = 5.0

    if 0.05 <= slope_ma10 <= 0.3:
        base += 30
    elif slope_ma10 <= 0.6:
        base += 25
    elif slope_ma10 <= 1.0:
        base += 15
    elif slope_ma10 <= 1.5:
        base += 5

    if slope_ma5 > 0:
        base += 5
    elif slope_ma5 > -0.5:
        base += 3

    if slope_ma5 <= 2.0:
        base += 10
    elif slope_ma5 <= 4.0:
        base += 5

    return min(max(base, 0), 100)


def _score_cohesion(spread_10_20: float, spread_10_30: float,
                    spread_20_30: float) -> float:
    """B. 均线粘合度分 (0-100)。"""
    if spread_10_20 <= 1.0:
        s1 = 45
    elif spread_10_20 <= 2.0:
        s1 = 40
    elif spread_10_20 <= 3.0:
        s1 = 30
    elif spread_10_20 <= 4.0:
        s1 = 20
    else:
        s1 = 10

    if spread_10_30 <= 2.0:
        s2 = 45
    elif spread_10_30 <= 4.0:
        s2 = 40
    elif spread_10_30 <= 6.0:
        s2 = 30
    else:
        s2 = 15

    bonus = 10 if spread_20_30 <= 2.0 else 0
    return min(s1 + s2 + bonus, 100)


def _score_pullback(vol_ratio: float, body_pct: float,
                    dist_abs: float, low_touch: bool) -> float:
    """C. 回踩质量分 (0-100)。"""
    if vol_ratio < 0.4:
        sv = 40
    elif vol_ratio < 0.6:
        sv = 30
    else:
        sv = 20

    if body_pct < 0.5:
        sb = 20
    elif body_pct < 1.0:
        sb = 15
    elif body_pct < 2.0:
        sb = 10
    else:
        sb = 5

    if dist_abs < 0.5:
        sd = 30
    elif dist_abs < 1.0:
        sd = 25
    elif dist_abs < 1.5:
        sd = 15
    else:
        sd = 5

    bonus = 10 if low_touch else 0
    return min(sv + sb + sd + bonus, 100)


def _score_surge_hist(surge_days: int, max_surge_pct: float,
                      days_since: int) -> float:
    """D. 放量历史分 (0-100)。"""
    if surge_days >= 3:
        s1 = 50
    elif surge_days == 2:
        s1 = 40
    elif surge_days == 1:
        s1 = 25
    else:
        return 0

    if max_surge_pct >= 7:
        s2 = 30
    elif max_surge_pct >= 5:
        s2 = 25
    else:
        s2 = 15

    if days_since <= 5:
        s3 = 20
    elif days_since <= 10:
        s3 = 15
    elif days_since <= 15:
        s3 = 10
    else:
        s3 = 5

    return min(s1 + s2 + s3, 100)


def _score_env(market_score: float) -> float:
    """E. 环境加分 (0-100)。"""
    if market_score >= 50:
        return min(market_score, 100)
    return market_score * 0.6


def _grade_from_score(score: float) -> str:
    if score >= 75:
        return 'S'
    if score >= 60:
        return 'A'
    if score >= 45:
        return 'B'
    return 'C'


# ══════════════════════════════════════════════════════════════
#  核心筛选器
# ══════════════════════════════════════════════════════════════

class PullbackMA10Screener:
    """多头排列回踩MA10缩量阴线策略筛选器。"""

    def __init__(self, data_manager: DataManager, fast_mode: bool = False):
        self.dm = data_manager
        self.fast_mode = fast_mode
        self._rt_lookup: Optional[Dict[str, pd.Series]] = None

    # ── 公开入口 ──

    def run(self, market_score: float = 50.0) -> PullbackScanResult:
        """扫描全市场，返回符合条件的回踩信号。"""
        result = PullbackScanResult(is_fast_mode=self.fast_mode)
        result.trade_date = self._get_trade_date()

        if self.fast_mode:
            self._build_rt_lookup()

        concept_map = build_stock_concept_map(
            self.dm.extra.get('concept_cons', {})
        )

        for key, df in self.dm.stocks_data.items():
            result.total_scanned += 1
            item = self._screen_one(key, df, market_score)
            if item is not None:
                item.concepts = concept_map.get(item.code, [])
                result.items.append(item)

        result.items.sort(key=lambda x: x.total_score, reverse=True)
        logging.info(
            'PullbackMA10: 扫描 %d 只, 信号 %d 只 (S=%d A=%d B=%d)',
            result.total_scanned, len(result.items),
            len(result.s_items), len(result.a_items), len(result.b_items),
        )
        return result

    # ── 单股筛选 ──

    def _screen_one(
        self, key: Tuple, df: pd.DataFrame, market_score: float,
    ) -> Optional[PullbackItem]:
        if df is None or len(df) < MIN_DATA_LEN:
            return None

        code, name = str(key[0]).zfill(6), str(key[1])
        if _is_excluded(code, name):
            return None

        df = self._prepare_df(code, df)
        if df is None or len(df) < MIN_DATA_LEN:
            return None

        pos = len(df) - 1
        closes = df['收盘'].values.astype(float)
        opens = df['开盘'].values.astype(float)
        highs = df['最高'].values.astype(float)
        lows = df['最低'].values.astype(float)
        volumes = df['成交量'].values.astype(float)
        changes = df['涨跌幅'].values.astype(float) if '涨跌幅' in df.columns else np.zeros(len(df))

        ma_dict = _compute_all_ma(closes)

        # 提取均线值
        ma_vals = {}
        for p in MA_PERIODS:
            v = ma_dict[p][pos]
            if np.isnan(v) or v <= 0:
                return None
            ma_vals[p] = v

        # ── 条件 1: 均线多头排列 + 斜率温和 ──
        if not _is_bullish_aligned(ma_vals[5], ma_vals[10], ma_vals[20], ma_vals[30]):
            return None

        # MA10 斜率 (T vs T-1)
        ma10_prev = ma_dict[10][pos - 1]
        if np.isnan(ma10_prev) or ma10_prev <= 0:
            return None
        slope_ma10 = (ma_vals[10] - ma10_prev) / ma10_prev * 100
        if not (SLOPE_MA10_MIN <= slope_ma10 <= SLOPE_MA10_MAX):
            return None

        # MA5 斜率 (3日)
        if pos < 3:
            return None
        ma5_3ago = ma_dict[5][pos - 3]
        if np.isnan(ma5_3ago) or ma5_3ago <= 0:
            return None
        slope_ma5 = (ma_vals[5] - ma5_3ago) / ma5_3ago * 100
        if slope_ma5 > SLOPE_MA5_MAX:
            return None

        # 均线方向数
        ma_up_count = 0
        for p in MA_PERIODS:
            prev_val = ma_dict[p][pos - 1]
            if not np.isnan(prev_val) and ma_dict[p][pos] > prev_val:
                ma_up_count += 1
        if ma_up_count < 2:
            return None

        # ── 条件 2: 均线粘合度 ──
        spread_10_20 = (ma_vals[10] - ma_vals[20]) / ma_vals[20] * 100
        spread_10_30 = (ma_vals[10] - ma_vals[30]) / ma_vals[30] * 100
        if spread_10_20 > SPREAD_10_20_MAX or spread_10_30 > SPREAD_10_30_MAX:
            return None
        spread_20_30 = (ma_vals[20] - ma_vals[30]) / ma_vals[30] * 100

        # ── 条件 3: 缩量阴线 + 回踩MA10 ──
        close_val = closes[pos]
        open_val = opens[pos]
        low_val = lows[pos]
        vol_today = volumes[pos]

        is_yin = close_val < open_val
        if not is_yin:
            return None

        body_pct = (open_val - close_val) / close_val * 100 if close_val > 0 else 0
        if body_pct > BODY_PCT_MAX:
            return None

        ma5_vol = np.nanmean(volumes[max(0, pos - 4):pos]) if pos >= 1 else vol_today
        vol_ratio = vol_today / ma5_vol if ma5_vol > 0 else 999
        if vol_ratio >= VOL_RATIO_MAX:
            return None

        dist_to_ma10 = (close_val - ma_vals[10]) / ma_vals[10] * 100
        dist_to_ma20 = (close_val - ma_vals[20]) / ma_vals[20] * 100

        upper_bound = ma_vals[10] * (1 + DIST_MA10_ABOVE_MAX / 100)
        lower_bound = ma_vals[20] * (1 - DIST_MA20_BELOW_MAX / 100)
        in_pullback_zone = lower_bound <= close_val <= upper_bound
        if not in_pullback_zone:
            return None

        if abs(close_val - ma_vals[10]) <= abs(close_val - ma_vals[20]):
            pullback_target = 'MA10'
        else:
            pullback_target = 'MA20'
        target_ma_val = ma_vals[10] if pullback_target == 'MA10' else ma_vals[20]
        dist_to_target = dist_to_ma10 if pullback_target == 'MA10' else dist_to_ma20

        low_touch = low_val < target_ma_val and close_val >= target_ma_val * 0.99

        # ── 条件 4: 过去20日有放量上涨 ──
        surge_days, max_surge_pct, days_since = self._count_surge_days(
            changes, volumes, pos,
        )
        if surge_days < 1:
            return None

        # ── 硬性否决 ──
        # MA20 方向向下
        ma20_prev = ma_dict[20][pos - 1]
        if not np.isnan(ma20_prev) and ma_vals[20] < ma20_prev:
            return None

        # 近 5 日有跌停
        if pos >= 5:
            recent_changes = changes[pos - 4:pos + 1]
            if np.any(recent_changes <= -9.5):
                return None

        # 均线排列持续不足 3 天
        if not self._alignment_persisted(ma_dict, pos, ALIGN_PERSIST_MIN):
            return None

        # 距 60 日低点涨幅过大
        rise_from_low = self._rise_from_low(lows, closes, pos, 60)
        if rise_from_low > RISE_FROM_LOW_MAX:
            return None

        # ── 20日最大回撤(距MA20) ──
        max_dd_ma20 = self._max_drawdown_from_ma20(lows, ma_dict[20], pos, 20)

        # ── 附加信号检测（仅标注，不影响筛选） ──
        is_div, div_detail = detect_top_divergence(closes, pos)
        is_tri, tri_detail = detect_triangle_support(ma_dict, pos)

        # ── 评分 ──
        s_a = _score_ma_quality(ma_up_count, slope_ma10, slope_ma5)
        s_b = _score_cohesion(spread_10_20, spread_10_30, spread_20_30)
        s_c = _score_pullback(vol_ratio, body_pct, abs(dist_to_target), low_touch)
        s_d = _score_surge_hist(surge_days, max_surge_pct, days_since)
        s_e = _score_env(market_score)

        total = round(
            s_a * 0.25 + s_b * 0.20 + s_c * 0.25 + s_d * 0.20 + s_e * 0.10,
            1,
        )
        grade = _grade_from_score(total)

        # 只返回 B 级及以上
        if grade == 'C':
            return None

        trade_date_str = str(df.iloc[pos]['日期'])[:10]
        is_rt = self.fast_mode and self._is_realtime_row(df, pos)

        return PullbackItem(
            code=code, name=name, trade_date=trade_date_str,
            close=round(close_val, 2), open_price=round(open_val, 2),
            high=round(highs[pos], 2), low=round(low_val, 2),
            change_pct=round(float(changes[pos]), 2),
            ma5=round(ma_vals[5], 2), ma10=round(ma_vals[10], 2),
            ma20=round(ma_vals[20], 2), ma30=round(ma_vals[30], 2),
            slope_ma10_pct=round(slope_ma10, 2),
            slope_ma5_pct=round(slope_ma5, 2),
            ma_up_count=ma_up_count,
            spread_10_20_pct=round(spread_10_20, 2),
            spread_10_30_pct=round(spread_10_30, 2),
            vol_ratio=round(vol_ratio, 2),
            body_pct=round(body_pct, 2),
            dist_to_ma10_pct=round(dist_to_ma10, 2),
            dist_to_ma20_pct=round(dist_to_ma20, 2),
            pullback_target=pullback_target,
            low_touch_ma10=low_touch,
            max_drawdown_ma20_pct=round(max_dd_ma20, 1),
            surge_days=surge_days,
            max_surge_pct=round(max_surge_pct, 1),
            days_since_surge=days_since,
            score_ma_quality=round(s_a, 1),
            score_cohesion=round(s_b, 1),
            score_pullback=round(s_c, 1),
            score_surge_hist=round(s_d, 1),
            score_env=round(s_e, 1),
            total_score=total,
            grade=grade,
            rise_from_low_pct=round(rise_from_low * 100, 1),
            is_realtime=is_rt,
            top_divergence=is_div,
            divergence_detail=div_detail,
            triangle_support=is_tri,
            triangle_detail=tri_detail,
        )

    # ── 放量上涨统计 ──

    @staticmethod
    def _count_surge_days(
        changes: np.ndarray, volumes: np.ndarray, pos: int,
    ) -> Tuple[int, float, int]:
        lookback = min(SURGE_LOOKBACK, pos)
        if lookback < 5:
            return 0, 0.0, 999

        start = pos - lookback
        end = pos  # 不含当日（当日是阴线）

        ma20_vol_arr = _compute_ma(volumes, 20)

        surge_indices = []
        max_pct = 0.0
        for i in range(start, end):
            ma20_v = ma20_vol_arr[i]
            if np.isnan(ma20_v) or ma20_v <= 0:
                continue
            if changes[i] >= SURGE_CHANGE_MIN and volumes[i] >= ma20_v * SURGE_VOL_MULT:
                surge_indices.append(i)
                if changes[i] > max_pct:
                    max_pct = changes[i]

        count = len(surge_indices)
        if count > 0:
            days_since = pos - surge_indices[-1]
        else:
            days_since = 999

        return count, max_pct, days_since

    # ── 多头排列持续性检查 ──

    @staticmethod
    def _alignment_persisted(
        ma_dict: Dict[int, np.ndarray], pos: int, min_days: int,
    ) -> bool:
        for i in range(pos - min_days + 1, pos + 1):
            if i < 0:
                return False
            vals = []
            for p in MA_PERIODS:
                v = ma_dict[p][i]
                if np.isnan(v):
                    return False
                vals.append(v)
            if not _is_bullish_aligned(vals[0], vals[1], vals[2], vals[3]):
                return False
        return True

    # ── 20日最大回撤(距MA20) ──

    @staticmethod
    def _max_drawdown_from_ma20(
        lows: np.ndarray, ma20_arr: np.ndarray, pos: int, days: int,
    ) -> float:
        """过去 N 日内最低价距 MA20 的最大下偏幅度 (%)，返回负值表示跌破。"""
        start = max(0, pos - days + 1)
        worst = 0.0
        for i in range(start, pos + 1):
            ma20_val = ma20_arr[i]
            if np.isnan(ma20_val) or ma20_val <= 0:
                continue
            dd = (lows[i] - ma20_val) / ma20_val * 100
            if dd < worst:
                worst = dd
        return worst

    # ── 距低点涨幅 ──

    @staticmethod
    def _rise_from_low(
        lows: np.ndarray, closes: np.ndarray, pos: int, days: int,
    ) -> float:
        start = max(0, pos - days + 1)
        low_min = np.nanmin(lows[start:pos + 1])
        if low_min <= 0 or np.isnan(low_min):
            return 0.0
        return (closes[pos] - low_min) / low_min

    # ── 数据准备 ──

    def _prepare_df(self, code: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """准备日K数据，快速模式下追加实时行情行。"""
        df = df.copy()
        df['日期'] = df['日期'].astype(str).str[:10]

        required = ['收盘', '开盘', '最高', '最低', '成交量']
        for col in required:
            if col not in df.columns:
                return None
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if '涨跌幅' in df.columns:
            df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
        else:
            df['涨跌幅'] = 0.0

        df = df.dropna(subset=['收盘']).sort_values('日期').reset_index(drop=True)

        if self.fast_mode:
            df = self._augment_with_realtime(code, df)

        return df

    def _build_rt_lookup(self):
        if self._rt_lookup is not None:
            return
        self._rt_lookup = {}
        if self.dm.all_data is not None:
            for _, row in self.dm.all_data.iterrows():
                c = str(row.get('代码', '')).zfill(6)
                self._rt_lookup[c] = row

    def _augment_with_realtime(
        self, code: str, df: pd.DataFrame,
    ) -> pd.DataFrame:
        """快速模式：如果K线缺少最新交易日数据，用实时行情补充。"""
        if self._rt_lookup is None:
            return df

        rt = self._rt_lookup.get(code)
        if rt is None:
            return df

        latest_price = pd.to_numeric(rt.get('最新价', 0), errors='coerce') or 0
        today_open = pd.to_numeric(rt.get('今开', 0), errors='coerce') or 0
        if latest_price <= 0 or today_open <= 0:
            return df

        today = date.today()
        if today.weekday() >= 5:
            return df

        last_kline_date_str = str(df.iloc[-1]['日期'])[:10]
        try:
            last_kline_date = datetime.strptime(last_kline_date_str, '%Y-%m-%d').date()
        except ValueError:
            return df

        if last_kline_date >= today:
            return df

        new_row = pd.DataFrame([{
            '日期': today.isoformat(),
            '开盘': today_open,
            '收盘': latest_price,
            '最高': pd.to_numeric(rt.get('最高', latest_price), errors='coerce') or latest_price,
            '最低': pd.to_numeric(rt.get('最低', latest_price), errors='coerce') or latest_price,
            '成交量': pd.to_numeric(rt.get('成交量', 0), errors='coerce') or 0,
            '涨跌幅': pd.to_numeric(rt.get('涨跌幅', 0), errors='coerce') or 0,
            '_realtime': True,
        }])

        df = pd.concat([df, new_row], ignore_index=True)
        return df

    @staticmethod
    def _is_realtime_row(df: pd.DataFrame, pos: int) -> bool:
        if '_realtime' in df.columns:
            return bool(df.iloc[pos].get('_realtime', False))
        return False

    # ── 工具 ──

    def _get_trade_date(self) -> str:
        for _key, df in self.dm.stocks_data.items():
            if df is not None and not df.empty and '日期' in df.columns:
                last_date = str(df['日期'].iloc[-1])[:10]
                if self.fast_mode and date.today().weekday() < 5:
                    today_str = date.today().isoformat()
                    if today_str > last_date:
                        return today_str
                return last_date
        return date.today().isoformat()


# ══════════════════════════════════════════════════════════════
#  报告生成
# ══════════════════════════════════════════════════════════════

def format_scan_result(result: PullbackScanResult) -> str:
    """生成 Markdown 格式报告。"""
    lines: List[str] = []
    mode_tag = ' ⚡实时' if result.is_fast_mode else ''
    lines.append(f'## 多头回踩MA10/MA20缩量阴线{mode_tag} ({result.trade_date})')
    lines.append('')
    lines.append(
        f'> 扫描 {result.total_scanned} 只（排除ST/北交所/科创/创业板）'
        f' → 信号 {len(result.items)} 只'
        f' (S={len(result.s_items)} A={len(result.a_items)} B={len(result.b_items)})'
    )
    lines.append('')

    if not result.items:
        lines.append('*当前无满足回踩MA10/MA20条件的标的。*')
        lines.append('')
        return '\n'.join(lines)

    def _build_signals(it: PullbackItem) -> str:
        """汇总该标的的附加信号标签。"""
        tags = []
        if it.triangle_support:
            tags.append('△架托')
        if it.top_divergence:
            tags.append('⚠顶背离')
        return ' '.join(tags) if tags else '—'

    def _table(items: List[PullbackItem]):
        lines.append(
            '| # | 代码 | 名称 | 收盘 | 涨跌% | 量比 | 回踩 | '
            '粘合度 | MA10斜率 | 放量 | 评分 | 信号 | 概念/板块 |'
        )
        lines.append(
            '|---|------|------|------|-------|------|------|'
            '--------|---------|------|------|------|-----------|'
        )
        for i, it in enumerate(items):
            rt_tag = '⚡' if it.is_realtime else ''
            concepts_str = ' / '.join(it.concepts[:3]) if it.concepts else '—'
            dist_pct = it.dist_to_ma10_pct if it.pullback_target == 'MA10' else it.dist_to_ma20_pct
            pullback_label = f'{it.pullback_target}{dist_pct:+.1f}%'
            signals_str = _build_signals(it)
            lines.append(
                f'| {i + 1} '
                f'| {it.code} '
                f'| {it.name}{rt_tag} '
                f'| {it.close:.2f} '
                f'| {it.change_pct:+.1f}% '
                f'| {it.vol_ratio:.2f} '
                f'| {pullback_label} '
                f'| {it.spread_10_20_pct:.1f}% '
                f'| {it.slope_ma10_pct:+.1f}% '
                f'| {it.surge_days}次 '
                f'| **{it.total_score:.0f}** '
                f'| {signals_str} '
                f'| {concepts_str} |'
            )
        lines.append('')

    if result.s_items:
        lines.append('### S级 — 经典回踩（>=75分，可半仓买入）')
        lines.append('')
        _table(result.s_items)

    if result.a_items:
        lines.append('### A级 — 良好回踩（60-74分，可1/3仓参与）')
        lines.append('')
        _table(result.a_items)

    if result.b_items:
        lines.append(
            '<details><summary>B级 — 可关注（45-59分，共 %d 只，点击展开）</summary>\n'
            % len(result.b_items)
        )
        _table(result.b_items)
        lines.append('</details>\n')

    if result.items:
        top = result.items[0]
        t_dist = top.dist_to_ma10_pct if top.pullback_target == 'MA10' else top.dist_to_ma20_pct
        lines.append('> **信号解读示例**')
        lines.append(
            f'> {top.name}({top.code}): MA均线多头排列{top.ma_up_count}线向上, '
            f'MA10斜率{top.slope_ma10_pct:+.1f}%, '
            f'MA10-MA20粘合{top.spread_10_20_pct:.1f}%, '
            f'今日缩量{top.vol_ratio:.2f}倍小阴线回踩{top.pullback_target}({t_dist:+.1f}%), '
            f'近{SURGE_LOOKBACK}日{top.surge_days}次放量上涨'
            f'(最大{top.max_surge_pct:.1f}%), '
            f'20日最大回撤{top.max_drawdown_ma20_pct:.1f}%(距MA20)'
        )
        lines.append('')

    tri_items = [x for x in result.items if x.triangle_support]
    if tri_items:
        lines.append('<details><summary>△ 三角架托详情（共 %d 只，点击展开）</summary>\n'
                      % len(tri_items))
        for it in tri_items:
            lines.append(f'- **{it.name}**({it.code}) {it.grade}级 {it.total_score:.0f}分: {it.triangle_detail}')
        lines.append('\n</details>\n')

    div_items_all = [x for x in result.items if x.top_divergence]
    if div_items_all:
        lines.append('<details><summary>⚠ DIFF顶背离详情（共 %d 只，点击展开）</summary>\n'
                      % len(div_items_all))
        for it in div_items_all:
            lines.append(f'- **{it.name}**({it.code}) {it.grade}级 {it.total_score:.0f}分: {it.divergence_detail}')
        lines.append('\n</details>\n')

    return '\n'.join(lines)
