# -*- encoding: UTF-8 -*-

"""缩量主升浪 + 趋势持股 — 选股筛选与报告

对全市场日K数据进行扫描，找出满足「缩量蓄力」条件的标的（观察池），
并检测已在观察池中的标的是否触发「放量突破」买入信号。

设计文档: docs/trend_surge_strategy.md

选股五条件:
    1. 缩量         — volume_ratio < 0.6
    2. 横盘整理     — 波动率比值 VR < 0.5
    3. MA20 向上    — 斜率 > 0 且持续 >= 5 天
    4. 重心上移     — 5日最低价均值逐窗口抬升
    5. 换手率适中   — 日均换手率 1%~5%

买入三信号:
    A. 放量突破     — 量 > MA5量 × 1.5 且收盘创N日新高
    B. 均线发散     — MA5 > MA10 > MA20 且距离在扩大
    C. K线实体饱满  — 收阳且实体占比 > 60%

数据依赖: DataManager.stocks_data（日K）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from data_manager import DataManager
except ModuleNotFoundError:
    from ..data_manager import DataManager  # type: ignore[no-redef]


# ══════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class ScreenResult:
    """单只股票的筛选结果。"""
    code: str = ''
    name: str = ''

    # 五条件原始值
    volume_ratio: float = 0.0
    vr: float = 0.0
    ma20_slope: float = 0.0
    ma20_up_days: int = 0
    low_rising: bool = False
    avg_turnover: float = 0.0

    # 评分
    score_shrink: float = 0.0       # A: 缩量质量
    score_consolidation: float = 0.0  # B: 横盘质量
    score_trend: float = 0.0        # C: 趋势强度
    score_vol_price: float = 0.0    # D: 量价配合
    score_breakout: float = 0.0     # E: 突破强度（仅触发买入时）
    score_env: float = 0.0          # F: 环境加分
    total_score: float = 0.0

    # 买入信号
    buy_signal: bool = False
    signal_detail: str = ''

    # 补充信息
    close: float = 0.0
    change_pct: float = 0.0
    mkt_cap: float = 0.0
    rise_from_low: float = 0.0


@dataclass
class TrendSurgeResult:
    """完整的趋势策略分析结果。"""
    trade_date: str = ''
    watch_pool: List[ScreenResult] = field(default_factory=list)
    buy_candidates: List[ScreenResult] = field(default_factory=list)
    total_scanned: int = 0
    passed_filter: int = 0


# ══════════════════════════════════════════════════════════════
#  核心筛选器
# ══════════════════════════════════════════════════════════════

class TrendSurgeScreener:
    """缩量主升浪选股筛选器。"""

    # 五条件阈值
    VOL_RATIO_MAX = 0.7
    VR_MAX = 0.6
    MA20_SLOPE_DAYS = 5
    TURNOVER_MIN = 1.0
    TURNOVER_MAX = 5.0
    MKT_CAP_MIN = 30e8
    MKT_CAP_MAX = 500e8
    RISE_FROM_LOW_MAX = 0.50

    def __init__(self, data_manager: DataManager):
        self.dm = data_manager

    def run(self, market_score: float = 50.0) -> TrendSurgeResult:
        """扫描全市场，返回观察池和买入候选。"""
        result = TrendSurgeResult()
        result.trade_date = self._get_trade_date()

        reject_counters = {
            'data_short': 0, 'excluded': 0, 'rise_high': 0,
            'vol_ratio': 0, 'vr': 0, 'ma20': 0,
            'low_not_rising': 0, 'turnover': 0,
        }

        for key, df in self.dm.stocks_data.items():
            result.total_scanned += 1
            sr = self._screen_one(key, df, market_score, reject_counters)
            if sr is None:
                continue
            result.passed_filter += 1
            if sr.buy_signal:
                result.buy_candidates.append(sr)
            else:
                result.watch_pool.append(sr)

        result.watch_pool.sort(key=lambda s: s.total_score, reverse=True)
        result.buy_candidates.sort(key=lambda s: s.total_score, reverse=True)

        logging.info(
            'TrendSurge: 扫描 %d 只, 通过 %d 只, 买入信号 %d 只',
            result.total_scanned, result.passed_filter,
            len(result.buy_candidates),
        )
        logging.info(
            'TrendSurge 淘汰分布: 数据不足=%d, 排除=%d, 涨幅过高=%d, '
            '缩量=%d, 横盘=%d, MA20=%d, 重心=%d, 换手=%d',
            reject_counters['data_short'], reject_counters['excluded'],
            reject_counters['rise_high'], reject_counters['vol_ratio'],
            reject_counters['vr'], reject_counters['ma20'],
            reject_counters['low_not_rising'], reject_counters['turnover'],
        )
        return result

    # ──────────────────────────────────────────────────────────
    #  单股筛选
    # ──────────────────────────────────────────────────────────

    def _screen_one(
        self, key: Tuple, df: pd.DataFrame, market_score: float,
        reject_counters: Optional[dict] = None,
    ) -> Optional[ScreenResult]:
        """对单只股票检查五条件 + 买入信号，返回 None 表示不通过。"""
        rc = reject_counters or {}

        if df is None or len(df) < 60:
            rc['data_short'] = rc.get('data_short', 0) + 1
            return None

        code, name = str(key[0]).zfill(6), str(key[1])

        if self._is_excluded(code, name):
            rc['excluded'] = rc.get('excluded', 0) + 1
            return None

        df = df.copy()
        df['日期'] = df['日期'].astype(str)
        df = df.sort_values('日期').reset_index(drop=True)

        for col in ('收盘', '开盘', '最高', '最低', '成交量', '涨跌幅'):
            if col not in df.columns:
                rc['data_short'] = rc.get('data_short', 0) + 1
                return None
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if '换手率' in df.columns:
            df['换手率'] = pd.to_numeric(df['换手率'], errors='coerce')

        pos = len(df) - 1
        if pos < 59:
            rc['data_short'] = rc.get('data_short', 0) + 1
            return None

        # ── 补充过滤 ──
        rise = self._rise_from_120d_low(df, pos)
        if rise > self.RISE_FROM_LOW_MAX:
            rc['rise_high'] = rc.get('rise_high', 0) + 1
            return None

        # ── 五条件 ──
        vol_ratio = self._calc_volume_ratio(df, pos)
        vol_pass = vol_ratio <= self.VOL_RATIO_MAX
        if not vol_pass:
            vol_declining = self._is_volume_declining(df, pos)
            vol_pass = vol_ratio <= 0.85 and vol_declining
        if not vol_pass:
            rc['vol_ratio'] = rc.get('vol_ratio', 0) + 1
            return None

        vr = self._calc_vr(df, pos)
        if vr > self.VR_MAX:
            rc['vr'] = rc.get('vr', 0) + 1
            return None

        slope, up_days = self._calc_ma20_slope(df, pos)
        if slope <= 0 or up_days < self.MA20_SLOPE_DAYS:
            rc['ma20'] = rc.get('ma20', 0) + 1
            return None

        low_rising = self._check_low_rising(df, pos)
        if not low_rising:
            rc['low_not_rising'] = rc.get('low_not_rising', 0) + 1
            return None

        avg_turnover = self._calc_avg_turnover(df, pos)
        if not (self.TURNOVER_MIN <= avg_turnover <= self.TURNOVER_MAX):
            rc['turnover'] = rc.get('turnover', 0) + 1
            return None

        # ── 构建结果 ──
        sr = ScreenResult(
            code=code, name=name,
            volume_ratio=round(vol_ratio, 3),
            vr=round(vr, 3),
            ma20_slope=round(slope * 100, 3),
            ma20_up_days=up_days,
            low_rising=low_rising,
            avg_turnover=round(avg_turnover, 2),
            close=float(df.iloc[pos]['收盘']),
            change_pct=float(df.iloc[pos].get('涨跌幅', 0)),
            rise_from_low=round(rise * 100, 1),
        )

        # ── 评分 ──
        sr.score_shrink = self._score_shrink(vol_ratio, avg_turnover, df, pos)
        sr.score_consolidation = self._score_consolidation(vr, df, pos)
        sr.score_trend = self._score_trend(slope, up_days, df, pos)
        sr.score_vol_price = self._score_vol_price(df, pos)
        sr.score_env = self._score_env(market_score)

        # ── 买入信号检测 ──
        buy, detail = self._check_buy_signal(df, pos)
        if buy:
            sr.buy_signal = True
            sr.signal_detail = detail
            sr.score_breakout = self._score_breakout(df, pos)

        sr.total_score = round(
            sr.score_shrink * 0.20
            + sr.score_consolidation * 0.20
            + sr.score_trend * 0.20
            + sr.score_vol_price * 0.15
            + sr.score_breakout * 0.15
            + sr.score_env * 0.10,
            1,
        )

        return sr

    # ──────────────────────────────────────────────────────────
    #  五条件计算
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _calc_volume_ratio(df: pd.DataFrame, pos: int) -> float:
        """缩量: 近10日均量 / 前50日均量（不重叠）。"""
        vols = df['成交量'].values
        short_start = max(0, pos - 9)
        short_ma = np.nanmean(vols[short_start:pos + 1])

        long_end = short_start
        long_start = max(0, long_end - 50)
        if long_end <= long_start:
            return 999.0
        long_ma = np.nanmean(vols[long_start:long_end])

        if long_ma == 0 or np.isnan(long_ma):
            return 999.0
        return short_ma / long_ma

    @staticmethod
    def _calc_vr(df: pd.DataFrame, pos: int) -> float:
        """横盘: 近15日收益率标准差 / 前60日收益率标准差。"""
        changes = df['涨跌幅'].values
        if pos < 59:
            return 999.0
        recent = changes[pos - 14:pos + 1]
        history = changes[pos - 59:pos - 14]
        std_recent = np.nanstd(recent)
        std_history = np.nanstd(history)
        if std_history == 0 or np.isnan(std_history):
            return 999.0
        return std_recent / std_history

    @staticmethod
    def _calc_ma20_slope(df: pd.DataFrame, pos: int) -> Tuple[float, int]:
        """MA20斜率和持续向上天数（容忍0.1%的微幅回落）。"""
        closes = df['收盘'].values
        if pos < 24:
            return 0.0, 0

        up_days = 0
        for i in range(pos, max(pos - 30, 24), -1):
            ma20_now = np.nanmean(closes[i - 19:i + 1])
            ma20_prev = np.nanmean(closes[i - 20:i])
            if ma20_now >= ma20_prev * 0.999:
                up_days += 1
            else:
                break

        ma20_today = np.nanmean(closes[pos - 19:pos + 1])
        ma20_5ago = np.nanmean(closes[pos - 24:pos - 4])
        if ma20_5ago == 0:
            return 0.0, up_days
        slope = (ma20_today - ma20_5ago) / ma20_5ago
        return slope, up_days

    @staticmethod
    def _check_low_rising(df: pd.DataFrame, pos: int) -> bool:
        """重心上移: 近5日低点均值 >= 前5日×0.995，或近10日 > 前10日。"""
        lows = df['最低'].values
        if pos < 9:
            return False
        recent_5 = np.nanmean(lows[pos - 4:pos + 1])
        prev_5 = np.nanmean(lows[pos - 9:pos - 4])
        if recent_5 >= prev_5 * 0.995:
            return True
        if pos >= 19:
            recent_10 = np.nanmean(lows[pos - 9:pos + 1])
            prev_10 = np.nanmean(lows[pos - 19:pos - 9])
            if recent_10 > prev_10:
                return True
        return False

    @staticmethod
    def _calc_avg_turnover(df: pd.DataFrame, pos: int) -> float:
        """换手率: 近10日均值。"""
        if '换手率' not in df.columns:
            return 3.0  # 无数据时返回中间值，不过滤
        turnover = df['换手率'].iloc[max(0, pos - 9):pos + 1]
        return float(np.nanmean(turnover.values))

    @staticmethod
    def _rise_from_120d_low(df: pd.DataFrame, pos: int) -> float:
        """距120日最低价的涨幅。"""
        start = max(0, pos - 119)
        low_120 = df['最低'].iloc[start:pos + 1].min()
        close = df['收盘'].iloc[pos]
        if low_120 == 0 or pd.isna(low_120):
            return 0.0
        return (close - low_120) / low_120

    @staticmethod
    def _is_volume_declining(df: pd.DataFrame, pos: int) -> bool:
        """量趋势递减: MA5量 < MA10量 < MA20量。"""
        if pos < 19:
            return False
        vols = df['成交量'].values
        ma5 = np.nanmean(vols[pos - 4:pos + 1])
        ma10 = np.nanmean(vols[pos - 9:pos + 1])
        ma20 = np.nanmean(vols[pos - 19:pos + 1])
        return ma5 < ma10 < ma20

    @staticmethod
    def _is_excluded(code: str, name: str) -> bool:
        if 'ST' in name or '*ST' in name:
            return True
        if code.startswith('8') or code.startswith('4'):
            return True
        return False

    # ──────────────────────────────────────────────────────────
    #  评分
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _score_shrink(vol_ratio: float, turnover: float,
                      df: pd.DataFrame, pos: int) -> float:
        """A. 缩量质量 (0-100)。"""
        if vol_ratio < 0.3:
            base = 90.0
        elif vol_ratio < 0.4:
            base = 75.0
        elif vol_ratio < 0.5:
            base = 55.0
        elif vol_ratio < 0.6:
            base = 35.0
        else:
            base = 0.0

        vols = df['成交量'].values

        # 连续缩量天数加分
        streak = 0
        if pos >= 5:
            ma5_v = np.nanmean(vols[pos - 4:pos + 1])
            for i in range(pos, max(pos - 15, 0), -1):
                if vols[i] < ma5_v:
                    streak += 1
                else:
                    break
        base += min(streak * 3, 15)

        # 缩量趋势加分: MA5量 < MA10量 → 量还在往下走
        if pos >= 9:
            ma5_vol = np.nanmean(vols[pos - 4:pos + 1])
            ma10_vol = np.nanmean(vols[pos - 9:pos + 1])
            if ma5_vol < ma10_vol:
                base += 10

        if turnover < 2:
            base += 10
        elif turnover > 4:
            base -= 10

        return min(max(base, 0), 100)

    @staticmethod
    def _score_consolidation(vr: float, df: pd.DataFrame, pos: int) -> float:
        """B. 横盘质量 (0-100)。"""
        base = max(0, min(1, (0.5 - vr) / 0.4)) * 70

        # 横盘持续天数（波动率低于阈值的连续天数）
        changes = df['涨跌幅'].values
        consol_days = 0
        if pos >= 30:
            std_ref = np.nanstd(changes[pos - 59:pos - 14]) if pos >= 59 else np.nanstd(changes[:pos - 14])
            if std_ref > 0:
                for window_end in range(pos, max(pos - 60, 14), -1):
                    window = changes[window_end - 14:window_end + 1]
                    if np.nanstd(window) / std_ref < 0.5:
                        consol_days += 1
                    else:
                        break

        if consol_days >= 45:
            base += 30
        elif consol_days >= 30:
            base += 25
        elif consol_days >= 20:
            base += 20
        elif consol_days >= 15:
            base += 10

        return min(base, 100)

    @staticmethod
    def _score_trend(slope: float, up_days: int,
                     df: pd.DataFrame, pos: int) -> float:
        """C. 趋势强度 (0-100)。"""
        slope_pct = slope * 100
        if slope_pct > 0.5:
            base = 70.0
        elif slope_pct > 0.3:
            base = 50.0
        elif slope_pct > 0.1:
            base = 30.0
        else:
            base = 10.0

        if up_days >= 20:
            base += 20
        elif up_days >= 10:
            base += 15
        elif up_days >= 5:
            base += 10

        closes = df['收盘'].values
        if pos >= 19:
            ma5 = np.nanmean(closes[pos - 4:pos + 1])
            ma10 = np.nanmean(closes[pos - 9:pos + 1])
            ma20 = np.nanmean(closes[pos - 19:pos + 1])
            if ma5 > ma10 > ma20:
                base += 10

        return min(base, 100)

    @staticmethod
    def _score_vol_price(df: pd.DataFrame, pos: int) -> float:
        """D. 量价配合 (0-100) — 涨时放量跌时缩量。"""
        if pos < 19:
            return 50.0

        up_vols, down_vols = [], []
        for i in range(pos - 19, pos + 1):
            close_val = df.iloc[i]['收盘']
            open_val = df.iloc[i]['开盘']
            vol = df.iloc[i]['成交量']
            if pd.isna(vol):
                continue
            if close_val > open_val:
                up_vols.append(vol)
            else:
                down_vols.append(vol)

        if not down_vols or not up_vols:
            return 50.0

        vp_ratio = np.mean(up_vols) / np.mean(down_vols)

        if vp_ratio > 1.5:
            return 90.0
        elif vp_ratio > 1.2:
            return 65.0
        elif vp_ratio > 1.0:
            return 40.0
        elif vp_ratio > 0.8:
            return 20.0
        else:
            return 0.0

    @staticmethod
    def _score_env(market_score: float) -> float:
        """F. 环境加分 (0-100) — 基于市场温度。"""
        return min(max(market_score, 0), 100)

    @staticmethod
    def _score_breakout(df: pd.DataFrame, pos: int) -> float:
        """E. 突破强度 (0-100)。"""
        score = 0.0
        vols = df['成交量'].values
        ma5_vol = np.nanmean(vols[max(0, pos - 4):pos]) if pos >= 1 else vols[pos]
        if ma5_vol > 0:
            vol_mult = vols[pos] / ma5_vol
            if vol_mult > 3.0:
                score += 40
            elif vol_mult > 2.0:
                score += 30
            elif vol_mult > 1.5:
                score += 20

        closes = df['收盘'].values
        lookback = min(20, pos)
        if lookback > 0:
            prev_high = np.nanmax(closes[pos - lookback:pos])
            if prev_high > 0:
                breakout_pct = (closes[pos] - prev_high) / prev_high
                if breakout_pct > 0.05:
                    score += 30
                elif breakout_pct > 0.03:
                    score += 20
                elif breakout_pct > 0.01:
                    score += 10

        close_val = df.iloc[pos]['收盘']
        open_val = df.iloc[pos]['开盘']
        high_val = df.iloc[pos]['最高']
        low_val = df.iloc[pos]['最低']
        amplitude = high_val - low_val
        if amplitude > 0:
            body_ratio = abs(close_val - open_val) / amplitude
            if body_ratio > 0.8:
                score += 30
            elif body_ratio > 0.6:
                score += 20
            else:
                score += 10

        return min(score, 100)

    # ──────────────────────────────────────────────────────────
    #  买入信号检测
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _check_buy_signal(df: pd.DataFrame, pos: int) -> Tuple[bool, str]:
        """检测放量突破 + 均线发散 + K线实体三合一买入信号。"""
        if pos < 19:
            return False, ''

        # A: 放量突破
        vols = df['成交量'].values
        ma5_vol = np.nanmean(vols[max(0, pos - 4):pos])
        vol_ok = vols[pos] > ma5_vol * 1.5 if ma5_vol > 0 else False

        closes = df['收盘'].values
        lookback = min(20, pos)
        prev_high = np.nanmax(closes[pos - lookback:pos])
        price_ok = closes[pos] > prev_high

        signal_a = vol_ok and price_ok

        # B: 均线发散
        ma5 = np.nanmean(closes[pos - 4:pos + 1])
        ma10 = np.nanmean(closes[pos - 9:pos + 1])
        ma20 = np.nanmean(closes[pos - 19:pos + 1])
        ma_order = ma5 > ma10 > ma20

        if pos >= 20:
            ma5_prev = np.nanmean(closes[pos - 5:pos])
            ma20_prev = np.nanmean(closes[pos - 20:pos])
            spread_expanding = (ma5 - ma20) > (ma5_prev - ma20_prev)
        else:
            spread_expanding = True

        signal_b = ma_order and spread_expanding

        # C: K线实体
        close_val = closes[pos]
        open_val = df.iloc[pos]['开盘']
        high_val = df.iloc[pos]['最高']
        low_val = df.iloc[pos]['最低']
        is_yang = close_val > open_val
        amplitude = high_val - low_val
        body_ratio = abs(close_val - open_val) / amplitude if amplitude > 0 else 0
        signal_c = is_yang and body_ratio > 0.6

        if signal_a and signal_b and signal_c:
            detail = (
                f"放量{vols[pos]/ma5_vol:.1f}倍, "
                f"突破{(closes[pos]-prev_high)/prev_high*100:.1f}%, "
                f"实体{body_ratio*100:.0f}%"
            )
            return True, detail

        return False, ''

    # ──────────────────────────────────────────────────────────
    #  工具
    # ──────────────────────────────────────────────────────────

    def _get_trade_date(self) -> str:
        for _key, df in self.dm.stocks_data.items():
            if df is not None and not df.empty and '日期' in df.columns:
                return str(df['日期'].iloc[-1])[:10]
        return date.today().isoformat()

    # ──────────────────────────────────────────────────────────
    #  报告输出
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def to_markdown(result: TrendSurgeResult) -> str:
        """生成 Markdown 格式报告。"""
        lines = []
        lines.append(f'## 缩量主升浪选股 ({result.trade_date})')
        lines.append('')
        lines.append(
            f'> 扫描 {result.total_scanned} 只 → '
            f'通过筛选 {result.passed_filter} 只 → '
            f'买入信号 {len(result.buy_candidates)} 只'
        )
        lines.append('')

        if result.buy_candidates:
            lines.append('### 买入信号（三合一确认）')
            lines.append('')
            lines.append(
                '| # | 代码 | 名称 | 收盘 | 涨跌% | 缩量比 | VR | '
                'MA20↑天 | 评分 | 信号 |'
            )
            lines.append(
                '|---|------|------|------|-------|--------|-----|'
                '---------|------|------|'
            )
            for i, s in enumerate(result.buy_candidates[:10]):
                lines.append(
                    f'| {i+1} '
                    f'| {s.code} '
                    f'| {s.name} '
                    f'| {s.close:.2f} '
                    f'| {s.change_pct:+.1f}% '
                    f'| {s.volume_ratio:.2f} '
                    f'| {s.vr:.2f} '
                    f'| {s.ma20_up_days} '
                    f'| **{s.total_score:.0f}** '
                    f'| {s.signal_detail} |'
                )
            lines.append('')

        if result.watch_pool:
            lines.append('### 观察池（蓄力中，等待突破）')
            lines.append('')
            lines.append(
                '| # | 代码 | 名称 | 收盘 | 涨跌% | 缩量比 | VR | '
                'MA20↑天 | 换手% | 评分 |'
            )
            lines.append(
                '|---|------|------|------|-------|--------|-----|'
                '---------|-------|------|'
            )
            top_n = min(20, len(result.watch_pool))
            for i, s in enumerate(result.watch_pool[:top_n]):
                lines.append(
                    f'| {i+1} '
                    f'| {s.code} '
                    f'| {s.name} '
                    f'| {s.close:.2f} '
                    f'| {s.change_pct:+.1f}% '
                    f'| {s.volume_ratio:.2f} '
                    f'| {s.vr:.2f} '
                    f'| {s.ma20_up_days} '
                    f'| {s.avg_turnover:.1f} '
                    f'| **{s.total_score:.0f}** |'
                )
            lines.append('')

        if not result.buy_candidates and not result.watch_pool:
            lines.append('*当前无满足条件的标的，耐心等待。*')
            lines.append('')

        return '\n'.join(lines)
