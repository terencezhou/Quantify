# -*- encoding: UTF-8 -*-

"""日K形态评分 — 连续评分函数实现

对一只股票的日K线进行系统性打分，返回结构化评分结果。
本模块只计算分数和理由，不生成报告、不发送消息。

设计文档: docs/daily_k.md

用法:
    from report.function_daily_k import compute_kline_score
    ks = compute_kline_score("002041", "登海种业", daily_df)
    print(ks.total)                    # 总分
    print(ks.to_markdown_table())      # 明细表格
    vp = ks.get_factor("量价配合度")    # 取单个因子
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════

@dataclass
class KlineFactor:
    """单个因子的评分结果"""
    name: str
    group: str          # "pattern" / "env" / "penalty"
    score: float        # 加权后的实际贡献分
    max_score: float    # 该因子的满分值
    raw_value: float    # 原始指标值（如 VR=0.35）
    reason: str         # 人类可读的理由


@dataclass
class KlinePatternMatch:
    """命中的形态"""
    pattern: str        # "consolidation_breakout" / "arc_bottom" / "none"
    label: str          # 中文标签
    window_days: int    # 最优窗口天数
    score: float        # 形态总分（不含环境因子和惩罚）


@dataclass
class KlineScore:
    """一只股票的K线完整评分结果"""
    code: str
    name: str
    total: float
    pattern: KlinePatternMatch
    factors: List[KlineFactor] = field(default_factory=list)

    def get_factor(self, name: str) -> Optional[KlineFactor]:
        for f in self.factors:
            if f.name == name:
                return f
        return None

    def to_markdown_table(self) -> str:
        lines = ["| 因子 | 得分 | 说明 |", "|------|------|------|"]

        current_group = None
        for f in self.factors:
            if f.group != current_group:
                current_group = f.group
                header = {"pattern": self.pattern.label or "形态",
                          "env": "环境因子",
                          "penalty": "惩罚"}.get(current_group, current_group)
                extra = ""
                if current_group == "pattern" and self.pattern.window_days > 0:
                    extra = f"窗口{self.pattern.window_days}日"
                lines.append(f"| **{header}** | | **{extra}** |")

            if f.max_score != 0:
                score_str = f"{f.score:+.1f}/{f.max_score:+.1f}"
            else:
                score_str = f"{f.score:+.1f}"
            lines.append(f"| 　{f.name} | {score_str} | {f.reason} |")

        lines.append(f"| **K线总分** | **{self.total:+.1f}** | |")
        return "\n".join(lines)

    def to_summary_line(self) -> str:
        tags = []
        if self.pattern.pattern != "none":
            tags.append(f"{self.pattern.label}")
        for f in self.factors:
            if f.group == "env" and abs(f.score) >= 2:
                tag = f.reason.split("，")[0] if "，" in f.reason else f.name
                tags.append(tag)
        detail = "·".join(tags) if tags else "无明显形态"
        return f"{self.pattern.label}{self.total:+.1f} ({detail})"


# ═══════════════════════════════════════════════
#  映射工具函数
# ═══════════════════════════════════════════════

def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_std(series: pd.Series) -> float:
    if len(series) < 3:
        return 0.0
    return float(series.std(ddof=1))


def _pct_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


# ═══════════════════════════════════════════════
#  形态①：横盘整理后突破
# ═══════════════════════════════════════════════

_CONSOL_FULL = 20.0
_CONSOL_W = {"vol_contraction": 0.35, "duration": 0.25,
             "breakout": 0.25, "volume": 0.15}


def _f_vol_contraction(recent_returns: pd.Series,
                       history_returns: pd.Series) -> tuple:
    """波动收敛度 → (f_value, VR)"""
    sigma_r = _safe_std(recent_returns)
    sigma_h = _safe_std(history_returns)
    if sigma_h < 1e-9:
        return 0.0, 999.0
    vr = sigma_r / sigma_h
    f = max(0.0, 1.0 - vr) ** 0.6
    return f, vr


def _f_duration(days: int) -> float:
    """横盘持续时间 → f_value"""
    if days <= 10:
        return 0.0
    return min(1.0, math.log(days / 10.0) / math.log(6.0))


def _f_breakout(today_close: float, window_high: float) -> tuple:
    """突破强度 → (f_value, breakout_pct)"""
    if window_high <= 0:
        return 0.0, 0.0
    brk = (today_close - window_high) / window_high
    return _clip(brk / 0.08, 0.0, 1.0), brk


def _f_volume_confirm(today_vol: float, window_avg_vol: float) -> tuple:
    """量能确认 → (f_value, vol_ratio)"""
    if window_avg_vol <= 0:
        return 0.0, 0.0
    ratio = today_vol / window_avg_vol
    return _clip((ratio - 1.0) / 2.0, 0.0, 1.0), ratio


def _score_consolidation_breakout(
    df: pd.DataFrame, today_idx: int
) -> tuple:
    """扫描所有窗口，返回 (best_score, best_window, factors_list)"""
    close = df["收盘"].values
    vol = df["成交量"].values.astype(float)

    history_end = today_idx
    history_start = max(0, history_end - 120)
    if history_end - history_start < 30:
        return 0.0, 0, []

    hist_returns = _pct_returns(pd.Series(close[history_start:history_end]))

    best_score = 0.0
    best_window = 0
    best_factors: List[KlineFactor] = []

    for win in range(15, min(61, today_idx)):
        w_start = today_idx - win
        if w_start < 0:
            break
        recent_close = pd.Series(close[w_start:today_idx])
        recent_returns = _pct_returns(recent_close)

        fv, vr = _f_vol_contraction(recent_returns, hist_returns)
        if fv < 0.15:
            continue

        fd = _f_duration(win)
        window_high = float(np.max(close[w_start:today_idx]))
        fb, brk_pct = _f_breakout(close[today_idx], window_high)

        w_avg_vol = float(np.mean(vol[w_start:today_idx])) if win > 0 else 1.0
        fvc, vr_vol = _f_volume_confirm(vol[today_idx], w_avg_vol)

        weighted = (
            _CONSOL_W["vol_contraction"] * fv
            + _CONSOL_W["duration"] * fd
            + _CONSOL_W["breakout"] * fb
            + _CONSOL_W["volume"] * fvc
        )
        score = _CONSOL_FULL * weighted

        if score > best_score:
            best_score = score
            best_window = win
            max_v = _CONSOL_FULL * _CONSOL_W["vol_contraction"]
            max_d = _CONSOL_FULL * _CONSOL_W["duration"]
            max_b = _CONSOL_FULL * _CONSOL_W["breakout"]
            max_vc = _CONSOL_FULL * _CONSOL_W["volume"]
            best_factors = [
                KlineFactor(
                    "波动收敛度", "pattern",
                    round(fv * max_v, 1), round(max_v, 1), round(vr, 3),
                    f"VR={vr:.2f}，{'极度收敛' if vr < 0.3 else '明显收敛' if vr < 0.6 else '轻度收敛'}",
                ),
                KlineFactor(
                    "横盘持续时间", "pattern",
                    round(fd * max_d, 1), round(max_d, 1), float(win),
                    f"横盘{win}个交易日",
                ),
                KlineFactor(
                    "突破强度", "pattern",
                    round(fb * max_b, 1), round(max_b, 1), round(brk_pct, 4),
                    f"突破窗口高点{brk_pct * 100:.1f}%"
                    + ("（涨停突破）" if brk_pct >= 0.08 else ""),
                ),
                KlineFactor(
                    "量能确认", "pattern",
                    round(fvc * max_vc, 1), round(max_vc, 1), round(vr_vol, 2),
                    f"量比{vr_vol:.1f}倍(窗口均量的{vr_vol:.1f}x)",
                ),
            ]

    return round(best_score, 2), best_window, best_factors


# ═══════════════════════════════════════════════
#  环境因子 H1：均线系统
# ═══════════════════════════════════════════════

def _score_ma_system(df: pd.DataFrame, today_idx: int) -> KlineFactor:
    close = df["收盘"].values
    n = today_idx + 1

    def _ma(period: int) -> float:
        if n < period:
            return float(np.mean(close[:n]))
        return float(np.mean(close[n - period: n]))

    def _ma_at(period: int, offset: int) -> float:
        end = n - offset
        if end < period:
            return float(np.mean(close[:max(end, 1)]))
        return float(np.mean(close[end - period: end]))

    ma5, ma10, ma20, ma60 = _ma(5), _ma(10), _ma(20), _ma(60)
    ma20_5ago = _ma_at(20, 5)
    slope_ma20 = (ma20 - ma20_5ago) / ma20_5ago if ma20_5ago > 0 else 0

    spread = max(ma5, ma10, ma20) - min(ma5, ma10, ma20)
    avg_ma = (ma5 + ma10 + ma20) / 3
    is_converged = (spread / avg_ma) < 0.02 if avg_ma > 0 else False

    if ma5 > ma10 > ma20 > ma60 and slope_ma20 > 0:
        return KlineFactor("均线系统", "env", 5.0, 5.0, round(slope_ma20, 5),
                           "均线完全多头排列，中期趋势向上")
    if ma5 > ma20 and slope_ma20 > 0 and ma20 < ma60:
        return KlineFactor("均线系统", "env", 2.0, 5.0, round(slope_ma20, 5),
                           "短期均线多头，中期趋势待确认")
    if is_converged:
        return KlineFactor("均线系统", "env", 0.0, 5.0, round(spread / avg_ma, 4),
                           "均线粘合，方向未明")
    if ma5 < ma10 < ma20 and slope_ma20 < 0:
        return KlineFactor("均线系统", "env", -5.0, 5.0, round(slope_ma20, 5),
                           "均线空头排列，逆势涨停风险高")

    return KlineFactor("均线系统", "env", 0.0, 5.0, 0.0, "均线无明确排列")


# ═══════════════════════════════════════════════
#  环境因子 H2：量价趋势配合度
# ═══════════════════════════════════════════════

def _score_volume_price_trend(df: pd.DataFrame, today_idx: int) -> KlineFactor:
    lookback = 20
    start = max(0, today_idx - lookback)
    seg = df.iloc[start: today_idx + 1]
    if len(seg) < 5:
        return KlineFactor("量价配合度", "env", 0.0, 5.0, 1.0, "数据不足")

    opens = seg["开盘"].values.astype(float)
    closes = seg["收盘"].values.astype(float)
    vols = seg["成交量"].values.astype(float)

    up_mask = closes > opens
    down_mask = ~up_mask

    avg_vol_up = float(np.mean(vols[up_mask])) if np.any(up_mask) else 1.0
    avg_vol_down = float(np.mean(vols[down_mask])) if np.any(down_mask) else 1.0

    if avg_vol_down < 1e-9:
        vp_ratio = 2.0
    else:
        vp_ratio = avg_vol_up / avg_vol_down

    f_vp = _clip((vp_ratio - 1.0) / 0.8, -1.0, 1.0) * 5.0
    f_vp = round(f_vp, 1)

    if vp_ratio >= 1.5:
        reason = f"vp_ratio={vp_ratio:.2f}，涨时放量跌时缩量，量价健康"
    elif vp_ratio >= 1.0:
        reason = f"vp_ratio={vp_ratio:.2f}，量价基本配合"
    elif vp_ratio >= 0.7:
        reason = f"vp_ratio={vp_ratio:.2f}，量价轻度背离"
    else:
        reason = f"vp_ratio={vp_ratio:.2f}，涨时缩量跌时放量，出货特征"

    return KlineFactor("量价配合度", "env", f_vp, 5.0, round(vp_ratio, 3), reason)


# ═══════════════════════════════════════════════
#  环境因子 H3：关键价位突破
# ═══════════════════════════════════════════════

def _score_key_level_breakout(df: pd.DataFrame, today_idx: int) -> KlineFactor:
    close = df["收盘"].values
    n = today_idx + 1
    today_close = close[today_idx]
    prev_close = close[today_idx - 1] if today_idx > 0 else today_close

    high_120 = float(np.max(close[max(0, n - 121): today_idx])) if today_idx > 0 else today_close

    def _ma_val(period: int) -> float:
        if n < period:
            return float(np.mean(close[:n]))
        return float(np.mean(close[n - period: n]))

    ma60 = _ma_val(60)
    ma120 = _ma_val(120)

    pts = 0.0
    reasons = []

    if prev_close < high_120 and today_close > high_120:
        pts += 3
        reasons.append(f"突破120日前高({high_120:.2f})")
    if n >= 120 and prev_close < ma120 and today_close > ma120:
        pts += 2
        reasons.append(f"站上MA120({ma120:.2f})")
    elif n >= 60 and prev_close < ma60 and today_close > ma60:
        pts += 1
        reasons.append(f"站上MA60({ma60:.2f})")

    score = min(pts, 5.0)
    reason = "，".join(reasons) if reasons else "未突破关键价位"
    return KlineFactor("关键价位突破", "env", score, 5.0, score, reason)


# ═══════════════════════════════════════════════
#  负面形态惩罚
# ═══════════════════════════════════════════════

def _score_penalties(df: pd.DataFrame, today_idx: int,
                     consol_score: float) -> List[KlineFactor]:
    close = df["收盘"].values
    penalties = []

    low_120 = float(np.min(close[max(0, today_idx - 120): today_idx + 1]))
    if low_120 > 0:
        rise = (close[today_idx] - low_120) / low_120
    else:
        rise = 0.0

    if rise > 0.80 and consol_score > 5:
        pen = round(consol_score * -0.5, 1)
        penalties.append(KlineFactor(
            "高位横盘突破", "penalty", pen, 0.0, round(rise, 3),
            f"距120日低点已涨{rise * 100:.0f}%，横盘突破可能是诱多",
        ))

    if today_idx >= 5:
        prev_close = close[today_idx - 1]
        today_pct = (close[today_idx] - prev_close) / prev_close if prev_close > 0 else 0
        is_limit_up = today_pct >= 0.098
        if is_limit_up:
            ret_5d = (close[today_idx - 1] - close[today_idx - 6]) / close[today_idx - 6]
            if ret_5d < -0.15:
                penalties.append(KlineFactor(
                    "连续急跌后涨停", "penalty", -5.0, 0.0, round(ret_5d, 4),
                    f"前5日累计跌{ret_5d * 100:.1f}%后涨停，超跌反弹持续性存疑",
                ))

    return penalties


# ═══════════════════════════════════════════════
#  总入口
# ═══════════════════════════════════════════════

def compute_kline_score(
    code: str,
    name: str,
    daily_df: pd.DataFrame,
) -> KlineScore:
    """对一只股票的日K线进行系统性打分。

    Parameters
    ----------
    code : str
        股票代码
    name : str
        股票名称
    daily_df : pd.DataFrame
        日K数据，需包含: 收盘, 开盘, 最高, 最低, 成交量
        按日期升序排列

    Returns
    -------
    KlineScore
        完整的评分结果，包含总分、形态匹配、因子明细
    """
    empty_result = KlineScore(
        code=code, name=name, total=0.0,
        pattern=KlinePatternMatch("none", "无明显形态", 0, 0.0),
    )

    if daily_df is None or daily_df.empty:
        return empty_result

    required_cols = {"收盘", "开盘", "成交量"}
    if not required_cols.issubset(daily_df.columns):
        return empty_result

    df = daily_df.copy()
    for col in ("收盘", "开盘", "成交量"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["收盘"])
    if len(df) < 30:
        return empty_result

    today_idx = len(df) - 1

    # ── 形态：横盘突破 ──
    consol_score, consol_win, consol_factors = _score_consolidation_breakout(
        df, today_idx
    )

    if consol_score > 0:
        pattern = KlinePatternMatch(
            "consolidation_breakout", "横盘整理后突破",
            consol_win, consol_score,
        )
        pattern_factors = consol_factors
        raw_pattern_score = consol_score
    else:
        pattern = KlinePatternMatch("none", "无明显形态", 0, 0.0)
        pattern_factors = []
        raw_pattern_score = 0.0

    # ── 负面形态惩罚 ──
    penalty_factors = _score_penalties(df, today_idx, consol_score)
    penalty_total = sum(f.score for f in penalty_factors)
    f1 = max(raw_pattern_score, 0) + penalty_total

    # ── 环境因子 ──
    h1 = _score_ma_system(df, today_idx)
    h2 = _score_volume_price_trend(df, today_idx)
    h3 = _score_key_level_breakout(df, today_idx)

    # ── 汇总 ──
    total = round(f1 + h1.score + h2.score + h3.score, 1)

    all_factors = pattern_factors + [h1, h2, h3] + penalty_factors

    return KlineScore(
        code=code, name=name, total=total,
        pattern=pattern, factors=all_factors,
    )
