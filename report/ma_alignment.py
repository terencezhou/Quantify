# -*- encoding: UTF-8 -*-

"""均线回归策略 — 多头排列首日捕捉

对全市场日K数据进行扫描，找出指定日期首次出现均线回归的股票。

设计文档: docs/ma_alignment_strategy.md

均线回归定义:
    1. 多头排列: MA5 > MA10 > MA20 > MA30
    2. MA5 斜率向上: MA5_T > MA5_(T-1)
    3. MA10 斜率向上: MA10_T > MA10_(T-1)

首日回归判定:
    is_first_alignment(T) = is_alignment(T) AND NOT is_alignment(T-1)

回归评分 (0-4):
    MA5/MA10/MA20/MA30 每条均线向上（较前一日）加1分

数据依赖: DataManager.stocks_data（日K，仅需收盘价）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from data_manager import DataManager
except ModuleNotFoundError:
    from ..data_manager import DataManager  # type: ignore[no-redef]


MA_PERIODS = [5, 10, 20, 30]
MIN_DATA_LEN = 35


# ══════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class MADetail:
    """单条均线的当日明细。"""
    period: int = 0
    value_today: float = 0.0
    value_prev: float = 0.0
    is_up: bool = False

    @property
    def direction_symbol(self) -> str:
        return "↑" if self.is_up else "↓"


@dataclass
class AlignmentResult:
    """单只股票在指定日期的均线回归检测结果。"""
    code: str = ''
    name: str = ''
    target_date: str = ''
    close: float = 0.0
    change_pct: float = 0.0

    is_alignment: bool = False
    is_first_day: bool = False
    score: int = 0

    ma5_slope_pct: float = 0.0
    ma_details: List[MADetail] = field(default_factory=list)

    # 追高风险指标
    bias5_pct: float = 0.0      # 收盘价对MA5的乖离率 (close-MA5)/MA5
    ma_spread_pct: float = 0.0  # 均线扩散度 (MA5-MA30)/MA30
    rise_5d_pct: float = 0.0    # 近5日累计涨幅
    chase_risk: str = ''        # "低" / "中" / "高"

    @property
    def score_label(self) -> str:
        labels = {4: "极强", 3: "强", 2: "中", 1: "弱", 0: "—"}
        return labels.get(self.score, "—")

    @property
    def stars(self) -> str:
        return "★" * self.score + "☆" * (4 - self.score)

    @property
    def chase_risk_symbol(self) -> str:
        return {"低": "🟢", "中": "🟡", "高": "🔴"}.get(self.chase_risk, "")


@dataclass
class ScanResult:
    """全市场扫描结果。"""
    target_date: str = ''
    total_scanned: int = 0
    items: List[AlignmentResult] = field(default_factory=list)

    @property
    def score_distribution(self) -> Dict[int, int]:
        dist: Dict[int, int] = {}
        for item in self.items:
            dist[item.score] = dist.get(item.score, 0) + 1
        return dist


# ══════════════════════════════════════════════════════════════
#  日期解析
# ══════════════════════════════════════════════════════════════

_DATE_FORMATS = ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']


def parse_date(date_str: str) -> str:
    """解析日期字符串，返回 YYYY-MM-DD 格式。

    支持: '2025-03-10', '20250310', '2025/03/10'
    """
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    raise ValueError(
        f"无法解析日期: {date_str}，支持格式: YYYY-MM-DD / YYYYMMDD / YYYY/MM/DD"
    )


# ══════════════════════════════════════════════════════════════
#  均线计算工具
# ══════════════════════════════════════════════════════════════

def _compute_ma_columns(closes: np.ndarray) -> Dict[int, np.ndarray]:
    """预计算全部 MA 序列，返回 {period: ma_array}。"""
    result = {}
    for p in MA_PERIODS:
        if len(closes) < p:
            result[p] = np.full(len(closes), np.nan)
        else:
            cumsum = np.cumsum(closes)
            cumsum = np.insert(cumsum, 0, 0.0)
            ma = (cumsum[p:] - cumsum[:-p]) / p
            padded = np.empty(len(closes))
            padded[:] = np.nan
            padded[p - 1:] = ma
            result[p] = padded
    return result


def _check_alignment_at(
    ma_dict: Dict[int, np.ndarray], idx: int, idx_prev: int
) -> Tuple[bool, bool, int, float, List[MADetail]]:
    """检查某个位置是否满足均线回归。

    Returns: (is_alignment, prev_is_alignment, score, ma5_slope_pct, details)
    """
    vals = {}
    prev_vals = {}
    for p in MA_PERIODS:
        v = ma_dict[p][idx]
        pv = ma_dict[p][idx_prev]
        if np.isnan(v) or np.isnan(pv):
            return False, False, 0, 0.0, []
        vals[p] = v
        prev_vals[p] = pv

    bull_order = vals[5] > vals[10] > vals[20] > vals[30]
    ma5_slope_up = vals[5] > prev_vals[5]
    ma10_slope_up = vals[10] > prev_vals[10]
    is_alignment = bull_order and ma5_slope_up and ma10_slope_up

    ma5_slope_pct = 0.0
    if prev_vals[5] > 0:
        ma5_slope_pct = (vals[5] - prev_vals[5]) / prev_vals[5] * 100

    score = 0
    details: List[MADetail] = []
    for p in MA_PERIODS:
        up = vals[p] > prev_vals[p]
        if up:
            score += 1
        details.append(MADetail(
            period=p,
            value_today=round(vals[p], 2),
            value_prev=round(prev_vals[p], 2),
            is_up=up,
        ))

    prev_bull = prev_vals[5] > prev_vals[10] > prev_vals[20] > prev_vals[30]
    if idx_prev >= 1:
        pprev_vals = {}
        valid = True
        for p in MA_PERIODS:
            pv2 = ma_dict[p][idx_prev - 1]
            if np.isnan(pv2):
                valid = False
                break
            pprev_vals[p] = pv2
        if valid:
            prev_ma5_slope_up = prev_vals[5] > pprev_vals[5]
            prev_ma10_slope_up = prev_vals[10] > pprev_vals[10]
            prev_is_alignment = prev_bull and prev_ma5_slope_up and prev_ma10_slope_up
        else:
            prev_is_alignment = False
    else:
        prev_is_alignment = False

    return is_alignment, prev_is_alignment, score, ma5_slope_pct, details


# ══════════════════════════════════════════════════════════════
#  过滤
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
#  追高风险评估
# ══════════════════════════════════════════════════════════════

# 阈值说明:
#   bias5 — 收盘价偏离MA5的幅度。均线回归首日理想情况是价格贴着MA5，
#           超过3%说明短线已有一定涨幅，超过5%明显偏高。
#   ma_spread — MA5与MA30的距离。首日回归时均线应刚刚分开，
#               超过5%说明排列已持续一段时间/急拉形成，超过8%明显追高。
#   rise_5d — 近5日累计涨幅。缓慢回归<5%，超过10%说明近期急拉。
_CHASE_THRESHOLDS = {
    'bias5_medium':     3.0,   # BIAS5 >= 3% → 中风险
    'bias5_high':       5.0,   # BIAS5 >= 5% → 高风险
    'spread_medium':    5.0,   # MA扩散度 >= 5% → 中风险
    'spread_high':      8.0,   # MA扩散度 >= 8% → 高风险
    'rise5d_medium':    8.0,   # 5日涨幅 >= 8% → 中风险
    'rise5d_high':     13.0,   # 5日涨幅 >= 13% → 高风险
}


def _compute_chase_metrics(
    closes: np.ndarray, ma_dict: Dict[int, np.ndarray], idx: int
) -> Tuple[float, float, float]:
    """计算追高风险的三个度量指标。

    Returns: (bias5_pct, ma_spread_pct, rise_5d_pct)
    """
    ma5_val = ma_dict[5][idx]
    ma30_val = ma_dict[30][idx]

    bias5 = 0.0
    if not np.isnan(ma5_val) and ma5_val > 0:
        bias5 = (closes[idx] - ma5_val) / ma5_val * 100

    spread = 0.0
    if not np.isnan(ma5_val) and not np.isnan(ma30_val) and ma30_val > 0:
        spread = (ma5_val - ma30_val) / ma30_val * 100

    rise_5d = 0.0
    if idx >= 5 and closes[idx - 5] > 0:
        rise_5d = (closes[idx] - closes[idx - 5]) / closes[idx - 5] * 100

    return round(bias5, 2), round(spread, 2), round(rise_5d, 2)


def _assess_chase_risk(bias5: float, spread: float, rise_5d: float) -> str:
    """综合三个指标评估追高风险等级。

    规则: 任一指标触及"高"阈值 → 高风险
          两个及以上指标触及"中"阈值 → 高风险
          一个指标触及"中"阈值 → 中风险
          全部低于"中"阈值 → 低风险
    """
    th = _CHASE_THRESHOLDS
    high_count = 0
    medium_count = 0

    if bias5 >= th['bias5_high']:
        high_count += 1
    elif bias5 >= th['bias5_medium']:
        medium_count += 1

    if spread >= th['spread_high']:
        high_count += 1
    elif spread >= th['spread_medium']:
        medium_count += 1

    if rise_5d >= th['rise5d_high']:
        high_count += 1
    elif rise_5d >= th['rise5d_medium']:
        medium_count += 1

    if high_count >= 1 or medium_count >= 2:
        return "高"
    if medium_count >= 1:
        return "中"
    return "低"


# ══════════════════════════════════════════════════════════════
#  基础过滤
# ══════════════════════════════════════════════════════════════

def _is_excluded(code: str, name: str) -> bool:
    """排除 ST、北交所、科创板、创业板。"""
    if 'ST' in name or '*ST' in name:
        return True
    if code.startswith('8') or code.startswith('4'):
        return True
    if code.startswith('68'):
        return True
    if code.startswith('3'):
        return True
    return False


# ══════════════════════════════════════════════════════════════
#  核心检测器
# ══════════════════════════════════════════════════════════════

class MAAlignmentDetector:
    """均线回归检测器。"""

    def __init__(self, data_manager: DataManager):
        self.dm = data_manager

    # ── 单只股票检测 ──

    def check_one(
        self, code: str, name: str, df: pd.DataFrame, target_date: str
    ) -> Optional[AlignmentResult]:
        """检测单只股票在目标日期的均线回归状态。

        Parameters
        ----------
        code : str
        name : str
        df : pd.DataFrame  日K数据，包含 '日期' 和 '收盘'
        target_date : str   格式 'YYYY-MM-DD'

        Returns
        -------
        AlignmentResult or None (数据不足时)
        """
        if df is None or len(df) < MIN_DATA_LEN:
            return None

        df = df.copy()
        df['日期'] = df['日期'].astype(str).str[:10]
        for col in ('收盘',):
            if col not in df.columns:
                return None
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['收盘']).sort_values('日期').reset_index(drop=True)

        if len(df) < MIN_DATA_LEN:
            return None

        mask = df['日期'] <= target_date
        if mask.sum() < MIN_DATA_LEN:
            return None

        df_up_to = df[mask].reset_index(drop=True)
        idx = len(df_up_to) - 1
        if idx < 1:
            return None

        actual_date = df_up_to.iloc[idx]['日期']

        closes = df_up_to['收盘'].values.astype(float)
        ma_dict = _compute_ma_columns(closes)

        is_align, prev_align, score, slope_pct, details = _check_alignment_at(
            ma_dict, idx, idx - 1
        )

        is_first = is_align and (not prev_align)

        change_pct = 0.0
        if '涨跌幅' in df_up_to.columns:
            change_pct = float(
                pd.to_numeric(df_up_to.iloc[idx].get('涨跌幅', 0), errors='coerce') or 0
            )

        bias5, spread, rise_5d = _compute_chase_metrics(closes, ma_dict, idx)
        chase_risk = _assess_chase_risk(bias5, spread, rise_5d) if is_align else ""

        return AlignmentResult(
            code=code,
            name=name,
            target_date=str(actual_date),
            close=round(float(closes[idx]), 2),
            change_pct=round(change_pct, 2),
            is_alignment=is_align,
            is_first_day=is_first,
            score=score,
            ma5_slope_pct=round(slope_pct, 4),
            ma_details=details,
            bias5_pct=bias5,
            ma_spread_pct=spread,
            rise_5d_pct=rise_5d,
            chase_risk=chase_risk,
        )

    # ── 指定股票检测 ──

    def check_by_code(self, stock_code: str, target_date: str) -> Optional[AlignmentResult]:
        """根据股票代码在全部数据中查找并检测。"""
        stock_code = stock_code.zfill(6)
        for (code, name), df in self.dm.stocks_data.items():
            if str(code).zfill(6) == stock_code:
                return self.check_one(stock_code, name, df, target_date)
        logging.warning("未找到股票: %s", stock_code)
        return None

    # ── 全市场扫描 ──

    def scan(self, target_date: str) -> ScanResult:
        """扫描全市场，返回目标日期所有首日回归的股票。"""
        result = ScanResult(target_date=target_date)

        for (code, name), df in self.dm.stocks_data.items():
            result.total_scanned += 1
            code_str = str(code).zfill(6)

            if _is_excluded(code_str, str(name)):
                continue

            ar = self.check_one(code_str, str(name), df, target_date)
            if ar is None:
                continue
            if ar.is_first_day:
                result.items.append(ar)

        _risk_order = {"低": 0, "中": 1, "高": 2}
        result.items.sort(
            key=lambda x: (_risk_order.get(x.chase_risk, 9), -x.score, -x.ma5_slope_pct)
        )

        logging.info(
            'MAAlignment: 扫描 %d 只, 首日回归 %d 只',
            result.total_scanned, len(result.items),
        )
        return result

    # ── 获取交易日期 ──

    def get_latest_trade_date(self) -> str:
        for _key, df in self.dm.stocks_data.items():
            if df is not None and not df.empty and '日期' in df.columns:
                return str(df['日期'].iloc[-1])[:10]
        return date.today().isoformat()


# ══════════════════════════════════════════════════════════════
#  报告生成
# ══════════════════════════════════════════════════════════════

def format_single_result(ar: AlignmentResult) -> str:
    """格式化单只股票的检测结果。"""
    lines = []
    w = 55
    lines.append("═" * w)
    lines.append(f"  均线回归检测 — {ar.name} ({ar.code})")
    lines.append(f"  检测日期: {ar.target_date}    收盘: {ar.close}    涨跌: {ar.change_pct:+.2f}%")
    lines.append("═" * w)
    lines.append("")

    align_str = "✅ 是" if ar.is_alignment else "❌ 否"
    lines.append(f"  ▸ 均线回归状态:  {align_str}")

    if ar.is_alignment:
        first_str = "✅ 是（昨日不满足，今日首次形成）" if ar.is_first_day else "否（已处于回归中）"
        lines.append(f"  ▸ 首日回归:      {first_str}")
        lines.append(
            f"  ▸ 回归评分:      {ar.stars} {ar.score}/4"
            f"（{ar.score_label}）"
        )
    lines.append("")

    lines.append(f"  {'均线':<8s}  {'今日值':>10s}  {'昨日值':>10s}  {'方向':>4s}")
    lines.append(f"  {'────':<8s}  {'──────':>10s}  {'──────':>10s}  {'──':>4s}")
    for d in ar.ma_details:
        lines.append(
            f"  MA{d.period:<5d}  {d.value_today:>10.2f}  {d.value_prev:>10.2f}"
            f"    {d.direction_symbol}"
        )
    lines.append("")

    if ar.is_alignment:
        order_str = " > ".join(
            f"MA{d.period}({d.value_today:.2f})" for d in ar.ma_details
        )
        lines.append(f"  排列验证: {order_str} ✓")
        lines.append(f"  MA5 斜率: {ar.ma5_slope_pct:+.4f}% ✓")
        lines.append("")

        lines.append(f"  ── 追高风险评估 ──  {ar.chase_risk_symbol} {ar.chase_risk}风险")
        lines.append(f"  ▸ 价格偏离MA5:   {ar.bias5_pct:+.2f}%"
                     f"{'  ⚠️ 偏高' if ar.bias5_pct >= 3 else ''}")
        lines.append(f"  ▸ 均线扩散度:     {ar.ma_spread_pct:+.2f}%"
                     f"  (MA5与MA30距离)"
                     f"{'  ⚠️ 过宽' if ar.ma_spread_pct >= 5 else ''}")
        lines.append(f"  ▸ 近5日累计涨幅:  {ar.rise_5d_pct:+.2f}%"
                     f"{'  ⚠️ 急涨' if ar.rise_5d_pct >= 8 else ''}")
        lines.append("")

        advice = {
            4: "评分4分，全周期共振的理想回归信号，可关注介入。",
            3: "评分3分，多数周期共振，留意落后均线方向变化。",
            2: "评分2分，中线支撑不足，建议观望。",
            1: "评分1分，回归质量不佳，不建议操作。",
        }
        base_advice = advice.get(ar.score, '—')
        if ar.chase_risk == "高":
            base_advice += "\n  ⚠️ 追高风险高：价格已明显偏离均线或近期涨幅过大，建议等回踩再介入。"
        elif ar.chase_risk == "中":
            base_advice += "\n  ⚠️ 追高风险中：轻仓参与，注意设好止损。"
        lines.append(f"  💡 操作建议: {base_advice}")
    else:
        lines.append("  均线未形成多头排列或 MA5 未向上，当前不满足回归条件。")

    lines.append("═" * w)
    return "\n".join(lines)


def format_scan_result(result: ScanResult) -> str:
    """生成全市场扫描的 Markdown 报告。"""
    lines = []
    lines.append(f"## 均线回归首日扫描 ({result.target_date})")
    lines.append("")
    lines.append(
        f"> 扫描 {result.total_scanned} 只（排除ST/北交所/科创/创业板）"
        f" → 首日回归 {len(result.items)} 只"
    )
    lines.append("")

    if not result.items:
        lines.append("*当前无满足首日回归条件的标的。*")
        lines.append("")
        return "\n".join(lines)

    safe = [x for x in result.items if x.chase_risk == "低" and x.score >= 3]
    caution = [x for x in result.items if x.chase_risk == "中" and x.score >= 3]
    risky = [x for x in result.items if x.chase_risk == "高" and x.score >= 3]
    low_score = [x for x in result.items if x.score <= 2]

    def _render_table(items: List[AlignmentResult], show_chase: bool = True):
        header = (
            "| # | 评分 | 风险 | 代码 | 名称 | 收盘 | 涨跌% "
            "| 偏离MA5 | 扩散度 | 5日涨幅 | MA5斜率 |"
        )
        sep = (
            "|---|------|------|------|------|------|-------"
            "|---------|--------|---------|---------|"
        )
        lines.append(header)
        lines.append(sep)
        for i, ar in enumerate(items):
            lines.append(
                f"| {i+1} "
                f"| {ar.stars} "
                f"| {ar.chase_risk_symbol}{ar.chase_risk} "
                f"| {ar.code} "
                f"| {ar.name} "
                f"| {ar.close:.2f} "
                f"| {ar.change_pct:+.1f}% "
                f"| {ar.bias5_pct:+.1f}% "
                f"| {ar.ma_spread_pct:+.1f}% "
                f"| {ar.rise_5d_pct:+.1f}% "
                f"| {ar.ma5_slope_pct:+.2f}% |"
            )
        lines.append("")

    if safe:
        lines.append("### 🟢 安全回归（评分≥3 + 追高风险低）")
        lines.append("")
        _render_table(safe)

    if caution:
        lines.append("### 🟡 注意回归（评分≥3 + 追高风险中）")
        lines.append("")
        _render_table(caution)

    if risky:
        lines.append("### 🔴 追高回归（评分≥3 + 追高风险高，建议等回踩）")
        lines.append("")
        _render_table(risky)

    if low_score:
        lines.append(
            "<details><summary>低质量回归（评分≤2，共 %d 只，点击展开）</summary>\n"
            % len(low_score)
        )
        lines.append(
            "| # | 评分 | 风险 | 代码 | 名称 | 收盘 | 涨跌% | 偏离MA5 | 5日涨幅 |"
        )
        lines.append(
            "|---|------|------|------|------|------|-------|---------|---------|"
        )
        for i, ar in enumerate(low_score):
            lines.append(
                f"| {i+1} "
                f"| {ar.stars} "
                f"| {ar.chase_risk_symbol}{ar.chase_risk} "
                f"| {ar.code} "
                f"| {ar.name} "
                f"| {ar.close:.2f} "
                f"| {ar.change_pct:+.1f}% "
                f"| {ar.bias5_pct:+.1f}% "
                f"| {ar.rise_5d_pct:+.1f}% |"
            )
        lines.append("\n</details>\n")

    dist = result.score_distribution
    dist_str = "  ".join(f"{s}分={dist.get(s, 0)}只" for s in [4, 3, 2, 1])
    risk_counts = {"低": 0, "中": 0, "高": 0}
    for item in result.items:
        if item.chase_risk in risk_counts:
            risk_counts[item.chase_risk] += 1
    risk_str = f"🟢低={risk_counts['低']}  🟡中={risk_counts['中']}  🔴高={risk_counts['高']}"
    lines.append(f"> 评分分布: {dist_str}")
    lines.append(f"> 追高风险: {risk_str}")
    lines.append("")

    return "\n".join(lines)
