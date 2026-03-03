# -*- encoding: UTF-8 -*-

"""持仓卖出决策分析（三层七维度模型）

基于 market_temperature / industry_temperature / leader_stock 三大上游报告，
结合个股 K 线、分时、资金流、实时价格，给出每只持仓的量化卖出建议。

三层结构：
    宏观层 — 市场情绪退潮 + 系统性风险
    中观层 — 行业板块退潮 + 龙头地位变化
    微观层 — 技术形态恶化 + 量价异动 + 盈亏管理

用法：
    mt = MarketTemperature(dm)
    temp_result = mt.run()
    sp = SellPointAnalyzer(dm, temp_result)
    report = sp.run()
    print(report.overview_md)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from data_manager import DataManager
except ModuleNotFoundError:
    from ..data_manager import DataManager  # type: ignore[no-redef]

from report.market_temperature import TemperatureResult
from report.industry_temperature import SectorHeat
from report.leader_stock import LeaderStock


# ══════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════

class Signal(Enum):
    STRONG_SELL = "强烈卖出"
    SELL = "建议卖出"
    WATCH = "关注观望"
    HOLD = "继续持有"
    BUY_MORE = "可加仓"


@dataclass
class Holding:
    code: str
    name: str
    buy_price: float
    buy_date: str
    shares: int = 0
    buy_reason: str = ''

    @classmethod
    def from_config(cls, d: dict) -> 'Holding':
        return cls(
            code=str(d['code']).strip(),
            name=d.get('name', ''),
            buy_price=float(d.get('buy_price', 0)),
            buy_date=str(d.get('buy_date', '')),
            shares=int(d.get('shares', 0)),
            buy_reason=str(d.get('buy_reason', '')),
        )


@dataclass
class SellDimension:
    name: str
    layer: str           # 宏观/中观/微观
    signal: Signal
    score: float         # 0-100
    weight: float        # 百分比权重 (0.0-1.0)
    reason: str
    details: str = ''


@dataclass
class RelatedStockInfo:
    code: str
    name: str
    relation: str        # 板块龙头 / 同板块涨停 / 同概念
    today_change: float
    status: str          # 涨停/炸板/下跌/正常


@dataclass
class StockSellAnalysis:
    holding: Holding
    current_price: float = 0.0
    pnl_pct: float = 0.0
    pnl_amount: float = 0.0
    holding_days: int = 0
    peak_price: float = 0.0
    drawdown_pct: float = 0.0

    dimensions: List[SellDimension] = field(default_factory=list)

    overall_score: float = 0.0
    overall_signal: Signal = Signal.HOLD
    hard_rule_triggered: str = ''
    bonus_applied: List[str] = field(default_factory=list)

    sector_name: str = ''
    sector_phase: str = ''
    sector_score: float = 0.0
    is_leader: bool = False
    leader_grade: str = '非龙头'
    related_stocks: List[RelatedStockInfo] = field(default_factory=list)

    action: str = ''
    risk_warnings: List[str] = field(default_factory=list)
    positive_factors: List[str] = field(default_factory=list)


@dataclass
class SellPointReport:
    trade_date: str = ''
    market_score: float = 0.0
    market_phase: str = ''
    r_yu: float = 0.0
    analyses: List[StockSellAnalysis] = field(default_factory=list)
    overview_md: str = ''


# ══════════════════════════════════════════════════════════════
#  辅助
# ══════════════════════════════════════════════════════════════

def _clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, val))


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if v != v else v
    except (TypeError, ValueError):
        return default


def _score_to_signal(score: float) -> Signal:
    if score >= 80:
        return Signal.STRONG_SELL
    if score >= 60:
        return Signal.SELL
    if score >= 40:
        return Signal.WATCH
    if score >= 20:
        return Signal.HOLD
    return Signal.BUY_MORE


def _signal_icon(sig: Signal) -> str:
    return {
        Signal.STRONG_SELL: '🔴',
        Signal.SELL: '🟠',
        Signal.WATCH: '🟡',
        Signal.HOLD: '🟢',
        Signal.BUY_MORE: '🔵',
    }.get(sig, '⚪')


# ══════════════════════════════════════════════════════════════
#  主类
# ══════════════════════════════════════════════════════════════

class SellPointAnalyzer:
    """持仓卖出决策分析器（三层七维度）。"""

    # 各维度权重
    _WEIGHTS = {
        '市场情绪退潮': 0.15,
        '系统性风险':   0.10,
        '行业板块退潮': 0.20,
        '龙头地位变化': 0.15,
        '技术形态恶化': 0.15,
        '量价异动':     0.15,
        '盈亏管理':     0.10,
    }

    def __init__(self, dm: DataManager, temp_result: TemperatureResult):
        self._dm = dm
        self._tr = temp_result
        self._config = dm.config

        # 预构建查找表
        self._rt_lookup: Dict[str, pd.Series] = {}
        self._sector_map: Dict[str, SectorHeat] = {}
        self._leader_map: Dict[str, LeaderStock] = {}
        self._leader_by_sector: Dict[str, LeaderStock] = {}

        self._build_lookups()

    def _build_lookups(self):
        self._rt_lookup = self._dm.build_rt_lookup()

        for sh in (self._tr.hot_sectors or []):
            self._sector_map[sh.sector] = sh

        for ls in (self._tr.leader_stocks or []):
            self._leader_map[ls.code] = ls
            existing = self._leader_by_sector.get(ls.sector)
            if existing is None or ls.lianban > existing.lianban:
                self._leader_by_sector[ls.sector] = ls

    # ══════════════════════════════════════════════
    #  公开入口
    # ══════════════════════════════════════════════

    def run(self) -> SellPointReport:
        holdings = self._parse_holdings()
        if not holdings:
            logging.warning("stock_config.yaml 中未配置 holdings")
            return SellPointReport()

        report = SellPointReport(
            trade_date=self._tr.trade_date or '',
            market_score=self._tr.score,
            market_phase=self._tr.phase,
            r_yu=self._tr.metrics.R_yu,
        )

        for h in holdings:
            analysis = self._analyze_one(h)
            report.analyses.append(analysis)

        report.overview_md = self._build_markdown(report)
        return report

    # ══════════════════════════════════════════════
    #  持仓解析
    # ══════════════════════════════════════════════

    def _parse_holdings(self) -> List[Holding]:
        raw = self._config.get('holdings') or []
        result = []
        for item in raw:
            try:
                result.append(Holding.from_config(item))
            except (KeyError, ValueError) as e:
                logging.warning("解析持仓失败: %s → %s", item, e)
        return result

    # ══════════════════════════════════════════════
    #  数据辅助
    # ══════════════════════════════════════════════

    def _get_daily(self, code: str) -> Optional[pd.DataFrame]:
        for cn, df in self._dm.stocks_data.items():
            if cn[0] == code:
                return df
        return None

    def _get_fund_flow(self, code: str) -> Optional[pd.DataFrame]:
        ff_data = self._dm.extra.get('fund_flow', {})
        for cn, df in ff_data.items():
            if cn[0] == code:
                return df
        return None

    def _get_intraday(self, code: str) -> Optional[pd.DataFrame]:
        intraday = self._dm.extra.get('intraday', {})
        for cn, df in intraday.items():
            if cn[0] == code or cn == code:
                return df
        return None

    def _find_sector(self, code: str) -> Tuple[str, Optional[SectorHeat]]:
        """查找持仓股所属行业及其 SectorHeat。"""
        zt_df = self._dm.extra.get('zt_pool')
        if zt_df is not None and not zt_df.empty and '代码' in zt_df.columns:
            match = zt_df[zt_df['代码'].astype(str) == code]
            if not match.empty and '所属行业' in match.columns:
                sector = str(match.iloc[0]['所属行业'])
                sh = self._sector_map.get(sector)
                if sh is None:
                    for k, v in self._sector_map.items():
                        if k.startswith(sector) or sector.startswith(k):
                            return sector, v
                return sector, sh

        # 从 stock_sector_map 查找
        import os
        map_path = os.path.join(self._dm._cache.cache_dir, 'stock_sector_map.parquet')
        if os.path.exists(map_path):
            try:
                sm = pd.read_parquet(map_path)
                match = sm[sm['code'].astype(str) == code]
                if not match.empty:
                    sector = str(match.iloc[0]['sector'])
                    sh = self._sector_map.get(sector)
                    if sh is None:
                        for k, v in self._sector_map.items():
                            if k.startswith(sector) or sector.startswith(k):
                                return sector, v
                    return sector, sh
            except Exception:
                pass

        return '', None

    def _calc_holding_days(self, buy_date: str, daily: Optional[pd.DataFrame]) -> int:
        if not buy_date or daily is None or daily.empty:
            return 0
        try:
            bd = str(buy_date).replace('-', '')
            bd_fmt = f"{bd[:4]}-{bd[4:6]}-{bd[6:]}"
            mask = daily['日期'].astype(str) >= bd_fmt
            return int(mask.sum())
        except Exception:
            return 0

    # ══════════════════════════════════════════════
    #  单只股票分析
    # ══════════════════════════════════════════════

    def _analyze_one(self, h: Holding) -> StockSellAnalysis:
        sa = StockSellAnalysis(holding=h)

        # 基本数据
        rt = self._rt_lookup.get(h.code)
        if rt is not None:
            sa.current_price = _safe_float(rt.get('最新价'))

        if sa.current_price > 0 and h.buy_price > 0:
            sa.pnl_pct = (sa.current_price - h.buy_price) / h.buy_price * 100
            sa.pnl_amount = (sa.current_price - h.buy_price) * h.shares

        daily = self._get_daily(h.code)
        sa.holding_days = self._calc_holding_days(h.buy_date, daily)

        # 持仓期最高价和回撤
        if daily is not None and h.buy_date:
            try:
                bd = str(h.buy_date).replace('-', '')
                bd_fmt = f"{bd[:4]}-{bd[4:6]}-{bd[6:]}"
                since = daily[daily['日期'].astype(str) >= bd_fmt]
                if not since.empty:
                    sa.peak_price = float(since['最高'].astype(float).max())
                    if sa.peak_price > 0 and sa.current_price > 0:
                        sa.drawdown_pct = (sa.peak_price - sa.current_price) / sa.peak_price * 100
            except Exception:
                pass

        # 行业/龙头
        sa.sector_name, sector_heat = self._find_sector(h.code)
        if sector_heat:
            sa.sector_phase = sector_heat.phase
            sa.sector_score = sector_heat.score

        leader = self._leader_map.get(h.code)
        if leader:
            sa.is_leader = True
            sa.leader_grade = leader.grade

        # 七维度分析
        sa.dimensions = [
            self._dim1_market_sentiment(h),
            self._dim2_systemic_risk(),
            self._dim3_sector_retreat(h, sector_heat),
            self._dim4_leader_status(h, sector_heat),
            self._dim5_technical(h, daily, sa),
            self._dim6_volume_price(h, daily, sa),
            self._dim7_pnl_management(h, sa),
        ]

        # 综合加权评分
        raw_score = sum(d.score * d.weight for d in sa.dimensions)
        sa.overall_score = _clamp(raw_score)

        # 加分规则（降低卖出评分）
        sa.bonus_applied, bonus = self._apply_bonus(h, sa, sector_heat, leader)
        sa.overall_score = _clamp(sa.overall_score + bonus)

        # 硬性规则一票否决
        sa.hard_rule_triggered = self._check_hard_rules(h, sa, sector_heat)
        if sa.hard_rule_triggered:
            if '强烈卖出' in sa.hard_rule_triggered:
                sa.overall_score = max(sa.overall_score, 85.0)
            elif '建议卖出' in sa.hard_rule_triggered:
                sa.overall_score = max(sa.overall_score, 65.0)

        sa.overall_signal = _score_to_signal(sa.overall_score)

        # 关联股票
        sa.related_stocks = self._find_related(h, sector_heat)

        # 风险和正面因素
        sa.risk_warnings = self._collect_risks(sa, leader)
        sa.positive_factors = self._collect_positives(sa, leader)

        # 操作建议
        sa.action = self._suggest_action(sa)

        return sa

    # ══════════════════════════════════════════════
    #  维度 1：市场情绪退潮（宏观，15%）
    # ══════════════════════════════════════════════

    def _dim1_market_sentiment(self, _h: Holding) -> SellDimension:
        w = self._WEIGHTS['市场情绪退潮']
        score = self._tr.score
        phase = self._tr.phase
        m = self._tr.metrics

        # 基础分：不仅看情绪阶段，还要用原始指标修正
        # 情绪总分可能被 R_yu 或空间溢价拉高，但 D/M 极端时不能掉以轻心
        sell_score = 10.0
        reason = f'市场情绪 {score:.0f} 分（{phase}），安全'

        if phase == '冰点期':
            sell_score = 75.0
            reason = f'市场冰点（{score:.0f}分），全市场赚钱效应极差'
        elif phase == '混沌/修复期':
            sell_score = 40.0
            reason = f'市场混沌（{score:.0f}分），方向不明'
        elif phase == '高潮期':
            sell_score = 35.0
            reason = f'市场高潮（{score:.0f}分），盛极必衰，准备兑现'
        elif phase == '主升/发酵期':
            sell_score = 10.0
            reason = f'市场主升（{score:.0f}分），赚钱效应好'

        # ── 叠加关键指标（与严重程度成比例） ──

        # R_yu < 0：接力链断裂
        if m.R_yu < 0:
            sell_score += 20
            reason += f'；R_yu={m.R_yu:+.1f}%（接力链断裂）'

        # D+M 恐慌指标：按严重程度分级，而非固定值
        panic = m.D + 0.5 * m.M
        if panic >= 60:
            sell_score += 40
            reason += f'；跌停{m.D}+大面{m.M}（Panic={panic:.0f}，极端恐慌）'
        elif panic >= 30:
            penalty = 20 + (panic - 30) / 30 * 20
            sell_score += penalty
            reason += f'；跌停{m.D}+大面{m.M}（Panic={panic:.0f}，恐慌蔓延）'
        elif panic >= 10:
            penalty = 5 + (panic - 10) / 20 * 15
            sell_score += penalty
            reason += f'；跌停{m.D}+大面{m.M}（偏恐慌）'

        # 空间龙断裂
        if m.H > 0:
            prev_df = self._dm.extra.get('zt_pool_previous')
            if prev_df is not None and not prev_df.empty:
                lb_col = '昨日连板数' if '昨日连板数' in prev_df.columns else '连板数'
                if lb_col in prev_df.columns:
                    prev_h = int(pd.to_numeric(prev_df[lb_col], errors='coerce').max())
                    if prev_h > 0 and m.H < prev_h - 2:
                        sell_score += 15
                        reason += f'；空间龙断裂（{prev_h}→{m.H}板）'

        sell_score = _clamp(sell_score)
        return SellDimension(
            name='市场情绪退潮', layer='宏观',
            signal=_score_to_signal(sell_score),
            score=sell_score, weight=w, reason=reason,
        )

    # ══════════════════════════════════════════════
    #  维度 2：系统性风险（宏观，10%）
    # ══════════════════════════════════════════════

    def _dim2_systemic_risk(self) -> SellDimension:
        w = self._WEIGHTS['系统性风险']
        m = self._tr.metrics

        sell_score = 5.0
        reason = '市场无异常'

        if m.D >= 30:
            sell_score = 95.0
            reason = f'跌停{m.D}家，极端恐慌'
        elif m.M >= 15:
            sell_score = 90.0
            reason = f'大面{m.M}家，市场崩溃信号'
        elif m.FR < 0.40 and m.U < 20:
            sell_score = 65.0
            reason = f'封板率{m.FR:.0%}+涨停仅{m.U}家，赚钱效应极差'
        elif m.PR < 0.05 and m.U > 0:
            sell_score = 45.0
            reason = f'晋级率{m.PR:.1%}，连板接力断裂'

        return SellDimension(
            name='系统性风险', layer='宏观',
            signal=_score_to_signal(sell_score),
            score=sell_score, weight=w, reason=reason,
        )

    # ══════════════════════════════════════════════
    #  维度 3：行业板块退潮（中观，20%）
    # ══════════════════════════════════════════════

    def _dim3_sector_retreat(
        self, h: Holding, sh: Optional[SectorHeat],
    ) -> SellDimension:
        w = self._WEIGHTS['行业板块退潮']

        if sh is None:
            return SellDimension(
                name='行业板块退潮', layer='中观',
                signal=Signal.WATCH, score=35.0, weight=w,
                reason='持仓股不在当前热点行业中，缺乏板块资金关注',
            )

        phase = sh.phase
        is_leader = h.code in self._leader_map

        sell_score = 15.0
        reason = f'{sh.sector}（热度{sh.score:.0f}，{phase}）'

        if phase == '退潮':
            sell_score = 90.0
            reason += ' — 板块退潮，资金撤退'
        elif phase == '高潮' and not is_leader:
            sell_score = 70.0
            reason += ' — 高潮期非龙头，最先被砸'
        elif phase == '高潮':
            sell_score = 45.0
            reason += ' — 高潮期，准备兑现'
        elif phase == '发酵':
            sell_score = 10.0
            reason += ' — 板块上升期，持股享溢价'
        elif phase == '启动':
            sell_score = 5.0
            reason += ' — 板块刚起步'
        else:
            sell_score = 15.0
            reason += ' — 平稳运行'

        # 板块热度骤降：比较今日与昨日涨停数
        prev_df = self._dm.extra.get('zt_pool_previous')
        if prev_df is not None and not prev_df.empty and '所属行业' in prev_df.columns:
            prev_snap = prev_df.copy()
            if '快照日期' in prev_snap.columns:
                prev_snap = prev_snap[prev_snap['快照日期'] == prev_snap['快照日期'].max()]
            prev_sector = prev_snap[prev_snap['所属行业'].astype(str).str.startswith(sh.sector[:3])]
            prev_zt_count = len(prev_sector)
            if prev_zt_count >= 3 and sh.zt_count <= prev_zt_count * 0.5:
                sell_score += 15
                reason += f'；涨停骤降（{prev_zt_count}→{sh.zt_count}）'

        sell_score = _clamp(sell_score)
        return SellDimension(
            name='行业板块退潮', layer='中观',
            signal=_score_to_signal(sell_score),
            score=sell_score, weight=w, reason=reason,
        )

    # ══════════════════════════════════════════════
    #  维度 4：龙头地位变化（中观，15%）
    # ══════════════════════════════════════════════

    def _dim4_leader_status(
        self, h: Holding, sh: Optional[SectorHeat],
    ) -> SellDimension:
        w = self._WEIGHTS['龙头地位变化']
        leader = self._leader_map.get(h.code)

        if leader is None:
            # 不在龙头列表中
            # 检查是否曾经是涨停股（连板断裂）
            prev_df = self._dm.extra.get('zt_pool_previous')
            was_zt = False
            if prev_df is not None and not prev_df.empty and '代码' in prev_df.columns:
                was_zt = h.code in set(prev_df['代码'].astype(str))

            if was_zt:
                return SellDimension(
                    name='龙头地位变化', layer='中观',
                    signal=Signal.STRONG_SELL, score=85.0, weight=w,
                    reason='昨日涨停今日掉出龙头池，连板断裂',
                )
            return SellDimension(
                name='龙头地位变化', layer='中观',
                signal=Signal.WATCH, score=40.0, weight=w,
                reason='持仓股不在龙头列表中，非热点标的',
            )

        grade = leader.grade
        sell_score = 15.0
        reason = f'{leader.grade}级龙头（{leader.lianban}板，{leader.lifecycle}）'

        if grade == 'S' and sh and sh.phase == '发酵':
            sell_score = 0.0
            reason += ' — 最强位置，持筹不动'
        elif grade == 'S' and sh and sh.phase == '高潮':
            sell_score = 35.0
            reason += ' — 高潮兑现窗口'
        elif grade == 'S':
            sell_score = 5.0
            reason += ' — S级龙头'
        elif grade == 'A':
            sell_score = 15.0
            reason += ' — 准龙头'
        elif grade in ('B+', 'B'):
            sell_score = 50.0
            reason += ' — 龙头身份存疑'
        elif grade in ('C', 'D'):
            sell_score = 65.0
            reason += ' — 跟风股特征，不是龙头就是跟风'

        # 龙头交接检测
        if sh and leader.lianban < sh.max_lianban:
            sector_top = self._leader_by_sector.get(sh.sector)
            if sector_top and sector_top.code != h.code:
                sell_score += 20
                reason += f'；龙头交接→{sector_top.name}({sector_top.lianban}板)'
                sell_score = min(sell_score, 100.0)

        # 风险信号叠加
        if leader.risk_flags:
            for flag in leader.risk_flags:
                if '退潮' in flag:
                    sell_score += 20
                elif '高位' in flag or '高潮' in flag:
                    sell_score += 10
                elif '一字板' in flag:
                    sell_score += 10

        sell_score = _clamp(sell_score)
        return SellDimension(
            name='龙头地位变化', layer='中观',
            signal=_score_to_signal(sell_score),
            score=sell_score, weight=w, reason=reason,
        )

    # ══════════════════════════════════════════════
    #  维度 5：技术形态恶化（微观，15%）
    # ══════════════════════════════════════════════

    def _dim5_technical(
        self, h: Holding, daily: Optional[pd.DataFrame],
        sa: StockSellAnalysis,
    ) -> SellDimension:
        w = self._WEIGHTS['技术形态恶化']

        if daily is None or len(daily) < 10:
            return SellDimension(
                name='技术形态恶化', layer='微观',
                signal=Signal.HOLD, score=0.0, weight=w,
                reason='日K数据不足，无法判断技术形态',
            )

        close = daily['收盘'].astype(float)
        high = daily['最高'].astype(float)
        low = daily['最低'].astype(float)
        opn = daily['开盘'].astype(float)
        last_close = float(close.iloc[-1])

        sub_scores: List[Tuple[float, str]] = []

        # 5a. 均线系统
        ma_score, ma_reason = self._check_ma(close, last_close)
        sub_scores.append((ma_score, ma_reason))

        # 5b. 关键支撑位
        if h.buy_date and sa.peak_price > 0:
            sup_score, sup_reason = self._check_support(daily, h.buy_date, sa.current_price, sa.peak_price)
            sub_scores.append((sup_score, sup_reason))

        # 5c. K线形态
        kline_score, kline_reason = self._check_kline_pattern(close, high, low, opn)
        sub_scores.append((kline_score, kline_reason))

        # 5d. 持仓天数与横盘
        if sa.holding_days > 20 and abs(sa.pnl_pct) < 5:
            sub_scores.append((10.0, f'持仓{sa.holding_days}天横盘，时间成本消耗'))
        if sa.holding_days > 30 and sa.pnl_pct < 0:
            sub_scores.append((15.0, f'持仓{sa.holding_days}天且亏损{sa.pnl_pct:.1f}%'))

        if not sub_scores:
            return SellDimension(
                name='技术形态恶化', layer='微观',
                signal=Signal.HOLD, score=5.0, weight=w, reason='技术面正常',
            )

        best = max(sub_scores, key=lambda x: x[0])
        sell_score = _clamp(best[0])
        all_reasons = '；'.join(r for _, r in sub_scores if _ >= 20)
        reason = all_reasons if all_reasons else best[1]

        return SellDimension(
            name='技术形态恶化', layer='微观',
            signal=_score_to_signal(sell_score),
            score=sell_score, weight=w, reason=reason,
        )

    @staticmethod
    def _check_ma(close: pd.Series, last_close: float) -> Tuple[float, str]:
        n = len(close)
        ma5 = float(close.iloc[-5:].mean()) if n >= 5 else last_close
        ma10 = float(close.iloc[-10:].mean()) if n >= 10 else last_close
        ma20 = float(close.iloc[-20:].mean()) if n >= 20 else last_close
        ma60 = float(close.iloc[-60:].mean()) if n >= 60 else ma20

        if last_close < ma60 and ma5 < ma20:
            return 90.0, f'跌破MA60({ma60:.2f})且均线空头，趋势破坏'
        if last_close < ma20 and ma5 < ma10:
            return 70.0, f'跌破MA20({ma20:.2f})且MA5<MA10，中期空头'
        if last_close < ma20:
            return 45.0, f'跌破MA20({ma20:.2f})，待确认'
        if ma5 < ma10:
            return 30.0, 'MA5下穿MA10，短期死叉'
        return 5.0, '均线多头排列，趋势健康'

    @staticmethod
    def _check_support(
        daily: pd.DataFrame, buy_date: str,
        current: float, peak: float,
    ) -> Tuple[float, str]:
        try:
            bd = str(buy_date).replace('-', '')
            bd_fmt = f"{bd[:4]}-{bd[4:6]}-{bd[6:]}"
            since = daily[daily['日期'].astype(str) >= bd_fmt]
            if since.empty or current <= 0:
                return 0.0, ''
            period_low = float(since['最低'].astype(float).min())
            half = (peak + period_low) / 2
            if current < period_low:
                return 85.0, f'跌破前低{period_low:.2f}，支撑完全失守'
            if current < half:
                return 55.0, f'跌破半分位{half:.2f}'
        except Exception:
            pass
        return 5.0, ''

    @staticmethod
    def _check_kline_pattern(
        close: pd.Series, high: pd.Series, low: pd.Series, opn: pd.Series,
    ) -> Tuple[float, str]:
        if len(close) < 3:
            return 0.0, ''

        c = float(close.iloc[-1])
        hi = float(high.iloc[-1])
        o = float(opn.iloc[-1])
        body = abs(c - o)
        upper = hi - max(c, o)
        prev_o = float(opn.iloc[-2])

        # 长上影线
        if body > 0 and upper >= 2 * body:
            return 50.0, f'长上影线（上影{upper:.2f}，实体{body:.2f}）'

        # 吞没阴线
        chg = (c - float(close.iloc[-2])) / float(close.iloc[-2]) * 100 if float(close.iloc[-2]) > 0 else 0
        if chg < -3 and c < prev_o:
            return 65.0, f'吞没阴线（跌{chg:.1f}%，吃掉前日阳线）'

        # 跳空低开
        prev_low = float(low.iloc[-2])
        if o < prev_low:
            return 60.0, f'跳空低开（开盘{o:.2f}<昨低{prev_low:.2f}）'

        # 连续阴线
        recent_3 = [(float(close.iloc[-i]) < float(opn.iloc[-i])) for i in range(1, 4)]
        if all(recent_3):
            return 55.0, '连续3日收阴'

        return 5.0, 'K线形态正常'

    # ══════════════════════════════════════════════
    #  维度 6：量价异动（微观，15%）
    # ══════════════════════════════════════════════

    def _dim6_volume_price(
        self, h: Holding, daily: Optional[pd.DataFrame],
        sa: StockSellAnalysis,
    ) -> SellDimension:
        w = self._WEIGHTS['量价异动']

        sub_scores: List[Tuple[float, str]] = []

        # 6a. 成交量分析
        if daily is not None and len(daily) >= 10:
            vol_score, vol_reason = self._check_volume(daily, sa.current_price)
            sub_scores.append((vol_score, vol_reason))

        # 6b. 分时图分析
        intraday = self._get_intraday(h.code)
        if intraday is not None and not intraday.empty:
            it_score, it_reason = self._check_intraday(intraday)
            sub_scores.append((it_score, it_reason))

        # 6c. 资金流分析
        ff = self._get_fund_flow(h.code)
        if ff is not None and not ff.empty:
            ff_score, ff_reason = self._check_fund_flow(ff)
            sub_scores.append((ff_score, ff_reason))

        # 6d. 大单追踪
        bd_score, bd_reason = self._check_big_deal(h.code)
        if bd_score > 0:
            sub_scores.append((bd_score, bd_reason))

        if not sub_scores:
            return SellDimension(
                name='量价异动', layer='微观',
                signal=Signal.HOLD, score=0.0, weight=w,
                reason='量价数据不足',
            )

        best = max(sub_scores, key=lambda x: x[0])
        sell_score = _clamp(best[0])
        all_reasons = '；'.join(r for _, r in sub_scores if _ >= 30)
        reason = all_reasons if all_reasons else best[1]

        return SellDimension(
            name='量价异动', layer='微观',
            signal=_score_to_signal(sell_score),
            score=sell_score, weight=w, reason=reason,
        )

    @staticmethod
    def _check_volume(daily: pd.DataFrame, _current_price: float) -> Tuple[float, str]:
        close = daily['收盘'].astype(float)
        vol = daily['成交量'].astype(float)
        high = daily['最高'].astype(float)

        avg10 = float(vol.iloc[-10:].mean()) if len(vol) >= 10 else float(vol.mean())
        today_vol = float(vol.iloc[-1])
        vol_ratio = today_vol / avg10 if avg10 > 0 else 1.0

        last_chg = 0.0
        if len(close) >= 2 and float(close.iloc[-2]) > 0:
            last_chg = (float(close.iloc[-1]) - float(close.iloc[-2])) / float(close.iloc[-2]) * 100

        # 放量大跌
        if vol_ratio > 2 and last_chg < -3:
            return 80.0, f'放量大跌{last_chg:.1f}%（量比{vol_ratio:.1f}x），恐慌出逃'

        # 量价顶背离
        if len(high) >= 5:
            h5_max = float(high.iloc[-5:].max())
            today_high = float(high.iloc[-1])
            if today_high >= h5_max * 0.99:
                vols = vol.iloc[-5:].values
                if len(vols) >= 3 and all(vols[i] >= vols[i + 1] for i in range(len(vols) - 1)):
                    return 70.0, '价格新高但量能连续递减，量价顶背离'

        # 放量滞涨
        if vol_ratio > 1.5 and len(close) >= 4:
            recent_chg = (float(close.iloc[-1]) - float(close.iloc[-4])) / float(close.iloc[-4]) * 100
            if abs(recent_chg) < 2:
                return 65.0, f'放量滞涨（量比{vol_ratio:.1f}x，近3日涨跌{recent_chg:.1f}%）'

        # 放量下跌
        if vol_ratio > 1.5 and last_chg < -2:
            return 50.0, f'温和放量下跌{last_chg:.1f}%（量比{vol_ratio:.1f}x）'

        # 缩量上涨
        if vol_ratio < 0.5 and last_chg > 0:
            return 30.0, '上涨但极度缩量，动能衰减'

        return 5.0, '量价关系正常'

    @staticmethod
    def _check_intraday(intraday: pd.DataFrame) -> Tuple[float, str]:
        if intraday.empty or len(intraday) < 5:
            return 0.0, ''

        try:
            close_col = '收盘' if '收盘' in intraday.columns else 'close'
            if close_col not in intraday.columns:
                return 0.0, ''

            prices = intraday[close_col].astype(float)
            if len(prices) < 5:
                return 0.0, ''

            first = float(prices.iloc[0])
            last = float(prices.iloc[-1])
            if first <= 0:
                return 0.0, ''

            # 高开低走
            early_prices = prices.iloc[:min(12, len(prices) // 3)]
            early_high_chg = (float(early_prices.max()) - first) / first * 100
            full_chg = (last - first) / first * 100
            if early_high_chg > 2 and full_chg < 0:
                return 60.0, f'高开低走（早盘涨{early_high_chg:.1f}%→收{full_chg:.1f}%）'

            # 尾盘急跌
            tail_start = max(0, len(prices) - 6)
            tail_chg = (float(prices.iloc[-1]) - float(prices.iloc[tail_start])) / float(prices.iloc[tail_start]) * 100
            if tail_chg < -2:
                return 55.0, f'尾盘急跌{tail_chg:.1f}%'

            # 全天低位运行
            mid = (float(prices.max()) + float(prices.min())) / 2
            below_mid_pct = (prices < mid).sum() / len(prices)
            if below_mid_pct > 0.7 and full_chg < -1:
                return 35.0, '全天低位横盘运行'

        except Exception:
            return 0.0, ''

        return 5.0, '分时正常'

    @staticmethod
    def _check_fund_flow(ff: pd.DataFrame) -> Tuple[float, str]:
        if '主力净流入-净额' not in ff.columns:
            return 0.0, '无资金流数据'

        recent = ff.tail(5)
        main_flow = recent['主力净流入-净额'].astype(float)

        # 连续3日主力净流出
        last3 = main_flow.iloc[-3:] if len(main_flow) >= 3 else main_flow
        if len(last3) >= 3 and (last3 < 0).all():
            total_out = float(last3.sum())
            if abs(total_out) > 50_000_000:
                return 75.0, f'连续3日主力净流出共{total_out / 1e4:.0f}万'
            return 60.0, f'连续3日主力净流出共{total_out / 1e4:.0f}万'

        # 超大单
        if '超大单净流入-净额' in recent.columns:
            super_flow = recent['超大单净流入-净额'].astype(float)
            last3_s = super_flow.iloc[-3:] if len(super_flow) >= 3 else super_flow
            if len(last3_s) >= 3 and (last3_s < 0).all():
                return 60.0, '超大单连续3日净流出'

        # 今日主力净流出
        today_flow = float(main_flow.iloc[-1]) if len(main_flow) > 0 else 0
        if today_flow < -100_000_000:
            return 55.0, f'今日主力净流出{today_flow / 1e4:.0f}万'

        # 近5日累计
        total_5d = float(main_flow.sum())
        if total_5d < -30_000_000:
            return 45.0, f'近5日主力累计净流出{total_5d / 1e4:.0f}万'

        if total_5d > 50_000_000:
            return 5.0, f'近5日主力净流入{total_5d / 1e4:.0f}万，安全'

        return 15.0, '资金流向中性'

    def _check_big_deal(self, code: str) -> Tuple[float, str]:
        bd = self._dm.extra.get('big_deal')
        if bd is None or (hasattr(bd, 'empty') and bd.empty):
            return 0.0, ''

        try:
            df = bd if isinstance(bd, pd.DataFrame) else None
            if df is None:
                return 0.0, ''
            if '代码' not in df.columns and '股票代码' not in df.columns:
                return 0.0, ''

            code_col = '代码' if '代码' in df.columns else '股票代码'
            stock_deals = df[df[code_col].astype(str) == code]
            if stock_deals.empty:
                return 0.0, ''

            buy_col = sell_col = None
            for c in stock_deals.columns:
                if '买' in c and '金额' in c:
                    buy_col = c
                if '卖' in c and '金额' in c:
                    sell_col = c

            if buy_col and sell_col:
                total_buy = _safe_float(stock_deals[buy_col].astype(float).sum())
                total_sell = _safe_float(stock_deals[sell_col].astype(float).sum())
                if total_buy > 0 and total_sell > total_buy * 2:
                    return 65.0, f'大单卖出{total_sell / 1e4:.0f}万 > 买入{total_buy / 1e4:.0f}万的2倍'
        except Exception:
            pass

        return 0.0, ''

    # ══════════════════════════════════════════════
    #  维度 7：盈亏管理（微观，10%）
    # ══════════════════════════════════════════════

    def _dim7_pnl_management(
        self, h: Holding, sa: StockSellAnalysis,
    ) -> SellDimension:
        w = self._WEIGHTS['盈亏管理']

        if h.buy_price <= 0 or sa.current_price <= 0:
            return SellDimension(
                name='盈亏管理', layer='微观',
                signal=Signal.HOLD, score=0.0, weight=w,
                reason='买入价或现价缺失',
            )

        pnl = sa.pnl_pct
        sub_scores: List[Tuple[float, str]] = []

        # 7a. 硬止损
        if pnl <= -8:
            sub_scores.append((95.0, f'浮亏{pnl:.1f}%，触及8%硬止损线'))
        elif pnl <= -5:
            sub_scores.append((70.0, f'浮亏{pnl:.1f}%，接近止损线'))
        elif pnl <= -2:
            sub_scores.append((35.0, f'浮亏{pnl:.1f}%'))

        # 7b. 移动止盈
        if sa.peak_price > 0 and sa.drawdown_pct > 0:
            threshold = self._trailing_stop_threshold(pnl)
            if sa.drawdown_pct >= threshold:
                sub_scores.append((
                    75.0,
                    f'从最高{sa.peak_price:.2f}回撤{sa.drawdown_pct:.1f}%，'
                    f'触发{threshold:.0f}%移动止盈',
                ))
            elif sa.drawdown_pct >= threshold - 2:
                sub_scores.append((
                    45.0,
                    f'从最高{sa.peak_price:.2f}回撤{sa.drawdown_pct:.1f}%，'
                    f'接近{threshold:.0f}%移动止盈线',
                ))

        # 7c. 盈利兑现
        if pnl >= 50 and sa.holding_days > 10:
            sub_scores.append((40.0, f'盈利{pnl:.1f}%，关注兑现时机'))
        elif pnl >= 30:
            sub_scores.append((25.0, f'盈利{pnl:.1f}%，关注止盈'))
        elif pnl >= 10:
            sub_scores.append((10.0, f'盈利{pnl:.1f}%，持股待涨'))

        if not sub_scores:
            return SellDimension(
                name='盈亏管理', layer='微观',
                signal=Signal.HOLD, score=0.0, weight=w,
                reason=f'盈亏{pnl:+.1f}%，无止损/止盈触发',
            )

        best = max(sub_scores, key=lambda x: x[0])
        sell_score = _clamp(best[0])
        all_reasons = '；'.join(r for _, r in sub_scores if _ >= 30)
        reason = all_reasons if all_reasons else best[1]

        return SellDimension(
            name='盈亏管理', layer='微观',
            signal=_score_to_signal(sell_score),
            score=sell_score, weight=w, reason=reason,
        )

    @staticmethod
    def _trailing_stop_threshold(pnl_pct: float) -> float:
        if pnl_pct > 50:
            return 25.0
        if pnl_pct > 20:
            return 15.0
        if pnl_pct > 10:
            return 10.0
        return 8.0

    # ══════════════════════════════════════════════
    #  硬性规则（一票否决）
    # ══════════════════════════════════════════════

    def _check_hard_rules(
        self, h: Holding, sa: StockSellAnalysis,
        sh: Optional[SectorHeat],
    ) -> str:
        # 浮亏 >= 8%
        if sa.pnl_pct <= -8:
            return '🔴 强烈卖出：浮亏≥8%硬止损'

        # 板块退潮 + 非S级龙头
        if sh and sh.phase == '退潮' and sa.leader_grade not in ('S',):
            return '🟠 建议卖出：板块退潮且非S级龙头'

        # 市场冰点 + 接力断裂
        if self._tr.score < 15 and self._tr.metrics.R_yu < -1:
            return '🟠 建议卖出：市场冰点+接力链断裂'

        # 跌停
        rt = self._rt_lookup.get(h.code)
        if rt is not None:
            chg = _safe_float(rt.get('涨跌幅'))
            if chg <= -9.5:
                return '🔴 强烈卖出：已跌停'

        return ''

    # ══════════════════════════════════════════════
    #  加分规则（降低卖出评分）
    # ══════════════════════════════════════════════

    def _apply_bonus(
        self, h: Holding, _sa: StockSellAnalysis,
        sh: Optional[SectorHeat], leader: Optional[LeaderStock],
    ) -> Tuple[List[str], float]:
        applied: List[str] = []
        bonus = 0.0

        # S级龙头 + 板块发酵/主升
        if leader and leader.grade == 'S' and sh and sh.phase in ('发酵', '启动'):
            applied.append('S级龙头+板块上升期(-20)')
            bonus -= 20

        # 市场情绪好 + 溢价高
        if self._tr.score > 60 and self._tr.metrics.R_yu > 2:
            applied.append('市场情绪好+溢价高(-10)')
            bonus -= 10

        # 今日涨停封板
        zt_df = self._dm.extra.get('zt_pool')
        if zt_df is not None and not zt_df.empty and '代码' in zt_df.columns:
            if h.code in set(zt_df['代码'].astype(str)):
                applied.append('今日涨停封板(-15)')
                bonus -= 15

        # 北向资金大幅净买入
        hsgt = self._dm.extra.get('hsgt_flow')
        if hsgt is not None and not (hasattr(hsgt, 'empty') and hsgt.empty):
            try:
                if isinstance(hsgt, pd.DataFrame) and not hsgt.empty:
                    for col in hsgt.columns:
                        if '北向' in col and '净' in col and '买' in col:
                            val = _safe_float(hsgt[col].iloc[-1])
                            if val > 50e8:
                                applied.append('北向大幅净买入(-5)')
                                bonus -= 5
                            break
            except Exception:
                pass

        return applied, bonus

    # ══════════════════════════════════════════════
    #  关联股票分析
    # ══════════════════════════════════════════════

    def _find_related(
        self, h: Holding, sh: Optional[SectorHeat],
    ) -> List[RelatedStockInfo]:
        related: List[RelatedStockInfo] = []

        if sh is None:
            return related

        # 板块龙头
        sector_leader = self._leader_by_sector.get(sh.sector)
        if sector_leader and sector_leader.code != h.code:
            rt = self._rt_lookup.get(sector_leader.code)
            chg = _safe_float(rt.get('涨跌幅')) if rt is not None else 0
            status = '涨停' if chg >= 9.5 else ('下跌' if chg < -1 else '正常')
            related.append(RelatedStockInfo(
                code=sector_leader.code, name=sector_leader.name,
                relation=f'板块龙头({sector_leader.lianban}板)',
                today_change=chg, status=status,
            ))

        # 同板块其他龙头股（最多3只）
        count = 0
        for ls in (self._tr.leader_stocks or []):
            if ls.sector == sh.sector and ls.code != h.code:
                if sector_leader and ls.code == sector_leader.code:
                    continue
                rt = self._rt_lookup.get(ls.code)
                chg = _safe_float(rt.get('涨跌幅')) if rt is not None else 0
                status = '涨停' if chg >= 9.5 else ('下跌' if chg < -1 else '正常')
                related.append(RelatedStockInfo(
                    code=ls.code, name=ls.name,
                    relation=f'同板块{ls.grade}级({ls.lianban}板)',
                    today_change=chg, status=status,
                ))
                count += 1
                if count >= 3:
                    break

        return related

    # ══════════════════════════════════════════════
    #  风险/正面收集
    # ══════════════════════════════════════════════

    def _collect_risks(
        self, sa: StockSellAnalysis, leader: Optional[LeaderStock],
    ) -> List[str]:
        risks: List[str] = []
        if sa.hard_rule_triggered:
            risks.append(sa.hard_rule_triggered)
        if leader and leader.risk_flags:
            risks.extend(leader.risk_flags)
        for d in sa.dimensions:
            if d.score >= 70:
                risks.append(f'{d.name}：{d.reason}')
        if self._tr.phase == '高潮期':
            risks.append('市场高潮期，注意高潮后分歧下杀')
        if sa.drawdown_pct > 10:
            risks.append(f'从最高点回撤{sa.drawdown_pct:.1f}%')
        return risks

    def _collect_positives(
        self, sa: StockSellAnalysis, leader: Optional[LeaderStock],
    ) -> List[str]:
        pos: List[str] = []
        if leader and leader.grade in ('S', 'A'):
            pos.append(f'{leader.grade}级龙头，市场共识强')
        if sa.sector_phase in ('发酵', '启动'):
            pos.append(f'板块{sa.sector_phase}期，上升趋势')
        if self._tr.score >= 50 and self._tr.metrics.R_yu > 0:
            pos.append(f'市场赚钱效应好（R_yu={self._tr.metrics.R_yu:+.1f}%）')
        if sa.pnl_pct > 0:
            pos.append(f'浮盈{sa.pnl_pct:.1f}%')
        for b in sa.bonus_applied:
            pos.append(b)
        return pos

    # ══════════════════════════════════════════════
    #  操作建议
    # ══════════════════════════════════════════════

    def _suggest_action(self, sa: StockSellAnalysis) -> str:
        sig = sa.overall_signal

        if sig == Signal.STRONG_SELL:
            if sa.pnl_pct <= -8:
                return '立即止损清仓，严守纪律'
            return '尽快卖出，不犹豫'

        if sig == Signal.SELL:
            if sa.sector_phase == '退潮':
                return '板块退潮，明日开盘择机卖出'
            if sa.drawdown_pct > 10:
                return f'回撤{sa.drawdown_pct:.1f}%，建议至少减半仓'
            return '择机卖出，至少减半仓'

        if sig == Signal.WATCH:
            parts = []
            if sa.peak_price > 0 and sa.pnl_pct > 0:
                # 移动止盈位
                threshold = self._trailing_stop_threshold(sa.pnl_pct)
                stop = sa.peak_price * (1 - threshold / 100)
                parts.append(f'设移动止盈位{stop:.2f}')
            if sa.leader_grade in ('S', 'A'):
                parts.append('龙头持筹，密切关注竞价和板块变化')
            else:
                parts.append('密切关注，准备随时撤退')
            return '；'.join(parts) if parts else '设好止损，密切观察'

        if sig == Signal.HOLD:
            if sa.is_leader and sa.sector_phase in ('发酵', '启动'):
                return '核心持股，享受板块上升溢价'
            return '安全持股，无需操作'

        # BUY_MORE
        return '多头信号强，可考虑加仓'

    # ══════════════════════════════════════════════
    #  Markdown 报告
    # ══════════════════════════════════════════════

    def _build_markdown(self, report: SellPointReport) -> str:
        lines: List[str] = []
        tr = self._tr

        # 标题
        lines.append('# 持仓卖出决策分析')
        lines.append('')
        lines.append(
            f'> 交易日：{report.trade_date} ｜ '
            f'市场情绪：{tr.score:.1f} 分（{tr.phase}）｜ '
            f'R_yu：{tr.metrics.R_yu:+.1f}%'
        )
        lines.append('')
        lines.append('---')
        lines.append('')

        # 持仓总览
        lines.append('## 持仓总览')
        lines.append('')
        lines.append('| 股票 | 买入价 | 现价 | 盈亏 | 持仓天数 | 最高回撤 | 信号 | 建议 |')
        lines.append('|------|-------|------|------|---------|---------|------|------|')

        for sa in report.analyses:
            icon = _signal_icon(sa.overall_signal)
            h = sa.holding
            pnl_str = f'{sa.pnl_pct:+.1f}%'
            dd_str = f'{sa.drawdown_pct:.1f}%' if sa.drawdown_pct > 0 else '—'
            short_action = sa.action[:20] + '…' if len(sa.action) > 20 else sa.action
            lines.append(
                f'| {icon} {h.name}({h.code}) '
                f'| {h.buy_price:.2f} '
                f'| {sa.current_price:.2f} '
                f'| {pnl_str} '
                f'| {sa.holding_days}天 '
                f'| {dd_str} '
                f'| {sa.overall_signal.value} '
                f'| {short_action} |'
            )

        lines.append('')
        lines.append('---')

        # 个股详细分析
        for sa in report.analyses:
            lines.append('')
            lines.extend(self._stock_detail_md(sa))
            lines.append('')
            lines.append('---')

        # 行动清单
        lines.append('')
        lines.append('## 行动清单')
        lines.append('')
        for i, sa in enumerate(report.analyses, 1):
            icon = _signal_icon(sa.overall_signal)
            lines.append(f'{i}. {icon} **{sa.holding.name}**：{sa.action}')

        lines.append('')
        lines.append(f'*报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}*')

        return '\n'.join(lines)

    def _stock_detail_md(self, sa: StockSellAnalysis) -> List[str]:
        icon = _signal_icon(sa.overall_signal)
        h = sa.holding
        lines: List[str] = []

        lines.append(f'## {icon} {h.name}({h.code}) — {sa.overall_signal.value}')
        lines.append('')

        # 基本信息
        pnl_str = f'{sa.pnl_pct:+.1f}%'
        dd_str = f'最高 {sa.peak_price:.2f} 回撤 {sa.drawdown_pct:.1f}%' if sa.peak_price > 0 else ''
        lines.append(
            f'**基本信息**：买入 {h.buy_price:.2f} → 现价 {sa.current_price:.2f}'
            f'（{pnl_str}）｜ 持仓 {sa.holding_days} 天'
            + (f' ｜ {dd_str}' if dd_str else '')
        )

        sector_info = f'{sa.sector_name}（{sa.sector_phase}，热度{sa.sector_score:.0f}）' \
            if sa.sector_name else '未知'
        lines.append(
            f'**行业**：{sector_info} ｜ '
            f'**龙头等级**：{sa.leader_grade}'
        )
        lines.append('')

        # 硬性规则
        if sa.hard_rule_triggered:
            lines.append(f'**⚡ 硬性规则触发**：{sa.hard_rule_triggered}')
            lines.append('')

        # 七维度评分表
        lines.append('### 七维度评分')
        lines.append('')
        lines.append('| 层级 | 维度 | 评分 | 权重 | 信号 | 说明 |')
        lines.append('|------|------|------|------|------|------|')
        for d in sa.dimensions:
            d_icon = _signal_icon(d.signal)
            reason_short = d.reason[:40] + '…' if len(d.reason) > 40 else d.reason
            lines.append(
                f'| {d.layer} '
                f'| {d.name} '
                f'| {d.score:.0f} '
                f'| {d.weight:.0%} '
                f'| {d_icon} {d.signal.value} '
                f'| {reason_short} |'
            )

        # 加分/减分
        if sa.bonus_applied:
            lines.append('')
            lines.append(f'**加分**：{"，".join(sa.bonus_applied)}')

        lines.append('')
        lines.append(f'**综合评分**：{sa.overall_score:.1f} / 100 → '
                      f'{icon} **{sa.overall_signal.value}**')

        # 关联股票
        if sa.related_stocks:
            lines.append('')
            lines.append('### 关联股票')
            lines.append('')
            lines.append('| 股票 | 关系 | 今日涨跌 | 状态 |')
            lines.append('|------|------|---------|------|')
            for rs in sa.related_stocks:
                lines.append(
                    f'| {rs.name}({rs.code}) '
                    f'| {rs.relation} '
                    f'| {rs.today_change:+.1f}% '
                    f'| {rs.status} |'
                )

        # 操作建议
        lines.append('')
        lines.append('### 操作建议')
        lines.append(f'> {sa.action}')

        # 风险提示
        if sa.risk_warnings:
            lines.append('')
            lines.append('**⚠️ 风险提示**：')
            for r in sa.risk_warnings[:5]:
                lines.append(f'- {r}')

        # 正面因素
        if sa.positive_factors:
            lines.append('')
            lines.append('**✅ 正面因素**：')
            for p in sa.positive_factors[:5]:
                lines.append(f'- {p}')

        return lines
