# -*- encoding: UTF-8 -*-

"""概念板块热度分析

基于 concept_cons（概念→个股映射）桥梁，聚合个股的多维度数据，
计算每个概念板块的热度得分、排名和发酵阶段。

数据来源（零新增 API，全部来自 DataManager 已有缓存）：
    concept_cons   — 概念→个股映射（桥梁）
    concept_board  — 概念板块快照（涨跌幅、上涨/下跌家数）
    daily          — 个股日K线（涨跌幅、成交额）
    fund_flow      — 个股资金流（主力净流入）
    zt_pool        — 涨停池（涨停数、连板高度、封板资金）
    lhb_detail     — 龙虎榜（净买入额）
    big_deal       — 大单追踪

评分模型（Phase 1 — 三维）：
    价格动量  30%  — 涨跌幅、上涨比例、创新高比例
    资金动向  30%  — 主力净流入、超大单、连续流入
    极端信号  40%  — 涨停数、连板高度、龙虎榜、大单

使用跨概念百分位排名法归一化各维度到 0-100，再加权求和。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    from data_manager import DataManager
except ModuleNotFoundError:
    from ..data_manager import DataManager  # type: ignore[no-redef]

from report.stock_group_calc import (
    calc_extreme_signals,
    calc_fund_flow_agg,
    calc_price_momentum,
    percentile_rank_scores,
)


# ══════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class ConceptHeatMetrics:
    """单个概念的原始聚合指标（由 stock_group_calc 原子函数计算）。"""

    # 价格动量
    avg_change_1d: float = 0.0
    avg_change_3d: float = 0.0
    avg_change_5d: float = 0.0
    up_ratio: float = 0.0
    new_high_ratio: float = 0.0

    # 资金动向
    sum_main_inflow: float = 0.0
    avg_main_pct: float = 0.0
    big_order_pct: float = 0.0
    inflow_streak_ratio: float = 0.0

    # 极端信号
    zt_count: int = 0
    max_lianban: int = 0
    total_seal_amt: float = 0.0
    lhb_net_buy: float = 0.0
    big_buy_amount: float = 0.0


@dataclass
class ConceptHeatScores:
    """各维度归一化得分（0-100）及加权总分。"""
    price_momentum: float = 0.0
    fund_flow: float = 0.0
    extreme_signal: float = 0.0
    total: float = 0.0


@dataclass
class ConceptHeat:
    """单个概念的完整热度评估结果。"""
    concept_name: str = ''
    concept_code: str = ''
    member_count: int = 0

    metrics: ConceptHeatMetrics = field(default_factory=ConceptHeatMetrics)
    scores: ConceptHeatScores = field(default_factory=ConceptHeatScores)

    phase: str = '平稳'
    rank: int = 0
    leader_code: str = ''
    leader_name: str = ''

    # concept_board 原始数据（直接透传，方便展示）
    board_change_pct: float = 0.0
    board_up_count: int = 0
    board_down_count: int = 0


@dataclass
class ConceptHotResult:
    """概念热度分析的完整输出。"""
    concepts: List[ConceptHeat] = field(default_factory=list)
    trade_date: str = ''


# ══════════════════════════════════════════════════════════════
#  权重配置
# ══════════════════════════════════════════════════════════════

_WEIGHTS = {
    'price_momentum': 0.30,
    'fund_flow': 0.30,
    'extreme_signal': 0.40,
}


# ══════════════════════════════════════════════════════════════
#  主类
# ══════════════════════════════════════════════════════════════

class ConceptHotAnalyzer:
    """概念板块热度分析器。

    用法::

        dm = DataManager(config)
        dm.refresh()
        analyzer = ConceptHotAnalyzer(dm)
        result = analyzer.run(top_n=20)
        print(ConceptHotAnalyzer.to_markdown(result))
    """

    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
        self._code_lookup: Dict[str, Tuple[str, str]] = {}

    # ──────────────────────────────────────────────────────────
    #  公开入口
    # ──────────────────────────────────────────────────────────

    def run(self, top_n: int = 20) -> ConceptHotResult:
        """计算概念热度，返回按热度降序的 Top N 结果。

        流程：
          1. 构建 code→(code,name) 快速查找表
          2. 确定今日/昨日交易日
          3. 今天和昨天分别走同一套评分管线 → 各自排名
          4. 对比两天排名变化 → 阶段判断
        """
        result = ConceptHotResult()

        concept_cons = self.dm.extra.get('concept_cons', {})
        if not concept_cons:
            logging.warning('ConceptHotAnalyzer: concept_cons 为空，跳过')
            return result

        # Step 1: 构建查找表
        self._code_lookup = self._build_code_lookup()

        logging.info(
            'ConceptHotAnalyzer: code_lookup %d 只, concept_cons %d 个',
            len(self._code_lookup), len(concept_cons),
        )

        # concept_board 用于展示信息
        board_map = self._build_board_map()

        # 确定今天/昨天的交易日
        today_date, prev_date = self._get_two_trade_dates()

        # Step 2: 逐概念收集成分股和基础数据（只收集一次）
        concept_members: List[Tuple[str, str, Set[str]]] = []
        for concept_code, cons_df in concept_cons.items():
            if cons_df is None or cons_df.empty:
                continue
            concept_name = self._extract_concept_name(
                cons_df, concept_code, board_map
            )
            if concept_name in _CONCEPT_BLACKLIST:
                continue
            member_codes = set(
                cons_df['代码'].astype(str).str.zfill(6).tolist()
            )
            if len(member_codes) < 3:
                continue
            concept_members.append((concept_name, concept_code, member_codes))

        if not concept_members:
            logging.warning('ConceptHotAnalyzer: 无有效概念数据')
            return result

        # Step 3: 用同一管线分别计算今天和昨天的指标 → 评分 → 排名
        raw_heats = self._compute_heats_for_date(
            concept_members, board_map, target_date=today_date,
        )
        if not raw_heats:
            logging.warning('ConceptHotAnalyzer: 今日无有效概念数据')
            return result

        self._score_and_rank(raw_heats)

        # 计算昨日排名（用同一套逻辑）
        prev_rank_map: Dict[str, int] = {}
        if prev_date:
            prev_heats = self._compute_heats_for_date(
                concept_members, board_map, target_date=prev_date,
            )
            if prev_heats:
                self._score_and_rank(prev_heats)
                prev_rank_map = {
                    h.concept_name: h.rank for h in prev_heats
                }
            logging.info(
                'ConceptHotAnalyzer: 昨日(%s) %d 个概念参与排名',
                prev_date, len(prev_rank_map),
            )
        else:
            logging.warning('ConceptHotAnalyzer: 无法确定前一交易日，跳过阶段判断')

        # Step 4: 阶段判断（今日 rank vs 昨日 rank，同一标尺）
        prev_board = self._build_prev_board_change_map()
        for heat in raw_heats:
            heat.phase = self._determine_phase(
                heat, prev_rank_map, prev_board,
            )

        result.concepts = raw_heats[:top_n]
        result.trade_date = self._get_trade_date_hint()

        logging.info(
            'ConceptHotAnalyzer: %d 个概念参与评分，输出 Top %d',
            len(raw_heats), len(result.concepts),
        )
        return result

    # ──────────────────────────────────────────────────────────
    #  核心计算管线（今天/昨天共用）
    # ──────────────────────────────────────────────────────────

    def _compute_heats_for_date(
        self,
        concept_members: List[Tuple[str, str, Set[str]]],
        board_map: Dict[str, dict],
        target_date: Optional[str] = None,
    ) -> List[ConceptHeat]:
        """对所有概念计算指定日期的原始指标，返回 ConceptHeat 列表。

        target_date=None 时取各股最新数据（等价于今天）。
        """
        zt_pool = self._get_snapshot_for_date('zt_pool', target_date)
        lhb = self._get_snapshot_for_date('lhb_detail', target_date)
        big_deal = self._get_snapshot_for_date('big_deal', target_date)

        heats: List[ConceptHeat] = []
        for concept_name, concept_code, member_codes in concept_members:
            daily_dfs = self._collect_daily(member_codes)
            ff_dfs = self._collect_fund_flow(member_codes)

            pm = calc_price_momentum(daily_dfs, target_date=target_date)
            ff = calc_fund_flow_agg(ff_dfs, target_date=target_date)
            es = calc_extreme_signals(member_codes, zt_pool, lhb, big_deal)

            heat = ConceptHeat(
                concept_name=concept_name,
                concept_code=concept_code,
                member_count=len(member_codes),
                metrics=ConceptHeatMetrics(
                    avg_change_1d=pm['avg_change_1d'],
                    avg_change_3d=pm['avg_change_3d'],
                    avg_change_5d=pm['avg_change_5d'],
                    up_ratio=pm['up_ratio'],
                    new_high_ratio=pm['new_high_ratio'],
                    sum_main_inflow=ff['sum_main_inflow'],
                    avg_main_pct=ff['avg_main_pct'],
                    big_order_pct=ff['big_order_pct'],
                    inflow_streak_ratio=ff['inflow_streak_ratio'],
                    zt_count=es['zt_count'],
                    max_lianban=es['max_lianban'],
                    total_seal_amt=es['total_seal_amt'],
                    lhb_net_buy=es['lhb_net_buy'],
                    big_buy_amount=es['big_buy_amount'],
                ),
            )

            # 展示信息只在今日填充
            if target_date is None:
                board_info = board_map.get(concept_name, {})
                heat.board_change_pct = board_info.get('涨跌幅', 0.0)
                heat.board_up_count = board_info.get('上涨家数', 0)
                heat.board_down_count = board_info.get('下跌家数', 0)
                if es['zt_codes']:
                    heat.leader_code, heat.leader_name = self._find_leader(
                        es['zt_codes'], zt_pool
                    )

            heats.append(heat)

        return heats

    @staticmethod
    def _score_and_rank(heats: List[ConceptHeat]) -> None:
        """百分位排名归一化 + 加权求总分 + 赋排名（原地修改）。

        今天和昨天都调用此方法，确保排名标尺一致。
        """
        n = len(heats)
        if n == 0:
            return

        raw_pm = []
        for h in heats:
            m = h.metrics
            raw_pm.append(
                m.avg_change_1d * 0.40
                + m.up_ratio * 100 * 0.30
                + m.new_high_ratio * 100 * 0.30
            )

        raw_ff = []
        for h in heats:
            m = h.metrics
            raw_ff.append(
                m.avg_main_pct * 0.50
                + m.big_order_pct * 0.30
                + m.inflow_streak_ratio * 100 * 0.20
            )

        raw_es = []
        for h in heats:
            m = h.metrics
            raw_es.append(
                m.zt_count * 10.0
                + m.max_lianban * 15.0
                + (1.0 if m.lhb_net_buy > 0 else 0.0) * 5.0
            )

        scores_pm = percentile_rank_scores(raw_pm)
        scores_ff = percentile_rank_scores(raw_ff)
        scores_es = percentile_rank_scores(raw_es)

        w = _WEIGHTS
        for i, h in enumerate(heats):
            h.scores.price_momentum = round(scores_pm[i], 1)
            h.scores.fund_flow = round(scores_ff[i], 1)
            h.scores.extreme_signal = round(scores_es[i], 1)
            h.scores.total = round(
                scores_pm[i] * w['price_momentum']
                + scores_ff[i] * w['fund_flow']
                + scores_es[i] * w['extreme_signal'],
                1,
            )

        heats.sort(key=lambda h: h.scores.total, reverse=True)
        for i, h in enumerate(heats):
            h.rank = i + 1

    # ──────────────────────────────────────────────────────────
    #  数据收集
    # ──────────────────────────────────────────────────────────

    def _build_code_lookup(self) -> Dict[str, Tuple[str, str]]:
        """构建 code → (code, name) 映射，用于从 stocks_data / extra 中取数据。"""
        lookup = {}
        for key in self.dm.stocks_data:
            code = key[0]
            lookup[str(code).zfill(6)] = key
        return lookup

    def _collect_daily(self, member_codes: Set[str]) -> Dict[str, pd.DataFrame]:
        """收集成分股的日K数据。"""
        result = {}
        for code in member_codes:
            key = self._code_lookup.get(code)
            if key and key in self.dm.stocks_data:
                result[code] = self.dm.stocks_data[key]
        return result

    def _collect_fund_flow(self, member_codes: Set[str]) -> Dict[str, pd.DataFrame]:
        """收集成分股的资金流数据。"""
        ff_dict = self.dm.extra.get('fund_flow', {})
        result = {}
        for code in member_codes:
            key = self._code_lookup.get(code)
            if key and key in ff_dict:
                result[code] = ff_dict[key]
        return result

    def _get_two_trade_dates(self) -> Tuple[Optional[str], Optional[str]]:
        """从 concept_board 快照中获取最近两个交易日。

        Returns:
            (today_date, prev_date)，today_date=None 表示取最新数据，
            prev_date=None 表示无法确定前一交易日。
        """
        cb = self.dm.extra.get('concept_board')
        if cb is None or cb.empty or '快照日期' not in cb.columns:
            return None, None

        dates = sorted(cb['快照日期'].unique())
        if len(dates) < 2:
            return None, None

        return dates[-1], dates[-2]

    def _get_snapshot_for_date(
        self, name: str, target_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """获取快照数据，按日期过滤。

        target_date=None 时取最新一天。
        对于不同快照，日期列名不同：
          zt_pool / big_deal → '快照日期'
          lhb_detail         → '上榜日'
        """
        df = self.dm.extra.get(name)
        if df is None or (hasattr(df, 'empty') and df.empty):
            return None
        if not isinstance(df, pd.DataFrame):
            return df

        date_col = None
        if '快照日期' in df.columns:
            date_col = '快照日期'
        elif '上榜日' in df.columns:
            date_col = '上榜日'

        if date_col is None:
            return df

        df = df.copy()
        df[date_col] = df[date_col].astype(str).str[:10]

        if target_date:
            filtered = df[df[date_col] == str(target_date)[:10]]
            return filtered if not filtered.empty else None

        latest = df[date_col].max()
        return df[df[date_col] == latest].copy()

    def _build_board_map(self) -> Dict[str, dict]:
        """从 concept_board 构建 概念名称→板块信息 的映射（最新一天）。"""
        cb = self.dm.extra.get('concept_board')
        if cb is None or cb.empty:
            return {}

        df = cb.copy()
        if '快照日期' in df.columns:
            df = df[df['快照日期'] == df['快照日期'].max()]

        result = {}
        for _, row in df.iterrows():
            name = str(row.get('板块名称', ''))
            if not name:
                continue
            result[name] = {
                '涨跌幅': _safe_float(row.get('涨跌幅')),
                '上涨家数': int(_safe_float(row.get('上涨家数'))),
                '下跌家数': int(_safe_float(row.get('下跌家数'))),
                '板块代码': str(row.get('板块代码', '')),
                '领涨股票': str(row.get('领涨股票', '')),
            }
        return result

    def _build_prev_board_change_map(self) -> Dict[str, float]:
        """从 concept_board 历史快照取前一天各概念的涨跌幅（用于退潮判断）。"""
        cb = self.dm.extra.get('concept_board')
        if cb is None or cb.empty or '快照日期' not in cb.columns:
            return {}

        dates = sorted(cb['快照日期'].unique())
        if len(dates) < 2:
            return {}

        prev_df = cb[cb['快照日期'] == dates[-2]]
        result = {}
        for _, row in prev_df.iterrows():
            name = str(row.get('板块名称', ''))
            if name:
                result[name] = _safe_float(row.get('涨跌幅'))
        return result

    @staticmethod
    def _extract_concept_name(
        cons_df: pd.DataFrame,
        concept_code: str,
        board_map: Dict[str, dict],
    ) -> str:
        """从成分股 DataFrame 或 board_map 中提取概念名称。"""
        if '板块名称' in cons_df.columns:
            names = cons_df['板块名称'].dropna().unique()
            if len(names) > 0:
                return str(names[0])

        for name, info in board_map.items():
            if info.get('板块代码') == concept_code:
                return name

        return concept_code

    @staticmethod
    def _find_leader(
        zt_codes: list,
        zt_pool_df: Optional[pd.DataFrame],
    ) -> Tuple[str, str]:
        """从涨停池中找到指定代码列表中连板最高的股票。"""
        if not zt_codes or zt_pool_df is None or zt_pool_df.empty:
            return '', ''

        zt = zt_pool_df.copy()
        zt['代码'] = zt['代码'].astype(str).str.zfill(6)
        matched = zt[zt['代码'].isin(zt_codes)]

        if matched.empty:
            return '', ''

        if '连板数' in matched.columns:
            matched = matched.copy()
            matched['连板数'] = pd.to_numeric(matched['连板数'], errors='coerce').fillna(1)
            best = matched.loc[matched['连板数'].idxmax()]
        else:
            best = matched.iloc[0]

        return str(best.get('代码', '')), str(best.get('名称', ''))

    # ──────────────────────────────────────────────────────────
    #  阶段判断
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _determine_phase(
        heat: ConceptHeat,
        prev_rank_map: Dict[str, int],
        prev_board_change: Dict[str, float],
    ) -> str:
        """根据指标判断概念所处阶段。

        排名体系：heat.rank（今日）和 prev_rank_map[name]（昨日）
        都由 _score_and_rank 用同一套管线（三维百分位 → 加权 → 排序）
        计算得出，rank=1 最强，标尺完全一致。

        阶段定义：
            高潮 — 涨停 >= 4 且连板 >= 3，或涨停 >= 3 且板块涨幅 >= 3%
            退潮 — 昨日板块涨幅 > 2% 但今日大幅回落且涨停稀少
            启动 — 今日涨停 >= 2 且昨日排名靠后（后 1/3）
            发酵 — 今日涨停 >= 2 且排名较昨日跃升 > 50 名
            平稳 — 其他
        """
        m = heat.metrics
        prev_rank = prev_rank_map.get(heat.concept_name, 999)
        prev_change = prev_board_change.get(heat.concept_name, 0.0)
        curr_rank = heat.rank

        # 高潮：涨停多 + 连板高 + 板块涨幅大
        if m.zt_count >= 4 and m.max_lianban >= 3:
            return '高潮'
        if m.zt_count >= 3 and heat.board_change_pct >= 3.0:
            return '高潮'

        # 退潮：板块涨幅明显回落 + 涨停稀少
        if (prev_change > 2.0
                and heat.board_change_pct < prev_change * 0.3
                and m.zt_count <= 1):
            return '退潮'

        # 启动：今天有涨停，但昨天排名在后 1/3（约 300 名之后）
        if m.zt_count >= 2 and prev_rank > 300:
            return '启动'

        # 发酵：排名跃升超过 50 名且有涨停
        if m.zt_count >= 2 and prev_rank - curr_rank > 50:
            return '发酵'

        return '平稳'

    # ──────────────────────────────────────────────────────────
    #  工具
    # ──────────────────────────────────────────────────────────

    def _get_trade_date_hint(self) -> str:
        """尽力获取交易日期标识。"""
        zt = self.dm.extra.get('zt_pool')
        if zt is not None and not zt.empty and '快照日期' in zt.columns:
            return str(zt['快照日期'].max())
        return ''

    # ──────────────────────────────────────────────────────────
    #  输出 / 展示
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def to_markdown(result: ConceptHotResult) -> str:
        """将概念热度结果格式化为 Markdown 报告。"""
        if not result.concepts:
            return '### 概念热度排名\n\n暂无有效概念数据。\n'

        date_str = f'（{result.trade_date}）' if result.trade_date else ''

        lines = [
            f'### 概念板块热度 Top {len(result.concepts)}{date_str}',
            '',
            '| # | 概念 | 涨跌幅 | 涨停 | 龙头(板) | 主力净流入 '
            '| 动量 | 资金 | 信号 | 热度 | 阶段 |',
            '|---|------|--------|------|---------|----------'
            '|------|------|------|------|------|',
        ]

        for i, h in enumerate(result.concepts, 1):
            m = h.metrics
            s = h.scores

            change_str = f'{h.board_change_pct:+.1f}%'

            zt_str = str(m.zt_count) if m.zt_count > 0 else '—'

            if h.leader_name:
                leader_str = f'{h.leader_name}({m.max_lianban})'
            else:
                leader_str = '—'

            inflow_yi = m.sum_main_inflow / 1e8
            if abs(inflow_yi) >= 0.01:
                sign = '+' if inflow_yi > 0 else ''
                inflow_str = f'{sign}{inflow_yi:.1f}亿'
            else:
                inflow_str = '—'

            phase_emoji = _PHASE_DISPLAY.get(h.phase, h.phase)

            lines.append(
                f'| {i} '
                f'| {h.concept_name} '
                f'| {change_str} '
                f'| {zt_str} '
                f'| {leader_str} '
                f'| {inflow_str} '
                f'| {s.price_momentum:.0f} '
                f'| {s.fund_flow:.0f} '
                f'| {s.extreme_signal:.0f} '
                f'| **{s.total:.0f}** '
                f'| {phase_emoji} |'
            )

        return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════
#  内部辅助
# ══════════════════════════════════════════════════════════════

_PHASE_DISPLAY = {
    '启动': '🚀启动',
    '发酵': '🔥发酵',
    '高潮': '⚡高潮',
    '退潮': '📉退潮',
    '平稳': '—',
}

# 非真实题材概念，不参与热度排名
_CONCEPT_BLACKLIST = {
    '昨日连板_含一字',
    '昨日涨停_含一字',
    '昨日首板',
    '东方财富热股',
    '最近多板',
    '昨日涨停',
}


def _safe_float(val) -> float:
    try:
        v = float(val)
        return v if pd.notna(v) else 0.0
    except (TypeError, ValueError):
        return 0.0
