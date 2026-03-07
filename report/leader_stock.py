# -*- encoding: UTF-8 -*-

"""龙头股识别与评分模块

基于市场情绪温度计和行业热度分析的输出，从热门行业中筛选并量化评估龙头股。

三层漏斗：
    market_temperature（全局情绪） → 仓位上限
    industry_temperature（板块热度）→ 关注方向
    leader_stock（龙头识别）       → 具体标的

五维评分（0-100）：
    连板高度 (0-30)  — 最直接的龙头身份证据
    板块领先度 (0-25) — 必须是板块内唯一最强
    封板质量 (0-20)  — 封板资金 + 炸板回封
    身份确认 (0-15)  — 连板股用昨日溢价，首板股用封板时间+竞价强度
    梯队支撑 (0-10)  — 板块阶段决定后援力度

六级分级：S / A / B+ / B / C / D
龙头生命周期：确立期 / 主升期 / 加速期 / 衰退期
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

try:
    from data_manager import DataManager
except ModuleNotFoundError:
    from ..data_manager import DataManager  # type: ignore[no-redef]

from report.industry_temperature import SectorHeat


# ══════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class LeaderStock:
    """单只龙头股的评估结果。"""
    code: str = ''
    name: str = ''
    sector: str = ''
    sector_phase: str = ''
    lianban: int = 0

    score_lianban: float = 0.0      # 连板高度分 (0-30)
    score_leading: float = 0.0      # 板块领先度分 (0-25)
    score_seal: float = 0.0         # 封板质量分 (0-20)
    score_confirm: float = 0.0      # 身份确认分 (0-15)
    score_support: float = 0.0      # 梯队支撑分 (0-10)

    score: float = 0.0              # 总分 (0-100)
    grade: str = ''                 # S / A / B+ / B / C / D
    lifecycle: str = ''             # 确立期 / 主升期 / 加速期 / 衰退期
    action: str = ''                # 操作建议
    risk_flags: List[str] = field(default_factory=list)

    seal_amount: float = 0.0        # 封板资金（元）
    first_seal_time: str = ''       # 首次封板时间
    zhaban_count: int = 0           # 炸板次数


# ══════════════════════════════════════════════════════════════
#  主类
# ══════════════════════════════════════════════════════════════

class LeaderStockFinder:
    """龙头股识别器。"""

    def __init__(self, dm: DataManager, mt):
        self.dm = dm
        self.mt = mt

    # ──────────────────────────────────────────────────────────
    #  公开入口
    # ──────────────────────────────────────────────────────────

    def run(
        self,
        hot_sectors: List[SectorHeat],
        market_score: float = 50.0,
        market_phase: str = '主升/发酵期',
        top_sectors: int = 8,
    ) -> List[LeaderStock]:
        if not hot_sectors:
            return []

        sectors = hot_sectors[:top_sectors]
        sector_map: Dict[str, SectorHeat] = {s.sector: s for s in sectors}
        sector_names = set(sector_map.keys())

        candidates = self._build_candidates(sector_names)
        if candidates.empty:
            logging.debug('LeaderStockFinder: 候选池为空')
            return []

        prev_map = self._build_prev_map()
        seal_rank = self._rank_by_seal(candidates)

        results: List[LeaderStock] = []
        for _, row in candidates.iterrows():
            code = str(row['代码'])
            name = str(row['名称'])
            sector = str(row['所属行业'])
            lianban = int(row['连板数'])
            sh = sector_map.get(sector)
            if sh is None:
                continue

            ls = LeaderStock(
                code=code, name=name, sector=sector,
                sector_phase=sh.phase, lianban=lianban,
            )

            if '封板资金' in row.index:
                ls.seal_amount = float(row['封板资金']) if pd.notna(row['封板资金']) else 0
            if '首次封板时间' in row.index:
                ls.first_seal_time = str(row['首次封板时间']) if pd.notna(row['首次封板时间']) else ''

            zhaban_count = int(row['炸板次数']) if '炸板次数' in row.index and pd.notna(row.get('炸板次数')) else 0
            ls.zhaban_count = zhaban_count

            ls.score_lianban = self._score_lianban(lianban)
            ls.score_leading = self._score_leading(lianban, sh)
            ls.score_seal = self._score_seal(code, seal_rank, zhaban_count)
            ls.score_confirm = self._score_confirm(
                code, lianban, prev_map, ls.first_seal_time, sh,
            )
            ls.score_support = self._score_support(sh)

            ls.score = min(100.0, max(0.0, (
                ls.score_lianban + ls.score_leading + ls.score_seal
                + ls.score_confirm + ls.score_support
            )))

            ls.grade = self._classify_grade(ls.score)
            ls.lifecycle = self._classify_lifecycle(lianban, sh)
            ls.risk_flags = self._detect_risks(ls, sh, market_score, market_phase)
            ls.action = self._suggest_action(
                ls.grade, ls.lifecycle, market_phase, sh.phase,
            )

            results.append(ls)

        results.sort(key=lambda x: x.score, reverse=True)
        logging.debug(
            'LeaderStockFinder: %d 只候选, %d 只 B 级以上',
            len(results),
            sum(1 for r in results if r.grade not in ('C', 'D')),
        )
        return results

    # ──────────────────────────────────────────────────────────
    #  Step 1：候选池
    # ──────────────────────────────────────────────────────────

    def _build_candidates(self, sector_names: set) -> pd.DataFrame:
        zt_df = self.dm.extra.get('zt_pool')
        if zt_df is None or zt_df.empty:
            return pd.DataFrame()

        df = zt_df.copy()
        df = df[df['所属行业'].isin(sector_names)]

        mask_st = df['名称'].str.upper().str.contains('ST', na=False)
        df = df[~mask_st]

        for col in ('连板数', '流通市值', '封板资金'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if '流通市值' in df.columns:
            has_mv = df['流通市值'].notna()
            mv = df['流通市值']
            df = df[~(has_mv & (mv > 500e8)) & ~(has_mv & (mv < 10e8))]

        if '连板数' in df.columns:
            df['连板数'] = df['连板数'].fillna(1).astype(int)
        else:
            df['连板数'] = 1

        return df.reset_index(drop=True)

    def _build_prev_map(self) -> Dict[str, float]:
        prev_df = self.dm.extra.get('zt_pool_previous')
        if prev_df is None or prev_df.empty:
            return {}
        prev = prev_df.copy()
        if '代码' not in prev.columns or '涨跌幅' not in prev.columns:
            return {}
        prev['涨跌幅'] = pd.to_numeric(prev['涨跌幅'], errors='coerce')
        return dict(zip(prev['代码'].astype(str), prev['涨跌幅']))

    def _rank_by_seal(self, candidates: pd.DataFrame) -> Dict[str, int]:
        if '封板资金' not in candidates.columns:
            return {}
        ranked = candidates.dropna(subset=['封板资金']).sort_values(
            '封板资金', ascending=False,
        )
        return {
            str(row['代码']): i + 1
            for i, (_, row) in enumerate(ranked.iterrows())
        }

    # ──────────────────────────────────────────────────────────
    #  Step 2：五维评分
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _score_lianban(lianban: int) -> float:
        """维度 1：连板高度分（0-30）。首板 8 分起步。"""
        mapping = {1: 8, 2: 14, 3: 18, 4: 22, 5: 26}
        if lianban >= 6:
            return 30.0
        return float(mapping.get(lianban, 8))

    @staticmethod
    def _score_leading(lianban: int, sh: SectorHeat) -> float:
        """维度 2：板块领先度分（0-25）。"""
        if lianban < sh.max_lianban:
            return 0.0
        count_at_max = sh.lianban_dist.get(sh.max_lianban, 0)
        if count_at_max == 1:
            return 25.0
        return 15.0

    @staticmethod
    def _score_seal(
        code: str,
        seal_rank: Dict[str, int],
        zhaban_count: int = 0,
    ) -> float:
        """维度 3：封板质量分（0-20）。

        3a 封板资金排名（0-12）
        3b 炸板回封加分（0 或 +8）：炸板次数>=1 说明盘中炸过板但最终封住
        3c 高炸板扣分（0 ~ -6）：炸板次数 >= 2 时扣分，分歧过大次日风险高
        """
        rank = seal_rank.get(code, 999)
        if rank <= 3:
            score_a = 12.0
        elif rank <= 10:
            score_a = 8.0
        elif rank <= 20:
            score_a = 4.0
        else:
            score_a = 0.0

        # 3b: 炸板回封 — 直接使用 zt_pool 的炸板次数字段
        score_b = 8.0 if zhaban_count >= 1 else 0.0

        # 3c: 高炸板扣分 — 炸板次数越多分歧越大，次日表现越不稳定
        if zhaban_count >= 3:
            score_c = -6.0
        elif zhaban_count == 2:
            score_c = -3.0
        else:
            score_c = 0.0

        return max(0.0, min(20.0, score_a + score_b + score_c))

    @staticmethod
    def _score_confirm(
        code: str,
        lianban: int,
        prev_map: Dict[str, float],
        first_seal_time: str,
        sector_heat: Optional['SectorHeat'] = None,
    ) -> float:
        """维度 4：身份确认分（0-15）。

        连板股（≥2）：昨日涨停溢价率，验证市场对其龙头身份的认可度。
        首板股（=1）：封板时间（4a）+ 板块地位（4b'），首板不存在
            "昨日溢价"数据，改为评估其在板块中的领先地位来确认潜力。
        """
        if lianban >= 2:
            chg = prev_map.get(code)
            if chg is None or pd.isna(chg):
                return 0.0
            if chg >= 5.0:
                return 15.0
            if chg >= 3.0:
                return 10.0
            if chg >= 0.0:
                return 5.0
            return 0.0

        # ── 首板确认度 ─────────────────────────────────────────
        score = 0.0

        # 4a: 封板时间（0-8 分）
        if first_seal_time:
            t = first_seal_time.replace(':', '')
            try:
                t_int = int(t[:6]) if len(t) >= 6 else int(t)
                if t_int <= 100000:      # 10:00 前
                    score += 8.0
                elif t_int <= 113000:    # 10:00-11:30
                    score += 5.0
                elif t_int < 143000:     # 下午开盘到 14:30
                    score += 2.0
            except (ValueError, TypeError):
                pass

        # 4b': 板块地位（0-7 分）— 替代连板股的"竞价强度"
        # 首板当日不存在昨日溢价数据，改为评估该股在热门板块中的地位：
        # 板块涨停数量越多且该股处于热门板块中，首板价值越高
        if sector_heat is not None:
            zt_n = sector_heat.zt_count
            phase = sector_heat.phase
            if zt_n >= 5 and phase in ('发酵', '高潮'):
                score += 7.0
            elif zt_n >= 4 and phase in ('启动', '发酵'):
                score += 5.0
            elif zt_n >= 3:
                score += 3.0

        return min(15.0, score)

    @staticmethod
    def _score_support(sh: SectorHeat) -> float:
        """维度 5：梯队支撑分（0-10）。"""
        if sh.phase == '发酵' and sh.score_ladder >= 15:
            return 10.0
        if sh.phase in ('启动', '高潮'):
            return 6.0
        if sh.phase == '平稳':
            return 3.0
        return 0.0

    # ──────────────────────────────────────────────────────────
    #  Step 3：分级、生命周期、风险、建议
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _classify_grade(score: float) -> str:
        if score >= 75:
            return 'S'
        if score >= 60:
            return 'A'
        if score >= 48:
            return 'B+'
        if score >= 35:
            return 'B'
        if score >= 25:
            return 'C'
        return 'D'

    @staticmethod
    def _classify_lifecycle(lianban: int, sh: SectorHeat) -> str:
        """龙头生命周期判断。"""
        if sh.phase == '退潮':
            return '衰退期'
        if lianban >= 5:
            return '加速期'
        if lianban >= 3:
            return '主升期'
        if lianban >= 2:
            return '确立期'
        return '确立期'

    @staticmethod
    def _detect_risks(
        ls: LeaderStock,
        sh: SectorHeat,
        market_score: float,
        market_phase: str,
    ) -> List[str]:
        flags: List[str] = []
        if sh.phase == '退潮':
            flags.append('板块退潮，注意离场')
        if sh.phase == '高潮' and ls.lianban >= 5:
            flags.append('板块高潮+高位龙头，关注次日冲高兑现窗口')
        if ls.lianban >= 7:
            flags.append(f'{ls.lianban}连板高位博弈')
        if market_phase == '高潮期':
            flags.append('市场高潮期，注意高潮后分歧')
        if market_score < 20:
            flags.append('市场冰点，极度谨慎')
        if ls.zhaban_count >= 3:
            flags.append(f'炸板{ls.zhaban_count}次，封板分歧大，次日不确定性高')
        elif ls.zhaban_count == 2:
            flags.append('炸板2次，封板有分歧')
        return flags

    # ── 操作建议矩阵 ─────────────────────────────────────────

    _ACTION_MATRIX = {
        # (grade, market_phase) → action
        ('S', '冰点期'):      '轻仓参与（逆势龙可能是反弹先锋）',
        ('S', '混沌/修复期'):  '半仓参与',
        ('S', '主升/发酵期'):  '重仓核心标的',
        ('S', '高潮期'):      '持筹享溢价，不追高',

        ('A', '冰点期'):      '轻仓跟踪，等放量确认',
        ('A', '混沌/修复期'):  '半仓参与',
        ('A', '主升/发酵期'):  '半仓参与',
        ('A', '高潮期'):      '持筹，逢高减仓',

        ('B+', '冰点期'):     '观望',
        ('B+', '混沌/修复期'): '轻仓参与，关注次日竞价确认',
        ('B+', '主升/发酵期'): '半仓参与',
        ('B+', '高潮期'):     '轻仓试错',

        ('B', '冰点期'):      '观望',
        ('B', '混沌/修复期'):  '观察不动手',
        ('B', '主升/发酵期'):  '轻仓试错',
        ('B', '高潮期'):      '观望',
    }

    @classmethod
    def _suggest_action(
        cls,
        grade: str,
        lifecycle: str,
        market_phase: str,
        sector_phase: str,
    ) -> str:
        if grade in ('C', 'D'):
            return '跟踪观察（板块人气参考）'
        if sector_phase == '退潮':
            return '回避（板块退潮）'

        action = cls._ACTION_MATRIX.get((grade, market_phase), '观望')

        # 生命周期叠加：确立期更积极，加速期更谨慎
        if lifecycle == '确立期' and '观望' in action:
            action = action.replace('观望', '关注，等待确认')
        if lifecycle == '加速期' and '重仓' in action:
            action = action.replace('重仓', '半仓')
        if lifecycle == '衰退期':
            action = '离场或回避'

        if sector_phase == '高潮' and '兑现' not in action and '溢价' not in action and '回避' not in action:
            action += '，板块高潮注意兑现'

        return action

    # ──────────────────────────────────────────────────────────
    #  输出
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def to_markdown(
        leaders: List[LeaderStock],
        market_score: float = 0.0,
        market_phase: str = '',
    ) -> str:
        if not leaders:
            return ''

        # D 级不显示；C 级以上全部输出
        show = [ls for ls in leaders if ls.grade != 'D']
        if not show:
            return ''

        lines = [
            '### 龙头股追踪',
            '',
            '| 级别 | 股票 | 行业 | 连板 | 封板时间 | 炸板 | 得分 | 周期 | 板块 | 操作建议 |',
            '|------|------|------|------|---------|------|------|------|------|---------|',
        ]

        for ls in show:
            seal_time = ls.first_seal_time if ls.first_seal_time else '—'
            zhaban_str = str(ls.zhaban_count) if ls.zhaban_count > 0 else '—'
            lines.append(
                f'| {ls.grade} '
                f'| {ls.name}({ls.code}) '
                f'| {ls.sector} '
                f'| {ls.lianban}板 '
                f'| {seal_time} '
                f'| {zhaban_str} '
                f'| {ls.score:.0f} '
                f'| {ls.lifecycle} '
                f'| {ls.sector_phase} '
                f'| {ls.action} |'
            )

        risk_lines = []
        for ls in show:
            if ls.risk_flags and ls.grade not in ('C', 'D'):
                risk_lines.append(f'- {ls.name}：{"；".join(ls.risk_flags)}')

        if market_phase:
            risk_lines.append(f'- 市场情绪 {market_score:.1f} 分（{market_phase}）')

        if risk_lines:
            lines.append('')
            lines.append('**风险提示：**')
            lines.extend(risk_lines)

        return '\n'.join(lines)
