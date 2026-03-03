# -*- encoding: UTF-8 -*-

"""行业板块热度分析

在市场情绪温度计的基础上，拆解到行业板块维度：
  1. 识别热点行业
  2. 量化每个行业的热度（0-100 分）
  3. 判断板块阶段（启动/发酵/高潮/退潮/平稳）

数据来源（零新增 API）：
    zt_pool          — 今日涨停池（所属行业、连板数、封板资金）
    zt_pool_previous — 昨日涨停池（用于阶段判断）
    stock_sector_map — 本地累积的 代码→行业 映射
    daily K线        — 成交额放量比（当日/近5日均值），替代不稳定的 sector_flow
    _get_market_df   — 全市场行情（realtime / daily 重建）
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

try:
    from data_manager import DataManager
except ModuleNotFoundError:
    from ..data_manager import DataManager  # type: ignore[no-redef]

from report.market_temperature import MarketTemperature


def _prefix_lookup(mapping: dict, key: str):
    """从 mapping 中查找 key：精确匹配 → 前缀匹配（zt_pool 行业名常被截断）。"""
    if key in mapping:
        return mapping[key]
    for full_name, val in mapping.items():
        if full_name.startswith(key) or key.startswith(full_name):
            return val
    return None


# ══════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class SectorHeat:
    """单个行业的热度评估结果。"""
    sector: str = ''
    zt_count: int = 0
    max_lianban: int = 0
    leader_code: str = ''
    leader_name: str = ''
    lianban_dist: Dict[int, int] = field(default_factory=dict)
    total_seal_amount: float = 0.0
    sector_turnover: float = 0.0       # 板块成交额（元）
    net_inflow: float = 0.0            # 主力净流入（元），来自 sector_flow
    net_inflow_pct: float = 0.0        # 主力净流入占比（%）
    follow_rate: float = 0.0           # 跟风率
    score_concentration: float = 0.0   # 涨停集中度得分 (0-30)
    score_ladder: float = 0.0          # 梯队完整度得分 (0-25)
    score_leader: float = 0.0          # 龙头高度得分 (0-20)
    score_fund: float = 0.0            # 资金流向得分 (0-15)
    score_follow: float = 0.0          # 跟风广度得分 (0-10)
    score: float = 0.0                 # 总分 (0-100)
    phase: str = '平稳'                # 启动/发酵/高潮/退潮/平稳
    fund_source: str = ''              # 资金数据来源: sector_flow / daily_volume


# ══════════════════════════════════════════════════════════════
#  主类
# ══════════════════════════════════════════════════════════════

class IndustryTemperature:
    """行业板块热度分析。

    用法::

        dm = DataManager(config)
        dm.refresh()
        mt = MarketTemperature(dm)      # 用于复用 _get_market_df / _get_trade_date
        it = IndustryTemperature(dm, mt)
        sectors = it.run()
        print(it.to_markdown(sectors))
    """

    def __init__(self, data_manager: DataManager, market_temp: MarketTemperature):
        self.dm = data_manager
        self.mt = market_temp

    # ──────────────────────────────────────────────────────────
    #  公开入口
    # ──────────────────────────────────────────────────────────

    def run(self, top_n: int = 10) -> List[SectorHeat]:
        """计算行业热度，返回按热度降序的 Top N 列表。"""
        trade_date = self.mt._get_trade_date()

        # Step 1: 按行业聚合涨停数据
        raw_sectors = self._aggregate_zt_pool()
        if not raw_sectors:
            logging.debug('IndustryTemperature: zt_pool 无有效行业数据')
            return []

        total_zt = sum(s.zt_count for s in raw_sectors.values())

        sector_map = self._load_sector_map()
        sector_names = list(raw_sectors.keys())

        # 资金排名：优先 sector_flow（净流入），回退到日K放量比
        fund_rank = self._build_fund_rank_from_sector_flow(sector_names)
        if not fund_rank:
            fund_rank = self._build_fund_rank_from_daily(trade_date, sector_map)
            logging.debug('IndustryTemperature: sector_flow 不可用，回退到日K放量比')

        # 成交额：始终从日K读取（sector_flow 无成交额字段）
        volume_rank = self._build_fund_rank_from_daily(trade_date, sector_map)

        market_df = self.mt._get_market_df(trade_date)

        # 昨日行业涨停分布（用于阶段判断）
        prev_sector_stats = self._aggregate_zt_pool_previous()

        results: List[SectorHeat] = []
        for sector, sh in raw_sectors.items():
            # Step 2: 梯队完整度
            sh.score_ladder = self._score_ladder(sh.lianban_dist)

            # Step 3: 资金活跃度（优先主力净流入排名，回退到放量比排名）
            sh.net_inflow, sh.net_inflow_pct, sh.score_fund, sh.fund_source = \
                self._score_fund(sector, fund_rank)
            # 成交额单独从日K取
            sh.sector_turnover = self._get_turnover(sector, volume_rank)

            # Step 4: 跟风广度
            sh.follow_rate, sh.score_follow = self._score_follow(
                sector, sector_map, market_df
            )

            # Step 5: 涨停集中度
            sh.score_concentration = self._score_concentration(sh.zt_count, total_zt)

            # Step 6: 龙头高度 + 合成总分
            sh.score_leader = self._score_leader(sh.max_lianban)
            sh.score = min(100.0, max(0.0,
                sh.score_concentration + sh.score_ladder
                + sh.score_leader + sh.score_fund + sh.score_follow
            ))

            # Step 7: 阶段判断
            prev = prev_sector_stats.get(sector, {})
            sh.phase = self._classify_phase(sh, prev)

            results.append(sh)

        results.sort(key=lambda s: s.score, reverse=True)
        return results[:top_n]

    # ──────────────────────────────────────────────────────────
    #  Step 1：按行业聚合涨停数据
    # ──────────────────────────────────────────────────────────

    def _aggregate_zt_pool(self) -> Dict[str, SectorHeat]:
        zt_df = self.dm.extra.get('zt_pool')
        if zt_df is None or zt_df.empty:
            return {}

        df = zt_df.copy()
        # 剔除 ST
        df = df[~df['名称'].str.upper().str.contains('ST', na=False)]

        if '所属行业' not in df.columns or '连板数' not in df.columns:
            return {}

        df['连板数'] = pd.to_numeric(df['连板数'], errors='coerce').fillna(1).astype(int)
        if '封板资金' in df.columns:
            df['封板资金'] = pd.to_numeric(df['封板资金'], errors='coerce').fillna(0)

        result: Dict[str, SectorHeat] = {}
        for sector, grp in df.groupby('所属行业'):
            if pd.isna(sector) or not sector:
                continue
            zt_count = len(grp)
            if zt_count < 2:
                continue

            lianban = grp['连板数']
            dist = dict(Counter(lianban))
            max_lb = int(lianban.max())

            # 龙头：连板最高的那只
            leader_row = grp.loc[lianban.idxmax()]
            leader_code = str(leader_row.get('代码', ''))
            leader_name = str(leader_row.get('名称', ''))

            seal = float(grp['封板资金'].sum()) if '封板资金' in grp.columns else 0.0

            result[sector] = SectorHeat(
                sector=sector,
                zt_count=zt_count,
                max_lianban=max_lb,
                leader_code=leader_code,
                leader_name=leader_name,
                lianban_dist=dist,
                total_seal_amount=seal,
            )

        logging.debug('_aggregate_zt_pool: %d 个行业（涨停>=2）', len(result))
        return result

    # ──────────────────────────────────────────────────────────
    #  Step 2：梯队完整度得分 (0-25)
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _score_ladder(dist: Dict[int, int]) -> float:
        """连板梯队完整度（0-25 分）。

        综合考虑龙头高度（决定潜力上限）和梯队连续性（断层惩罚）。
        完美梯队（1→2→3→4 无断层）拿满分；
        严重断层（如 1板+4板，中间全空）按连续性比例折扣。
        """
        if not dist:
            return 0.0

        levels = sorted(dist.keys())
        max_lb = max(levels)

        if len(levels) == 1 and max_lb == 1:
            return 5.0

        # 潜力上限：由龙头高度决定
        if max_lb >= 4:
            potential = 25.0
        elif max_lb == 3:
            potential = 20.0
        elif max_lb == 2:
            potential = 12.0
        else:
            return 5.0

        # 连续性：1~max_lb 中实际填充的比例
        expected = set(range(1, max_lb + 1))
        filled = len(set(levels) & expected)
        continuity = filled / len(expected)

        # 得分 = 潜力 × (基准0.3 + 连续性贡献0.7)
        # 完美连续 → ×1.0；全断层(仅首尾) → ×~0.3+
        score = potential * (0.3 + 0.7 * continuity)
        return round(min(25.0, max(5.0, score)), 1)

    # ──────────────────────────────────────────────────────────
    #  Step 3：资金活跃度得分 (0-15)
    #    优先使用 sector_flow（主力净流入排名），不可用时回退到日K放量比
    # ──────────────────────────────────────────────────────────

    def _build_fund_rank_from_sector_flow(
        self, sector_names: List[str],
    ) -> Dict[str, dict]:
        """从 extra['sector_flow'] 构建行业资金排名（主力净流入）。

        sector_names 来自 zt_pool 聚合后的行业名（可能被截断），
        需要对 sector_flow 做前缀模糊匹配。
        """
        sf = self.dm.extra.get('sector_flow')
        if sf is None or (hasattr(sf, 'empty') and sf.empty):
            return {}

        df = sf.copy()
        if '快照日期' in df.columns:
            df = df[df['快照日期'] == df['快照日期'].max()].copy()

        inflow_col = None
        pct_col = None
        for c in df.columns:
            if '主力净流入' in c and '净额' in c:
                inflow_col = c
            if '主力净流入' in c and '净占比' in c:
                pct_col = c

        if inflow_col is None or '名称' not in df.columns:
            return {}

        df[inflow_col] = pd.to_numeric(df[inflow_col], errors='coerce').fillna(0)
        if pct_col:
            df[pct_col] = pd.to_numeric(df[pct_col], errors='coerce').fillna(0)

        df = df.sort_values(inflow_col, ascending=False).reset_index(drop=True)
        total = len(df)

        # 完整名称 → {inflow, rank, ...}
        full_name_map: Dict[str, dict] = {}
        for i, row in df.iterrows():
            name = str(row['名称'])
            inflow = float(row[inflow_col])
            pct = float(row[pct_col]) if pct_col else 0.0
            full_name_map[name] = {
                'inflow': inflow,
                'inflow_pct': pct,
                'rank': int(i) + 1,
                'total': total,
                'source': 'sector_flow',
            }

        # zt_pool 行业名可能被截断（如"炼化及贸"→"炼化及贸易"），
        # 对无法精确匹配的名称做前缀匹配
        result: Dict[str, dict] = {}
        full_names = list(full_name_map.keys())
        matched = 0
        for sec in sector_names:
            if sec in full_name_map:
                result[sec] = full_name_map[sec]
                matched += 1
            else:
                candidates = [fn for fn in full_names if fn.startswith(sec) or sec.startswith(fn)]
                if len(candidates) == 1:
                    result[sec] = full_name_map[candidates[0]]
                    matched += 1

        logging.debug(
            '_build_fund_rank_from_sector_flow: sector_flow %d 行业, '
            'zt_pool %d 行业, 匹配 %d',
            total, len(sector_names), matched,
        )
        return result

    def _build_fund_rank_from_daily(
        self,
        trade_date: str,
        sector_map: Dict[str, List[str]],
    ) -> Dict[str, dict]:
        """sector_flow 不可用时，从日K成交额重建行业资金排名。

        使用 "成交额放量比"（当日成交额 / 近5日均成交额）代替主力净流入，
        衡量行业异常活跃度。
        """
        daily_dir = os.path.join(self.dm._cache.cache_dir, 'daily')
        if not os.path.exists(daily_dir) or not sector_map:
            return {}

        td = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"

        def _read_one(fname: str):
            code = fname[:-8]
            try:
                df = pd.read_parquet(os.path.join(daily_dir, fname))
                if '成交额' not in df.columns or '日期' not in df.columns:
                    return None
                df['日期'] = df['日期'].astype(str)
                idx_list = df.index[df['日期'] == td].tolist()
                if not idx_list:
                    return None
                pos = df.index.get_loc(idx_list[0])
                today_vol = float(df.iloc[pos]['成交额']) if pd.notna(df.iloc[pos]['成交额']) else 0
                start = max(0, pos - 5)
                avg_vol = float(df.iloc[start:pos]['成交额'].mean()) if pos > 0 else today_vol
                return code, today_vol, avg_vol
            except Exception:
                return None

        files = [f for f in os.listdir(daily_dir) if f.endswith('.parquet')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(_read_one, files))

        stock_data = {r[0]: (r[1], r[2]) for r in results if r is not None}
        if not stock_data:
            return {}

        sector_stats: Dict[str, dict] = {}
        for sector, codes in sector_map.items():
            total_vol = 0.0
            total_avg = 0.0
            for c in codes:
                if c in stock_data:
                    total_vol += stock_data[c][0]
                    total_avg += stock_data[c][1]
            if total_avg > 0:
                ratio = total_vol / total_avg
            else:
                ratio = 1.0
            sector_stats[sector] = {
                'inflow': total_vol,
                'volume_ratio': ratio,
            }

        # 按放量比降序排名
        ranked = sorted(sector_stats.items(), key=lambda x: x[1]['volume_ratio'], reverse=True)
        total = len(ranked)
        result = {}
        for i, (sec, info) in enumerate(ranked):
            result[sec] = {
                'inflow': info['inflow'],
                'rank': i + 1,
                'total': total,
                'source': 'daily_volume',
            }

        logging.debug(
            '_build_fund_rank_from_daily: %d 个行业, 放量比 Top3: %s',
            total,
            [(s, f"{d['volume_ratio']:.2f}x") for s, d in ranked[:3]],
        )
        return result

    @staticmethod
    def _get_turnover(sector: str, volume_rank: Dict[str, dict]) -> float:
        """从日K成交额排名中取板块成交额。"""
        info = _prefix_lookup(volume_rank, sector)
        return info['inflow'] if info else 0.0

    def _score_fund(
        self, sector: str, fund_rank: Dict[str, dict]
    ) -> tuple[float, float, float, str]:
        """返回 (主力净流入, 净流入占比, 资金活跃度得分, 数据来源)。"""
        info = _prefix_lookup(fund_rank, sector)
        if info is None:
            return 0.0, 0.0, 0.0, ''

        rank = info['rank']
        total = info['total']
        source = info.get('source', 'daily_volume')

        net_inflow = info.get('inflow', 0.0) if source == 'sector_flow' else 0.0
        net_inflow_pct = info.get('inflow_pct', 0.0) if source == 'sector_flow' else 0.0

        if rank <= 5:
            score = 15.0
        elif rank <= 10:
            score = 12.0
        elif rank <= 20:
            score = 8.0
        elif rank <= total * 0.5:
            score = 4.0
        else:
            score = 0.0
        return net_inflow, net_inflow_pct, score, source

    # ──────────────────────────────────────────────────────────
    #  Step 4：跟风广度得分 (0-10)
    # ──────────────────────────────────────────────────────────

    def _load_sector_map(self) -> Dict[str, List[str]]:
        """加载 stock_sector_map.parquet，返回 行业→[代码列表] 映射。"""
        path = os.path.join(self.dm._cache.cache_dir, 'stock_sector_map.parquet')
        if not os.path.exists(path):
            return {}
        try:
            df = pd.read_parquet(path)
            result: Dict[str, List[str]] = {}
            for _, row in df.iterrows():
                sec = str(row['sector'])
                code = str(row['code'])
                result.setdefault(sec, []).append(code)
            return result
        except Exception as e:
            logging.debug('_load_sector_map: %s', e)
            return {}

    @staticmethod
    def _score_follow(
        sector: str,
        sector_map: Dict[str, List[str]],
        market_df: Optional[pd.DataFrame],
    ) -> tuple[float, float]:
        """返回 (跟风率, 跟风得分)。"""
        codes = _prefix_lookup(sector_map, sector) or []
        if len(codes) < 5 or market_df is None or market_df.empty:
            return 0.0, 0.0

        mkt = market_df[market_df['代码'].astype(str).isin(codes)].copy()
        if mkt.empty:
            return 0.0, 0.0

        mkt['涨跌幅'] = pd.to_numeric(mkt['涨跌幅'], errors='coerce')
        valid = mkt.dropna(subset=['涨跌幅'])
        if valid.empty:
            return 0.0, 0.0

        follow_count = int((valid['涨跌幅'] > 3.0).sum())
        follow_rate = follow_count / len(valid)

        if follow_rate >= 0.30:
            score = 10.0
        elif follow_rate <= 0.10:
            score = 0.0
        else:
            score = (follow_rate - 0.10) / (0.30 - 0.10) * 10.0

        return follow_rate, score

    # ──────────────────────────────────────────────────────────
    #  Step 5：涨停集中度得分 (0-30)
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _score_concentration(zt_count: int, total_zt: int) -> float:
        if total_zt <= 0:
            return 0.0
        ratio = zt_count / total_zt
        if ratio >= 0.15:
            return 30.0
        if ratio <= 0.03:
            return 0.0
        return (ratio - 0.03) / (0.15 - 0.03) * 30.0

    # ──────────────────────────────────────────────────────────
    #  Step 6：龙头高度得分 (0-20)
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _score_leader(max_lianban: int) -> float:
        mapping = {1: 4.0, 2: 8.0, 3: 12.0, 4: 16.0}
        if max_lianban >= 5:
            return 20.0
        return mapping.get(max_lianban, 0.0)

    # ──────────────────────────────────────────────────────────
    #  Step 7：板块阶段判断
    # ──────────────────────────────────────────────────────────

    def _aggregate_zt_pool_previous(self) -> Dict[str, dict]:
        """聚合昨日涨停的行业分布，返回 {行业: {zt_count, max_lianban}}。"""
        prev_df = self.dm.extra.get('zt_pool_previous')
        if prev_df is None or prev_df.empty:
            return {}

        df = prev_df.copy()
        df = df[~df['名称'].str.upper().str.contains('ST', na=False)]

        if '所属行业' not in df.columns:
            return {}

        # 取最新快照
        if '快照日期' in df.columns:
            df = df[df['快照日期'] == df['快照日期'].max()].copy()

        lb_col = '昨日连板数' if '昨日连板数' in df.columns else '连板数'
        if lb_col in df.columns:
            df['_lb'] = pd.to_numeric(df[lb_col], errors='coerce').fillna(1).astype(int)
        else:
            df['_lb'] = 1

        result = {}
        for sector, grp in df.groupby('所属行业'):
            if pd.isna(sector) or not sector:
                continue
            result[sector] = {
                'zt_count': len(grp),
                'max_lianban': int(grp['_lb'].max()),
            }
        return result

    @staticmethod
    def _classify_phase(sh: SectorHeat, prev: dict) -> str:
        prev_zt = prev.get('zt_count', 0)
        prev_max_lb = prev.get('max_lianban', 0)

        # 退潮：龙头断板 或 涨停数骤减 >= 50%
        if prev_max_lb > 0 and sh.max_lianban < prev_max_lb:
            return '退潮'
        if prev_zt >= 3 and sh.zt_count <= prev_zt * 0.5:
            return '退潮'

        # 高潮：龙头 >= 5板 且 跟风率高
        if sh.max_lianban >= 5 and sh.follow_rate >= 0.20:
            return '高潮'

        # 启动：昨日 <= 1 只涨停，今日 >= 3 只
        if prev_zt <= 1 and sh.zt_count >= 3:
            return '启动'

        # 发酵：连续 2 天 >= 3 只涨停，且龙头在晋级
        if prev_zt >= 3 and sh.zt_count >= 3 and sh.max_lianban > prev_max_lb:
            return '发酵'

        return '平稳'

    # ──────────────────────────────────────────────────────────
    #  输出/展示
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def to_markdown(sectors: List[SectorHeat]) -> str:
        if not sectors:
            return '### 热门行业\n\n暂无涨停集中的热点行业。\n'

        has_inflow = any(s.fund_source == 'sector_flow' for s in sectors)

        header_cols = '| 排名 | 行业 | 涨停数 | 龙头(连板) | 梯队 |'
        sep_cols = '|------|------|--------|-----------|------|'
        if has_inflow:
            header_cols += ' 主力净流入 |'
            sep_cols += '------------|'
        header_cols += ' 成交额 | 跟风率 | 热度 | 阶段 |'
        sep_cols += '--------|--------|------|------|'

        lines = [
            '### 热门行业 Top {}'.format(len(sectors)),
            '',
            header_cols,
            sep_cols,
        ]
        for i, sh in enumerate(sectors, 1):
            ladder_parts = []
            for lb in sorted(sh.lianban_dist):
                ladder_parts.append(f'{lb}板×{sh.lianban_dist[lb]}')
            ladder_str = ' → '.join(ladder_parts)

            vol_yi = sh.sector_turnover / 1e8
            vol_str = f'{vol_yi:.0f}亿' if vol_yi >= 1 else '—'

            follow_str = f'{sh.follow_rate:.0%}' if sh.follow_rate > 0 else '—'

            row = (
                f'| {i} '
                f'| {sh.sector} '
                f'| {sh.zt_count} '
                f'| {sh.leader_name}({sh.max_lianban}板) '
                f'| {ladder_str} '
            )
            if has_inflow:
                inflow_yi = sh.net_inflow / 1e8
                if abs(inflow_yi) >= 0.01:
                    sign = '+' if inflow_yi > 0 else ''
                    row += f'| {sign}{inflow_yi:.1f}亿 '
                else:
                    row += '| — '
            row += (
                f'| {vol_str} '
                f'| {follow_str} '
                f'| {sh.score:.0f} '
                f'| {sh.phase} |'
            )
            lines.append(row)
        return '\n'.join(lines)
