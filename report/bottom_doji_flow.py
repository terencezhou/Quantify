# -*- encoding: UTF-8 -*-

"""底部十字星 + 低估筹码流程。

筛选条件：
1. 出现底部十字星
2. 信号日收盘价低于筹码平均成本
3. 70%集中度足够低（默认 < 4%）

输出分两部分：
1. 今日新信号：最新交易日新出现的候选
2. 今日确认信号：前 1~3 日十字星在今日完成确认
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

try:
    from data_manager import DataManager
except ModuleNotFoundError:
    from ..data_manager import DataManager  # type: ignore[no-redef]


@dataclass
class DojiFlowItem:
    """单只股票的输出结果。"""

    code: str = ''
    name: str = ''
    signal_date: str = ''
    confirm_date: str = ''
    industry: str = ''
    concepts: List[str] = field(default_factory=list)

    close: float = 0.0
    change_pct: float = 0.0
    amplitude_pct: float = 0.0
    turnover_pct: float = 0.0
    amount_yi: float = 0.0
    total_mv_yi: float = 0.0
    float_mv_yi: float = 0.0

    avg_cost: float = 0.0
    discount_pct: float = 0.0
    chip_conc70_pct: float = 0.0
    profit_ratio_pct: float = 0.0

    body_pct: float = 0.0
    upper_pct: float = 0.0
    lower_pct: float = 0.0
    dist_low_pct: float = 0.0
    pre5_pct: float = 0.0

    confirm_pct: float = 0.0
    confirm_vol_ratio: float = 0.0
    summary: str = ''


@dataclass
class BottomDojiFlowResult:
    """完整流程结果。"""

    trade_date: str = ''
    today_signals: List[DojiFlowItem] = field(default_factory=list)
    today_confirmed: List[DojiFlowItem] = field(default_factory=list)
    scanned: int = 0


class BottomDojiFlow:
    """底部十字星 + 低估筹码流程。"""

    CHIP_CONC_MAX = 0.06

    def __init__(self, dm: DataManager):
        self.dm = dm
        self._data_dir = dm.config.get('data_dir', 'stock_data')
        self._sector_map = self._load_sector_map()
        self._realtime_map = self._build_realtime_map()
        self._concept_map = self._build_concept_map()
        self._chips_map = self._build_chips_map()

    def run(self) -> BottomDojiFlowResult:
        """扫描全市场，返回今日信号与今日确认。"""
        result = BottomDojiFlowResult(trade_date=self._get_trade_date())

        for key, df in self.dm.stocks_data.items():
            result.scanned += 1
            code, name = str(key[0]).zfill(6), str(key[1])
            if self._is_excluded(code, name):
                continue
            item = self._scan_one(code, name, df, result.trade_date)
            if item is None:
                continue
            if item.signal_date == result.trade_date:
                result.today_signals.append(item)
            if item.confirm_date == result.trade_date:
                result.today_confirmed.append(item)

        result.today_signals.sort(key=lambda x: (x.discount_pct, -x.chip_conc70_pct))
        result.today_confirmed.sort(key=lambda x: (x.confirm_pct, x.discount_pct), reverse=True)

        logging.info(
            'BottomDojiFlow: 扫描%d只, 今日信号%d只, 今日确认%d只',
            result.scanned, len(result.today_signals), len(result.today_confirmed),
        )
        return result

    def _scan_one(
        self,
        code: str,
        name: str,
        df: pd.DataFrame,
        trade_date: str,
    ) -> Optional[DojiFlowItem]:
        """扫描单只股票。"""
        if df is None or len(df) < 40:
            return None

        work = df.copy()
        work['日期'] = work['日期'].astype(str)
        work = work.sort_values('日期').reset_index(drop=True)
        for col in ('开盘', '收盘', '最高', '最低', '成交额', '振幅', '涨跌幅', '换手率'):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors='coerce')

        last_pos = len(work) - 1
        start = max(20, last_pos - 3)
        best_item = None
        for pos in range(start, last_pos + 1):
            signal_date = str(work.iloc[pos]['日期'])[:10]
            raw = self._detect_bottom_doji(work, pos)
            if raw is None:
                continue
            chip_raw = self._get_chip_raw(code, signal_date, float(work.iloc[pos]['收盘']))
            if chip_raw is None:
                continue

            confirm = self._detect_confirmation(work, pos)
            item = self._build_item(code, name, work, pos, raw, chip_raw, confirm)
            if best_item is None:
                best_item = item
                continue
            # 优先保留今日确认，其次今日信号，再次最新日期
            best_key = (
                best_item.confirm_date == trade_date,
                best_item.signal_date == trade_date,
                best_item.signal_date,
            )
            cur_key = (
                item.confirm_date == trade_date,
                item.signal_date == trade_date,
                item.signal_date,
            )
            if cur_key > best_key:
                best_item = item
        return best_item

    @staticmethod
    def _detect_bottom_doji(df: pd.DataFrame, pos: int) -> Optional[dict]:
        """判断某日是否为底部十字星。"""
        if pos < 5:
            return None

        open_p = df.iloc[pos]['开盘']
        close_p = df.iloc[pos]['收盘']
        high_p = df.iloc[pos]['最高']
        low_p = df.iloc[pos]['最低']
        if pd.isna(open_p) or pd.isna(close_p) or pd.isna(high_p) or pd.isna(low_p):
            return None

        amplitude = high_p - low_p
        if amplitude <= 0:
            return None

        body_pct = abs(close_p - open_p) / amplitude * 100
        upper_pct = (high_p - max(open_p, close_p)) / amplitude * 100
        lower_pct = (min(open_p, close_p) - low_p) / amplitude * 100
        if body_pct >= 15 or upper_pct < 15 or lower_pct < 15:
            return None

        low20 = df['最低'].iloc[max(0, pos - 19):pos + 1].min()
        if pd.isna(low20) or low20 <= 0:
            return None
        dist_low_pct = (low_p - low20) / low20 * 100
        if dist_low_pct > 10:
            return None

        prev_close_5 = df.iloc[pos - 5]['收盘']
        prev_close_1 = df.iloc[pos - 1]['收盘']
        if pd.isna(prev_close_5) or pd.isna(prev_close_1) or prev_close_5 <= 0:
            return None
        pre5_pct = (prev_close_1 - prev_close_5) / prev_close_5 * 100
        if pre5_pct > -3:
            return None

        return {
            'body_pct': round(body_pct, 1),
            'upper_pct': round(upper_pct, 1),
            'lower_pct': round(lower_pct, 1),
            'dist_low_pct': round(dist_low_pct, 1),
            'pre5_pct': round(pre5_pct, 1),
        }

    @staticmethod
    def _detect_confirmation(df: pd.DataFrame, pos: int) -> Optional[dict]:
        """检查十字星后 1~3 日是否出现确认。"""
        if pos >= len(df) - 1:
            return None

        base_close = df.iloc[pos]['收盘']
        if pd.isna(base_close) or base_close <= 0:
            return None

        base_vol = pd.to_numeric(df.iloc[pos].get('成交量'), errors='coerce')
        for i in range(pos + 1, min(pos + 4, len(df))):
            row = df.iloc[i]
            open_p = row['开盘']
            close_p = row['收盘']
            if pd.isna(open_p) or pd.isna(close_p):
                continue

            confirm_pct = (close_p - base_close) / base_close * 100
            is_yang = close_p > open_p
            vol_ratio = 0.0
            cur_vol = pd.to_numeric(row.get('成交量'), errors='coerce')
            if pd.notna(base_vol) and base_vol > 0 and pd.notna(cur_vol):
                vol_ratio = float(cur_vol / base_vol)
            if confirm_pct >= 3 and is_yang:
                return {
                    'confirm_date': str(row['日期'])[:10],
                    'confirm_pct': round(confirm_pct, 1),
                    'confirm_vol_ratio': round(vol_ratio, 2),
                }
        return None

    def _get_chip_raw(self, code: str, signal_date: str, close_p: float) -> Optional[dict]:
        """读取信号日筹码条件。"""
        cdf = self._chips_map.get(code)
        if cdf is None or cdf.empty:
            return None
        sub = cdf[cdf['日期'].astype(str).str[:10] == signal_date]
        if sub.empty:
            return None
        row = sub.iloc[-1]
        avg_cost = pd.to_numeric(row.get('平均成本'), errors='coerce')
        conc70 = pd.to_numeric(row.get('70集中度'), errors='coerce')
        profit_ratio = pd.to_numeric(row.get('获利比例'), errors='coerce')
        if pd.isna(avg_cost) or avg_cost <= 0 or pd.isna(conc70):
            return None
        if close_p >= avg_cost:
            return None
        if conc70 >= self.CHIP_CONC_MAX:
            return None
        return {
            'avg_cost': round(float(avg_cost), 2),
            'discount_pct': round((close_p - float(avg_cost)) / float(avg_cost) * 100, 1),
            'chip_conc70_pct': round(float(conc70) * 100, 2),
            'profit_ratio_pct': round(float(profit_ratio) * 100, 1) if pd.notna(profit_ratio) else 0.0,
        }

    def _build_item(
        self,
        code: str,
        name: str,
        df: pd.DataFrame,
        pos: int,
        raw: dict,
        chip_raw: dict,
        confirm: Optional[dict],
    ) -> DojiFlowItem:
        """组装单只结果。"""
        signal_date = str(df.iloc[pos]['日期'])[:10]
        latest = df.iloc[-1]
        rt = self._realtime_map.get(code, {})
        concepts = self._concept_map.get(code, [])[:3]

        item = DojiFlowItem(
            code=code,
            name=name,
            signal_date=signal_date,
            confirm_date=confirm['confirm_date'] if confirm else '',
            industry=self._sector_map.get(code, ''),
            concepts=concepts,
            close=round(float(latest.get('收盘', 0) or 0), 2),
            change_pct=round(float(latest.get('涨跌幅', 0) or 0), 1),
            amplitude_pct=round(float(latest.get('振幅', 0) or 0), 1),
            turnover_pct=round(float(latest.get('换手率', 0) or 0), 1),
            amount_yi=round(float(latest.get('成交额', 0) or 0) / 1e8, 2),
            total_mv_yi=round(float(rt.get('总市值', 0) or 0) / 1e8, 1),
            float_mv_yi=round(float(rt.get('流通市值', 0) or 0) / 1e8, 1),
            avg_cost=chip_raw['avg_cost'],
            discount_pct=chip_raw['discount_pct'],
            chip_conc70_pct=chip_raw['chip_conc70_pct'],
            profit_ratio_pct=chip_raw['profit_ratio_pct'],
            body_pct=raw['body_pct'],
            upper_pct=raw['upper_pct'],
            lower_pct=raw['lower_pct'],
            dist_low_pct=raw['dist_low_pct'],
            pre5_pct=raw['pre5_pct'],
            confirm_pct=confirm['confirm_pct'] if confirm else 0.0,
            confirm_vol_ratio=confirm['confirm_vol_ratio'] if confirm else 0.0,
        )
        item.summary = self._build_summary(item)
        return item

    @staticmethod
    def _build_summary(item: DojiFlowItem) -> str:
        """一句话摘要。"""
        parts = [
            f'低于均价{item.discount_pct:.1f}%',
            f'70集中{item.chip_conc70_pct:.2f}%',
            f'前5日{item.pre5_pct:+.1f}%',
        ]
        if item.confirm_date:
            parts.append(f'确认{item.confirm_pct:+.1f}%')
        return ' / '.join(parts)

    def _build_realtime_map(self) -> Dict[str, dict]:
        """构建实时快照映射。"""
        df = self.dm.extra.get('realtime')
        if df is None or getattr(df, 'empty', True):
            return {}
        work = df.copy()
        work['代码'] = work['代码'].astype(str).str.zfill(6)
        return {str(row['代码']): row.to_dict() for _, row in work.iterrows()}

    def _load_sector_map(self) -> Dict[str, str]:
        """读取 code -> 行业 映射。"""
        path = os.path.join(self._data_dir, 'stock_sector_map.parquet')
        if not os.path.exists(path):
            return {}
        try:
            df = pd.read_parquet(path)
        except (OSError, ValueError):
            return {}
        if df is None or df.empty or 'code' not in df.columns or 'sector' not in df.columns:
            return {}
        work = df.copy()
        work['code'] = work['code'].astype(str).str.zfill(6)
        return dict(zip(work['code'], work['sector'].astype(str)))

    def _build_concept_map(self) -> Dict[str, List[str]]:
        """构建 code -> [概念...] 映射。"""
        concept_cons = self.dm.extra.get('concept_cons') or {}
        result: Dict[str, List[str]] = {}
        for _board_code, cons_df in concept_cons.items():
            if cons_df is None or cons_df.empty or '代码' not in cons_df.columns:
                continue
            board_name = ''
            if '板块名称' in cons_df.columns and not cons_df['板块名称'].dropna().empty:
                board_name = str(cons_df['板块名称'].dropna().iloc[0])
            for code in cons_df['代码'].astype(str).str.zfill(6):
                result.setdefault(code, [])
                if board_name and board_name not in result[code]:
                    result[code].append(board_name)
        return result

    def _build_chips_map(self) -> Dict[str, pd.DataFrame]:
        """构建 code -> chips_df 映射。"""
        data = self.dm.extra.get('chips') or {}
        result = {}
        for key, df in data.items():
            code = str(key[0]).zfill(6) if isinstance(key, tuple) else str(key).zfill(6)
            if df is None or df.empty:
                continue
            result[code] = df
        return result

    def _get_trade_date(self) -> str:
        """返回日K最新交易日。"""
        for _key, df in self.dm.stocks_data.items():
            if df is not None and not df.empty and '日期' in df.columns:
                return str(df['日期'].iloc[-1])[:10]
        return ''

    @staticmethod
    def _is_excluded(code: str, name: str) -> bool:
        return (
            ('ST' in name.upper())
            or code.startswith(('8', '4', '92', '30', '68'))
        )

    @staticmethod
    def _format_concepts(concepts: List[str]) -> str:
        return '、'.join(concepts[:3]) if concepts else '-'

    @classmethod
    def _items_to_lines(cls, items: List[DojiFlowItem], limit: int) -> List[str]:
        lines: List[str] = []
        if not items:
            lines.append('*无*')
            return lines
        lines.append('| # | 代码 | 名称 | 行业 | 概念 | 收盘/涨跌 | 换手/成交额 | 筹码 | 信号 |')
        lines.append('|---|---|---|---|---|---|---|---|---|')
        for idx, item in enumerate(items[:limit], 1):
            lines.append(
                f'| {idx} | {item.code} | {item.name} | {item.industry or "-"} | '
                f'{cls._format_concepts(item.concepts)} | '
                f'{item.close:.2f} / {item.change_pct:+.1f}% | '
                f'{item.turnover_pct:.1f}% / {item.amount_yi:.2f}亿 | '
                f'均价{item.avg_cost:.2f}, 偏离{item.discount_pct:+.1f}%, '
                f'70集中{item.chip_conc70_pct:.2f}%, 获利{item.profit_ratio_pct:.1f}% | '
                f'{item.summary} |'
            )
        return lines

    @classmethod
    def to_markdown(cls, result: BottomDojiFlowResult, limit: int = 20) -> str:
        """输出 Markdown 报告。"""
        lines: List[str] = []
        lines.append(f'## 底部十字星低估筹码流程 ({result.trade_date})')
        lines.append('')
        lines.append(
            f'> 扫描 {result.scanned} 只，今日新信号 {len(result.today_signals)} 只，'
            f'今日确认 {len(result.today_confirmed)} 只'
        )
        lines.append('')
        lines.append('### 今日新信号')
        lines.append('')
        lines.extend(cls._items_to_lines(result.today_signals, limit))
        lines.append('')
        lines.append('### 今日确认信号')
        lines.append('')
        lines.extend(cls._items_to_lines(result.today_confirmed, limit))
        lines.append('')
        return '\n'.join(lines)
