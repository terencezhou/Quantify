# -*- encoding: UTF-8 -*-

"""箱体跌幅 + 底部十字星流程（宽松版 Doji）。

与 bottom_doji_flow.py 的区别：
- 增加箱体预筛：2个月内距高点跌幅≥20%
- 十字星搜索窗口：最近7个交易日（原版4日）
- 下跌条件：距十字星前7日最高点跌≥7%（原版：前5日跌≥3%）
- 确认窗口：十字星后至当前最新日（原版：后1~3日）
- 筹码集中度上限：10%（原版6%）
- 输出增加"近期候选"列表

用法：
    python main_new.py doji2
    python main_new.py doji2 --fast
    python main_new.py doji2 --no-refresh
"""

from __future__ import annotations

import datetime
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
class BoxDojiFlowItem:
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

    box_high: float = 0.0
    box_decline_pct: float = 0.0

    body_pct: float = 0.0
    upper_pct: float = 0.0
    lower_pct: float = 0.0
    dist_low_pct: float = 0.0
    pre_decline_pct: float = 0.0

    confirm_pct: float = 0.0
    confirm_vol_ratio: float = 0.0
    summary: str = ''


@dataclass
class BoxDojiFlowResult:
    """完整流程结果。"""

    trade_date: str = ''
    today_signals: List[BoxDojiFlowItem] = field(default_factory=list)
    today_confirmed: List[BoxDojiFlowItem] = field(default_factory=list)
    recent_candidates: List[BoxDojiFlowItem] = field(default_factory=list)
    scanned: int = 0


class BoxDojiFlow:
    """箱体跌幅 + 底部十字星流程。"""

    CHIP_CONC_MAX = 0.10
    BOX_DAYS = 40
    BOX_DECLINE_MIN = 20
    DOJI_WINDOW = 7
    PRE_DECLINE_MIN = 7

    def __init__(self, dm: DataManager, fast_mode: bool = False):
        self.dm = dm
        self._data_dir = dm.config.get('data_dir', 'stock_data')
        self._sector_map = self._load_sector_map()
        self._realtime_map = self._build_realtime_map()
        self._concept_map = self._build_concept_map()
        self._chips_map = self._build_chips_map()
        self._fast_mode = fast_mode and bool(self._realtime_map)
        self._today_str = self._resolve_fast_date() if self._fast_mode else ''

    def run(self) -> BoxDojiFlowResult:
        result = BoxDojiFlowResult(trade_date=self._get_trade_date())

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
            result.recent_candidates.append(item)

        result.today_signals.sort(key=lambda x: (x.discount_pct, -x.chip_conc70_pct))
        result.today_confirmed.sort(key=lambda x: (x.confirm_pct, x.discount_pct), reverse=True)
        result.recent_candidates.sort(
            key=lambda x: (x.confirm_date != '', x.box_decline_pct), reverse=True,
        )

        mode_tag = '[快速]' if self._fast_mode else ''
        logging.info(
            'BoxDojiFlow%s: 扫描%d只, 近期候选%d只, 今日信号%d只, 今日确认%d只',
            mode_tag, result.scanned, len(result.recent_candidates),
            len(result.today_signals), len(result.today_confirmed),
        )
        return result

    # ── 核心扫描 ────────────────────────────────────────────

    def _scan_one(
        self,
        code: str,
        name: str,
        df: pd.DataFrame,
        trade_date: str,
    ) -> Optional[BoxDojiFlowItem]:
        if df is None or len(df) < 40:
            return None

        work = df.copy()
        work['日期'] = work['日期'].astype(str)
        work = work.sort_values('日期').reset_index(drop=True)
        for col in ('开盘', '收盘', '最高', '最低', '成交额', '振幅', '涨跌幅', '换手率'):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors='coerce')

        if self._fast_mode and self._today_str:
            last_date = str(work.iloc[-1]['日期'])[:10]
            if last_date < self._today_str:
                rt_kline = self._build_realtime_kline(code, self._today_str)
                if rt_kline is not None:
                    work = pd.concat(
                        [work, pd.DataFrame([rt_kline])], ignore_index=True,
                    )

        last_pos = len(work) - 1

        box_info = self._check_box_decline(
            work, last_pos, self.BOX_DAYS, self.BOX_DECLINE_MIN,
        )
        if box_info is None:
            return None

        start = max(20, last_pos - self.DOJI_WINDOW + 1)
        best_item = None
        for pos in range(start, last_pos + 1):
            signal_date = str(work.iloc[pos]['日期'])[:10]
            raw = self._detect_bottom_doji(work, pos, self.PRE_DECLINE_MIN)
            if raw is None:
                continue
            chip_raw = self._get_chip_raw(code, signal_date, float(work.iloc[pos]['收盘']))
            if chip_raw is None:
                continue

            confirm = self._detect_confirmation(work, pos)
            item = self._build_item(code, name, work, pos, raw, chip_raw, confirm, box_info)
            if best_item is None:
                best_item = item
                continue
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

    # ── 条件检测 ────────────────────────────────────────────

    @staticmethod
    def _check_box_decline(
        df: pd.DataFrame, pos: int, box_days: int = 40, min_decline: float = 20,
    ) -> Optional[dict]:
        start = max(0, pos - box_days + 1)
        box_slice = df.iloc[start:pos + 1]
        box_high = pd.to_numeric(box_slice['最高'], errors='coerce').max()
        close_p = pd.to_numeric(df.iloc[pos]['收盘'], errors='coerce')
        if pd.isna(box_high) or pd.isna(close_p) or box_high <= 0:
            return None
        decline_pct = (box_high - close_p) / box_high * 100
        if decline_pct < min_decline:
            return None
        return {
            'box_high': round(float(box_high), 2),
            'box_decline_pct': round(float(decline_pct), 1),
        }

    @staticmethod
    def _detect_bottom_doji(
        df: pd.DataFrame, pos: int, pre_decline_min: float = 7,
    ) -> Optional[dict]:
        if pos < 7:
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

        high7 = df['收盘'].iloc[max(0, pos - 7):pos].max()
        if pd.isna(high7) or high7 <= 0:
            return None
        pre_decline_pct = (high7 - close_p) / high7 * 100
        if pre_decline_pct < pre_decline_min:
            return None

        return {
            'body_pct': round(body_pct, 1),
            'upper_pct': round(upper_pct, 1),
            'lower_pct': round(lower_pct, 1),
            'dist_low_pct': round(dist_low_pct, 1),
            'pre_decline_pct': round(pre_decline_pct, 1),
        }

    @staticmethod
    def _detect_confirmation(df: pd.DataFrame, pos: int) -> Optional[dict]:
        if pos >= len(df) - 1:
            return None

        base_close = df.iloc[pos]['收盘']
        if pd.isna(base_close) or base_close <= 0:
            return None

        base_vol = pd.to_numeric(df.iloc[pos].get('成交量'), errors='coerce')
        for i in range(pos + 1, len(df)):
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

    # ── 筹码 ────────────────────────────────────────────────

    def _get_chip_raw(self, code: str, signal_date: str, close_p: float) -> Optional[dict]:
        cdf = self._chips_map.get(code)
        if cdf is None or cdf.empty:
            return None
        date_col = cdf['日期'].astype(str).str[:10]
        sub = cdf[date_col == signal_date]
        if sub.empty and self._fast_mode:
            earlier = cdf[date_col < signal_date]
            if not earlier.empty:
                sub = earlier.sort_values('日期').tail(1)
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

    # ── 组装结果 ────────────────────────────────────────────

    def _build_item(
        self,
        code: str,
        name: str,
        df: pd.DataFrame,
        pos: int,
        raw: dict,
        chip_raw: dict,
        confirm: Optional[dict],
        box_info: Optional[dict] = None,
    ) -> BoxDojiFlowItem:
        signal_date = str(df.iloc[pos]['日期'])[:10]
        latest = df.iloc[-1]
        rt = self._realtime_map.get(code, {})
        concepts = self._concept_map.get(code, [])[:3]

        item = BoxDojiFlowItem(
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
            box_high=box_info['box_high'] if box_info else 0.0,
            box_decline_pct=box_info['box_decline_pct'] if box_info else 0.0,
            avg_cost=chip_raw['avg_cost'],
            discount_pct=chip_raw['discount_pct'],
            chip_conc70_pct=chip_raw['chip_conc70_pct'],
            profit_ratio_pct=chip_raw['profit_ratio_pct'],
            body_pct=raw['body_pct'],
            upper_pct=raw['upper_pct'],
            lower_pct=raw['lower_pct'],
            dist_low_pct=raw['dist_low_pct'],
            pre_decline_pct=raw['pre_decline_pct'],
            confirm_pct=confirm['confirm_pct'] if confirm else 0.0,
            confirm_vol_ratio=confirm['confirm_vol_ratio'] if confirm else 0.0,
        )
        item.summary = self._build_summary(item)
        return item

    @staticmethod
    def _build_summary(item: BoxDojiFlowItem) -> str:
        parts = [
            f'箱高{item.box_high:.2f}跌{item.box_decline_pct:.1f}%',
            f'低于均价{item.discount_pct:.1f}%',
            f'70集中{item.chip_conc70_pct:.2f}%',
            f'距7日高点跌{item.pre_decline_pct:.1f}%',
        ]
        if item.confirm_date:
            parts.append(f'确认{item.confirm_pct:+.1f}%')
        return ' / '.join(parts)

    # ── 数据映射构建 ────────────────────────────────────────

    def _build_realtime_map(self) -> Dict[str, dict]:
        df = self.dm.extra.get('realtime')
        if df is None or getattr(df, 'empty', True):
            return {}
        work = df.copy()
        work['代码'] = work['代码'].astype(str).str.zfill(6)
        return {str(row['代码']): row.to_dict() for _, row in work.iterrows()}

    def _load_sector_map(self) -> Dict[str, str]:
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
        data = self.dm.extra.get('chips') or {}
        result = {}
        for key, df in data.items():
            code = str(key[0]).zfill(6) if isinstance(key, tuple) else str(key).zfill(6)
            if df is None or df.empty:
                continue
            result[code] = df
        return result

    def _resolve_fast_date(self) -> str:
        now = datetime.datetime.now()
        if now.weekday() < 5 and now.hour >= 9:
            return now.date().isoformat()
        return ''

    def _build_realtime_kline(self, code: str, date_str: str) -> Optional[dict]:
        rt = self._realtime_map.get(code)
        if rt is None:
            return None
        open_p = pd.to_numeric(rt.get('今开'), errors='coerce')
        close_p = pd.to_numeric(rt.get('最新价'), errors='coerce')
        high_p = pd.to_numeric(rt.get('最高'), errors='coerce')
        low_p = pd.to_numeric(rt.get('最低'), errors='coerce')
        if pd.isna(open_p) or pd.isna(close_p) or pd.isna(high_p) or pd.isna(low_p):
            return None
        if open_p <= 0 or close_p <= 0:
            return None
        return {
            '日期': date_str,
            '开盘': float(open_p),
            '收盘': float(close_p),
            '最高': float(high_p),
            '最低': float(low_p),
            '成交量': float(pd.to_numeric(rt.get('成交量', 0), errors='coerce') or 0),
            '成交额': float(pd.to_numeric(rt.get('成交额', 0), errors='coerce') or 0),
            '涨跌幅': float(pd.to_numeric(rt.get('涨跌幅', 0), errors='coerce') or 0),
            '振幅': float(pd.to_numeric(rt.get('振幅', 0), errors='coerce') or 0),
            '换手率': float(pd.to_numeric(rt.get('换手率', 0), errors='coerce') or 0),
        }

    def _get_trade_date(self) -> str:
        if self._fast_mode and self._today_str:
            return self._today_str
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

    # ── 输出 ────────────────────────────────────────────────

    @staticmethod
    def _format_concepts(concepts: List[str]) -> str:
        return '、'.join(concepts[:3]) if concepts else '-'

    @classmethod
    def _items_to_lines(cls, items: List[BoxDojiFlowItem], limit: int) -> List[str]:
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
    def to_markdown(cls, result: BoxDojiFlowResult, limit: int = 30) -> str:
        lines: List[str] = []
        lines.append(f'## 箱体十字星流程 ({result.trade_date})')
        lines.append('')
        lines.append(
            f'> 扫描 {result.scanned} 只，近期候选 {len(result.recent_candidates)} 只，'
            f'今日新信号 {len(result.today_signals)} 只，'
            f'今日确认 {len(result.today_confirmed)} 只'
        )
        lines.append('')
        lines.append('### 近期候选（7日内十字星 + 箱体跌幅≥20%）')
        lines.append('')
        lines.extend(cls._items_to_lines(result.recent_candidates, limit))
        lines.append('')
        lines.append('### 今日确认信号')
        lines.append('')
        lines.extend(cls._items_to_lines(result.today_confirmed, limit))
        lines.append('')
        return '\n'.join(lines)
