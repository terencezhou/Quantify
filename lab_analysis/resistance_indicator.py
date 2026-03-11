# -*- encoding: UTF-8 -*-
"""日内阻力指标实验。

将每个交易日拆成：
  1. 隔夜缺口 gap_pct = (open - prev_close) / prev_close * 100
  2. 日内方向 direction = 1(收>开) / -1(收<开) / 0
  3. 日内阻力 resistance = volume / range_pct (越大越难推动)
  4. 相对阻力 rel_resistance = resistance / MA20(resistance)
  5. 日类型   day_type: 一字板 / 十字星 / 正常

用法:
    python lab_analysis/resistance_indicator.py [股票代码] [--days N]

示例:
    python lab_analysis/resistance_indicator.py 601789
    python lab_analysis/resistance_indicator.py 601789 --days 40
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from data_cache import DataCache


def classify_day(row):
    h, l = row['最高'], row['最低']
    o, c = row['开盘'], row['收盘']
    if h == l:
        return '一字板'
    if (h - l) > 0 and abs(c - o) / (h - l) < 0.15:
        return '十字星'
    return '正常'


def compute_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """在日K DataFrame 上计算阻力指标，返回增强后的 DataFrame。"""
    w = df.copy()
    for col in ('开盘', '收盘', '最高', '最低', '成交量'):
        w[col] = pd.to_numeric(w[col], errors='coerce')
    w['日期'] = pd.to_datetime(w['日期'])
    w = w.sort_values('日期').reset_index(drop=True)

    w['prev_close'] = w['收盘'].shift(1)

    # 隔夜缺口
    w['gap_pct'] = (w['开盘'] - w['prev_close']) / w['prev_close'] * 100

    # 方向：基于收盘价 vs 前一天收盘价（用涨跌幅判断，避免四舍五入精度问题）
    chg_pct = (w['收盘'] - w['prev_close']) / w['prev_close'] * 100
    w['direction'] = np.where(
        chg_pct > 0.01, 1,
        np.where(chg_pct < -0.01, -1, 0),
    )
    w['direction'] = w['direction'].fillna(0).astype(int)

    # 日类型
    w['day_type'] = w.apply(classify_day, axis=1)

    # 日内振幅(%) — 保留供参考
    w['range_pct'] = (w['最高'] - w['最低']) / w['最低'] * 100

    # 日内实体变动(%) = |收盘 - 开盘| / 开盘
    w['body_pct'] = abs(w['收盘'] - w['开盘']) / w['开盘'] * 100

    # 日内阻力：一字板=0，十字星(body_pct极小)=inf→用NaN，其他=volume/body_pct
    w['resistance'] = np.where(
        w['day_type'] == '一字板',
        0,
        np.where(w['body_pct'] > 0.05, w['成交量'] / w['body_pct'], np.nan),
    )

    # 20日均阻力 & 相对阻力（排除一字板参与均值）
    normal_res = w['resistance'].where(w['day_type'] != '一字板')
    w['res_ma20'] = normal_res.rolling(20, min_periods=5).mean()
    w['rel_resistance'] = np.where(
        w['res_ma20'] > 0,
        w['resistance'] / w['res_ma20'],
        np.nan,
    )

    # 日内实体占比
    w['body_ratio'] = np.where(
        (w['最高'] - w['最低']) > 0,
        abs(w['收盘'] - w['开盘']) / (w['最高'] - w['最低']) * 100,
        0,
    )

    # ── 归一化（跨股票可比）──
    valid = w.loc[w['day_type'] != '一字板', 'resistance']

    # 百分位排名 0~100：该日阻力在自身历史中的位置
    w['res_pct'] = w['resistance'].rank(pct=True) * 100
    w.loc[w['day_type'] == '一字板', 'res_pct'] = 0.0

    # Z-score：偏离历史均值几个标准差
    mu, sigma = valid.mean(), valid.std()
    w['res_z'] = np.where(
        (w['day_type'] != '一字板') & (sigma > 0),
        (w['resistance'] - mu) / sigma,
        np.nan,
    )
    w.loc[w['day_type'] == '一字板', 'res_z'] = np.nan

    return w


def print_report(w: pd.DataFrame, code: str, days: int):
    """打印分析报告。"""
    show = w.tail(days).copy()

    print(f"\n{'='*120}")
    print(f"  {code} 日内阻力指标  (最近 {days} 个交易日)")
    print(f"{'='*120}")

    cols = ['日期', '收盘', 'gap_pct', 'direction', 'day_type',
            'body_pct', 'range_pct', '成交量', 'resistance', 'rel_resistance',
            'res_pct', 'res_z']
    disp = show[cols].copy()
    disp['日期'] = disp['日期'].dt.strftime('%m-%d')
    disp['gap_pct'] = disp['gap_pct'].round(2)
    disp['body_pct'] = disp['body_pct'].round(2)
    disp['range_pct'] = disp['range_pct'].round(2)
    disp['成交量'] = (disp['成交量'] / 1e4).round(1)
    disp['resistance'] = disp['resistance'].round(0)
    disp['rel_resistance'] = disp['rel_resistance'].round(2)
    disp['res_pct'] = disp['res_pct'].round(1)
    disp['res_z'] = disp['res_z'].round(2)

    disp.columns = ['日期', '收盘', '缺口%', '方向', '类型',
                    '实体%', '振幅%', '量(万)', '阻力', '相对阻力',
                    '百分位', 'Z值']

    dir_map = {1: '↑', -1: '↓', 0: '—'}
    disp['方向'] = disp['方向'].map(dir_map)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    print(disp.to_string(index=False))

    # 关键信号标注
    print(f"\n{'─'*120}")
    print("关键信号:")
    for _, r in show.iterrows():
        d = r['日期'].strftime('%m-%d')
        dt = r['day_type']
        rel = r['rel_resistance']
        dirn = '↑' if r['direction'] > 0 else ('↓' if r['direction'] < 0 else '—')
        gap = r['gap_pct']

        signals = []
        if dt == '一字板':
            signals.append('一字板(极端信号)')
        elif pd.notna(rel):
            if rel > 2.0:
                signals.append(f'高阻力({rel:.1f}x)')
            elif rel < 0.5:
                signals.append(f'低阻力({rel:.1f}x)')

        if abs(gap) > 2:
            tag = '高开' if gap > 0 else '低开'
            signals.append(f'{tag}{gap:+.1f}%')

        if dt == '十字星':
            signals.append('十字星')

        if signals:
            print(f"  {d} {dirn} {' + '.join(signals)}")

    # 阻力变化趋势
    last5 = show.tail(5)
    normal_last5 = last5[last5['day_type'] != '一字板']
    if len(normal_last5) >= 2:
        first_rel = normal_last5.iloc[0]['rel_resistance']
        last_rel = normal_last5.iloc[-1]['rel_resistance']
        if pd.notna(first_rel) and pd.notna(last_rel):
            delta = last_rel - first_rel
            if delta < -0.5:
                trend = '阻力下降(趋势启动/加速)'
            elif delta > 0.5:
                trend = '阻力上升(趋势减速/胶着)'
            else:
                trend = '阻力平稳'
            print(f"\n  近5日阻力趋势: {trend} ({first_rel:.2f} → {last_rel:.2f})")


def compare_stocks(codes: list[str], cache: DataCache, date: str | None = None):
    """多股票横向对比某一天（默认最新）的归一化阻力。"""
    rows = []
    for code in codes:
        daily = cache.get_daily(code)
        if daily is None or daily.empty:
            continue
        w = compute_resistance(daily)
        if date:
            target = w[w['日期'] == pd.Timestamp(date)]
        else:
            target = w.tail(1)
        if target.empty:
            continue
        r = target.iloc[-1]
        rows.append({
            '代码': code,
            '日期': r['日期'].strftime('%Y-%m-%d'),
            '收盘': r['收盘'],
            '方向': {1: '↑', -1: '↓', 0: '—'}.get(r['direction'], '?'),
            '类型': r['day_type'],
            '阻力(原值)': int(r['resistance']),
            '相对阻力': round(r['rel_resistance'], 2) if pd.notna(r['rel_resistance']) else '-',
            '百分位': round(r['res_pct'], 1),
            'Z值': round(r['res_z'], 2) if pd.notna(r['res_z']) else '-',
        })

    if not rows:
        print('无有效数据'); return

    result = pd.DataFrame(rows)
    result = result.sort_values('百分位', ascending=False)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    title_date = rows[0]['日期']
    print(f"\n{'='*100}")
    print(f"  多股票阻力横向对比  ({title_date})")
    print(f"  百分位: 0=历史最低阻力  100=历史最高阻力  (跨股票可直接比较)")
    print(f"{'='*100}")
    print(result.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='日内阻力指标实验')
    parser.add_argument('code', nargs='*', default=['601789'],
                        help='股票代码（支持多个，空格分隔）')
    parser.add_argument('--days', type=int, default=30, help='显示天数')
    parser.add_argument('--date', type=str, default=None,
                        help='对比日期 (YYYY-MM-DD)，仅多股票模式生效')
    args = parser.parse_args()

    codes = [str(c).zfill(6) for c in args.code]
    cache = DataCache('stock_data')

    if len(codes) == 1:
        daily = cache.get_daily(codes[0])
        if daily is None or daily.empty:
            print(f'{codes[0]} 无日K缓存'); sys.exit(1)
        w = compute_resistance(daily)
        print_report(w, codes[0], args.days)
    else:
        compare_stocks(codes, cache, args.date)


if __name__ == '__main__':
    main()
