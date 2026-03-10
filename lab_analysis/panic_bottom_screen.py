# -*- encoding: UTF-8 -*-
"""恐慌底/洗盘底 筛选器

触发条件（全部满足）：
  1. 单日跌幅 > 7%，大实体阴线
  2. 获利比例骤降至 < 10%（几乎全员套牢）
  3. 收盘价远低于平均成本（偏离 > 7%）
  4. 股价仍在 MA60 附近或之上（收盘 >= MA60 * 0.95）
  5. 次日急剧缩量（成交量 < 信号日的 70%）

用法：
  python lab_analysis/panic_bottom_screen.py
"""

import os
import sys
import pandas as pd
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'stock_data')

def load_daily(code):
    path = os.path.join(DATA_DIR, 'daily', f'{code}.parquet')
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期').reset_index(drop=True)
    return df

def load_chips(code):
    path = os.path.join(DATA_DIR, 'chips', f'{code}.parquet')
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期').reset_index(drop=True)
    return df

def screen_one(code, start_date):
    daily = load_daily(code)
    chips = load_chips(code)
    if daily is None or chips is None:
        return []

    daily['MA60'] = daily['收盘'].rolling(60).mean()
    daily_after = daily[daily['日期'] >= start_date].copy()

    chips_map = {}
    for _, row in chips.iterrows():
        chips_map[row['日期'].date()] = row

    signals = []
    for i, row in daily_after.iterrows():
        d = row['日期'].date()

        # 条件1: 跌幅 > 7%
        if row['涨跌幅'] > -7:
            continue

        # 条件4: MA60 存在且收盘 >= MA60 * 0.95
        ma60 = row.get('MA60')
        if pd.isna(ma60) or row['收盘'] < ma60 * 0.95:
            continue

        # 条件2+3: 筹码
        chip = chips_map.get(d)
        if chip is None:
            continue
        profit_ratio = chip.get('获利比例', 1.0)
        avg_cost = chip.get('平均成本', 0)

        if profit_ratio >= 0.10:
            continue
        if avg_cost <= 0 or row['收盘'] >= avg_cost * 0.93:
            continue

        # 条件5: 次日缩量（< 信号日的 70%）
        if i + 1 >= len(daily):
            continue
        next_row = daily.iloc[i + 1]
        if row['成交量'] <= 0:
            continue
        vol_ratio = next_row['成交量'] / row['成交量']
        if vol_ratio >= 0.70:
            continue

        # 触发日 = T+1（缩量确认日），后续收益从 T+1 收盘价起算
        t1 = i + 1
        t1_row = next_row
        t1_date = t1_row['日期']
        if hasattr(t1_date, 'date'):
            t1_date = t1_date.date()
        entry_close = t1_row['收盘']

        # 5日内最大收益（触发日后 1~5 个交易日的最高价 vs 触发日收盘）
        max_5d = None
        slice_5d = daily.iloc[t1 + 1: t1 + 6]
        if not slice_5d.empty:
            max_5d = (slice_5d['最高'].max() / entry_close - 1) * 100

        # 5日后收益（触发日后第5个交易日收盘 vs 触发日收盘）
        ret_5d = None
        if t1 + 5 < len(daily):
            ret_5d = (daily.iloc[t1 + 5]['收盘'] / entry_close - 1) * 100

        # 10日内最大收益
        max_10d = None
        slice_10d = daily.iloc[t1 + 1: t1 + 11]
        if not slice_10d.empty:
            max_10d = (slice_10d['最高'].max() / entry_close - 1) * 100

        signals.append({
            '代码': code,
            '大跌日': d,
            '触发日': t1_date,
            '大跌日收盘': row['收盘'],
            '触发日收盘': entry_close,
            '跌幅%': round(row['涨跌幅'], 2),
            '获利比例%': round(profit_ratio * 100, 2),
            '成本偏离%': round((row['收盘'] / avg_cost - 1) * 100, 2),
            '缩量比': round(vol_ratio, 2),
            '5日内最大收益%': round(max_5d, 2) if max_5d is not None else None,
            '5日后收益%': round(ret_5d, 2) if ret_5d is not None else None,
            '10日内最大收益%': round(max_10d, 2) if max_10d is not None else None,
        })

    return signals


def main():
    start_date = pd.Timestamp('2026-01-01')

    daily_dir = os.path.join(DATA_DIR, 'daily')
    chips_dir = os.path.join(DATA_DIR, 'chips')

    daily_codes = {f.replace('.parquet', '') for f in os.listdir(daily_dir) if f.endswith('.parquet')}
    chips_codes = {f.replace('.parquet', '') for f in os.listdir(chips_dir) if f.endswith('.parquet')}
    codes = sorted(daily_codes & chips_codes)

    print(f"扫描 {len(codes)} 只股票，起始日期: {start_date.date()}")
    print()

    all_signals = []
    for i, code in enumerate(codes):
        if i % 500 == 0 and i > 0:
            print(f"  已扫描 {i}/{len(codes)}...")
        sigs = screen_one(code, start_date)
        all_signals.extend(sigs)

    if not all_signals:
        print("未找到符合条件的信号")
        return

    result = pd.DataFrame(all_signals)
    result = result.sort_values(['触发日', '跌幅%']).reset_index(drop=True)

    out_path = os.path.join(os.path.dirname(__file__), 'panic_bottom_signals.csv')
    result.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"共找到 {len(result)} 个信号，已保存到 {out_path}\n")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 300)
    pd.set_option('display.max_rows', None)

    for d, grp in result.groupby('触发日'):
        print(f"━━━ 触发日 {d} ({len(grp)} 只) ━━━")
        print(grp.drop(columns=['触发日']).to_string(index=False))
        print()

    print("=" * 80)
    print(f"信号统计: 共 {len(result)} 个")
    print(f"按触发日分布:")
    for d, grp in result.groupby('触发日'):
        print(f"  {d}: {len(grp)} 只")
    print()
    v = result['5日内最大收益%'].dropna()
    if not v.empty:
        print(f"  5日内最大收益均值: {v.mean():.2f}%  中位数: {v.median():.2f}%  正收益率: {(v>0).sum()}/{len(v)} = {(v>0).mean()*100:.1f}%")
    v = result['5日后收益%'].dropna()
    if not v.empty:
        print(f"  5日后收益均值: {v.mean():.2f}%  中位数: {v.median():.2f}%  正收益率: {(v>0).sum()}/{len(v)} = {(v>0).mean()*100:.1f}%")
    v = result['10日内最大收益%'].dropna()
    if not v.empty:
        print(f"  10日内最大收益均值: {v.mean():.2f}%  中位数: {v.median():.2f}%  正收益率: {(v>0).sum()}/{len(v)} = {(v>0).mean()*100:.1f}%")


if __name__ == '__main__':
    main()
