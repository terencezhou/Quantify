# -*- encoding: UTF-8 -*-
"""扫描指定概念板块，按 realtime 数据对 Doji 各条件打分排序。"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from data_cache import DataCache

CONCEPT_CODE = sys.argv[1] if len(sys.argv) > 1 else 'BK0581'

cache = DataCache('stock_data')

# 1. 板块成分股
cons = cache.get_concept_cons(CONCEPT_CODE)
if cons is None or cons.empty:
    print(f"概念 {CONCEPT_CODE} 无成分股数据"); sys.exit(1)
codes = cons['代码'].astype(str).str.zfill(6).tolist()
names = dict(zip(cons['代码'].astype(str).str.zfill(6), cons.get('名称', cons.get('代码', ''))))
board_name = cons['板块名称'].iloc[0] if '板块名称' in cons.columns else CONCEPT_CODE
print(f"概念板块: {board_name} ({CONCEPT_CODE}), 成分股 {len(codes)} 只\n")

# 2. 加载 realtime
rt_df = cache.get_snapshot_latest('realtime_quotes')
if rt_df is not None and not rt_df.empty:
    rt_df['代码'] = rt_df['代码'].astype(str).str.zfill(6)
    rt_map = {row['代码']: row for _, row in rt_df.iterrows()}
else:
    rt_map = {}

rows = []
for code in codes:
    daily = cache.get_daily(code)
    if daily is None or len(daily) < 25:
        continue
    daily['日期'] = pd.to_datetime(daily['日期'])
    daily = daily.sort_values('日期').reset_index(drop=True)
    for col in ('开盘','收盘','最高','最低','涨跌幅'):
        if col in daily.columns:
            daily[col] = pd.to_numeric(daily[col], errors='coerce')

    # 用 realtime 合成今天的 K 线
    rt = rt_map.get(code)
    if rt is not None:
        open_p = pd.to_numeric(rt.get('今开'), errors='coerce')
        close_p = pd.to_numeric(rt.get('最新价'), errors='coerce')
        high_p = pd.to_numeric(rt.get('最高'), errors='coerce')
        low_p = pd.to_numeric(rt.get('最低'), errors='coerce')
        chg = pd.to_numeric(rt.get('涨跌幅'), errors='coerce')
        if all(pd.notna([open_p, close_p, high_p, low_p])) and open_p > 0:
            today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
            last_date = daily['日期'].iloc[-1].strftime('%Y-%m-%d')
            if last_date < today_str:
                new_row = {'日期': pd.Timestamp(today_str), '开盘': open_p,
                           '收盘': close_p, '最高': high_p, '最低': low_p,
                           '涨跌幅': chg if pd.notna(chg) else 0}
                daily = pd.concat([daily, pd.DataFrame([new_row])], ignore_index=True)

    pos = len(daily) - 1
    if pos < 5:
        continue
    r = daily.iloc[pos]
    o, c, h, l = r['开盘'], r['收盘'], r['最高'], r['最低']
    if pd.isna(o) or pd.isna(c) or pd.isna(h) or pd.isna(l) or o <= 0:
        continue

    amplitude = h - l
    if amplitude <= 0:
        continue

    body_pct = abs(c - o) / amplitude * 100
    upper_pct = (h - max(o, c)) / amplitude * 100
    lower_pct = (min(o, c) - l) / amplitude * 100

    low20 = daily['最低'].iloc[max(0, pos-19):pos+1].min()
    dist_low_pct = (l - low20) / low20 * 100 if pd.notna(low20) and low20 > 0 else 999

    prev5_close = daily.iloc[pos-5]['收盘'] if pos >= 5 else None
    prev1_close = daily.iloc[pos-1]['收盘'] if pos >= 1 else None
    pre5_pct = ((prev1_close - prev5_close) / prev5_close * 100
                if prev5_close and prev1_close and prev5_close > 0 else 0)

    # 筹码
    chips = cache.get_chips(code)
    avg_cost = np.nan
    conc70 = np.nan
    profit_ratio = np.nan
    if chips is not None and not chips.empty:
        chips['日期'] = pd.to_datetime(chips['日期'])
        last_chip = chips.sort_values('日期').iloc[-1]
        avg_cost = pd.to_numeric(last_chip.get('平均成本'), errors='coerce')
        conc70 = pd.to_numeric(last_chip.get('70集中度'), errors='coerce')
        profit_ratio = pd.to_numeric(last_chip.get('获利比例'), errors='coerce')

    cost_discount = ((c - avg_cost) / avg_cost * 100) if pd.notna(avg_cost) and avg_cost > 0 else 0

    # 逐条打分: 越接近阈值得分越高, 满足得 1.0
    def score_le(val, thresh, soft=5):
        """val <= thresh 得 1.0, 超出 soft 范围得 0"""
        if val <= thresh: return 1.0
        return max(0, 1 - (val - thresh) / soft)

    def score_ge(val, thresh, soft=5):
        if val >= thresh: return 1.0
        return max(0, 1 - (thresh - val) / soft)

    s_body   = score_le(body_pct, 15, soft=15)
    s_upper  = score_ge(upper_pct, 15, soft=15)
    s_lower  = score_ge(lower_pct, 15, soft=15)
    s_dist   = score_le(dist_low_pct, 10, soft=10)
    s_pre5   = score_le(pre5_pct, -3, soft=5)
    s_cost   = score_le(cost_discount, 0, soft=5)
    s_conc   = score_le(conc70 * 100 if pd.notna(conc70) else 99, 6, soft=4) 

    total = s_body + s_upper + s_lower + s_dist + s_pre5 + s_cost + s_conc
    name = names.get(code, code)

    rows.append({
        '代码': code, '名称': name, '收盘': round(c, 2),
        '涨跌幅%': round(r.get('涨跌幅', 0) or 0, 2),
        '实体%': round(body_pct, 1), '上影%': round(upper_pct, 1),
        '下影%': round(lower_pct, 1), '距低点%': round(dist_low_pct, 1),
        '前5日%': round(pre5_pct, 1),
        '成本偏离%': round(cost_discount, 1),
        '70集中%': round(conc70 * 100, 2) if pd.notna(conc70) else None,
        '获利比%': round(profit_ratio * 100, 1) if pd.notna(profit_ratio) else None,
        '总分': round(total, 2),
        '_s': [round(s_body,2), round(s_upper,2), round(s_lower,2),
               round(s_dist,2), round(s_pre5,2), round(s_cost,2), round(s_conc,2)],
    })

result = pd.DataFrame(rows).sort_values('总分', ascending=False).reset_index(drop=True)

pd.set_option('display.max_rows', 60)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 12)

print(f"{'='*100}")
print(f"  Doji 匹配排名 (满分 7.0)  —  {board_name}")
print(f"  条件: 实体小 + 上下影长 + 近20日低位 + 前5日下跌 + 低于均价 + 筹码集中")
print(f"{'='*100}\n")

show = result.drop(columns=['_s'])
print(show.head(40).to_string(index=True))

print(f"\n共 {len(result)} 只, 显示前 40 只")
print(f"\n满分股 (>=6.0): {len(result[result['总分']>=6.0])} 只")
print(f"高分股 (>=5.0): {len(result[result['总分']>=5.0])} 只")
print(f"中分股 (>=4.0): {len(result[result['总分']>=4.0])} 只")
