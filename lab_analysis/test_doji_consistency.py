# -*- encoding: UTF-8 -*-
"""对比 BottomDojiFlow 普通模式 vs fast 模式的输出一致性。

普通模式: dm.refresh()        + BottomDojiFlow(fast_mode=False)
fast模式: dm.load_from_cache() + BottomDojiFlow(fast_mode=True)

两条完全不同的数据加载路径 + 分析路径，验证结果是否一致。
"""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
import yaml
from data_manager import DataManager
from report.bottom_doji_flow import BottomDojiFlow

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')

config_path = Path(__file__).parent.parent / 'stock_config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f) or {}


def run_one(label, dm, fast_mode):
    t0 = time.time()
    flow = BottomDojiFlow(dm, fast_mode=fast_mode)
    result = flow.run()
    elapsed = time.time() - t0
    print(f"\n>>> {label} (fast_mode={fast_mode})  耗时 {elapsed:.1f}s")
    print(f"    trade_date={result.trade_date}, "
          f"信号={len(result.today_signals)}, 确认={len(result.today_confirmed)}")
    return result


# ── 1. 普通模式: dm.refresh() ──
print("=" * 60)
print("Step 1: 普通模式 - dm.refresh()")
print("=" * 60)
t0 = time.time()
dm_normal = DataManager(config)
ok = dm_normal.refresh()
print(f"dm.refresh() 耗时 {time.time()-t0:.1f}s, stocks={len(dm_normal.stocks_data)}")
if not ok:
    print("refresh 失败"); sys.exit(1)
r_normal = run_one("普通模式", dm_normal, fast_mode=False)

# ── 2. fast模式: dm.load_from_cache() ──
print("\n" + "=" * 60)
print("Step 2: fast模式 - dm.load_from_cache()")
print("=" * 60)
t0 = time.time()
dm_fast = DataManager(config)
ok = dm_fast.load_from_cache()
print(f"dm.load_from_cache() 耗时 {time.time()-t0:.1f}s, stocks={len(dm_fast.stocks_data)}")
if not ok:
    print("cache 加载失败"); sys.exit(1)
r_fast = run_one("fast模式", dm_fast, fast_mode=True)

# ── 3. 数据层对比 ──
print("\n" + "=" * 60)
print("数据层对比:")
normal_codes = set(k[0] for k in dm_normal.stocks_data)
fast_codes = set(k[0] for k in dm_fast.stocks_data)
print(f"  日K股票数: normal={len(normal_codes)}  fast={len(fast_codes)}  "
      f"交集={len(normal_codes & fast_codes)}  "
      f"差集={len(normal_codes ^ fast_codes)}")

normal_chips = set(k[0] for k in (dm_normal.extra.get('chips') or {}))
fast_chips = set(k[0] for k in (dm_fast.extra.get('chips') or {}))
print(f"  筹码股票数: normal={len(normal_chips)}  fast={len(fast_chips)}  "
      f"交集={len(normal_chips & fast_chips)}  "
      f"差集={len(normal_chips ^ fast_chips)}")

# ── 4. 信号对比 ──
print("\n" + "=" * 60)
print("信号对比:")
print(f"  trade_date:  normal={r_normal.trade_date}  fast={r_fast.trade_date}  "
      f"{'✓' if r_normal.trade_date == r_fast.trade_date else '✗'}")
print(f"  今日信号数:  normal={len(r_normal.today_signals)}  fast={len(r_fast.today_signals)}  "
      f"{'✓' if len(r_normal.today_signals) == len(r_fast.today_signals) else '✗'}")
print(f"  今日确认数:  normal={len(r_normal.today_confirmed)}  fast={len(r_fast.today_confirmed)}  "
      f"{'✓' if len(r_normal.today_confirmed) == len(r_fast.today_confirmed) else '✗'}")

normal_sig = {(i.code, i.signal_date) for i in r_normal.today_signals}
fast_sig = {(i.code, i.signal_date) for i in r_fast.today_signals}
normal_conf = {(i.code, i.confirm_date) for i in r_normal.today_confirmed}
fast_conf = {(i.code, i.confirm_date) for i in r_fast.today_confirmed}

for label, s1, s2, n1, n2 in [
    ("信号", normal_sig, fast_sig, "普通", "fast"),
    ("确认", normal_conf, fast_conf, "普通", "fast"),
]:
    only1, only2 = s1 - s2, s2 - s1
    if only1:
        print(f"\n  仅{n1}有的{label} ({len(only1)}):")
        for code, d in sorted(only1):
            print(f"    {code} @ {d}")
    if only2:
        print(f"\n  仅{n2}有的{label} ({len(only2)}):")
        for code, d in sorted(only2):
            print(f"    {code} @ {d}")

if normal_sig == fast_sig and normal_conf == fast_conf:
    print("\n  ✓ 信号列表完全一致")

# ── 5. 数值字段逐只对比 ──
print("\n" + "-" * 60)
print("数值字段逐只对比（信号）:")
nm = {i.code: i for i in r_normal.today_signals}
fm = {i.code: i for i in r_fast.today_signals}
diffs = 0
for code in sorted(set(nm) & set(fm)):
    n, f = nm[code], fm[code]
    for fld in ['close', 'change_pct', 'avg_cost', 'discount_pct',
                'chip_conc70_pct', 'profit_ratio_pct', 'body_pct',
                'upper_pct', 'lower_pct', 'dist_low_pct', 'pre5_pct']:
        nv, fv = getattr(n, fld), getattr(f, fld)
        if abs(nv - fv) > 0.01:
            print(f"  {code} {fld}: normal={nv} fast={fv}")
            diffs += 1
if diffs == 0:
    print("  ✓ 所有数值字段完全一致")
else:
    print(f"  共 {diffs} 处差异")
