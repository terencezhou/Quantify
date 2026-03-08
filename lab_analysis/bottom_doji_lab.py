# -*- encoding: UTF-8 -*-

"""底部十字星实验脚本。

只依赖本地 daily 缓存数据，不依赖 main_new.py 或实时网络刷新。

功能：
1. 扫描最近 N 个交易日出现的底部十字星
2. 输出失败样本（十字星后未明显上涨）
3. 输出确认样本（十字星后出现放量/阳线确认）

用法：
    python lab_analysis/bottom_doji_lab.py
    python lab_analysis/bottom_doji_lab.py --mode failed --limit 20
    python lab_analysis/bottom_doji_lab.py --mode confirmed --days 20
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml


@dataclass
class DojiSignal:
    code: str
    name: str
    signal_date: str
    signal_close: float
    body_pct: float
    upper_pct: float
    lower_pct: float
    dist_low_pct: float
    pre5_pct: float
    days_after: int
    after_pct: float
    max_after_pct: float
    min_after_pct: float
    confirm_date: str = ""
    confirm_pct: float = 0.0
    confirm_vol_ratio: float = 0.0
    avg_cost: float = 0.0
    conc70_pct: float = 0.0
    discount_pct: float = 0.0


def load_config(config_path: str) -> dict:
    """读取 yaml 配置；不存在时返回空配置。"""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def resolve_daily_dir(base_dir: str, config_path: str) -> str:
    """解析 daily 缓存目录。"""
    cfg = load_config(config_path)
    data_dir = cfg.get("data_dir", "stock_data")
    return os.path.join(base_dir, data_dir, "daily")


def resolve_chips_dir(base_dir: str, config_path: str) -> str:
    """解析 chips 缓存目录。"""
    cfg = load_config(config_path)
    data_dir = cfg.get("data_dir", "stock_data")
    return os.path.join(base_dir, data_dir, "chips")


def iter_daily_files(daily_dir: str) -> Iterable[str]:
    """遍历 daily parquet 文件。"""
    if not os.path.isdir(daily_dir):
        return []
    for name in sorted(os.listdir(daily_dir)):
        if name.endswith(".parquet"):
            yield os.path.join(daily_dir, name)


def load_daily_frame(path: str) -> Optional[pd.DataFrame]:
    """读取单只股票日K。"""
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    need = ["日期", "开盘", "收盘", "最高", "最低", "涨跌幅"]
    if not all(c in df.columns for c in need):
        return None
    df = df.copy()
    df["日期"] = df["日期"].astype(str)
    df = df.sort_values("日期").reset_index(drop=True)
    for col in need[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "成交量" in df.columns:
        df["成交量"] = pd.to_numeric(df["成交量"], errors="coerce")
    return df


def load_chips_map(chips_dir: str, code: str) -> dict[str, tuple[float, float]]:
    """读取单只股票筹码映射: 日期 -> (平均成本, 70集中度)。"""
    path = os.path.join(chips_dir, f"{code}.parquet")
    if not os.path.exists(path):
        path = os.path.join(chips_dir, f"{code.lstrip('0')}.parquet")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception:
        return {}
    if df is None or df.empty:
        return {}
    need = ["日期", "平均成本", "70集中度"]
    if not all(c in df.columns for c in need):
        return {}
    df = df.copy()
    df["日期"] = df["日期"].astype(str)
    df["平均成本"] = pd.to_numeric(df["平均成本"], errors="coerce")
    df["70集中度"] = pd.to_numeric(df["70集中度"], errors="coerce")
    return {
        str(row["日期"])[:10]: (float(row["平均成本"]), float(row["70集中度"]))
        for _, row in df.iterrows()
        if pd.notna(row["平均成本"]) and pd.notna(row["70集中度"])
    }


def get_stock_meta(path: str, df: pd.DataFrame) -> tuple[str, str]:
    """返回 code/name。"""
    code = os.path.basename(path).replace(".parquet", "").zfill(6)
    name = str(df["名称"].iloc[-1]) if "名称" in df.columns else code
    return code, name


def is_excluded(code: str, name: str) -> bool:
    """过滤 ST / 北交所。"""
    return ("ST" in name) or code.startswith(("8", "4", "92"))


def get_market_latest_date(daily_dir: str) -> str:
    """从缓存中找出全市场最新交易日。"""
    latest = ""
    for path in iter_daily_files(daily_dir):
        df = load_daily_frame(path)
        if df is None or df.empty:
            continue
        cur = str(df.iloc[-1]["日期"])[:10]
        if cur > latest:
            latest = cur
    return latest


def detect_bottom_doji(df: pd.DataFrame, pos: int) -> Optional[dict]:
    """判断某日是否为底部十字星。"""
    if pos < 5:
        return None

    open_p = df.iloc[pos]["开盘"]
    close_p = df.iloc[pos]["收盘"]
    high_p = df.iloc[pos]["最高"]
    low_p = df.iloc[pos]["最低"]

    if pd.isna(open_p) or pd.isna(close_p) or pd.isna(high_p) or pd.isna(low_p):
        return None

    amplitude = high_p - low_p
    if amplitude <= 0:
        return None

    body_pct = abs(close_p - open_p) / amplitude * 100
    upper_pct = (high_p - max(open_p, close_p)) / amplitude * 100
    lower_pct = (min(open_p, close_p) - low_p) / amplitude * 100

    # 十字星定义：实体小，上下影都明显
    if body_pct >= 15:
        return None
    if upper_pct < 15 or lower_pct < 15:
        return None

    low20 = df["最低"].iloc[max(0, pos - 19):pos + 1].min()
    if pd.isna(low20) or low20 <= 0:
        return None
    dist_low_pct = (low_p - low20) / low20 * 100
    if dist_low_pct > 5:
        return None

    prev_close_5 = df.iloc[pos - 5]["收盘"]
    prev_close_1 = df.iloc[pos - 1]["收盘"]
    if pd.isna(prev_close_5) or pd.isna(prev_close_1) or prev_close_5 <= 0:
        return None
    pre5_pct = (prev_close_1 - prev_close_5) / prev_close_5 * 100
    if pre5_pct > -3:
        return None

    return {
        "body_pct": round(body_pct, 1),
        "upper_pct": round(upper_pct, 1),
        "lower_pct": round(lower_pct, 1),
        "dist_low_pct": round(dist_low_pct, 1),
        "pre5_pct": round(pre5_pct, 1),
    }


def detect_confirmation(df: pd.DataFrame, pos: int) -> Optional[dict]:
    """检查十字星后 1~3 日是否出现确认。"""
    if pos >= len(df) - 1:
        return None

    base_close = df.iloc[pos]["收盘"]
    doji_high = df.iloc[pos]["最高"]
    if pd.isna(base_close) or base_close <= 0:
        return None

    # 更强确认：次日开盘必须直接站上十字星上影线之上
    next_open = df.iloc[pos + 1]["开盘"]
    if pd.isna(doji_high) or pd.isna(next_open) or next_open <= doji_high:
        return None

    base_vol = None
    if "成交量" in df.columns:
        base_vol = df.iloc[pos]["成交量"]

    for i in range(pos + 1, min(pos + 4, len(df))):
        row = df.iloc[i]
        open_p = row["开盘"]
        close_p = row["收盘"]
        if pd.isna(open_p) or pd.isna(close_p):
            continue

        confirm_pct = (close_p - base_close) / base_close * 100
        is_yang = close_p > open_p

        vol_ratio = 0.0
        if base_vol is not None and not pd.isna(base_vol) and base_vol > 0:
            vol_ratio = float(row.get("成交量", 0) / base_vol)

        # 确认标准：1~3日内收盘涨超3%，且阳线；量能放大更优
        if confirm_pct >= 3 and is_yang:
            return {
                "confirm_date": str(row["日期"])[:10],
                "confirm_pct": round(confirm_pct, 1),
                "confirm_vol_ratio": round(vol_ratio, 2),
            }
    return None


def build_signal(
    code: str,
    name: str,
    df: pd.DataFrame,
    pos: int,
    raw: dict,
    chip_raw: Optional[dict] = None,
) -> DojiSignal:
    """组装信号对象。"""
    last_pos = len(df) - 1
    signal_close = float(df.iloc[pos]["收盘"])
    last_close = float(df.iloc[last_pos]["收盘"])
    after_pct = (last_close - signal_close) / signal_close * 100 if signal_close > 0 else 0.0

    highs = df["最高"].iloc[pos + 1:last_pos + 1]
    lows = df["最低"].iloc[pos + 1:last_pos + 1]
    max_after = (highs.max() - signal_close) / signal_close * 100 if not highs.empty else 0.0
    min_after = (lows.min() - signal_close) / signal_close * 100 if not lows.empty else 0.0

    signal = DojiSignal(
        code=code,
        name=name,
        signal_date=str(df.iloc[pos]["日期"])[:10],
        signal_close=round(signal_close, 2),
        body_pct=raw["body_pct"],
        upper_pct=raw["upper_pct"],
        lower_pct=raw["lower_pct"],
        dist_low_pct=raw["dist_low_pct"],
        pre5_pct=raw["pre5_pct"],
        days_after=last_pos - pos,
        after_pct=round(after_pct, 1),
        max_after_pct=round(max_after, 1),
        min_after_pct=round(min_after, 1),
    )

    confirm = detect_confirmation(df, pos)
    if confirm:
        signal.confirm_date = confirm["confirm_date"]
        signal.confirm_pct = confirm["confirm_pct"]
        signal.confirm_vol_ratio = confirm["confirm_vol_ratio"]
    if chip_raw:
        signal.avg_cost = chip_raw["avg_cost"]
        signal.conc70_pct = chip_raw["conc70_pct"]
        signal.discount_pct = chip_raw["discount_pct"]
    return signal


def scan_signals(
    daily_dir: str,
    days: int,
    start_date: str = "",
    chips_dir: str = "",
    chip_below_avg: bool = False,
    chip_conc_max: float = 0.0,
) -> List[DojiSignal]:
    """扫描最近 N 个交易日的底部十字星。"""
    results: List[DojiSignal] = []
    market_latest = get_market_latest_date(daily_dir)
    for path in iter_daily_files(daily_dir):
        df = load_daily_frame(path)
        if df is None or len(df) < 40:
            continue
        code, name = get_stock_meta(path, df)
        if is_excluded(code, name):
            continue
        last_date = str(df.iloc[-1]["日期"])[:10]
        if market_latest and last_date != market_latest:
            continue

        chips_map = load_chips_map(chips_dir, code) if chips_dir else {}

        last_pos = len(df) - 1
        start = max(20, last_pos - days + 1)
        for pos in range(start, last_pos + 1):
            signal_date = str(df.iloc[pos]["日期"])[:10]
            if start_date and signal_date < start_date:
                continue
            raw = detect_bottom_doji(df, pos)
            if raw is None:
                continue

            chip_raw = None
            if chips_dir:
                chip_val = chips_map.get(signal_date)
                if chip_val is None:
                    continue
                avg_cost, conc70 = chip_val
                close_p = float(df.iloc[pos]["收盘"])
                discount_pct = (close_p - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0.0
                chip_raw = {
                    "avg_cost": round(avg_cost, 2),
                    "conc70_pct": round(conc70 * 100, 2),
                    "discount_pct": round(discount_pct, 1),
                }
                if chip_below_avg and close_p >= avg_cost:
                    continue
                if chip_conc_max > 0 and conc70 >= chip_conc_max:
                    continue

            results.append(build_signal(code, name, df, pos, raw, chip_raw))
    return results


def pick_latest_per_stock(signals: List[DojiSignal]) -> List[DojiSignal]:
    """每只股票只保留最近一次信号。"""
    out = {}
    for s in sorted(signals, key=lambda x: x.signal_date, reverse=True):
        out.setdefault(s.code, s)
    return list(out.values())


def filter_failed(signals: List[DojiSignal]) -> List[DojiSignal]:
    """失败样本：后续没明显涨起来。"""
    out = []
    for s in signals:
        if s.days_after < 3:
            continue
        if s.after_pct > 2:
            continue
        if s.max_after_pct > 6:
            continue
        out.append(s)
    out.sort(key=lambda x: (x.after_pct, x.max_after_pct, x.dist_low_pct))
    return out


def filter_confirmed(signals: List[DojiSignal]) -> List[DojiSignal]:
    """确认样本：1~3日内出现有效确认。"""
    out = [s for s in signals if s.confirm_date]
    out.sort(key=lambda x: (x.confirm_pct, x.confirm_vol_ratio), reverse=True)
    return out


def sort_raw(signals: List[DojiSignal]) -> List[DojiSignal]:
    """原始信号排序：按日期倒序，再按筹码集中和价格偏离排序。"""
    out = list(signals)
    out.sort(key=lambda x: (x.signal_date, -x.discount_pct, -x.conc70_pct), reverse=True)
    return out


def print_table(title: str, signals: List[DojiSignal], limit: int) -> None:
    """打印结果表。"""
    print(f"\n## {title} ({min(limit, len(signals))}/{len(signals)})\n")
    if not signals:
        print("无结果\n")
        return

    show_chip = any(s.avg_cost > 0 for s in signals)
    header = (
        f"{'代码':<8} {'名称':<10} {'信号日':<12} {'收盘':>6} "
        + (f"{'均价':>6} {'偏离':>7} {'70集中':>8} " if show_chip else "")
        + f"{'前5日':>6} {'距底':>5} {'至今':>6} {'最大反弹':>8} {'确认日':<12} {'确认涨幅':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in signals[:limit]:
        print(
            f"{s.code:<8} {s.name[:10]:<10} {s.signal_date:<12} {s.signal_close:>6.2f} "
            + (
                f"{s.avg_cost:>6.2f} {s.discount_pct:>+6.1f}% {s.conc70_pct:>7.2f}% "
                if show_chip else ""
            )
            + f"{s.pre5_pct:>+5.1f}% {s.dist_low_pct:>4.1f}% {s.after_pct:>+5.1f}% "
            f"{s.max_after_pct:>+7.1f}% {s.confirm_date or '-':<12} "
            f"{(f'{s.confirm_pct:+.1f}%') if s.confirm_date else '-':>8}"
        )
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="底部十字星实验脚本（只读缓存）")
    parser.add_argument(
        "--mode",
        choices=["all", "raw", "failed", "confirmed"],
        default="all",
        help="输出模式",
    )
    parser.add_argument("--days", type=int, default=20, help="扫描最近多少个交易日")
    parser.add_argument("--limit", type=int, default=20, help="输出条数")
    parser.add_argument("--start-date", default="", help="只保留该日期及之后的信号，如 2026-02-23")
    parser.add_argument("--chip-below-avg", action="store_true", help="要求信号日收盘低于筹码平均成本")
    parser.add_argument("--chip-conc-max", type=float, default=0.0, help="70集中度上限，按小数传入，如 0.04 表示 4%")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "stock_config.yaml"),
        help="配置文件路径（只读取 data_dir）",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.abspath(args.config)
    daily_dir = resolve_daily_dir(base_dir, config_path)
    chips_dir = resolve_chips_dir(base_dir, config_path)

    print(f"使用缓存目录: {daily_dir}")
    signals = scan_signals(
        daily_dir,
        args.days,
        start_date=args.start_date,
        chips_dir=chips_dir if (args.chip_below_avg or args.chip_conc_max > 0) else "",
        chip_below_avg=args.chip_below_avg,
        chip_conc_max=args.chip_conc_max,
    )
    latest = pick_latest_per_stock(signals)

    print(f"最近 {args.days} 个交易日底部十字星: {len(signals)} 个信号, 去重后 {len(latest)} 只股票")

    if args.mode in ("all", "raw"):
        raw_signals = sort_raw(signals)
        print_table("原始命中信号", raw_signals, args.limit)

    if args.mode in ("all", "failed"):
        failed = filter_failed(latest)
        print_table("失败样本", failed, args.limit)

    if args.mode in ("all", "confirmed"):
        confirmed = filter_confirmed(latest)
        print_table("确认样本", confirmed, args.limit)


if __name__ == "__main__":
    main()
