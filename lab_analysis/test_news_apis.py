# -*- encoding: UTF-8 -*-
"""测试 AkShare 新闻相关接口可用性。

重点测试：
1) stock_news_em
2) stock_research_report_em
3) stock_news_main_cx
4) stock_js_weibo_report
5) index_news_sentiment_scope

用法:
    /opt/anaconda3/envs/stock/bin/python lab_analysis/test_news_apis.py
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, Dict, Any

import akshare as ak
import pandas as pd


def _print_df_brief(name: str, df: pd.DataFrame, elapsed: float) -> None:
    """打印 DataFrame 简要信息。"""
    print(f"\n{'=' * 90}")
    print(f"[{name}] 成功, 耗时: {elapsed:.2f}s")
    print(f"行数: {len(df)}, 列数: {len(df.columns)}")
    print(f"字段: {list(df.columns)}")
    if not df.empty:
        print("\n样例(前3行):")
        print(df.head(3).to_string(index=False))
    else:
        print("返回为空 DataFrame")


def _run_case(name: str, fn: Callable[[], pd.DataFrame]) -> Dict[str, Any]:
    """执行单个测试用例，返回结构化结果。"""
    start = time.time()
    try:
        df = fn()
        elapsed = time.time() - start
        if df is None:
            print(f"\n[{name}] 返回 None, 耗时: {elapsed:.2f}s")
            return {"name": name, "ok": False, "elapsed": elapsed, "rows": 0, "error": "return None"}
        _print_df_brief(name, df, elapsed)
        return {"name": name, "ok": True, "elapsed": elapsed, "rows": len(df), "error": ""}
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - start
        print(f"\n{'=' * 90}")
        print(f"[{name}] 失败, 耗时: {elapsed:.2f}s")
        print(f"错误: {exc}")
        return {"name": name, "ok": False, "elapsed": elapsed, "rows": 0, "error": str(exc)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测试 AkShare 新闻相关接口")
    parser.add_argument("--symbol", default="601789", help="个股代码，默认 601789")
    parser.add_argument(
        "--period",
        default="CNHOUR12",
        choices=["CNHOUR2", "CNHOUR6", "CNHOUR12", "CNHOUR24", "CNDAY7", "CNDAY30"],
        help="微博舆情周期，默认 CNHOUR12",
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="跳过 index_news_sentiment_scope（该接口偶发不可用）",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 50)

    print("开始测试 AkShare 新闻相关接口...")
    print("说明: 某些接口受源站限制，失败不代表代码错误。")
    print(f"测试参数: symbol={args.symbol}, period={args.period}, skip_sentiment={args.skip_sentiment}")

    stock_code = str(args.symbol).zfill(6)

    test_cases = [
        ("stock_news_em", lambda: ak.stock_news_em(symbol=stock_code)),
        ("stock_research_report_em", lambda: ak.stock_research_report_em(symbol=stock_code)),
        ("stock_news_main_cx", ak.stock_news_main_cx),
        ("stock_js_weibo_report", lambda: ak.stock_js_weibo_report(time_period=args.period)),
    ]
    if not args.skip_sentiment:
        test_cases.append(("index_news_sentiment_scope", ak.index_news_sentiment_scope))

    results = []
    for name, fn in test_cases:
        results.append(_run_case(name, fn))

    print(f"\n{'#' * 90}")
    print("测试汇总")
    print(f"{'#' * 90}")
    ok_count = sum(1 for r in results if r["ok"])
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(
            f"{status:4} | {r['name']:<28} | "
            f"rows={r['rows']:<6} | {r['elapsed']:.2f}s"
        )
        if r["error"]:
            print(f"      error: {r['error']}")

    print(f"\n通过: {ok_count}/{len(results)}")


if __name__ == "__main__":
    main()

