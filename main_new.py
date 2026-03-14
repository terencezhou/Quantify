# -*- encoding: UTF-8 -*-

"""Sequoia 选股系统 — 新版入口

用法：
    # 选股预测（默认模式）
    python main_new.py buy

    # 持仓卖出信号分析
    python main_new.py sell

    # 缩量主升浪选股
    python main_new.py trend

    # 底部十字星低估筹码流程
    python main_new.py doji

    # 策略回测（暂未实现）
    python main_new.py backtest --strategy 放量上涨 --start 2024-01-01

    # 仅刷新数据，不运行任何分析
    python main_new.py refresh

    # 查看缓存状态
    python main_new.py status

    # 多头回踩MA10缩量阴线（标准模式，使用日K数据）
    python main_new.py pullback

    # 多头回踩MA10缩量阴线（快速模式，盘中/盘后用实时行情）
    python main_new.py pullback -f

    # 均线回归首日扫描（默认最新交易日）
    python main_new.py ma_align

    # 均线回归检测指定日期
    python main_new.py ma_align --date 2025-03-10

    # 均线回归检测单只股票
    python main_new.py ma_align --date 2025-03-10 --code 600519

    # 定时模式（每日15:15自动运行 buy）
    python main_new.py buy --cron

数据加载模式（全局参数，所有子命令通用）：
    # 默认：全量刷新（从远程拉取最新数据）
    python main_new.py doji

    # 快速模式：仅拉取实时行情，其余走本地缓存（适合盘中快速扫描）
    python main_new.py doji --fast

    # 纯缓存模式：不发起任何网络请求（适合二次分析 / 离线调试）
    python main_new.py doji --no-refresh
"""

import argparse
import logging
import sys
import os
import yaml
import time
import schedule
from datetime import datetime
from pathlib import Path

from data_manager import DataManager


# ══════════════════════════════════════════════════════════
#  配置加载
# ══════════════════════════════════════════════════════════

def load_config(config_path: str = None) -> dict:
    """加载配置文件，返回配置字典"""
    if config_path is None:
        root = Path(__file__).parent
        config_path = str(root / 'stock_config.yaml')

    if not os.path.exists(config_path):
        logging.error("配置文件不存在: %s", config_path)
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config or {}


# ══════════════════════════════════════════════════════════
#  DataManager 初始化（统一处理刷新模式）
# ══════════════════════════════════════════════════════════

def init_dm(config: dict, args) -> DataManager | None:
    """根据 --fast / --no-refresh 选择合适的数据加载模式。

    返回初始化好的 DataManager；加载失败返回 None。
    """
    dm = DataManager(config)
    dm.report_cache_status()

    no_refresh = getattr(args, 'no_refresh', False)
    fast = getattr(args, 'fast', False)

    if no_refresh:
        ok = dm.load_from_cache()
    elif fast:
        ok = dm.refresh_fast()
    else:
        ok = dm.refresh()

    if not ok:
        logging.error("数据加载失败，终止")
        return None
    return dm


# ══════════════════════════════════════════════════════════
#  日志初始化
# ══════════════════════════════════════════════════════════

def setup_logging(verbose: bool = False):
    """统一日志配置：文件 + 控制台"""
    level = logging.DEBUG if verbose else logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    # 文件输出
    fh = logging.FileHandler('sequoia.log', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    # 控制台输出
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)


# ══════════════════════════════════════════════════════════
#  子命令: refresh
# ══════════════════════════════════════════════════════════

def cmd_refresh(config: dict, args):
    """仅刷新数据，不运行分析（始终走全量刷新）"""
    dm = DataManager(config)
    dm.report_cache_status()
    logging.info("开始数据刷新...")
    ok = dm.refresh()
    dm.report_cache_status()
    if ok:
        logging.info("数据刷新完成: %d 只股票", len(dm.stocks_data))
    else:
        logging.error("数据刷新失败")
    return ok


# ══════════════════════════════════════════════════════════
#  子命令: status
# ══════════════════════════════════════════════════════════

def cmd_status(config: dict, args):
    """查看缓存状态（含有效/未更新分类 + 最新数据时间）"""
    from data_cache import DataCache

    cache = DataCache(config.get('data_dir', 'data'))
    print("正在统计缓存状态，请稍候...")
    s = cache.get_detailed_stats()

    W = 62

    def _bar(label, total, valid, stale, latest):
        stale_str = f"  未更新{stale:4d}只" if stale else ""
        latest_str = f"  最新 {latest}" if latest else ""
        print(f"  {label:<10}  {total:4d}只  有效{valid:4d}只{stale_str}{latest_str}")

    def _snap(label, info):
        if not info['exists']:
            print(f"  {label:<10}  — 无缓存")
            return
        tag = "✓ 新鲜" if info['fresh'] else "✗ 过期"
        latest_str = f"  最新 {info['latest']}" if info['latest'] else ""
        print(f"  {label:<10}  {tag}  {info['rows']:6d}条{latest_str}")

    print("=" * W)
    print(f"  Sequoia 本地缓存状态   (最近交易日: {s['last_trade_day']})")
    print("=" * W)
    print("  ── 个股数据 ─────────────────────────────────────────")
    d = s['daily']
    _bar("日K线",   d['total'], d['valid'], d['stale'], d['latest'])
    d = s['fund_flow']
    _bar("个股资金流", d['total'], d['valid'], d['stale'], d['latest'])
    d = s['intraday']
    _bar("分时K线",  d['total'], d['valid'], d['stale'], d['latest'])
    d = s['chips']
    _bar("筹码分布",  d['total'], d['valid'], d['stale'], d['latest'])
    d = s['hsgt_hold']
    _bar("北向持股",  d['total'], d['valid'], d['stale'], d['latest'])
    d = s['stock_info']
    _bar("个股基本信息", d['total'], d['valid'], d['stale'], d['latest'])
    print("  ── 全市场快照 ───────────────────────────────────────")
    snaps = s['snapshots']
    _snap("涨停池",    snaps['zt_pool'])
    _snap("强势股池",  snaps['zt_pool_strong'])
    _snap("龙虎榜",    snaps['lhb_detail'])
    _snap("行业资金流", snaps['sector_flow'])
    _snap("大单追踪",  snaps['big_deal'])
    _snap("北向资金",  snaps['hsgt_flow'])
    _snap("概念板块",  snaps['concept_board'])
    _snap("业绩报表",  snaps['financial_report'])
    _snap("资产负债表", snaps['financial_balance'])
    print("=" * W)


# ══════════════════════════════════════════════════════════
#  子命令: buy
# ══════════════════════════════════════════════════════════

def cmd_buy(config: dict, args):
    """每日复盘：刷新数据 → 市场情绪 + 行业热度 + 龙头识别 → 推送报告"""
    logging.info("=" * 60)
    logging.info("Zhoumi 每日复盘 启动  %s", datetime.now().strftime('%Y-%m-%d %H:%M'))
    logging.info("=" * 60)

    dm = init_dm(config, args)
    if dm is None:
        return

    # Step 2: 初始化推送
    import push
    push.init(config)

    # Step 3: 运行三大报告（market_temperature 内部串联 industry + leader）
    from report.market_temperature import MarketTemperature
    mt = MarketTemperature(dm)
    result = mt.run()
    report_md = mt.to_markdown(result)

    # Step 4: 推送
    push_cfg = config.get('push', {})
    if push_cfg.get('enable', False):
        try:
            push.markdown(report_md)
        except Exception as e:
            logging.warning("推送失败: %s", e)
    else:
        print(report_md)

    logging.info("复盘完成: 情绪 %.1f 分（%s），龙头 %d 只",
                 result.score, result.phase, len(result.leader_stocks))
    logging.info("=" * 60)

# ══════════════════════════════════════════════════════════
#  子命令: sell
# ══════════════════════════════════════════════════════════

def cmd_sell(config: dict, args):
    """持仓分析：全量刷新 → 三大报告 → 卖出决策 → 推送"""
    holdings_raw = config.get('holdings') or []
    if not holdings_raw:
        logging.warning("stock_config.yaml 中未配置 holdings，无法分析卖出信号")
        print("\n请在 stock_config.yaml 中添加持仓配置，格式：")
        print("  holdings:")
        print('    - code: "000001"')
        print('      name: "平安银行"')
        print("      buy_price: 10.50")
        print('      buy_date: "2025-01-15"')
        print("      shares: 1000")
        return

    logging.info("=" * 60)
    logging.info("Zhoumi 持仓卖出决策分析 启动  %s", datetime.now().strftime('%Y-%m-%d %H:%M'))
    logging.info("=" * 60)

    dm = init_dm(config, args)
    if dm is None:
        return

    # Step 2: 运行三大上游报告（市场情绪 → 行业热度 → 龙头识别）
    from report.market_temperature import MarketTemperature
    mt = MarketTemperature(dm)
    temp_result = mt.run()

    logging.info("市场情绪: %.1f 分（%s），R_yu=%.2f%%",
                 temp_result.score, temp_result.phase, temp_result.metrics.R_yu)

    # Step 3: 运行卖出决策分析
    from report.sell_point import SellPointAnalyzer
    sp = SellPointAnalyzer(dm, temp_result)
    report = sp.run()

    # Step 4: 输出
    if report.overview_md:
        print(report.overview_md)

    for sa in report.analyses:
        logging.info("%s(%s) 盈亏%+.1f%% → %s（%.0f分）",
                     sa.holding.name, sa.holding.code,
                     sa.pnl_pct, sa.overall_signal.value, sa.overall_score)

    # Step 5: 推送
    import push
    push.init(config)
    push_cfg = config.get('push', {})
    if report.overview_md and push_cfg.get('enable', False):
        try:
            push.markdown(report.overview_md)
        except Exception as e:
            logging.warning("推送失败: %s", e)

    logging.info("持仓分析完成: %d 只持仓", len(report.analyses))
    logging.info("=" * 60)


# ══════════════════════════════════════════════════════════
#  子命令: trend（缩量主升浪选股）
# ══════════════════════════════════════════════════════════

def cmd_trend(config: dict, args):
    """缩量主升浪 + 趋势持股：扫描全市场，输出观察池与买入信号"""
    logging.info("=" * 60)
    logging.info("缩量主升浪选股 启动  %s", datetime.now().strftime('%Y-%m-%d %H:%M'))
    logging.info("=" * 60)

    dm = init_dm(config, args)
    if dm is None:
        return

    # 获取市场温度作为环境加分（可选）
    market_score = 50.0
    try:
        from report.market_temperature import MarketTemperature
        mt = MarketTemperature(dm)
        mt_result = mt.run()
        market_score = mt_result.score
        logging.info("市场温度: %.1f (%s)", mt_result.score, mt_result.phase)
    except Exception as e:
        logging.warning("市场温度获取失败，使用默认值: %s", e)

    from report.trend_surge import TrendSurgeScreener
    screener = TrendSurgeScreener(dm)
    result = screener.run(market_score=market_score)
    report_md = screener.to_markdown(result)

    import push
    push.init(config)
    push_cfg = config.get('push', {})
    if push_cfg.get('enable', False):
        try:
            push.markdown(report_md)
        except Exception as e:
            logging.warning("推送失败: %s", e)
    else:
        print(report_md)

    logging.info(
        "选股完成: 观察池 %d 只, 买入信号 %d 只",
        len(result.watch_pool), len(result.buy_candidates),
    )
    logging.info("=" * 60)


# ══════════════════════════════════════════════════════════
#  子命令: doji（底部十字星低估筹码流程）
# ══════════════════════════════════════════════════════════

def cmd_doji(config: dict, args):
    """底部十字星 + 低估筹码流程：扫描全市场，输出今日新信号与确认信号"""
    logging.info("=" * 60)
    logging.info("底部十字星低估筹码流程 启动  %s", datetime.now().strftime('%Y-%m-%d %H:%M'))
    logging.info("=" * 60)

    dm = init_dm(config, args)
    if dm is None:
        return

    from report.bottom_doji_flow import BottomDojiFlow
    use_fast = getattr(args, 'fast', False)
    flow = BottomDojiFlow(dm, fast_mode=use_fast)
    result = flow.run()
    report_md = flow.to_markdown(result)

    import push
    push.init(config)
    push_cfg = config.get('push', {})
    if push_cfg.get('enable', False):
        try:
            push.markdown(report_md)
        except Exception as e:
            logging.warning("推送失败: %s", e)
            print(report_md)
    else:
        print(report_md)

    logging.info(
        "流程完成: 今日新信号 %d 只, 今日确认 %d 只",
        len(result.today_signals), len(result.today_confirmed),
    )
    logging.info("=" * 60)


# ══════════════════════════════════════════════════════════
#  子命令: doji2（箱体跌幅 + 底部十字星，宽松版）
# ══════════════════════════════════════════════════════════

def cmd_doji2(config: dict, args):
    """箱体十字星流程：箱体跌幅≥20% + 7日内十字星 + 筹码低估 + 涨幅确认"""
    logging.info("=" * 60)
    logging.info("箱体十字星流程 启动  %s", datetime.now().strftime('%Y-%m-%d %H:%M'))
    logging.info("=" * 60)

    dm = init_dm(config, args)
    if dm is None:
        return

    from report.box_doji_flow import BoxDojiFlow
    use_fast = getattr(args, 'fast', False)
    flow = BoxDojiFlow(dm, fast_mode=use_fast)
    result = flow.run()
    report_md = flow.to_markdown(result)

    import push
    push.init(config)
    push_cfg = config.get('push', {})
    if push_cfg.get('enable', False):
        try:
            push.markdown(report_md)
        except Exception as e:
            logging.warning("推送失败: %s", e)
            print(report_md)
    else:
        print(report_md)

    logging.info(
        "流程完成: 近期候选 %d 只, 今日确认 %d 只",
        len(result.recent_candidates), len(result.today_confirmed),
    )
    logging.info("=" * 60)


# ══════════════════════════════════════════════════════════
#  子命令: pullback（多头回踩MA10缩量阴线）
# ══════════════════════════════════════════════════════════

def cmd_pullback(config: dict, args):
    """多头排列回踩MA10缩量阴线：扫描全市场趋势回踩买入信号

    -f 模式：使用实时行情替代最新一日K线，适合盘中/盘后快速扫描。
    非交易日运行时自动以上一个交易日为目标。
    """
    use_fast = getattr(args, 'fast', False)
    mode_str = '⚡快速(实时)' if use_fast else '标准(K线)'

    logging.info("=" * 60)
    logging.info("多头回踩MA10缩量阴线 启动  %s  模式=%s",
                 datetime.now().strftime('%Y-%m-%d %H:%M'), mode_str)
    logging.info("=" * 60)

    dm = init_dm(config, args)
    if dm is None:
        return

    market_score = 50.0
    try:
        from report.market_temperature import MarketTemperature
        mt = MarketTemperature(dm)
        mt_result = mt.run()
        market_score = mt_result.score
        logging.info("市场温度: %.1f (%s)", mt_result.score, mt_result.phase)
    except Exception as e:
        logging.warning("市场温度获取失败，使用默认值: %s", e)

    from report.pullback_ma10 import PullbackMA10Screener, format_scan_result
    screener = PullbackMA10Screener(dm, fast_mode=use_fast)
    result = screener.run(market_score=market_score)
    report_md = format_scan_result(result)

    import push
    push.init(config)
    push_cfg = config.get('push', {})
    if push_cfg.get('enable', False):
        try:
            push.markdown(report_md)
        except Exception as e:
            logging.warning("推送失败: %s", e)
    else:
        print(report_md)

    logging.info(
        "扫描完成: 信号 %d 只 (S=%d A=%d B=%d)",
        len(result.items),
        len(result.s_items), len(result.a_items), len(result.b_items),
    )
    logging.info("=" * 60)


# ══════════════════════════════════════════════════════════
#  子命令: ma_align（均线回归首日扫描）
# ══════════════════════════════════════════════════════════

def cmd_ma_align(config: dict, args):
    """均线回归首日扫描：检测指定日期全市场（或单只）的均线回归信号"""
    logging.info("=" * 60)
    logging.info("均线回归首日扫描 启动  %s", datetime.now().strftime('%Y-%m-%d %H:%M'))
    logging.info("=" * 60)

    dm = init_dm(config, args)
    if dm is None:
        return

    from report.ma_alignment import (
        MAAlignmentDetector, parse_date,
        format_single_result, format_scan_result,
    )

    detector = MAAlignmentDetector(dm)

    target_date = getattr(args, 'date', None)
    if target_date:
        target_date = parse_date(target_date)
    else:
        target_date = detector.get_latest_trade_date()

    stock_code = getattr(args, 'code', None)

    if stock_code:
        ar = detector.check_by_code(stock_code, target_date)
        if ar is None:
            print(f"\n未找到 {stock_code} 的数据或数据不足，无法检测。")
            return
        report_text = format_single_result(ar)
        print(f"\n{report_text}")
        logging.info(
            "检测完成: %s(%s) 日期=%s 回归=%s 首日=%s 评分=%d",
            ar.name, ar.code, ar.target_date,
            ar.is_alignment, ar.is_first_day, ar.score,
        )
    else:
        result = detector.scan(target_date)
        report_md = format_scan_result(result)

        import push
        push.init(config)
        push_cfg = config.get('push', {})
        if push_cfg.get('enable', False):
            try:
                push.markdown(report_md)
            except Exception as e:
                logging.warning("推送失败: %s", e)
        else:
            print(report_md)

        logging.info(
            "扫描完成: 共 %d 只, 首日回归 %d 只",
            result.total_scanned, len(result.items),
        )

    logging.info("=" * 60)


# ══════════════════════════════════════════════════════════
#  子命令: backtest
# ══════════════════════════════════════════════════════════

def cmd_backtest(config: dict, args):
    """策略回测（预留接口）"""
    logging.info("=" * 60)
    logging.info("Sequoia 策略回测 启动")
    logging.info("=" * 60)

    dm = init_dm(config, args)
    if dm is None:
        return

    from test_strategy import StrategyBacktester
    bt = StrategyBacktester(dm)
    result = bt.run(
        strategy_name=args.strategy,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
    )
    print(result.summary())


# ══════════════════════════════════════════════════════════
#  定时执行包装
# ══════════════════════════════════════════════════════════

def run_with_cron(func, config, args, exec_time="15:15"):
    """每日定时执行"""
    import utils

    def job():
        if utils.is_weekday():
            try:
                func(config, args)
            except Exception as e:
                logging.error("定时任务异常: %s", e, exc_info=True)

    logging.info("定时模式启动: 每日 %s 执行", exec_time)
    schedule.every().day.at(exec_time).do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)


# ══════════════════════════════════════════════════════════
#  CLI 解析
# ══════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='sequoia',
        description='Sequoia 智能选股系统',
    )
    parser.add_argument('-c', '--config', default=None,
                        help='配置文件路径（默认: stock_config.yaml）')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='详细日志输出')

    sub = parser.add_subparsers(dest='command', help='子命令')

    def _add_refresh_flags(p):
        """为子命令添加 --fast / --no-refresh 互斥参数。"""
        g = p.add_mutually_exclusive_group()
        g.add_argument('-f', '--fast', action='store_true',
                       help='快速模式：仅拉取实时行情，其余走本地缓存')
        g.add_argument('-n', '--no-refresh', action='store_true',
                       help='纯缓存模式：不发起任何网络请求')

    # buy
    p_buy = sub.add_parser('buy', help='全市场选股预测')
    _add_refresh_flags(p_buy)
    p_buy.add_argument('--cron', action='store_true',
                       help='定时模式（每日15:15自动执行）')
    p_buy.add_argument('--time', default='15:15',
                       help='定时执行时间（默认 15:15）')

    # sell
    p_sell = sub.add_parser('sell', help='持仓卖出信号分析')
    _add_refresh_flags(p_sell)
    p_sell.add_argument('--cron', action='store_true',
                        help='定时模式')
    p_sell.add_argument('--time', default='15:15',
                        help='定时执行时间')

    # backtest
    p_bt = sub.add_parser('backtest', help='策略历史回测（开发中）')
    _add_refresh_flags(p_bt)
    p_bt.add_argument('--strategy', default='放量上涨',
                      help='策略名称')
    p_bt.add_argument('--start', default='2024-01-01',
                      help='回测起始日期')
    p_bt.add_argument('--end', default='',
                      help='回测截止日期（默认至今）')
    p_bt.add_argument('--capital', type=float, default=100000,
                      help='初始资金')

    # trend
    p_trend = sub.add_parser('trend', help='缩量主升浪选股')
    _add_refresh_flags(p_trend)
    p_trend.add_argument('--cron', action='store_true',
                         help='定时模式')
    p_trend.add_argument('--time', default='15:15',
                         help='定时执行时间')

    # doji
    p_doji = sub.add_parser('doji', help='底部十字星低估筹码流程（严格版）')
    _add_refresh_flags(p_doji)
    p_doji.add_argument('--cron', action='store_true',
                        help='定时模式')
    p_doji.add_argument('--time', default='15:15',
                        help='定时执行时间')

    # ma_align
    p_ma = sub.add_parser('ma_align', help='均线回归首日扫描（MA5>MA10>MA20>MA30多头排列）')
    _add_refresh_flags(p_ma)
    p_ma.add_argument('--date', '-d', default=None,
                       help='检测日期（默认最新交易日），支持 YYYY-MM-DD / YYYYMMDD / YYYY/MM/DD')
    p_ma.add_argument('--code', default=None,
                       help='指定股票代码（如 600519），不指定则全市场扫描')
    p_ma.add_argument('--cron', action='store_true',
                       help='定时模式')
    p_ma.add_argument('--time', default='15:15',
                       help='定时执行时间')

    # pullback
    p_pb = sub.add_parser('pullback', help='多头回踩MA10缩量阴线策略（趋势回踩买入）')
    _add_refresh_flags(p_pb)
    p_pb.add_argument('--code', default=None,
                      help='指定股票代码（如 601012），不指定则全市场扫描')
    p_pb.add_argument('--cron', action='store_true',
                      help='定时模式')
    p_pb.add_argument('--time', default='15:15',
                      help='定时执行时间')

    # doji2
    p_doji2 = sub.add_parser('doji2', help='箱体十字星流程（宽松版：箱体跌幅+7日窗口）')
    _add_refresh_flags(p_doji2)
    p_doji2.add_argument('--cron', action='store_true',
                         help='定时模式')
    p_doji2.add_argument('--time', default='15:15',
                         help='定时执行时间')

    # refresh（始终全量刷新，不需要 --fast / --no-refresh）
    sub.add_parser('refresh', help='仅刷新数据（不运行分析）')

    # status
    sub.add_parser('status', help='查看本地缓存状态')

    return parser


# ══════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    setup_logging(verbose=args.verbose)
    config = load_config(args.config)

    dispatch = {
        'buy':      cmd_buy,
        'sell':     cmd_sell,
        'trend':    cmd_trend,
        'doji':     cmd_doji,
        'doji2':    cmd_doji2,
        'pullback': cmd_pullback,
        'ma_align': cmd_ma_align,
        'backtest': cmd_backtest,
        'refresh':  cmd_refresh,
        'status':   cmd_status,
    }

    cmd_func = dispatch.get(args.command)
    if cmd_func is None:
        parser.print_help()
        sys.exit(1)

    # 定时模式
    use_cron = getattr(args, 'cron', False)
    if use_cron:
        exec_time = getattr(args, 'time', '15:15')
        run_with_cron(cmd_func, config, args, exec_time)
    else:
        try:
            cmd_func(config, args)
        except KeyboardInterrupt:
            logging.info("用户中断")
        except Exception as e:
            logging.error("执行异常: %s", e, exc_info=True)
            sys.exit(1)


if __name__ == '__main__':
    main()
