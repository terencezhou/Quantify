# -*- encoding: UTF-8 -*-

"""Sequoia 选股系统 — 新版入口

用法：
    # 选股预测（默认模式）
    python main_new.py buy

    # 持仓卖出信号分析
    python main_new.py sell

    # 策略回测（暂未实现）
    python main_new.py backtest --strategy 放量上涨 --start 2024-01-01

    # 仅刷新数据，不运行任何分析
    python main_new.py refresh

    # 查看缓存状态
    python main_new.py status

    # 定时模式（每日15:15自动运行 buy）
    python main_new.py buy --cron
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
    """仅刷新数据，不运行分析"""
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

    # Step 1: 数据刷新
    dm = DataManager(config)
    dm.report_cache_status()
    ok = dm.refresh()
    if not ok:
        logging.error("数据刷新失败，终止")
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

    # Step 1: 全量刷新（三大上游报告需要全市场数据）
    dm = DataManager(config)
    dm.report_cache_status()
    ok = dm.refresh()
    if not ok:
        logging.error("数据刷新失败，终止")
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
#  子命令: backtest
# ══════════════════════════════════════════════════════════

def cmd_backtest(config: dict, args):
    """策略回测（预留接口）"""
    logging.info("=" * 60)
    logging.info("Sequoia 策略回测 启动")
    logging.info("=" * 60)

    dm = DataManager(config)
    ok = dm.refresh()
    if not ok:
        logging.error("数据刷新失败，终止回测")
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

    # buy
    p_buy = sub.add_parser('buy', help='全市场选股预测')
    p_buy.add_argument('--cron', action='store_true',
                       help='定时模式（每日15:15自动执行）')
    p_buy.add_argument('--time', default='15:15',
                       help='定时执行时间（默认 15:15）')

    # sell
    p_sell = sub.add_parser('sell', help='持仓卖出信号分析')
    p_sell.add_argument('--cron', action='store_true',
                        help='定时模式')
    p_sell.add_argument('--time', default='15:15',
                        help='定时执行时间')

    # backtest
    p_bt = sub.add_parser('backtest', help='策略历史回测（开发中）')
    p_bt.add_argument('--strategy', default='放量上涨',
                      help='策略名称')
    p_bt.add_argument('--start', default='2024-01-01',
                      help='回测起始日期')
    p_bt.add_argument('--end', default='',
                      help='回测截止日期（默认至今）')
    p_bt.add_argument('--capital', type=float, default=100000,
                      help='初始资金')

    # refresh
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
