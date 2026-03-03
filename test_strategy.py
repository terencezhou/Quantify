# -*- encoding: UTF-8 -*-

"""策略回测框架 — 基于历史数据验证策略有效性

职责：
  1. 加载历史数据（指定日期范围）
  2. 在历史数据上模拟运行策略
  3. 计算回测指标（收益率、最大回撤、夏普比率等）
  4. 输出回测报告

使用方式：
    dm = DataManager(config)
    dm.refresh()
    bt = StrategyBacktester(dm)
    result = bt.run(
        strategy_name='放量上涨',
        start_date='2024-01-01',
        end_date='2025-01-01',
        initial_capital=100000,
    )
    print(result.summary())

注意：此模块为框架定义，具体实现留作后续迭代。
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import date

from data_manager import DataManager


@dataclass
class Trade:
    """单笔交易记录"""
    code: str
    name: str
    buy_date: str
    buy_price: float
    sell_date: str = ''
    sell_price: float = 0.0
    shares: int = 0
    pnl: float = 0.0           # 盈亏金额
    pnl_pct: float = 0.0       # 盈亏比例
    hold_days: int = 0
    strategy: str = ''


@dataclass
class BacktestMetrics:
    """回测统计指标"""
    total_return: float = 0.0       # 总收益率 %
    annual_return: float = 0.0      # 年化收益率 %
    max_drawdown: float = 0.0       # 最大回撤 %
    sharpe_ratio: float = 0.0       # 夏普比率
    win_rate: float = 0.0           # 胜率 %
    profit_factor: float = 0.0      # 盈亏比
    total_trades: int = 0           # 总交易次数
    avg_hold_days: float = 0.0      # 平均持仓天数
    avg_pnl_pct: float = 0.0        # 平均每笔收益 %


@dataclass
class BacktestResult:
    """完整回测结果"""
    strategy_name: str = ''
    start_date: str = ''
    end_date: str = ''
    initial_capital: float = 0.0
    final_capital: float = 0.0
    metrics: BacktestMetrics = field(default_factory=BacktestMetrics)
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)  # 每日净值

    def summary(self) -> str:
        """返回回测摘要文本"""
        m = self.metrics
        return (
            f"策略: {self.strategy_name}\n"
            f"区间: {self.start_date} ~ {self.end_date}\n"
            f"初始资金: {self.initial_capital:,.0f}\n"
            f"最终资金: {self.final_capital:,.0f}\n"
            f"总收益率: {m.total_return:+.2f}%\n"
            f"年化收益: {m.annual_return:+.2f}%\n"
            f"最大回撤: {m.max_drawdown:.2f}%\n"
            f"夏普比率: {m.sharpe_ratio:.2f}\n"
            f"胜率: {m.win_rate:.1f}%\n"
            f"盈亏比: {m.profit_factor:.2f}\n"
            f"交易次数: {m.total_trades}\n"
            f"平均持仓: {m.avg_hold_days:.1f}天\n"
        )


class StrategyBacktester:
    """策略回测引擎。

    在历史数据上模拟策略运行，验证策略有效性。
    """

    def __init__(self, dm: DataManager):
        self._dm = dm
        self._config = dm.config

    # ══════════════════════════════════════════════
    #  主流程
    # ══════════════════════════════════════════════

    def run(self,
            strategy_name: str,
            start_date: str = '2024-01-01',
            end_date: str = '',
            initial_capital: float = 100_000,
            max_positions: int = 10,
            stop_loss: float = -8.0,
            take_profit: float = 30.0,
            ) -> BacktestResult:
        """运行单策略回测。

        Args:
            strategy_name: 策略名称（对应 strategy/ 模块中的策略）
            start_date:    回测起始日期
            end_date:      回测截止日期（空字符串表示至今）
            initial_capital: 初始资金
            max_positions:   最大同时持仓数量
            stop_loss:       止损比例（如 -8.0 表示亏损8%止损）
            take_profit:     止盈比例（如 30.0 表示盈利30%止盈）

        Returns:
            BacktestResult 包含全部交易记录和统计指标
        """
        # TODO: 后续实现
        logging.warning("策略回测功能尚未实现，敬请期待")
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date or str(date.today()),
            initial_capital=initial_capital,
            final_capital=initial_capital,
        )

    def run_multi(self,
                  strategy_names: List[str],
                  **kwargs) -> Dict[str, BacktestResult]:
        """多策略对比回测"""
        # TODO: 后续实现
        results = {}
        for name in strategy_names:
            results[name] = self.run(strategy_name=name, **kwargs)
        return results

    # ══════════════════════════════════════════════
    #  内部方法（后续实现）
    # ══════════════════════════════════════════════

    def _get_strategy_func(self, name: str):
        """根据策略名获取对应的检查函数"""
        # TODO: 实现策略名称 → 函数的映射
        raise NotImplementedError

    def _simulate(self, strategy_func, stocks_data, start, end,
                  capital, max_pos, sl, tp) -> BacktestResult:
        """核心模拟循环：逐日遍历，触发买卖信号"""
        # TODO: 实现逐日模拟
        raise NotImplementedError

    def _calc_metrics(self, trades: List[Trade],
                      equity_curve: List[float],
                      initial_capital: float) -> BacktestMetrics:
        """根据交易记录和净值曲线计算统计指标"""
        # TODO: 实现指标计算
        raise NotImplementedError
