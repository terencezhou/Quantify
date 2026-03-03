# -*- encoding: UTF-8 -*-

"""持仓卖出信号分析器 — 分析当前持仓是否应该卖出

职责：
  1. 读取配置中的持仓列表
  2. 对每只持仓股进行多维度卖出信号分析
  3. 输出卖出建议 & 推送报告

持仓配置格式（config.yaml）：
    holdings:
      - code: "000001"
        name: "平安银行"
        buy_price: 10.50
        buy_date: "2025-01-15"
        shares: 1000
      - code: "600519"
        name: "贵州茅台"
        buy_price: 1680.00
        buy_date: "2025-02-01"
        shares: 100

使用方式：
    dm = DataManager(config)
    dm.refresh_for_stocks(holdings)
    analyzer = SellAnalyzer(dm)
    report = analyzer.run()
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from data_manager import DataManager


class Signal(Enum):
    """卖出信号强度"""
    STRONG_SELL = "强烈卖出"
    SELL = "建议卖出"
    WATCH = "关注观望"
    HOLD = "继续持有"
    BUY_MORE = "可加仓"


@dataclass
class Holding:
    """单只持仓信息"""
    code: str
    name: str
    buy_price: float
    buy_date: str
    shares: int = 0

    @classmethod
    def from_config(cls, d: dict) -> 'Holding':
        return cls(
            code=str(d['code']),
            name=d.get('name', ''),
            buy_price=float(d.get('buy_price', 0)),
            buy_date=str(d.get('buy_date', '')),
            shares=int(d.get('shares', 0)),
        )


@dataclass
class SellSignalDetail:
    """单个维度的信号详情"""
    dimension: str       # 维度名称：止损/止盈/趋势/资金/基本面
    signal: Signal
    score: float         # 0~100，越高越应该卖
    reason: str          # 一句话解释


@dataclass
class StockSellReport:
    """单只持仓的卖出分析报告"""
    holding: Holding
    current_price: float = 0.0
    pnl_pct: float = 0.0            # 盈亏比例
    overall_signal: Signal = Signal.HOLD
    overall_score: float = 0.0       # 综合卖出评分 0~100
    details: List[SellSignalDetail] = field(default_factory=list)
    summary: str = ''


@dataclass
class SellReport:
    """完整卖出分析报告"""
    stock_reports: List[StockSellReport] = field(default_factory=list)
    overview_md: str = ''


class SellAnalyzer:
    """持仓卖出信号分析器。

    对配置文件中的每只持仓股进行多维度分析，输出是否应该卖出的建议。
    """

    def __init__(self, dm: DataManager):
        self._dm = dm
        self._config = dm.config
        self._report: Optional[SellReport] = None

    @property
    def report(self) -> Optional[SellReport]:
        return self._report

    # ══════════════════════════════════════════════
    #  主流程
    # ══════════════════════════════════════════════

    def run(self) -> SellReport:
        """执行持仓卖出信号分析，返回 SellReport。

        流程：
          1. 解析持仓列表
          2. 对每只持仓运行多维卖出信号分析
          3. 汇总信号，给出综合建议
          4. 构建报告 & 推送
        """
        holdings = self._parse_holdings()
        if not holdings:
            logging.warning("配置中无持仓信息（holdings 为空）")
            return SellReport()

        report = SellReport()
        for h in holdings:
            sr = self._analyze_one(h)
            report.stock_reports.append(sr)

        report.overview_md = self._build_report_md(report)
        self._report = report
        return report

    # ══════════════════════════════════════════════
    #  持仓解析
    # ══════════════════════════════════════════════

    def _parse_holdings(self) -> List[Holding]:
        """从配置中解析持仓列表"""
        raw = self._config.get('holdings') or []
        holdings = []
        for item in raw:
            try:
                holdings.append(Holding.from_config(item))
            except (KeyError, ValueError) as e:
                logging.warning("解析持仓项失败: %s, %s", item, e)
        return holdings

    # ══════════════════════════════════════════════
    #  单只股票分析
    # ══════════════════════════════════════════════

    def _analyze_one(self, holding: Holding) -> StockSellReport:
        """对单只持仓进行多维度卖出分析"""
        sr = StockSellReport(holding=holding)

        # 获取当前价格
        rt_lookup = self._dm.build_rt_lookup()
        rt = rt_lookup.get(holding.code)
        if rt is not None:
            try:
                sr.current_price = float(rt.get('最新价', 0))
            except (TypeError, ValueError):
                pass

        if sr.current_price > 0 and holding.buy_price > 0:
            sr.pnl_pct = (sr.current_price - holding.buy_price) / holding.buy_price * 100

        # 多维度信号分析
        sr.details.append(self._check_stop_loss(holding, sr))
        sr.details.append(self._check_take_profit(holding, sr))
        sr.details.append(self._check_trend_breakdown(holding))
        sr.details.append(self._check_volume_divergence(holding))
        sr.details.append(self._check_fund_flow_exit(holding))
        sr.details.append(self._check_fundamental_change(holding))

        # 综合评分
        sr.overall_score = self._calc_overall_score(sr.details)
        sr.overall_signal = self._score_to_signal(sr.overall_score)
        sr.summary = self._make_summary(sr)

        return sr

    # ══════════════════════════════════════════════
    #  数据辅助
    # ══════════════════════════════════════════════

    def _get_daily(self, holding: Holding):
        """获取持仓股的日K线数据"""
        for cn, df in self._dm.stocks_data.items():
            if cn[0] == holding.code:
                return df
        return None

    # ══════════════════════════════════════════════
    #  卖出信号维度（每个返回 SellSignalDetail）
    # ══════════════════════════════════════════════

    def _check_stop_loss(self, holding: Holding, sr: StockSellReport) -> SellSignalDetail:
        """固定止损 + 移动止盈止损"""
        pnl = sr.pnl_pct

        # 固定止损：亏损 ≥ 8% 强烈卖出
        if pnl <= -8:
            return SellSignalDetail(
                dimension='止损', signal=Signal.STRONG_SELL, score=95,
                reason=f'浮亏{pnl:.1f}%，触及8%硬止损线')

        # 亏损 ≥ 5%：建议卖出
        if pnl <= -5:
            return SellSignalDetail(
                dimension='止损', signal=Signal.SELL, score=70,
                reason=f'浮亏{pnl:.1f}%，接近止损线')

        # 移动止盈：从最高点回撤 ≥ 15%
        data = self._get_daily(holding)
        if data is not None and len(data) >= 5 and holding.buy_date:
            try:
                mask = data['日期'] >= str(holding.buy_date)
                since_buy = data.loc[mask]
                if len(since_buy) > 0:
                    peak = float(since_buy['最高'].max())
                    if peak > 0 and sr.current_price > 0:
                        drawdown = (peak - sr.current_price) / peak * 100
                        if drawdown >= 15:
                            return SellSignalDetail(
                                dimension='止损', signal=Signal.SELL, score=75,
                                reason=f'从最高{peak:.2f}回撤{drawdown:.1f}%，触发移动止盈')
                        if drawdown >= 10:
                            return SellSignalDetail(
                                dimension='止损', signal=Signal.WATCH, score=45,
                                reason=f'从最高{peak:.2f}回撤{drawdown:.1f}%，关注')
            except (KeyError, TypeError):
                pass

        # 小幅亏损
        if pnl < 0:
            return SellSignalDetail(
                dimension='止损', signal=Signal.WATCH, score=30,
                reason=f'浮亏{pnl:.1f}%，未触及止损线')

        return SellSignalDetail(
            dimension='止损', signal=Signal.HOLD, score=5,
            reason=f'浮盈{pnl:.1f}%，安全')

    def _check_take_profit(self, holding: Holding, sr: StockSellReport) -> SellSignalDetail:
        """止盈检查：盈利达标 + 放量滞涨等见顶信号"""
        pnl = sr.pnl_pct
        data = self._get_daily(holding)

        # 盈利 ≥ 30% 且近3日放量滞涨
        if pnl >= 30 and data is not None and len(data) >= 5:
            close = data['收盘'].astype(float)
            volume = data['成交量'].astype(float)
            recent_change = (float(close.iloc[-1]) - float(close.iloc[-4])) / float(close.iloc[-4]) * 100
            vol_ratio = float(volume.iloc[-3:].mean()) / float(volume.iloc[-10:].mean()) if float(volume.iloc[-10:].mean()) > 0 else 1
            if abs(recent_change) < 2 and vol_ratio > 1.5:
                return SellSignalDetail(
                    dimension='止盈', signal=Signal.SELL, score=80,
                    reason=f'盈利{pnl:.1f}%且近3日放量滞涨（量比{vol_ratio:.1f}x）')

        if pnl >= 50:
            return SellSignalDetail(
                dimension='止盈', signal=Signal.WATCH, score=50,
                reason=f'盈利{pnl:.1f}%，高位需关注分批止盈')

        if pnl >= 30:
            return SellSignalDetail(
                dimension='止盈', signal=Signal.WATCH, score=35,
                reason=f'盈利{pnl:.1f}%，达到止盈关注区')

        if pnl >= 10:
            return SellSignalDetail(
                dimension='止盈', signal=Signal.HOLD, score=10,
                reason=f'盈利{pnl:.1f}%，持股待涨')

        return SellSignalDetail(
            dimension='止盈', signal=Signal.HOLD, score=0,
            reason=f'盈利{pnl:.1f}%，未到止盈区')

    def _check_trend_breakdown(self, holding: Holding) -> SellSignalDetail:
        """趋势止损：收盘跌破均线系统"""
        data = self._get_daily(holding)
        if data is None or len(data) < 60:
            return SellSignalDetail(
                dimension='趋势', signal=Signal.HOLD, score=0,
                reason='数据不足，无法判断趋势')

        close = data['收盘'].astype(float)
        last_close = float(close.iloc[-1])
        ma5 = float(close.iloc[-5:].mean())
        ma10 = float(close.iloc[-10:].mean())
        ma20 = float(close.iloc[-20:].mean())
        ma60 = float(close.iloc[-60:].mean())

        # 跌破 MA60：强烈卖出
        if last_close < ma60 and ma5 < ma20:
            return SellSignalDetail(
                dimension='趋势', signal=Signal.STRONG_SELL, score=90,
                reason=f'收盘{last_close:.2f}跌破MA60({ma60:.2f})，趋势完全破坏')

        # 跌破 MA20 且 MA5 < MA10：建议卖出
        if last_close < ma20 and ma5 < ma10:
            return SellSignalDetail(
                dimension='趋势', signal=Signal.SELL, score=70,
                reason=f'收盘{last_close:.2f}跌破MA20({ma20:.2f})且均线空头')

        # 跌破 MA20 但未确认
        if last_close < ma20:
            return SellSignalDetail(
                dimension='趋势', signal=Signal.WATCH, score=45,
                reason=f'收盘{last_close:.2f}跌破MA20({ma20:.2f})，待确认')

        # MA5 下穿 MA10（短期死叉）
        if ma5 < ma10:
            return SellSignalDetail(
                dimension='趋势', signal=Signal.WATCH, score=30,
                reason='MA5下穿MA10，短期趋势走弱')

        return SellSignalDetail(
            dimension='趋势', signal=Signal.HOLD, score=5,
            reason=f'均线多头排列，趋势健康')

    def _check_volume_divergence(self, holding: Holding) -> SellSignalDetail:
        """量价背离：价格创新高但量能萎缩 / 放量下跌"""
        data = self._get_daily(holding)
        if data is None or len(data) < 20:
            return SellSignalDetail(
                dimension='量价', signal=Signal.HOLD, score=0,
                reason='数据不足')

        close = data['收盘'].astype(float)
        volume = data['成交量'].astype(float)

        # 价格创5日新高但成交量逐日递减（顶背离）
        high_5 = float(data['最高'].iloc[-5:].max())
        today_high = float(data['最高'].iloc[-1])

        if today_high >= high_5 * 0.99:
            vol_trend = volume.iloc[-5:].values
            decreasing = all(vol_trend[i] >= vol_trend[i+1] for i in range(len(vol_trend)-1))
            if decreasing and len(vol_trend) >= 3:
                return SellSignalDetail(
                    dimension='量价', signal=Signal.SELL, score=70,
                    reason='价格在高位，成交量连续递减，量价顶背离')

        # 放量下跌（恐慌出逃）
        last_change = float(data.iloc[-1].get('涨跌幅', 0))
        vol_ratio = float(volume.iloc[-1]) / float(volume.iloc[-10:].mean()) if float(volume.iloc[-10:].mean()) > 0 else 1
        if last_change < -3 and vol_ratio > 2:
            return SellSignalDetail(
                dimension='量价', signal=Signal.SELL, score=75,
                reason=f'放量下跌{last_change:.1f}%（量比{vol_ratio:.1f}x），恐慌出逃信号')

        # 温和放量下跌
        if last_change < -2 and vol_ratio > 1.5:
            return SellSignalDetail(
                dimension='量价', signal=Signal.WATCH, score=40,
                reason=f'温和放量下跌{last_change:.1f}%（量比{vol_ratio:.1f}x）')

        # 缩量上涨（动能减弱）
        if last_change > 0 and vol_ratio < 0.5:
            return SellSignalDetail(
                dimension='量价', signal=Signal.WATCH, score=25,
                reason='上涨但极度缩量，动能减弱')

        return SellSignalDetail(
            dimension='量价', signal=Signal.HOLD, score=5,
            reason='量价关系正常')

    def _check_fund_flow_exit(self, holding: Holding) -> SellSignalDetail:
        """资金流向：主力是否在撤退"""
        extra = self._dm.extra
        ff_data = extra.get('fund_flow', {})

        code_name = None
        for cn in self._dm.stocks_data.keys():
            if cn[0] == holding.code:
                code_name = cn
                break

        ff = ff_data.get(code_name) if code_name else None
        if ff is None or ff.empty or '主力净流入-净额' not in ff.columns:
            return SellSignalDetail(
                dimension='资金', signal=Signal.HOLD, score=0,
                reason='无资金流数据')

        recent = ff.tail(5)
        main_flow = recent['主力净流入-净额'].astype(float)

        # 连续 3 日主力净流出
        last3 = main_flow.iloc[-3:] if len(main_flow) >= 3 else main_flow
        if len(last3) >= 3 and (last3 < 0).all():
            total_out = float(last3.sum())
            return SellSignalDetail(
                dimension='资金', signal=Signal.SELL, score=70,
                reason=f'连续3日主力净流出共{total_out/10000:.0f}万')

        # 超大单持续净卖出
        if '超大单净流入-净额' in recent.columns:
            super_flow = recent['超大单净流入-净额'].astype(float)
            last3_super = super_flow.iloc[-3:] if len(super_flow) >= 3 else super_flow
            if len(last3_super) >= 3 and (last3_super < 0).all():
                return SellSignalDetail(
                    dimension='资金', signal=Signal.WATCH, score=45,
                    reason='超大单连续3日净卖出')

        # 近5日主力累计净流出
        total_5d = float(main_flow.sum())
        if total_5d < -5_000_000:
            return SellSignalDetail(
                dimension='资金', signal=Signal.WATCH, score=35,
                reason=f'近5日主力累计净流出{total_5d/10000:.0f}万')

        if total_5d > 5_000_000:
            return SellSignalDetail(
                dimension='资金', signal=Signal.HOLD, score=5,
                reason=f'近5日主力累计净流入{total_5d/10000:.0f}万')

        return SellSignalDetail(
            dimension='资金', signal=Signal.HOLD, score=15,
            reason='资金流向中性')

    def _check_fundamental_change(self, holding: Holding) -> SellSignalDetail:
        """基本面变化：业绩恶化 / 财务风险"""
        extra = self._dm.extra
        report_df = extra.get('financial_report')
        if report_df is None or report_df.empty:
            return SellSignalDetail(
                dimension='基本面', signal=Signal.HOLD, score=0,
                reason='无财报数据')

        report_by_code = dict(list(report_df.groupby('股票代码')))
        fin = report_by_code.get(holding.code)
        if fin is None or fin.empty:
            return SellSignalDetail(
                dimension='基本面', signal=Signal.HOLD, score=0,
                reason='无该股财报')

        fin = fin.sort_values('报告期').reset_index(drop=True)
        latest = fin.iloc[-1]

        def _sf(val):
            try:
                v = float(val)
                return 0.0 if v != v else v
            except (TypeError, ValueError):
                return 0.0

        np_g = _sf(latest.get('净利润-同比增长'))
        rev_g = _sf(latest.get('营业总收入-同比增长'))
        roe = _sf(latest.get('净资产收益率'))
        net_profit = _sf(latest.get('净利润-净利润'))
        ocf = _sf(latest.get('每股经营现金流量'))

        # 净利润为负 + 营收下滑：强烈卖出
        if net_profit < 0 and rev_g < 0:
            return SellSignalDetail(
                dimension='基本面', signal=Signal.STRONG_SELL, score=85,
                reason=f'净利润为负且营收下滑{rev_g:.1f}%，基本面严重恶化')

        # ROE 大幅下降或转负
        if roe < 0:
            return SellSignalDetail(
                dimension='基本面', signal=Signal.SELL, score=70,
                reason=f'ROE为{roe:.1f}%（负值），盈利能力恶化')

        # 净利同比大幅下滑
        if np_g < -30:
            return SellSignalDetail(
                dimension='基本面', signal=Signal.SELL, score=65,
                reason=f'净利润同比下滑{np_g:.1f}%')

        # 营收/净利双降
        if np_g < 0 and rev_g < 0:
            return SellSignalDetail(
                dimension='基本面', signal=Signal.WATCH, score=45,
                reason=f'营收({rev_g:.1f}%)和净利({np_g:.1f}%)双降')

        # 经营现金流为负
        if ocf < 0:
            return SellSignalDetail(
                dimension='基本面', signal=Signal.WATCH, score=35,
                reason='每股经营现金流为负，造血能力不足')

        return SellSignalDetail(
            dimension='基本面', signal=Signal.HOLD, score=5,
            reason=f'基本面稳健（ROE={roe:.1f}% 净利增{np_g:.1f}%）')

    # ══════════════════════════════════════════════
    #  评分汇总
    # ══════════════════════════════════════════════

    def _calc_overall_score(self, details: List[SellSignalDetail]) -> float:
        """根据各维度评分计算综合卖出评分 (0~100)"""
        if not details:
            return 0.0
        # 加权平均，止损/止盈权重更高
        weights = {'止损': 3.0, '止盈': 2.5, '趋势': 2.0,
                   '量价': 1.5, '资金': 1.5, '基本面': 1.0}
        total_w = 0.0
        total_s = 0.0
        for d in details:
            w = weights.get(d.dimension, 1.0)
            total_w += w
            total_s += d.score * w
        return total_s / total_w if total_w > 0 else 0.0

    @staticmethod
    def _score_to_signal(score: float) -> Signal:
        if score >= 80:
            return Signal.STRONG_SELL
        if score >= 60:
            return Signal.SELL
        if score >= 40:
            return Signal.WATCH
        if score >= 20:
            return Signal.HOLD
        return Signal.BUY_MORE

    @staticmethod
    def _make_summary(sr: StockSellReport) -> str:
        """生成一句话总结"""
        sig = sr.overall_signal.value
        pnl = f"{'盈利' if sr.pnl_pct >= 0 else '亏损'}{abs(sr.pnl_pct):.1f}%"
        return f"{sr.holding.name}({sr.holding.code}) {pnl} → {sig}"

    # ══════════════════════════════════════════════
    #  报告构建
    # ══════════════════════════════════════════════

    def _build_report_md(self, report: SellReport) -> str:
        """构建持仓分析 Markdown 报告"""
        # TODO: 实现完整的 Markdown 报告构建
        lines = ["# 持仓信号分析\n"]
        for sr in report.stock_reports:
            emoji = {'强烈卖出': '🔴', '建议卖出': '🟠', '关注观望': '🟡',
                     '继续持有': '🟢', '可加仓': '🔵'}
            icon = emoji.get(sr.overall_signal.value, '⚪')
            lines.append(f"{icon} **{sr.holding.name}**({sr.holding.code})")
            lines.append(f"  买入 {sr.holding.buy_price} → 现价 {sr.current_price}"
                         f" ({sr.pnl_pct:+.1f}%)")
            lines.append(f"  综合评分: {sr.overall_score:.0f}/100 → **{sr.overall_signal.value}**")
            for d in sr.details:
                lines.append(f"  - {d.dimension}: {d.reason}")
            lines.append("")
        return "\n".join(lines)
