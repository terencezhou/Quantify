# -*- encoding: UTF-8 -*-

"""股票组聚合计算 — 可复用的原子计算层

给定任意一组股票的数据，聚合计算多维度指标。
每个函数都是纯函数（无副作用），输入输出明确，可被不同业务模块复用。

适用场景：
    - 概念热度分析（concept_hot.py）
    - 行业热度分析（industry_temperature.py，未来可迁移）
    - 自选股组合分析
    - 持仓组合风控

五大维度：
    1. 价格动量  — 基于日K线聚合
    2. 资金动向  — 基于个股资金流聚合
    3. 极端信号  — 基于涨停池 / 龙虎榜 / 大单
    4. 筹码结构  — 基于筹码分布聚合
    5. 聪明钱    — 基于北向持股聚合
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import pandas as pd


# ══════════════════════════════════════════════════════════════
#  维度一：价格动量
# ══════════════════════════════════════════════════════════════

def calc_price_momentum(
    daily_dfs: Dict[str, pd.DataFrame],
    target_date: Optional[str] = None,
) -> dict:
    """计算一组股票的价格动量聚合指标。

    Args:
        daily_dfs: {股票代码: 日K DataFrame}，DataFrame 需包含
                   '日期'、'收盘'、'涨跌幅'、'最高' 列。
        target_date: 基准日期（'YYYY-MM-DD' 格式），None 时取各股最新日期。

    Returns:
        dict 包含：
            member_count  — 有效成分股数量
            avg_change_1d — 今日涨跌幅均值（%）
            avg_change_3d — 近3日累计涨跌幅均值（%）
            avg_change_5d — 近5日累计涨跌幅均值（%）
            avg_change_10d— 近10日累计涨跌幅均值（%）
            up_ratio      — 今日上涨家数 / 总家数
            new_high_ratio— 创20日新高家数 / 总家数
    """
    if not daily_dfs:
        return _empty_price_momentum()

    changes_1d = []
    changes_3d = []
    changes_5d = []
    changes_10d = []
    up_count = 0
    new_high_count = 0
    valid_count = 0

    for _code, df in daily_dfs.items():
        if df is None or df.empty:
            continue
        if '涨跌幅' not in df.columns or '收盘' not in df.columns:
            continue

        df = df.copy()
        df['日期'] = df['日期'].astype(str)
        df = df.sort_values('日期').reset_index(drop=True)

        if target_date:
            idx_list = df.index[df['日期'] == target_date].tolist()
            if not idx_list:
                continue
            pos = idx_list[-1]
        else:
            pos = len(df) - 1

        valid_count += 1

        chg_1d = _safe_float(df.iloc[pos].get('涨跌幅'))
        changes_1d.append(chg_1d)

        if chg_1d > 0:
            up_count += 1

        changes_3d.append(_cum_change(df, pos, 3))
        changes_5d.append(_cum_change(df, pos, 5))
        changes_10d.append(_cum_change(df, pos, 10))

        if '最高' in df.columns:
            high_today = _safe_float(df.iloc[pos].get('最高'))
            lookback_start = max(0, pos - 20)
            if lookback_start < pos:
                hist_high = pd.to_numeric(
                    df.iloc[lookback_start:pos]['最高'], errors='coerce'
                ).max()
                if high_today >= hist_high and hist_high > 0:
                    new_high_count += 1

    if valid_count == 0:
        return _empty_price_momentum()

    return {
        'member_count': valid_count,
        'avg_change_1d': _safe_mean(changes_1d),
        'avg_change_3d': _safe_mean(changes_3d),
        'avg_change_5d': _safe_mean(changes_5d),
        'avg_change_10d': _safe_mean(changes_10d),
        'up_ratio': up_count / valid_count,
        'new_high_ratio': new_high_count / valid_count,
    }


def _empty_price_momentum() -> dict:
    return {
        'member_count': 0,
        'avg_change_1d': 0.0, 'avg_change_3d': 0.0,
        'avg_change_5d': 0.0, 'avg_change_10d': 0.0,
        'up_ratio': 0.0, 'new_high_ratio': 0.0,
    }


# ══════════════════════════════════════════════════════════════
#  维度二：资金动向
# ══════════════════════════════════════════════════════════════

def calc_fund_flow_agg(
    fund_flow_dfs: Dict[str, pd.DataFrame],
    target_date: Optional[str] = None,
    streak_days: int = 3,
) -> dict:
    """计算一组股票的资金流聚合指标。

    Args:
        fund_flow_dfs: {股票代码: 资金流 DataFrame}，DataFrame 需包含
                       '日期'、'主力净流入-净额'、'主力净流入-净占比' 等列。
        target_date: 基准日期（'YYYY-MM-DD'），None 时取各股最新日期。
        streak_days: 连续净流入天数阈值，用于计算 inflow_streak_ratio。

    Returns:
        dict 包含：
            member_count        — 有效成分股数量
            sum_main_inflow     — 主力净流入合计（元）
            avg_main_pct        — 主力净流入占比均值（%）
            big_order_pct       — 超大单净流入占比均值（%）
            inflow_streak_ratio — 连续 N 日主力净流入的成分股比例
    """
    if not fund_flow_dfs:
        return _empty_fund_flow()

    main_inflows = []
    main_pcts = []
    big_pcts = []
    streak_count = 0
    valid_count = 0

    for _code, df in fund_flow_dfs.items():
        if df is None or df.empty:
            continue
        if '主力净流入-净额' not in df.columns:
            continue

        df = df.copy()
        df['日期'] = df['日期'].astype(str)
        df = df.sort_values('日期').reset_index(drop=True)

        if target_date:
            idx_list = df.index[df['日期'] == target_date].tolist()
            if not idx_list:
                continue
            pos = idx_list[-1]
        else:
            pos = len(df) - 1

        valid_count += 1

        inflow = _safe_float(df.iloc[pos].get('主力净流入-净额'))
        main_inflows.append(inflow)

        pct = _safe_float(df.iloc[pos].get('主力净流入-净占比'))
        main_pcts.append(pct)

        big = _safe_float(df.iloc[pos].get('超大单净流入-净占比'))
        big_pcts.append(big)

        start = max(0, pos - streak_days + 1)
        if start <= pos:
            recent = df.iloc[start:pos + 1]
            vals = pd.to_numeric(recent['主力净流入-净额'], errors='coerce').fillna(0)
            if len(vals) >= streak_days and (vals > 0).all():
                streak_count += 1

    if valid_count == 0:
        return _empty_fund_flow()

    return {
        'member_count': valid_count,
        'sum_main_inflow': sum(main_inflows),
        'avg_main_pct': _safe_mean(main_pcts),
        'big_order_pct': _safe_mean(big_pcts),
        'inflow_streak_ratio': streak_count / valid_count,
    }


def _empty_fund_flow() -> dict:
    return {
        'member_count': 0,
        'sum_main_inflow': 0.0, 'avg_main_pct': 0.0,
        'big_order_pct': 0.0, 'inflow_streak_ratio': 0.0,
    }


# ══════════════════════════════════════════════════════════════
#  维度三：极端信号（涨停 / 龙虎榜 / 大单）
# ══════════════════════════════════════════════════════════════

def calc_extreme_signals(
    member_codes: Set[str],
    zt_pool_df: Optional[pd.DataFrame] = None,
    lhb_df: Optional[pd.DataFrame] = None,
    big_deal_df: Optional[pd.DataFrame] = None,
) -> dict:
    """从全市场快照中过滤指定股票组的极端信号。

    Args:
        member_codes: 成分股代码集合（6位字符串）。
        zt_pool_df: 涨停池 DataFrame，需含 '代码'、'连板数'、'封板资金' 列。
        lhb_df: 龙虎榜 DataFrame，需含 '代码'、'龙虎榜净买额' 列。
        big_deal_df: 大单追踪 DataFrame，需含 '股票代码' 或 '代码' 列。

    Returns:
        dict 包含：
            zt_count       — 概念内涨停股数量
            max_lianban    — 最高连板数
            total_seal_amt — 封板资金合计（元）
            lhb_net_buy    — 龙虎榜净买入合计（元）
            big_buy_amount — 大单买入金额合计（元）
            zt_codes       — 涨停股代码列表（供下游使用）
    """
    result = {
        'zt_count': 0, 'max_lianban': 0, 'total_seal_amt': 0.0,
        'lhb_net_buy': 0.0, 'big_buy_amount': 0.0, 'zt_codes': [],
    }

    if not member_codes:
        return result

    codes_set = {str(c).zfill(6) for c in member_codes}

    # 涨停池
    if zt_pool_df is not None and not zt_pool_df.empty:
        zt = zt_pool_df.copy()
        zt['代码'] = zt['代码'].astype(str).str.zfill(6)
        matched = zt[zt['代码'].isin(codes_set)]

        if not matched.empty:
            result['zt_count'] = len(matched)
            result['zt_codes'] = matched['代码'].tolist()

            if '连板数' in matched.columns:
                lb = pd.to_numeric(matched['连板数'], errors='coerce').fillna(1)
                result['max_lianban'] = int(lb.max())

            if '封板资金' in matched.columns:
                seal = pd.to_numeric(matched['封板资金'], errors='coerce').fillna(0)
                result['total_seal_amt'] = float(seal.sum())

    # 龙虎榜
    if lhb_df is not None and not lhb_df.empty:
        lhb = lhb_df.copy()
        lhb['代码'] = lhb['代码'].astype(str).str.zfill(6)
        matched = lhb[lhb['代码'].isin(codes_set)]

        if not matched.empty and '龙虎榜净买额' in matched.columns:
            net = pd.to_numeric(matched['龙虎榜净买额'], errors='coerce').fillna(0)
            result['lhb_net_buy'] = float(net.sum())

    # 大单追踪
    if big_deal_df is not None and not big_deal_df.empty:
        bd = big_deal_df.copy()
        code_col = '股票代码' if '股票代码' in bd.columns else '代码'
        if code_col in bd.columns:
            bd[code_col] = bd[code_col].astype(str).str.zfill(6)
            matched = bd[bd[code_col].isin(codes_set)]
            if not matched.empty:
                amt_col = next(
                    (c for c in matched.columns if '成交金额' in c or '金额' in c),
                    None,
                )
                if amt_col:
                    amt = pd.to_numeric(matched[amt_col], errors='coerce').fillna(0)
                    result['big_buy_amount'] = float(amt.sum())

    return result


# ══════════════════════════════════════════════════════════════
#  维度四：筹码结构
# ══════════════════════════════════════════════════════════════

def calc_chips_agg(
    chips_dfs: Dict[str, pd.DataFrame],
) -> dict:
    """计算一组股票的筹码结构聚合指标。

    Args:
        chips_dfs: {股票代码: 筹码 DataFrame}，DataFrame 需包含
                   '日期'、'获利比例'、'70%集中度' 列。

    Returns:
        dict 包含：
            member_count         — 有效成分股数量
            avg_profit_ratio     — 平均获利比例（0~1）
            concentration_change — 70%集中度近5日平均变化
                                   （负值=筹码集中=主力收集）
    """
    if not chips_dfs:
        return {'member_count': 0, 'avg_profit_ratio': 0.0,
                'concentration_change': 0.0}

    profit_ratios = []
    conc_changes = []
    valid_count = 0

    for _code, df in chips_dfs.items():
        if df is None or df.empty:
            continue

        df = df.copy()
        df = df.sort_values('日期').reset_index(drop=True)

        valid_count += 1

        if '获利比例' in df.columns:
            latest_profit = _safe_float(df.iloc[-1].get('获利比例'))
            profit_ratios.append(latest_profit)

        if '70%集中度' in df.columns and len(df) >= 2:
            conc = pd.to_numeric(df['70%集中度'], errors='coerce').dropna()
            if len(conc) >= 2:
                recent_len = min(5, len(conc))
                recent = conc.iloc[-recent_len:]
                change = float(recent.iloc[-1] - recent.iloc[0])
                conc_changes.append(change)

    if valid_count == 0:
        return {'member_count': 0, 'avg_profit_ratio': 0.0,
                'concentration_change': 0.0}

    return {
        'member_count': valid_count,
        'avg_profit_ratio': _safe_mean(profit_ratios),
        'concentration_change': _safe_mean(conc_changes),
    }


# ══════════════════════════════════════════════════════════════
#  维度五：聪明钱（北向资金）
# ══════════════════════════════════════════════════════════════

def calc_northbound_agg(
    hsgt_dfs: Dict[str, pd.DataFrame],
) -> dict:
    """计算一组股票的北向持股聚合指标。

    Args:
        hsgt_dfs: {股票代码: 北向持股 DataFrame}，DataFrame 需包含
                  '日期'、'持股市值'（或 '持股市值变化1日'）列。

    Returns:
        dict 包含：
            member_count    — 有数据的成分股数量
            north_add_count — 今日北向净增持的成分股数量
            north_add_value — 今日北向净增持市值合计（元）
    """
    if not hsgt_dfs:
        return {'member_count': 0, 'north_add_count': 0,
                'north_add_value': 0.0}

    add_count = 0
    add_value = 0.0
    valid_count = 0

    for _code, df in hsgt_dfs.items():
        if df is None or df.empty:
            continue

        df = df.copy()
        df = df.sort_values('日期').reset_index(drop=True)

        valid_count += 1

        chg_col = next(
            (c for c in df.columns if '持股市值变化' in c and '1日' in c),
            None,
        )

        if chg_col:
            change = _safe_float(df.iloc[-1].get(chg_col))
        elif '持股市值' in df.columns and len(df) >= 2:
            v_today = _safe_float(df.iloc[-1].get('持股市值'))
            v_prev = _safe_float(df.iloc[-2].get('持股市值'))
            change = v_today - v_prev
        else:
            change = 0.0

        if change > 0:
            add_count += 1
            add_value += change

    return {
        'member_count': valid_count,
        'north_add_count': add_count,
        'north_add_value': add_value,
    }


# ══════════════════════════════════════════════════════════════
#  通用评分工具
# ══════════════════════════════════════════════════════════════

def percentile_rank_scores(
    values: List[float],
    reverse: bool = False,
) -> List[float]:
    """将一组值转换为百分位排名得分（0-100）。

    适用于跨概念/跨行业的相对排名评分：
    排名第一得 100 分，最后一名得 0 分。

    Args:
        values: 原始值列表，长度 N。
        reverse: True 表示值越小排名越高（如获利比例，越低越健康）。

    Returns:
        长度 N 的得分列表，每项 0-100。
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [50.0]

    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1], reverse=(not reverse))

    scores = [0.0] * n
    for rank, (orig_idx, _) in enumerate(indexed):
        scores[orig_idx] = (n - 1 - rank) / (n - 1) * 100.0

    return scores


def normalize_score(
    value: float,
    min_val: float,
    max_val: float,
    clip: bool = True,
) -> float:
    """线性归一化到 0-100。

    Args:
        value: 原始值。
        min_val: 映射为 0 分的原始值。
        max_val: 映射为 100 分的原始值。
        clip: 是否裁剪到 [0, 100]。

    Returns:
        归一化得分 (0-100)。
    """
    if max_val <= min_val:
        return 0.0
    score = (value - min_val) / (max_val - min_val) * 100.0
    if clip:
        score = max(0.0, min(100.0, score))
    return round(score, 2)


# ══════════════════════════════════════════════════════════════
#  内部辅助
# ══════════════════════════════════════════════════════════════

def _safe_float(val) -> float:
    """安全转换为 float，失败返回 0.0。"""
    try:
        v = float(val)
        return v if pd.notna(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _safe_mean(values: list) -> float:
    """安全计算均值，空列表返回 0.0。"""
    valid = [v for v in values if v is not None and pd.notna(v)]
    return sum(valid) / len(valid) if valid else 0.0


def _cum_change(df: pd.DataFrame, pos: int, days: int) -> float:
    """计算从 pos-days+1 到 pos 的累计涨跌幅（%）。

    直接用首尾收盘价计算：(end_close - start_prev_close) / start_prev_close。
    需要取 pos-days 那天的收盘价作为基准（即区间第一天的前一天）。
    """
    if days <= 1 or pos < 1:
        return _safe_float(df.iloc[pos].get('涨跌幅'))

    base_pos = max(0, pos - days)
    close_end = pd.to_numeric(df.iloc[pos].get('收盘'), errors='coerce')
    close_base = pd.to_numeric(df.iloc[base_pos].get('收盘'), errors='coerce')

    if pd.isna(close_end) or pd.isna(close_base) or close_base == 0:
        return 0.0
    return float((close_end - close_base) / close_base * 100)


# ══════════════════════════════════════════════════════════════
#  概念板块工具
# ══════════════════════════════════════════════════════════════

# 与具体行业/题材无关的平台性标签，反向构建时过滤掉
CONCEPT_PLATFORM_TAGS: Set[str] = {
    # 指数成份
    '沪深300', '上证50', '中证500', '中证100', '中证1000',
    '科创50', '创业板50', '深证100', '中小板指',
    'MSCI中国', 'MSCI成份股', '富时罗素', '标普中国A股大中盘指数',
    # 互联互通
    '沪股通', '深股通', '港股通', '北向资金重仓',
    # 融资交易
    '融资融券', '可融资标的', '融资标的',
    # 股权/回购/分红
    '股权激励', '回购预案', '高股息', '参股银行', 'AB股',
    # 机构/资金持仓标签
    '机构重仓', '基金重仓', '社保重仓', '券商重仓', 'QFII重仓',
    '外资持股', '北向持股',
    # 财报/业绩类标签（与行业无关的时效性标签）
    '2025中报扭亏', '2024中报扭亏', '2024年报扭亏', '2025年报扭亏',
    '2025中报预增', '2024中报预增', '2025年报预增', '2024年报预增',
    '业绩预增', '业绩扭亏', '高送转',
    # 宏观/政策/地域概念（非行业题材）
    '长江三角', '长三角', '珠三角', '京津冀', '粤港澳大湾区',
    '雄安新区', '自贸区', '海南自贸港', '西部大开发',
    '贬值受益', '升值受益', '通胀受益',
    # 涨停/技术类标签
    '昨日连板_含一字', '昨日涨停_含一字', '昨日首板', '昨日涨停',
    '东方财富热股', '最近多板',
    # 国资/治理类
    '国企改革', '央企改革', '中字头', '国资委概念', '央企国资',
    '混合所有制', '壳资源',
}


def build_stock_concept_map(
    concept_cons: dict,
    blacklist: Optional[Set[str]] = None,
    max_per_stock: int = 5,
) -> Dict[str, List[str]]:
    """从 concept_cons 反向构建「股票代码 → 概念列表」映射。

    Args:
        concept_cons: DataManager.extra['concept_cons']，结构为
                      {concept_code: DataFrame(含'代码'列, 可选'板块名称'列)}
        blacklist:    需要过滤的平台性/非题材标签集合，默认使用 CONCEPT_PLATFORM_TAGS
        max_per_stock: 每只股票最多保留的概念数量

    Returns:
        {stock_code_6digit: [concept_name, ...]}
    """
    if blacklist is None:
        blacklist = CONCEPT_PLATFORM_TAGS

    stock_concepts: Dict[str, List[str]] = {}

    for concept_code, cons_df in concept_cons.items():
        if cons_df is None or cons_df.empty or '代码' not in cons_df.columns:
            continue

        # 提取概念名称
        if '板块名称' in cons_df.columns:
            names = cons_df['板块名称'].dropna().unique()
            concept_name = str(names[0]) if len(names) > 0 else str(concept_code)
        else:
            concept_name = str(concept_code)

        if concept_name in blacklist:
            continue

        codes = cons_df['代码'].astype(str).str.zfill(6).tolist()
        for code in codes:
            stock_concepts.setdefault(code, [])
            if concept_name not in stock_concepts[code]:
                stock_concepts[code].append(concept_name)

    # 每只股票取前 max_per_stock 个概念（保持原始顺序，即按概念遍历顺序）
    return {
        code: concepts[:max_per_stock]
        for code, concepts in stock_concepts.items()
    }
