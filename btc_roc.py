import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import traceback
from tqdm import tqdm
import concurrent.futures


def fetch_binance_klines(symbol, interval, start_date_str, end_date_str):
    """
    获取指定日期范围内的K线数据。
    :param symbol: 交易对, e.g., "BTCUSDT"
    :param interval: K线周期, e.g., "1m"
    :param start_date_str: 开始日期, "YYYY-MM-DD"格式
    :param end_date_str: 结束日期, "YYYY-MM-DD"格式
    """
    try:
        start_time = int(datetime.strptime(
            start_date_str, "%Y-%m-%d").timestamp() * 1000)
        end_time = int((datetime.strptime(
            end_date_str, "%Y-%m-%d") + timedelta(days=1)).timestamp() * 1000)
    except ValueError:
        print("错误：日期格式不正确，请使用 'YYYY-MM-DD' 格式。")
        return pd.DataFrame()

    url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    data = []
    print(
        f"正在从币安获取 {symbol} 从 {start_date_str} 到 {end_date_str} 的 {interval} K线数据...")

    current_start_time = start_time
    while current_start_time < end_time:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": current_start_time, "limit": limit, "endTime": end_time}
        try:
            resp = requests.get(url, params=params, timeout=10).json()
            if not resp or "code" in resp:
                print(f"API错误或无数据返回: {resp}")
                break
            data.extend(resp)
            print(f"已获取 {len(data)} 条数据...")
            current_start_time = resp[-1][0] + 60_000
            if len(resp) < limit:
                break
        except requests.exceptions.RequestException as e:
            print(f"网络请求失败: {e}")
            break

    if not data:
        print("未能获取任何数据。")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume",
                                     "close_time", "qav", "trades", "tb_base", "tb_quote", "ignore"])
    df = df[["open_time", "open", "high", "low", "close"]].astype(float)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    print("数据获取完毕！")
    return df


def add_indicators(df, period=720):
    """
    为DataFrame添加技术指标：
    - MA: 移动平均线
    """
    print(f"正在计算 {period} 分钟的移动平均线...")
    df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
    print("指标计算完毕！")
    return df


def build_levels(lower, upper, n_grids):
    """
    【方案A】根据给定的上下限和网格数量，生成一个标准的网格线列表。
    步长是动态计算的。
    :param n_grids: 您期望的网格数量 (e.g., 20)
    """
    if lower >= upper or n_grids <= 0:
        return [], 0  # 返回空的levels和一个step=0

    # 动态计算步长
    step = (upper - lower) / n_grids

    levels = []
    # 使用 n_grids+1 来确保包含上限
    levels = [round(lower + step * i, 2) for i in range(int(n_grids) + 1)]

    return levels, step  # 【重要】同时返回生成的levels和计算出的step


def load_from_sqlite(db_path, symbol, start_date, end_date):
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT
            datetime(open_time/1000, 'unixepoch') as datetime,
            open, high, low, close, volume
        FROM {symbol}_1m
        WHERE datetime(open_time/1000, 'unixepoch') BETWEEN ? AND ?
        ORDER BY open_time ASC
    """
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    conn.close()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


class GridTrader:
    def __init__(self, capital, fee_rate, n_grids, initial_lower, initial_upper,
                 ma_period, strategy_params, verbose=False):
        """
        初始化交易引擎的所有状态和参数。
        """
        # --- 核心参数 ---
        self.capital = capital
        self.fee_rate = fee_rate
        self.n_grids = n_grids
        self.initial_lower = initial_lower
        self.initial_upper = initial_upper
        self.ma_period = ma_period
        self.verbose = verbose

        # --- 策略状态 ---
        self.cash = capital
        self.total_qty = 0
        self.open_positions = []
        self.bought_levels = set()
        self.trades = []
        self.levels = []
        self.lower = 0
        self.upper = 0
        self.step = 0
        self.trade_qty_per_grid = 0.0

        # --- 盈利与成本跟踪 ---
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.profit_pool = 0.0

        # --- 网格移动控制 ---
        self.shift_count = 0
        self.reference_ma = 0.0
        self.highest_sell_level_watermark = 0.0
        # 使用 .get() 方法，如果字典中没有这个键，就使用一个安全的默认值
        self.REINVESTMENT_THRESHOLD = strategy_params.get(
            "reinvest_threshold", 70)
        self.FORCE_MOVE_BARS = strategy_params.get("force_move_bars", 360)
        self.breakout_buffer = strategy_params.get("breakout_buffer", 0.01)
        self.ma_change_threshold = strategy_params.get(
            "ma_change_threshold", 0.01)
        self.shift_ratio = strategy_params.get("shift_ratio", 0.01)
        self.outside_bars = 0

    def _log_trade(self, timestamp, side, level_price, linked_info, watermark_snapshot, avg_cost,
                   qty, amount_usdt, profit, close_price, positions_snapshot, levels_snapshot, profit_pool_snapshot, cash_snapshot, single_trade_fee, total_fees_snapshot):
        # 在 GridTrader 类内部

        def _format_positions_snapshot(positions_list):
            """将持仓列表格式化为一个易读的字符串。"""
            if not positions_list:
                return "[]"

            # 使用我们之前讨论的方法二
            formatted_items = [
                f"({price:.2f}, {qty:.3f})" for price, qty in positions_list]
            return f"[{', '.join(formatted_items)}]"

        """统一的交易日志记录函数"""
        total_qty_snapshot = sum(item[1] for item in positions_snapshot)
        grid_range_str = f"{levels_snapshot[0]:.2f}-{levels_snapshot[-1]:.2f}" if levels_snapshot else "N/A"

        formatted_positions_str = _format_positions_snapshot(
            positions_snapshot)

        clean_levels = [float(lv)
                        for lv in levels_snapshot] if levels_snapshot else []
        levels_snapshot_str = str(clean_levels)
        # ✅ 新增计算：计算当时的总资产
        # =========================================================================
        positions_value = total_qty_snapshot * close_price
        total_capital = cash_snapshot + positions_value + profit_pool_snapshot
        # =========================================================================

        log_entry = (
            timestamp,          # 交易时间
            side,               # 买/卖/事件
            level_price,        # 网格价位
            linked_info,        # 买入时填 market@xx，卖出时填开仓价
            watermark_snapshot,  # ✅ 新增: 水位线快照
            avg_cost,           # 平均成本
            qty,                # 成交数量
            amount_usdt,        # 成交金额 (不含手续费)
            cash_snapshot,          # 当前现金余额
            total_qty_snapshot,  # 当前持仓总量
            profit,             # 卖出利润 (买入时为 None)
            profit_pool_snapshot,  # ✅ 新增: 利润池快照
            single_trade_fee,  # ✅ 新增: 当笔手续费
            total_fees_snapshot,
            total_capital,      # ✅ 新增: 总资产快照
            close_price,        # 当前收盘价
            grid_range_str,     # 网格区间
            formatted_positions_str,  # 持仓快照
            levels_snapshot_str     # 网格快照
        )
        self.trades.append(log_entry)

    def _execute_buy(self, level_price, buy_price, qty_to_buy, timestamp, positions_snapshot, levels_snapshot, close_price, side="BUY", modify_global_state=True):
        if buy_price <= 0 or qty_to_buy <= 0:
            return None
        watermark_snapshot_before = self.highest_sell_level_watermark

        raw_cost = qty_to_buy * buy_price
        fee = raw_cost * self.fee_rate
        total_cost = raw_cost + fee

        if self.cash < total_cost:
            # ✅ 新增：调用 _log_trade 记录失败事件
            self._log_trade(
                timestamp=timestamp, side="BUY_FAIL", level_price=level_price,
                linked_info=f"Cash needed: {total_cost:.2f}",
                watermark_snapshot=self.highest_sell_level_watermark,
                avg_cost=None, qty=qty_to_buy, amount_usdt=None, profit=None,
                close_price=close_price,
                positions_snapshot=sorted(
                    [(p['price'], p['qty']) for p in self.open_positions]),
                levels_snapshot=self.levels, profit_pool_snapshot=self.profit_pool,
                cash_snapshot=self.cash, single_trade_fee=None, total_fees_snapshot=self.total_fees
            )
            return None  # 保持原有逻辑，返回 None 表示失败
        self.cash -= total_cost
        self.total_fees += fee

        new_position = {"price": level_price,
                        "qty": qty_to_buy, "cost": total_cost, "avg_cost": total_cost / qty_to_buy if qty_to_buy > 0 else 0}
        if modify_global_state:
            self.open_positions.append(new_position)
            self.bought_levels.add(level_price)
        final_positions_snapshot = positions_snapshot + \
            [(level_price, qty_to_buy)]
        cash_snapshot_after = self.cash
        profit_pool_snapshot_after = self.profit_pool
        total_fees_snapshot_after = self.total_fees
        # 日志记录交易
        self._log_trade(timestamp, side, level_price, buy_price, watermark_snapshot_before,
                        new_position["avg_cost"], qty_to_buy, raw_cost,
                        None, close_price, sorted(final_positions_snapshot), levels_snapshot, profit_pool_snapshot=profit_pool_snapshot_after, cash_snapshot=cash_snapshot_after, single_trade_fee=fee,
                        total_fees_snapshot=total_fees_snapshot_after)
        return new_position

    def _execute_sell(self, position, sell_price, timestamp, positions_snapshot_before, levels_snapshot, close_price, profit_to_pool, side="SELL", modify_global_state=True):
        trade_qty = position["qty"]
        proceeds = trade_qty * sell_price
        fee = proceeds * self.fee_rate
        self.total_fees += fee
        # 净收入（已扣手续费），把它加回 cash（为下次买入保留手续费）
        net_proceeds = proceeds - fee

        # 本次卖出的净利润（相对于开仓成本 position['cost']）
        single_profit = net_proceeds - position["cost"]

        # 将净收入计入 cash
        self.cash += net_proceeds

        # 如果要把利润划入利润池，通常只划正利润（更保守）
        if profit_to_pool and single_profit > 0:
            # 从 cash 中划出正利润到 profit_pool
            self.cash -= single_profit
            self.profit_pool += single_profit

        # 记录已实现盈亏（净利润，可能为负）
        self.realized_pnl += single_profit

        position_price_to_remove = position["price"]
        final_positions_snapshot = [
            (price, qty) for price, qty in positions_snapshot_before if price != position_price_to_remove
        ]
        cash_snapshot_after = self.cash
        profit_pool_snapshot_after = self.profit_pool
        total_fees_snapshot_after = self.total_fees

        self.highest_sell_level_watermark = max(
            self.highest_sell_level_watermark, sell_price)
        watermark_snapshot_after = self.highest_sell_level_watermark

        self._log_trade(timestamp, side, round(sell_price, 2), position["price"], watermark_snapshot_after,
                        position["avg_cost"], trade_qty, proceeds,
                        single_profit, close_price, final_positions_snapshot, levels_snapshot, profit_pool_snapshot_after,
                        cash_snapshot_after, single_trade_fee=fee,
                        total_fees_snapshot=total_fees_snapshot_after)

        # === 是否修改全局仓位 ===
        if modify_global_state:
            try:
                self.open_positions.remove(position)
            except ValueError:
                pass
            self.bought_levels.discard(position["price"])

        return True

    def _compute_trade_qty_per_grid(self, capital, price, fee_rate, deploy, reserve, safety_factor=0.999):
        """
        (V3 - 混合稳健版)
        根据总资本，使用混合成本模型计算标准交易单位。
        - 部署区(deploy): 按当前市价(price)计算成本，因为这是立即发生的交易。
        - 储备区(reserve): 按各自的格子价(lv)计算成本，以更精确地预估未来资金需求。
        - 引入安全系数(safety_factor)来抵消模型的乐观性，保证稳健。

        :param capital: 可用于规划的总资产
        :param price: 当前市价，用于计算部署区的成本
        :param fee_rate: 手续费率
        :param deploy: 一个包含【需要立即持仓】的格子价格的集合
        :param reserve: 一个包含【需要预留现金】的格子价格的集合
        :param safety_factor: 安全系数，用于轻微下调最终结果以增加稳健性
        """

        # ✅ 核心修正：增加防御性校验和日志
        # =========================================================================
        deploy = set(deploy)
        reserve = set(reserve)

        # 确保 deploy 和 reserve 没有交集
        assert deploy.isdisjoint(reserve), "逻辑错误: deploy 和 reserve 集合存在交集！"

        # 1. 计算部署区的总成本系数
        # 这部分的成本是确定的，必须按当前价计算
        deploy_cost_factor = len(deploy) * price * (1 + fee_rate)

        # 2. 计算储备区的总成本系数
        # 这部分的成本是基于未来的、更优价格的估算
        reserve_cost_factor = sum(lv * (1 + fee_rate) for lv in reserve)

        # 3. 计算总成本系数
        total_cost_factor = deploy_cost_factor + reserve_cost_factor

        if total_cost_factor > 1e-9:
            # 4. 解出标准交易单位 Q
            Q = capital / total_cost_factor
            # 5. ✅ 关键：应用安全系数，为模型的乐观性买一份保险
            Q_adjusted = Q * safety_factor
        else:
            Q_adjusted = 0

        return Q_adjusted

    def _redistribute_positions(self, current_price, timestamp, old_levels_snapshot):
        """
        (V6 - 数量本位 + 渐进式迁移)
        1. 根据总净值，计算出新的“标准交易单位”(trade_qty_per_grid)。
        2. 以此为标准，通过“渐进式”的迁移（优先利用旧仓位），完成对新持仓的部署。
        """
        # === Step 1 & 2: 资产盘点并计算新的“标准交易单位” ===
        total_positions_value = sum(
            p['qty'] * current_price for p in self.open_positions)
        total_asset_value = self.cash + total_positions_value

        # ✅ 核心修正：根据不同的场景，定义不同的“应持仓”范围
        # =========================================================================
        highest_level = self.levels[-1]

        deploy_levels = {
            lv for lv in self.levels
            if lv >= current_price and lv != highest_level}
        reserve_levels = {
            lv for lv in self.levels if lv not in deploy_levels and lv != highest_level}

        old_qty = self.trade_qty_per_grid
        self.trade_qty_per_grid = self._compute_trade_qty_per_grid(
            total_asset_value, current_price, self.fee_rate, deploy_levels, reserve_levels)

        # ✅ [新增] 记录 Q 值计算的详细依据
        # =========================================================================
        if self.verbose:
            self._log_trade(
                timestamp=timestamp,
                side="Q_CALC_INFO",
                level_price=f"Deploy {len(deploy_levels)}:({sorted(deploy_levels)})",
                linked_info=f"Reserve {len(reserve_levels)}:({sorted(reserve_levels)})",
                watermark_snapshot=self.highest_sell_level_watermark,
                avg_cost=None, qty=self.trade_qty_per_grid, amount_usdt=None, profit=None,
                close_price=current_price,
                # 为了简洁，这里的快照可以简化或传递当时的状态
                positions_snapshot=sorted(
                    [(p['price'], p['qty']) for p in self.open_positions]),
                levels_snapshot=self.levels,
                profit_pool_snapshot=self.profit_pool,
                cash_snapshot=self.cash,
                single_trade_fee=None, total_fees_snapshot=self.total_fees
            )
        # =========================================================================

        # ✅ 替换 print: 记录“宏观调仓”的启动事件
        # =========================================================================
        if self.verbose:
            self._log_trade(
                timestamp=timestamp, side="REDIST_START",
                level_price=f"Q:{old_qty:.9f} -> {self.trade_qty_per_grid:.9f}",
                linked_info=f"{old_levels_snapshot[0]:.2f}-{old_levels_snapshot[-1]:.2f} -> {self.levels[0]:.2f}-{self.levels[-1]:.2f}",
                watermark_snapshot=self.highest_sell_level_watermark,
                avg_cost=None, qty=None, amount_usdt=None, profit=None,
                close_price=current_price,
                positions_snapshot=sorted(
                    [(p['price'], p['qty']) for p in self.open_positions]),
                levels_snapshot=self.levels,
                # 此时 profit_pool 已为 0
                profit_pool_snapshot=self.profit_pool,
                cash_snapshot=self.cash,
                single_trade_fee=None,
                total_fees_snapshot=self.total_fees
            )
        # =========================================================================

        # === ✅ [核心修改] Step 3: "渐进式迁移" 执行逻辑 ===

        # 3.1 打包现有持仓，作为“可分配的资产池”
        survivors_pool = sorted(
            [{"qty": p["qty"], "price": p["price"], "cost": p["cost"], "avg_cost": p["avg_cost"]}
             for p in self.open_positions if p["qty"] > 1e-9],
            key=lambda x: x["avg_cost"]  # 按成本从低到高排序，优先保留低成本仓位
        )

        new_positions = []

        # 3.2 遍历所有【新的持仓目标格】，用“资产池”和“现金”去填满它们
        for lv in sorted(list(deploy_levels), reverse=True):
            qty_needed = self.trade_qty_per_grid  # 每个目标格都需要这么多数量的币

            cost_from_survivors = 0.0
            qty_from_survivors = 0.0

            # 优先从“资产池”里分配
            while qty_needed > 1e-9 and survivors_pool:
                sp = survivors_pool[0]  # 从成本最低的旧仓位开始拿
                take_qty = min(sp["qty"], qty_needed)
                take_cost = take_qty * sp["avg_cost"]  # 成本按旧仓位的平均成本计算

                qty_from_survivors += take_qty
                cost_from_survivors += take_cost
                sp["qty"] -= take_qty
                qty_needed -= take_qty

                if sp["qty"] < 1e-9:
                    survivors_pool.pop(0)  # 如果这个旧仓位被掏空了，就扔掉

            # 如果“资产池”不够用，就动用现金去市场上补仓
            bought_position_part = None
            if qty_needed > 1e-9:
                # ✅ 核心修正：在这里“内联”执行买入逻辑，不再调用 _execute_buy
                # =================================================================
                raw_cost = qty_needed * current_price
                fee = raw_cost * self.fee_rate
                total_cost = raw_cost + fee

                if self.cash + 1e-9 >= total_cost:
                    self.cash -= total_cost
                    self.total_fees += fee
                    bought_position_part = {
                        "price": lv, "qty": qty_needed, "cost": total_cost,
                        "avg_cost": total_cost / qty_needed if qty_needed > 0 else 0
                    }
                else:
                    # ✅ 替换 print: 记录“补仓失败”的日志
                    # =============================================================
                    self._log_trade(
                        timestamp, "REDIST_BUY_FAIL", lv, f"Cash needed: {total_cost:.2f}", self.highest_sell_level_watermark,
                        None, qty_needed, None, None, current_price,
                        sorted([(p['price'], p['qty'])
                               for p in new_positions]),
                        self.levels, self.profit_pool, self.cash, single_trade_fee=None,
                        total_fees_snapshot=self.total_fees
                    )
                    # =============================================================

            # 汇总新仓位的信息
            final_qty = qty_from_survivors + \
                (bought_position_part['qty'] if bought_position_part else 0)
            final_cost = cost_from_survivors + \
                (bought_position_part['cost'] if bought_position_part else 0)

            if final_qty > 1e-9:
                final_avg_cost = final_cost / final_qty
                new_positions.append({
                    "price": lv,
                    "qty": final_qty,
                    "cost": final_cost,
                    "avg_cost": final_avg_cost
                })

        # 3.3 将“资产池”里剩余的、未被分配的旧仓位全部卖掉
        if survivors_pool:
            # ✅ [新增] 用于汇总总账的变量
            # =====================================================================
            total_qty_sold = 0
            total_proceeds = 0
            total_cost_of_sold = 0
            sold_positions_info = []  # 用于记录被卖掉的仓位的细节
            # =====================================================================

            for sp in survivors_pool:
                # “内联”执行卖出逻辑，只更新资金，不修改全局持仓
                proceeds = sp['qty'] * current_price
                fee = proceeds * self.fee_rate
                net_proceeds = proceeds - fee

                self.cash += net_proceeds
                self.total_fees += fee

                # ✅ [新增] 累加总账数据
                # =====================================================================
                total_qty_sold += sp['qty']
                total_proceeds += proceeds
                cost_of_this_position = sp['qty'] * sp['avg_cost']
                total_cost_of_sold += cost_of_this_position
                sold_positions_info.append(
                    f"({sp['price']:.2f}, {sp['qty']:.3f})")

                # 计算这笔“模拟”卖出的利润，并累加到全局的 realized_pnl
                single_profit = net_proceeds - cost_of_this_position
                self.realized_pnl += single_profit
                # =====================================================================

            # ✅ [新增] 在循环结束后，记录一笔总账
            # =====================================================================
            if total_qty_sold > 1e-9:
                total_profit = (total_proceeds *
                                (1 - self.fee_rate)) - total_cost_of_sold
                avg_cost_of_sold = total_cost_of_sold / \
                    total_qty_sold if total_qty_sold > 0 else 0

                # 构建一个“交易后”的快照。此时 new_positions 已经是最终持仓了。
                final_positions_snapshot = sorted(
                    [(p['price'], p['qty']) for p in new_positions])
                total_fee_for_leftovers = total_proceeds * self.fee_rate

                if self.verbose:
                    self._log_trade(
                        timestamp=timestamp,
                        side="REDIST_SELL_LEFTOVER",
                        # 借用列显示描述
                        level_price=f"Sold {len(sold_positions_info)} leftover(s) @{current_price}",
                        linked_info=', '.join(
                            sold_positions_info),  # 借用列显示被卖掉的仓位
                        watermark_snapshot=self.highest_sell_level_watermark,
                        avg_cost=avg_cost_of_sold,  # 记录被卖掉部分的总平均成本
                        qty=total_qty_sold,
                        amount_usdt=total_proceeds,
                        profit=total_profit,
                        close_price=current_price,
                        positions_snapshot=final_positions_snapshot,
                        levels_snapshot=self.levels,  # 记录当时的旧网格
                        profit_pool_snapshot=self.profit_pool,  # profit_pool 在此函数开头已清零
                        cash_snapshot=self.cash,  # 传递更新后的 cash
                        single_trade_fee=total_fee_for_leftovers,  # ✅ 新增: 传递这批卖出的总手续费
                        total_fees_snapshot=self.total_fees
                    )
            # =====================================================================

        # === Step 5: 更新仓位 ===
        self.open_positions = new_positions
        self.bought_levels = {p["price"] for p in self.open_positions}

    def _process_bar_trades(self, o, h, l, c, ts):
        # 提前排序仓位，减少循环中的开销
        sorted_positions = sorted(
            self.open_positions, key=lambda x: x['price'])

        # 分段遍历价格路径
        segments = [(o, h), (h, l), (l, c)]
        for seg_start, seg_end in segments:
            if seg_start == seg_end:
                continue

            # ========= 上涨段：检查卖出 =========
            if seg_start < seg_end:
                for p in sorted_positions:
                    if p not in self.open_positions:
                        continue  # 可能已被卖出，跳过

                    # --- 优先用 levels.index() 查找下一个格子 ---
                    next_level = None
                    try:
                        idx = self.levels.index(p['price'])
                        if idx + 1 < len(self.levels):
                            next_level = self.levels[idx + 1]
                    except ValueError:
                        # --- fallback：用 min() 查找更大的格子 ---
                        next_level = min(
                            (lv for lv in self.levels if lv > p['price']), default=None)

                    # 如果 next_level 被价格路径穿越 → 卖出
                    if next_level and seg_start < next_level <= seg_end:
                        if self._execute_sell(position=p,
                                              sell_price=next_level,
                                              timestamp=ts,
                                              positions_snapshot_before=sorted(
                                                  [(pos['price'], pos['qty']) for pos in self.open_positions if pos != p]),
                                              levels_snapshot=self.levels,
                                              close_price=c,
                                              profit_to_pool=True):

                            self.levels_sold_this_bar.add(next_level)

            # ========= 下跌段：检查买入 =========
            else:
                highest_level = self.levels[-1]  # 获取最高网格线
                touched = [
                    lv for lv in self.levels if seg_end <= lv < seg_start]
                for lv in sorted(touched, reverse=True):
                    if lv == highest_level or lv in self.bought_levels or lv in self.levels_sold_this_bar or lv in self.levels_bought_this_bar or lv >= self.highest_sell_level_watermark:
                        continue
                    if self._execute_buy(level_price=lv,
                                         buy_price=lv,
                                         qty_to_buy=self.trade_qty_per_grid,
                                         timestamp=ts,
                                         # 快照参数应该反映“即将发生”交易前的状态
                                         positions_snapshot=sorted(
                                             [(p['price'], p['qty']) for p in self.open_positions]),
                                         levels_snapshot=self.levels,
                                         close_price=c,  # <--- 补上这个缺失的参数
                                         side="BUY"
                                         ):
                        self.levels_bought_this_bar.add(lv)

    def _initial_setup(self, df):
        """
        (V2 - 独立且清晰版)
        处理初始建仓，并记录详细日志。
        """
        if df.empty:
            return

        init_price = df.iloc[0]['close']
        init_ts = df.iloc[0]['datetime']
        self.highest_sell_level_watermark = init_price

        # === Step 1: 确定建仓目标 ===
        highest_level = self.levels[-1]
        deploy_levels = {lv for lv in self.levels if lv >=
                         init_price and lv != highest_level}
        reserve_levels = {
            lv for lv in self.levels if lv not in deploy_levels and lv != highest_level}

        # === Step 2: 计算初始的“标准交易单位” ===
        # 初始建仓时，总资产就是初始资本
        self.trade_qty_per_grid = self._compute_trade_qty_per_grid(
            self.capital, init_price, self.fee_rate,
            deploy_levels, reserve_levels
        )

        # ✅ 记录“初始化开始”事件
        # =========================================================================
        self._log_trade(
            timestamp=init_ts, side="INIT_START",
            level_price=f"Initial Q={self.trade_qty_per_grid:.6f}",
            linked_info=f"Deploying on {len(deploy_levels)} grids",
            watermark_snapshot=self.highest_sell_level_watermark,
            avg_cost=None, qty=None, amount_usdt=None, profit=None,
            close_price=init_price,
            positions_snapshot=[],  # 初始时没有持仓
            levels_snapshot=self.levels,
            profit_pool_snapshot=self.profit_pool,
            cash_snapshot=self.cash,
            single_trade_fee=None,
            total_fees_snapshot=self.total_fees
        )
        # =========================================================================

        # === Step 3: 逐个执行建仓 ===
        if deploy_levels:
            if self.verbose:
                print(f"初始建仓   -> 计划在 {len(deploy_levels)} 个网格上建仓...")

            for lv in sorted(list(deploy_levels)):
                # 这里我们【必须】调用 _execute_buy，因为它能正确处理资金和日志
                # 我们传递 modify_global_state=True，让它直接更新全局持仓
                self._execute_buy(
                    level_price=lv,
                    buy_price=init_price,
                    qty_to_buy=self.trade_qty_per_grid,
                    timestamp=init_ts,
                    # 传递【交易前】的快照，_execute_buy 内部会构建【交易后】的快照
                    positions_snapshot=sorted(
                        [(p['price'], p['qty']) for p in self.open_positions]),
                    levels_snapshot=self.levels,
                    close_price=init_price,
                    side="INIT_BUY",
                    modify_global_state=True
                )

        # === Step 4: 初始化MA参考点 ===
        first_ma = df.iloc[0][f'ma_{self.ma_period}']
        self.reference_ma = first_ma if not pd.isna(first_ma) else init_price
        if self.verbose:
            print(f"{init_ts} 📌 初始 MA 参考点设为: {self.reference_ma:.2f}")

        # ✅ 记录“初始化完成”事件
        # =========================================================================
        if self.verbose:
            self._log_trade(
                timestamp=init_ts, side="INIT_DONE",
                level_price="Initial setup complete",
                linked_info=None, watermark_snapshot=self.highest_sell_level_watermark, avg_cost=None, qty=None, amount_usdt=None, profit=None,
                close_price=init_price,
                # 传递【最终】的快照
                positions_snapshot=sorted(
                    [(p['price'], p['qty']) for p in self.open_positions]),
                levels_snapshot=self.levels,
                profit_pool_snapshot=self.profit_pool,
                cash_snapshot=self.cash,
                single_trade_fee=None,
                total_fees_snapshot=self.total_fees
            )
        # =========================================================================

    def _check_and_handle_grid_shift(self, h, l, c, ts, current_ma):
        """
        检查并处理网格移动的逻辑。
        返回 True 如果发生了移动，否则返回 False。
        """
        boundary_changed, shift_direction = False, None
        # --- 主逻辑：边界突破 + 动能过滤 ---
        if self.reference_ma is not None and not pd.isna(current_ma):
            ma_roc_from_ref = (
                current_ma - self.reference_ma) / self.reference_ma
            if h > self.upper * (1 + self.breakout_buffer) and ma_roc_from_ref >= self.ma_change_threshold:
                shift_direction = "UP"
            elif l < self.lower * (1 - self.breakout_buffer) and ma_roc_from_ref <= -self.ma_change_threshold:
                shift_direction = "DOWN"
        # --- 兜底逻辑：长时间悬空强制移动 ---
        self.outside_bars += 1 if (c >
                                   self.upper or c < self.lower) else 0
        if self.outside_bars >= self.FORCE_MOVE_BARS:
            if c > self.upper:
                shift_direction = "UP_FORCED"
            elif c < self.lower:
                shift_direction = "DOWN_FORCED"
            self.outside_bars = 0  # 重置计数器

        # --- 如果确定要移动，则执行 ---
        if shift_direction:
            old_levels = self.levels
            old_reference_ma = self.reference_ma  # ✅ [新增] 在改变前，记下旧值
            old_watermark = self.highest_sell_level_watermark  # ✅ [新增] 记下旧值

            if shift_direction == "UP" or shift_direction == "UP_FORCED":
                target_lower, target_upper = self.lower * \
                    (1+self.shift_ratio), self.upper * (1+self.shift_ratio)
            else:  # SHIFT_DOWN
                target_lower, target_upper = self.lower * \
                    (1-self.shift_ratio), self.upper * (1-self.shift_ratio)

            self.levels, self.step = build_levels(
                target_lower, target_upper, self.n_grids)

            if not self.levels:
                return False

            self.lower, self.upper = self.levels[0], self.levels[-1]
            self.highest_sell_level_watermark = c

            self._redistribute_positions(
                c, ts, old_levels)

            self.shift_count += 1
            boundary_changed = True
            self.reference_ma = current_ma
            # ✅ [新增] 专门为 reference_ma 的变化记录一条日志
            # =====================================================================
            if self.verbose and old_reference_ma != self.reference_ma:
                self._log_trade(
                    timestamp=ts,
                    side="MA_REF_UPDATE",
                    level_price=f"{old_reference_ma:.2f} -> {self.reference_ma:.2f}",
                    watermark_snapshot=self.highest_sell_level_watermark,
                    linked_info=f"Triggered by {shift_direction}",
                    # --- 以下都是占位符 ---
                    avg_cost=None, qty=None, amount_usdt=None, profit=None,
                    close_price=c,
                    positions_snapshot=sorted(
                        [(p['price'], p['qty']) for p in self.open_positions]),
                    levels_snapshot=self.levels,
                    profit_pool_snapshot=self.profit_pool,
                    cash_snapshot=self.cash,
                    single_trade_fee=None,
                    total_fees_snapshot=self.total_fees
                )
            # =====================================================================
            # 在所有状态更新完毕后，记录一个总的 SHIFT 事件
            if self.verbose:
                arrow = "▲" if "UP" in shift_direction else "▼"
                self._log_trade(timestamp=ts, side=f"SHIFT_{shift_direction}_DONE", level_price=f"{arrow} Grid Shifted", watermark_snapshot=self.highest_sell_level_watermark,
                                linked_info=f"{old_levels[0]:.2f}-{old_levels[-1]:.2f} -> {self.lower:.2f}-{self.upper:.2f}",
                                avg_cost=None, qty=None, amount_usdt=None, profit=None, close_price=c, positions_snapshot=sorted([(p['price'], p['qty']) for p in self.open_positions]),
                                levels_snapshot=self.levels,  profit_pool_snapshot=self.profit_pool, cash_snapshot=self.cash, single_trade_fee=None,  # ✅ 新增: 非交易事件，无单笔费用
                                total_fees_snapshot=self.total_fees
                                )

        return boundary_changed

    def _check_and_handle_reinvestment(self, c, ts):
        """
        (V3 - 基于独立的 compute_trade_qty_per_grid 函数)
        检查并处理利润复投的逻辑。
        1. 优先用利润补足现有持仓至新的“标准交易单位”。
        2. 剩余利润自动并入现金池，增强未来购买力。
        """
        if self.profit_pool >= self.REINVESTMENT_THRESHOLD:

            # --- 暂存旧状态以供日志记录 ---
            reinvest_amount = self.profit_pool
            old_qty = self.trade_qty_per_grid

            # --- Step 1: 统一资金来源 (采纳您的核心建议) ---
            # 立即将利润池资金并入主现金池，并清空利润池
            self.cash += self.profit_pool
            self.profit_pool = 0.0

            # ✅ 替换 print: 用一个 "BEFORE" 事件来记录
            if self.verbose:
                self._log_trade(
                    ts, "REINVEST_START", f"Pool: {reinvest_amount:.2f}",
                    f"Threshold: {self.REINVESTMENT_THRESHOLD}",
                    self.highest_sell_level_watermark,
                    None, None, None, None, c,
                    sorted([(p['price'], p['qty'])
                           for p in self.open_positions]),
                    self.levels, reinvest_amount, self.cash, single_trade_fee=None, total_fees_snapshot=self.total_fees
                )

            # 1.1 假设利润已全部注入，计算理想中的总资产
            total_positions_value = sum(
                p['qty'] * c for p in self.open_positions)
            temp_total_asset_value = self.cash + total_positions_value

            # 1.2【调用函数】计算新的“目标标准交易单位” (new_target_qty)
            #     我们需要为你的函数准备正确的 deploy 和 reserve 集合
            highest_level = self.levels[-1]
            #     在复投场景下，我们假设所有低于最高格的格子都是目标
            all_potential_grids = {
                lv for lv in self.levels if lv != highest_level}
            deploy_for_calc = {p['price'] for p in self.open_positions}
            reserve_for_calc = {
                lv for lv in all_potential_grids if lv not in deploy_for_calc}

            new_target_qty = self._compute_trade_qty_per_grid(
                temp_total_asset_value, c, self.fee_rate,
                deploy_for_calc, reserve_for_calc
            )
            # ✅ [新增] 记录 Q 值计算的详细依据
            # =========================================================================
            if self.verbose:
                self._log_trade(
                    timestamp=ts,
                    side="Q_CALC_INFO",
                    level_price=f"Deploy {len(deploy_for_calc)}:({sorted(deploy_for_calc)})",
                    linked_info=f"Reserve {len(reserve_for_calc)}:({sorted(reserve_for_calc)})",
                    watermark_snapshot=self.highest_sell_level_watermark,
                    avg_cost=None, qty=new_target_qty, amount_usdt=None, profit=None,
                    close_price=c,
                    positions_snapshot=sorted(
                        [(p['price'], p['qty']) for p in self.open_positions]),
                    levels_snapshot=self.levels,
                    profit_pool_snapshot=reinvest_amount,  # 这是复投前的
                    cash_snapshot=self.cash,
                    single_trade_fee=None, total_fees_snapshot=self.total_fees
                )
            # =========================================================================

            # === Step 2: 计算补足现有持仓所需的成本 ===
            cash_needed_for_add_on = 0
            qty_to_add_per_position = {}

            for p in self.open_positions:
                qty_diff = new_target_qty - p['qty']
                if qty_diff > 1e-9:
                    qty_to_add_per_position[p['price']] = qty_diff
                    cash_needed_for_add_on += (qty_diff * c) * \
                        (1 + self.fee_rate)

            # === Step 3: 决策与执行 ===
            executed_reinvestment = False  # 用于日志记录的标志位
            eps = 1e-8
            if self.cash + eps >= cash_needed_for_add_on:
                executed_reinvestment = True
                # # 利润充足，执行补仓
                # 3.2 遍历并执行补仓
                for p in self.open_positions:
                    if p['price'] in qty_to_add_per_position:
                        qty_to_add = qty_to_add_per_position[p['price']]

                        raw_cost = qty_to_add * c
                        fee = raw_cost * self.fee_rate
                        total_cost = raw_cost + fee

                        if self.cash + 1e-9 >= total_cost:
                            # 1. 记录【交易前】的快照
                            positions_snapshot_before = sorted(
                                [(pos['price'], pos['qty']) for pos in self.open_positions])
                            # 1. 更新资金
                            self.cash -= total_cost
                            self.total_fees += fee

                            # 2. 构建“交易后”的快照，用于日志

                            snapshot_after_add = []
                            for price, qty in positions_snapshot_before:
                                if price == p['price']:
                                    snapshot_after_add.append(
                                        (price, qty + qty_to_add))
                                else:
                                    snapshot_after_add.append((price, qty))

                            # 3. 记录日志！
                            if self.verbose:
                                self._log_trade(
                                    ts, "REINVEST_ADD", p['price'], c,
                                    self.highest_sell_level_watermark,
                                    (p['cost'] + total_cost) /
                                    (p['qty'] + qty_to_add),  # 预估的新avg_cost
                                    qty_to_add, raw_cost, None, c,
                                    snapshot_after_add, self.levels,
                                    profit_pool_snapshot=self.profit_pool,  # 此时 profit_pool 已为 0
                                    cash_snapshot=self.cash,  # 传递扣款后的 cash
                                    single_trade_fee=fee,  # <-- 传递当笔补仓的手续费
                                    total_fees_snapshot=self.total_fees
                                )

                            # 4. 最后，才更新真实的仓位信息
                            p['qty'] += qty_to_add
                            p['cost'] += total_cost
                            p['avg_cost'] = p['cost'] / \
                                p['qty'] if p['qty'] > 0 else 0
                        # =============================================================

                # 3.3 更新全局标准
                old_qty = self.trade_qty_per_grid
                self.trade_qty_per_grid = new_target_qty

            else:
                # 利润不足，本次跳过，等待下次
                pass

            # 记录一个完成/跳过事件
            if self.verbose:
                if executed_reinvestment:
                    log_side = "REINVEST_DONE"
                    log_msg = f"Q: {old_qty:.9f} -> {self.trade_qty_per_grid:.9f} Pool used: {reinvest_amount:.2f}"
                else:
                    log_side = "REINVEST_SKIPPED"
                    log_msg = f"Cash insufficient. Needed ${cash_needed_for_add_on:.2f}, Have ${self.cash:.2f}; Pool was {reinvest_amount:.2f}"
                self._log_trade(
                    ts, log_side, log_msg,
                    None, self.highest_sell_level_watermark, None, None, None, None, c,
                    positions_snapshot=sorted(
                        [(p['price'], p['qty']) for p in self.open_positions]),
                    levels_snapshot=self.levels,
                    profit_pool_snapshot=self.profit_pool,
                    cash_snapshot=self.cash,
                    single_trade_fee=None, total_fees_snapshot=self.total_fees
                )

    def simulate(self, df):
        # 1. 初始化网格和状态
        self.levels, self.step = build_levels(
            self.initial_lower, self.initial_upper, self.n_grids)
        if not self.levels:
            return self.trades, self.realized_pnl, self.capital, self.total_fees, self.shift_count, self.open_positions
        self.lower, self.upper = self.levels[0], self.levels[-1]

        if self.verbose:
            print("回测开始...")

        # 2. 初始建仓 (使用一个专门的私有方法)
        self._initial_setup(df)
        # --- 结束初始建仓 ---

        # 3. 主循环 (现在是纯粹的流程编排)
        ma_col_name = f'ma_{self.ma_period}'
        data_arrays = {col: df[col].to_numpy() for col in [
            'open', 'high', 'low', 'close', 'datetime', ma_col_name]}

        for i in range(1, len(df)):
            o, h, l, c, ts, current_ma = (data_arrays[col][i] for col in [
                'open', 'high', 'low', 'close', 'datetime', ma_col_name])

            # 【核心修正】将常规交易逻辑的调用放在这里
            self.levels_sold_this_bar = set()
            self.levels_bought_this_bar = set()
            # 步骤 3.1: 检查并处理网格移动
            self._check_and_handle_grid_shift(h, l, c, ts, current_ma)
            # 步骤 3.3: 检查并处理利润复投
            self._check_and_handle_reinvestment(c, ts)

            # 步骤 3.2: 执行常规的买卖交易
            self._process_bar_trades(o, h, l, c, ts)

        # === 最终结算 ===
        final_equity = self.cash + self.profit_pool + sum(p['qty'] * df.iloc[-1]['close']
                                                          for p in self.open_positions)
        return self.trades, self.realized_pnl, final_equity, self.total_fees, self.shift_count, self.open_positions

# ==============================================================================
# 5. 主程序/业务流程编排 (重构版)
# ==============================================================================


def setup_backtest_data(config):
    """
    数据准备的总函数：负责加载、缓存、计算指标并切片。

    :param config: 包含所有配置的字典。
    :return: 准备好的、可用于回测的 DataFrame，如果失败则返回 None。
    """
    try:
        # --- 1. 数据加载与缓存 ---
        preload_start_date = datetime.strptime(
            config["start_date"], "%Y-%m-%d") - timedelta(minutes=config["ma_period"], days=1)
        preload_start_date_str = preload_start_date.strftime("%Y-%m-%d")

        data_filename = f"{config['symbol']}_{config['interval']}_{preload_start_date_str}_to_{config['end_date']}.csv"
        db_filename = f"{config['symbol'].lower()}_{config['interval']}_data.db"

        if os.path.exists(data_filename):
            print(f"加载本地CSV缓存: '{data_filename}'")
            df_full = pd.read_csv(data_filename)
            df_full['datetime'] = pd.to_datetime(df_full['datetime'])
        else:
            print(f"CSV缓存不存在, 从SQLite数据库加载...")
            df_full = load_from_sqlite(
                db_filename,  # 数据库名可以加入config
                config["symbol"],
                preload_start_date_str,
                config["end_date"]
            )
            if not df_full.empty:
                print(f"数据已加载并缓存到 '{data_filename}'")
                df_full.to_csv(data_filename, index=False)

            # ✅ 核心修正：在保存CSV之前，先计算指标！
            # =================================================================
            print("首次加载数据，正在计算并缓存指标...")
            df_full = add_indicators(df_full, period=config["ma_period"])

            print(f"数据及指标已加载，正在创建缓存文件 '{data_filename}'")
            df_full.to_csv(data_filename, index=False)
            # =================================================================

        if df_full.empty:
            print("错误:未能获取任何K线数据。")
            return None

        # --- 2. 指标计算 ---
        ma_col = f"ma_{config['ma_period']}"
        if ma_col not in df_full.columns or df_full[ma_col].isnull().all():
            print(f"警告：CSV缓存 '{data_filename}' 中缺少指标，正在重新计算...")
            df_full = add_indicators(df_full, period=config["ma_period"])

        # --- 3. 数据切片 ---
        start_bound = pd.to_datetime(config["start_date"])
        end_bound = pd.to_datetime(config["end_date"]) + pd.Timedelta(days=1)
        df_backtest = df_full[
            (df_full['datetime'] >= start_bound) & (
                df_full['datetime'] < end_bound)
        ].copy().reset_index(drop=True)

        print(f"数据准备完成: {len(df_backtest)} 条K线已就绪。")
        return df_backtest

    except Exception as e:
        print(f"❌ 数据准备阶段发生错误: {e}")
        traceback.print_exc()
        return None


def generate_summary(params, realized, final_equity, total_fees, shift_count, final_positions, trade_df):
    """ (最终修复版) 生成包含所有测试参数、使用中文键的汇总字典 """
    total_pnl = final_equity - params["capital"]
    unrealized_pnl = total_pnl - realized
    sell_trades_count = len(trade_df[trade_df['side'] == 'SELL'])
    avg_profit_per_sell = realized / sell_trades_count if sell_trades_count > 0 else 0

    # ==================================================================
    # ✅ 关键修复：明确地创建包含中文键的字典
    # ==================================================================
    summary = {
        # --- 结果指标 ---
        '总盈亏(%)': total_pnl / params["capital"] * 100,
        '已实现盈亏': realized,
        '未实现盈亏': unrealized_pnl,
        '卖出次数': sell_trades_count,
        '单次均利': avg_profit_per_sell,
        '当前持仓': len(final_positions),
        '总手续费': total_fees,
        '移动次数': shift_count,

        # --- 策略参数 (从英文键映射到中文键) ---
        '下限': params.get('lower'),
        '上限': params.get('upper'),
        '网格数量': params.get('n_grids'),
    }

    # 动态添加其他在CSV中优化的参数，以便它们也出现在报告中
    # (如果CSV中有'shift_ratio'列，报告中就会多一列'shift_ratio')
    for key in ['shift_ratio', 'ma_change_threshold', 'breakout_buffer', 'force_move_bars']:
        if key in params:
            summary[key] = params[key]

    return summary


def print_summary_report(results_df):
    """
    (最终简化版) 接收一个已经排好列序的DataFrame，将其格式化并打印到控制台。
    """
    df_to_print = results_df.copy()

    # --- 格式化数字显示 ---
    formatters = {
        '总盈亏(%)': "{:,.2f}%".format,  # 加上百分号更清晰
        '已实现盈亏': "{:,.2f}".format,
        '未实现盈亏': "{:,.2f}".format,
        '卖出次数': "{:d}".format,
        '单次均利': "{:,.2f}".format,
        '当前持仓': "{:d}".format,
        '总手续费': "{:,.2f}".format,
        '移动次数': "{:d}".format
    }
    for col, formatter in formatters.items():
        if col in df_to_print:
            df_to_print[col] = df_to_print[col].apply(formatter)

    # --- 打印报告 ---
    print("\n" + "="*30 + " 回测参数对比报告 " + "="*30)
    # to_string() 默认会处理好对齐，并且我们不希望打印索引
    report_string = df_to_print.to_string(index=False)
    print(report_string)
    # 动态调整分隔线长度以匹配报告宽度
    print("=" * len(report_string.split('\n')[0]))


def run_single_backtest(params):
    """
    为单个参数组合运行一次完整的回测。 (核心引擎/Worker)
    """
    config, df_backtest, param_row = params

    # 🚀 优化点 1: 更优雅地合并参数
    # 首先，从全局config中复制一份基础参数
    final_params = config.copy()
    # 然后，用 param_row (来自CSV) 中的特定值覆盖基础参数
    final_params.update(param_row)

    # 从合并后的参数中提取所需变量
    lower = final_params["lower"]
    upper = final_params["upper"]
    n_grids = int(final_params["n_grids"])

    run_id = f"L{lower}_U{upper}_N{n_grids}"

    try:
        trader = GridTrader(
            capital=config["capital"],
            fee_rate=config["fee_rate"],
            n_grids=n_grids,
            initial_lower=lower,
            initial_upper=upper,
            ma_period=config["ma_period"],
            strategy_params=final_params,
            verbose=config.get("verbose", False)
        )

        trades, realized, final_equity, total_fees, shift_count, final_positions = trader.simulate(
            df_backtest)

        # 4. 转换交易记录为DataFrame
        trade_df_columns = [
            "time", "side", "level price", "linked_buy_price", "watermark",
            "average cost", "trade_qty", "amount_usdt", "cash_balance", "total_qty", "profit",
            "profit_pool", "fee", "total_fee", "total_capital",
            "close_price", "grid_range", "positions", "levels_snapshot"
        ]
        trade_df = pd.DataFrame(trades, columns=trade_df_columns)

        # 5. 生成汇总
        summary = generate_summary(
            final_params, realized, final_equity,
            total_fees, shift_count, final_positions, trade_df
        )

        # 6. 返回结果
        return (run_id, summary, trade_df, None)

    except Exception as e:
        # 7. 返回错误
        error_msg = f"参数组 {run_id} 发生错误: {e}\n{traceback.format_exc()}"
        return (run_id, None, None, error_msg)


def run_parameter_scan_refactored(config, df_backtest):
    """
    (重构版)
    执行单参数（n_grids）扫描回测，并生成Excel报告。
    这个函数通过串行调用核心引擎 run_single_backtest 来工作。
    """
    results_list = []
    output_filename = f"backtest_{config['symbol']}_report.xlsx"

    try:
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            grid_values = config["grid_n_range"]

            # 准备要迭代的任务参数
            tasks = []
            for n_grids_value in grid_values:
                # 为每个任务构建一个与 run_single_backtest 兼容的参数行
                param_row = {
                    "lower": config["lower_bound"],
                    "upper": config["upper_bound"],
                    "n_grids": n_grids_value
                }
                tasks.append((config, df_backtest, param_row))

            print(f"\n--- 开始单参数扫描 ({len(tasks)} 组参数) ---")
            for task in tqdm(tasks, desc="单参数扫描中"):
                # 在循环中直接调用核心引擎
                run_id, summary, trade_df, error = run_single_backtest(task)

                if error:
                    print(f"\n⚠️ 参数组 {run_id} 出错: {error}")
                    continue

                results_list.append(summary)

                # 在主进程中安全地写入详细交易日志
                sheet_name = f"Details_{summary['网格数量']}"
                trade_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # ==================================================================
            # ✅ 关键修复：在这里加入与 batch 模式完全一致的列排序逻辑
            # ==================================================================
            print("\n--- 所有回测计算完成，正在生成最终汇总报告 ---")
            if results_list:
                # 将汇总结果列表转换为DataFrame
                summary_df = pd.DataFrame(results_list)
                summary_df.sort_values(
                    by='总盈亏(%)', ascending=False, inplace=True)

                # 1) 结果指标（我们希望这些指标排在最前面）
                result_cols = [
                    '总盈亏(%)', '已实现盈亏', '未实现盈亏', '卖出次数',
                    '单次均利', '当前持仓', '总手续费', '移动次数'
                ]

                # 2) 我们希望紧接在结果指标后面的两个特定参数（如果存在）
                special_params = ['shift_ratio', 'ma_change_threshold']

                # 3) 核心参数（中文）排在参数区前面
                core_param_cols = ['下限', '上限', '网格数量']

                # 4) 其余的参数列（动态检测）
                other_param_cols = [col for col in summary_df.columns
                                    if col not in (result_cols + special_params + core_param_cols)]

                # 5) 最终列顺序：结果指标 -> special_params -> 核心参数 -> 其它参数
                final_cols = result_cols + [p for p in special_params if p in summary_df.columns] + \
                    [c for c in core_param_cols if c in summary_df.columns] + \
                    other_param_cols

                # 应用最终顺序（只保留实际存在的列）
                summary_df = summary_df[[
                    col for col in final_cols if col in summary_df.columns]]

                # 写入 Excel
                summary_df.to_excel(
                    writer, sheet_name='Summary', index=False, float_format='%.2f')

                print("\n--- 批量汇总报告 ---")
                print(summary_df.to_string())

        print(f"\n✅ 完整回测报告已成功保存到: {output_filename}")

    except Exception as e:
        print(f"\n❌ 回测或报告生成阶段发生错误: {e}")
        traceback.print_exc()

# ==============================================================================
# 🚀 正确的并行批量扫描函数 (The Manager)
# ==============================================================================


def run_batch_scan_parallel(config, df_backtest, param_csv):
    """
    (并行版)
    从CSV文件加载多组参数，使用多进程并行执行回测，并生成一份包含
    汇总报告和各组参数详细交易记录的Excel文件。

    Args:
        config (dict): 全局配置字典。
        df_backtest (pd.DataFrame): 用于回测的K线数据。
        param_csv (str): 包含参数组合的CSV文件路径。
    """
    # 1. 检查并读取参数文件
    if not os.path.exists(param_csv):
        print(f"❌ 参数CSV文件不存在: {param_csv}")
        return

    param_df = pd.read_csv(param_csv)
    required_cols = {"lower", "upper", "n_grids"}
    if not required_cols.issubset(param_df.columns):
        print(f"❌ 参数CSV缺少必须列: {required_cols}")
        return

    # 2. 准备要分发给每个工作进程的任务列表
    # 每个任务是一个元组，包含所有`run_single_backtest`需要的参数
    tasks = [(config, df_backtest, row) for _, row in param_df.iterrows()]

    results_list = []
    output_filename = f"batch_parallel_{config['symbol']}_report.xlsx"

    # 3. 创建并管理进程池
    # max_workers不指定时，默认为系统的CPU核心数。可以根据内存情况适当调低。
    with concurrent.futures.ProcessPoolExecutor() as executor:
        print(f"\n--- 开始并行批量回测 ({len(tasks)} 组参数) ---")

        # 使用executor.map将tasks列表中的每个任务分发给一个工作进程
        # executor.map会按顺序返回结果
        results_iterator = executor.map(run_single_backtest, tasks)

        # 4. 在主进程中，安全地打开一个Excel文件写入器
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:

            # 5. 遍历从工作进程返回的结果，并用tqdm显示进度条
            for result in tqdm(results_iterator, total=len(tasks), desc="并行批量回测中"):
                run_id, summary, trade_df, error = result

                # 检查工作进程是否返回了错误
                if error:
                    print(f"\n⚠️ 跳过出错的参数组 {run_id}。")
                    # 如果需要看详细错误，可以取消下面这行注释
                    # print(f"错误详情: {error}")
                    continue

                # 6. 将成功的汇总结果收集起来
                results_list.append(summary)

                # 7. 将详细的交易记录DataFrame写入Excel的一个独立工作表
                # 这是安全的操作，因为只有主进程在执行写入
                sheet_name = f"Details_{run_id}"
                # 确保工作表名称长度不超过Excel的31个字符限制
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:31]
                trade_df.to_excel(writer, sheet_name=sheet_name, index=False)

           # ==================================================================
            # ✅ 关键修复：在这里使用与 single 模式完全一致的列排序逻辑
            # ==================================================================
            print("\n--- 所有回测计算完成，正在生成最终汇总报告 ---")
            if results_list:
                summary_df = pd.DataFrame(results_list)
                summary_df.sort_values(
                    by='总盈亏(%)', ascending=False, inplace=True)

                # 1. 定义核心结果指标的顺序
                result_cols = [
                    '总盈亏(%)', '已实现盈亏', '未实现盈亏', '卖出次数',
                    '单次均利', '当前持仓', '总手续费', '移动次数'
                ]

                # 2. 定义核心参数的顺序
                core_param_cols = ['下限', '上限', '网格数量']

                # 3. 动态查找所有其他的参数列
                all_summary_cols = summary_df.columns.tolist()
                other_param_cols = [col for col in all_summary_cols
                                    if col not in result_cols and col not in core_param_cols]

                # 4. 拼接成最终的列顺序
                final_cols = result_cols + core_param_cols + other_param_cols

                # 5. 应用新的列顺序
                summary_df = summary_df[[
                    col for col in final_cols if col in summary_df.columns]]

                # 写入 Excel
                summary_df.to_excel(
                    writer, sheet_name='Summary', index=False, float_format='%.2f')

                # 打印报告
                print("\n--- 批量汇总报告 ---")
                print_summary_report(summary_df)
            # ==================================================================

    print(f"\n✅ 批量并行回测报告已成功保存到: {output_filename}")


def main():
    """
    主程序入口，负责编排整个回测流程。
    """
    config = {
        "mode": "single",          # "single" 或 "batch"

        "symbol": "ETHUSDT",
        "start_date": "2021-01-04",
        "end_date": "2025-10-02",
        "interval": "1m",
        "ma_period": 720,
        "capital": 10000,
        "fee_rate": 0.00026,
        "verbose": True,  # 回测引擎是否打印详细日志
        "param_csv": "param_grid.csv",  # batch 模式下需要的文件

        # --- "single" 模式下的网格参数 ---
        "lower_bound": 1272.8,
        "upper_bound": 4060.82,
        "grid_n_range": [10],  # 可以测试多个参数
        # ==================================================================
        # ✅ 这里就是您要修改的地方：统一的策略参数配置区
        # ==================================================================
        # 您可以自由修改下面的值，回测时会自动生效

        # --- 利润复投参数 ---
        "reinvest_threshold": 70,  # 旧值: 100, 新值: 150 (利润池超过150才复投)

        # --- 网格移动参数 ---
        "force_move_bars": 360,     # 旧值: 360, 新值: 720 (价格在网格外12小时才强制移动)
        "breakout_buffer": 0.01,    # 旧值: 0.01, 新值: 0.02 (价格需突破上下轨2%才触发移动)
        "ma_change_threshold": 0.01,  # 旧值: 0.01, 新值: 0.03 (MA均线变化超过3%才确认趋势)
        "shift_ratio": 0.01       # 旧值: 0.01, 新值: 0.015 (每次网格移动1.5%)
    }

    # 步骤 1: 准备数据
    df_for_backtest = setup_backtest_data(config)

    # 步骤 2: 执行回测与分析
    if config["mode"] == "single":
        run_parameter_scan_refactored(config, df_for_backtest)
    elif config["mode"] == "batch":
        run_batch_scan_parallel(config, df_for_backtest, config["param_csv"])
    else:
        print(f"❌ 未知模式: {config['mode']}, 请选择 'single' 或 'batch'")


if __name__ == "__main__":
    main()
