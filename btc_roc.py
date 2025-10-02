import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import traceback


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
                 ma_period, reinvest_threshold=100, force_move_bars=360, verbose=False):
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
        self.breakout_buffer = 0.01
        self.highest_sell_level_watermark = 0.0
        self.REINVESTMENT_THRESHOLD = reinvest_threshold
        self.FORCE_MOVE_BARS = force_move_bars
        self.outside_bars = 0
        self.ma_change_threshold = 0.01
        self.shift_ratio = 0.01

    def _log_trade(self, timestamp, side, level_price, linked_info, avg_cost,
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
            levels_snapshot     # 网格快照
        )
        self.trades.append(log_entry)

    def _execute_buy(self, level_price, buy_price, qty_to_buy, timestamp, positions_snapshot, levels_snapshot, close_price, side="BUY", modify_global_state=True):
        if buy_price <= 0 or qty_to_buy <= 0:
            return None
        raw_cost = qty_to_buy * buy_price
        fee = raw_cost * self.fee_rate
        total_cost = raw_cost + fee

        if self.cash < total_cost:
            if self.verbose:
                print(f"{timestamp} 下单失败: {level_price:.2f}, "
                      f"新标准交易单位: {self.trade_qty_per_grid:.6f}",
                      f"目前现金: {self.cash:.9f}",
                      f"所需资金: {total_cost:.9f}")
            return None
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

        self._log_trade(timestamp, side, level_price, buy_price,
                        new_position["avg_cost"], qty_to_buy, raw_cost,
                        # ✅ 新增
                        None, close_price, sorted(final_positions_snapshot), levels_snapshot, profit_pool_snapshot=profit_pool_snapshot_after, cash_snapshot=cash_snapshot_after, single_trade_fee=fee,
                        total_fees_snapshot=total_fees_snapshot_after)
        return new_position

    def _execute_sell(self, position, sell_price, timestamp, positions_snapshot_before, levels_snapshot, close_price, profit_to_pool, side="SELL", modify_global_state=True):
        trade_qty = position["qty"]
        proceeds = trade_qty * sell_price
        fee = proceeds * self.fee_rate
        self.total_fees += fee

        net_proceeds = proceeds - fee

        # 【修改】计算单笔利润
        single_profit = net_proceeds - position["cost"]
        if profit_to_pool:
            # 常规交易，利润进池
            # 1. 净收入全部进入现金池
            self.cash += net_proceeds
            # 2. 然后，从现金池中，将利润“划转”到利润池
            self.cash -= single_profit
            self.profit_pool += single_profit
        else:
            self.cash += net_proceeds
        self.realized_pnl += single_profit

        position_price_to_remove = position["price"]
        final_positions_snapshot = [
            (price, qty) for price, qty in positions_snapshot_before if price != position_price_to_remove
        ]
        cash_snapshot_after = self.cash
        profit_pool_snapshot_after = self.profit_pool
        total_fees_snapshot_after = self.total_fees  # ✅ 获取更新后的 total_fees

        self._log_trade(timestamp, side, round(sell_price, 2), position["price"],
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

        self.highest_sell_level_watermark = max(
            self.highest_sell_level_watermark, sell_price)
        return True

    '''
    # def compute_trade_qty_per_grid(self, capital, price, fee_rate, deploy, reserve):
    #     # 成本：deploy 用当前价计，reserve 用格价计
    #     deploy_cost = len(deploy) * price * (1 + fee_rate)
    #     reserve_cost = sum(lv * (1 + fee_rate) for lv in reserve)

    #     total_cost_factor = (deploy_cost + reserve_cost)*1.00001
    #     if total_cost_factor > 1e-9:
    #         Q = capital / total_cost_factor
    #     else:
    #         Q = 0

    #     # print("需预留资金的仓位如下：")
    #     # for lv in reserve:
    #     #     print(lv)
    #     # print("需即刻部署的仓位如下：")
    #     # for lv in deploy:
    #     #     print(lv)
    #     # print("新标准交易单位：", Q)
    #     return Q

    # # 在 GridTrader 类内部
    '''

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

        # if self.verbose:
        #     print("需预留资金的仓位如下：")
        #     for lv in reserve:
        #         print(lv)
        #     print("需即刻部署的仓位如下：")
        #     for lv in deploy:
        #         print(lv)
        #     print("新旧标准交易单位：", self.trade_qty_per_grid, Q)

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
        total_asset_value = self.cash + total_positions_value  # ✅ 利润池在这里正式并入计算

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

        # ✅ 替换 print: 记录“宏观调仓”的启动事件
        # =========================================================================
        self._log_trade(
            timestamp=timestamp, side="REDIST_START",
            level_price=f"New Q={self.trade_qty_per_grid:.9f}",
            linked_info=f"Old Q={old_qty:.9f}",
            avg_cost=None, qty=None, amount_usdt=None, profit=None,
            close_price=current_price,
            positions_snapshot=sorted(
                [(p['price'], p['qty']) for p in self.open_positions]),
            levels_snapshot=old_levels_snapshot,
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

                if self.cash >= total_cost:
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
                        timestamp, "REDIST_BUY_FAIL", lv, f"Cash needed: {total_cost:.2f}",
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

                self._log_trade(
                    timestamp=timestamp,
                    side="REDIST_SELL_LEFTOVER",
                    # 借用列显示描述
                    level_price=f"Sold {len(sold_positions_info)} leftover(s)",
                    linked_info=', '.join(sold_positions_info),  # 借用列显示被卖掉的仓位
                    avg_cost=avg_cost_of_sold,  # 记录被卖掉部分的总平均成本
                    qty=total_qty_sold,
                    amount_usdt=total_proceeds,
                    profit=total_profit,
                    close_price=current_price,
                    positions_snapshot=final_positions_snapshot,
                    levels_snapshot=old_levels_snapshot,  # 记录当时的旧网格
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
                touched = [
                    lv for lv in self.levels if seg_end <= lv < seg_start]
                for lv in sorted(touched, reverse=True):
                    if lv in self.bought_levels or lv in self.levels_sold_this_bar or lv >= self.highest_sell_level_watermark:
                        continue
                    self._execute_buy(level_price=lv,
                                      buy_price=lv,
                                      qty_to_buy=self.trade_qty_per_grid,
                                      timestamp=ts,
                                      # 快照参数应该反映“即将发生”交易前的状态
                                      positions_snapshot=sorted(
                                          [(p['price'], p['qty']) for p in self.open_positions]),
                                      levels_snapshot=self.levels,
                                      close_price=c,  # <--- 补上这个缺失的参数
                                      side="BUY"
                                      )

    # 在 GridTrader 类内部

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
        self._log_trade(
            timestamp=init_ts, side="INIT_DONE",
            level_price="Initial setup complete",
            linked_info=None, avg_cost=None, qty=None, amount_usdt=None, profit=None,
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

            # 在所有状态更新完毕后，记录一个总的 SHIFT 事件
            arrow = "▲" if "UP" in shift_direction else "▼"
            self._log_trade(timestamp=ts, side=f"SHIFT_{shift_direction}", level_price=f"{arrow} Grid Shifted",
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

            # === Step 1: 暂存利润并进行“沙盘推演” ===
            reinvest_amount = self.profit_pool

            # ✅ 替换 print: 用一个 "BEFORE" 事件来记录
            self._log_trade(
                ts, "REINVEST_START", f"Pool: {reinvest_amount:.2f}",
                f"Threshold: {self.REINVESTMENT_THRESHOLD}",
                None, None, None, None, c,
                sorted([(p['price'], p['qty']) for p in self.open_positions]),
                self.levels, self.profit_pool, self.cash, single_trade_fee=None, total_fees_snapshot=self.total_fees
            )

            # 1.1 假设利润已全部注入，计算理想中的总资产
            total_positions_value = sum(
                p['qty'] * c for p in self.open_positions)
            temp_total_asset_value = self.cash + reinvest_amount + total_positions_value

            # 1.2【调用你的函数】计算新的“目标标准交易单位” (new_target_qty)
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
            if reinvest_amount >= cash_needed_for_add_on:
                # 3.1 利润正式注入现金池
                self.profit_pool = 0.0
                self.cash += reinvest_amount

                # # 利润充足，执行补仓
                # if self.verbose and cash_needed_for_add_on > 0:
                #     print(
                #         f"    -> 使用 {cash_needed_for_add_on:.2f} USDT 利润补足 {len(qty_to_add_per_position)} 个现有持仓...")

                # 3.2 遍历并执行补仓
                for p in self.open_positions:
                    if p['price'] in qty_to_add_per_position:
                        qty_to_add = qty_to_add_per_position[p['price']]

                        raw_cost = qty_to_add * c
                        fee = raw_cost * self.fee_rate
                        total_cost = raw_cost + fee

                        if self.cash >= total_cost:
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
                            self._log_trade(
                                ts, "REINVEST_ADD", p['price'], c,
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
                # if self.verbose:
                #     print(
                #         f"    -> 利润 {reinvest_amount:.2f} 不足以支付补仓所需 {cash_needed_for_add_on:.2f}, 本次跳过复投。")
                # 利润没有被使用，所以 profit_pool 保持不变 (之前暂存了)
                pass  # profit_pool 在开头被暂存，这里不做操作，它就保持原样

            # 记录一个完成/跳过事件
            log_side = "REINVEST_DONE" if reinvest_amount >= cash_needed_for_add_on else "REINVEST_SKIPPED"
            log_msg = f"Q: {old_qty:.9f}->{self.trade_qty_per_grid:.9f}" if log_side == "REINVEST_DONE" else f"Needed {cash_needed_for_add_on:.2f}"
            self._log_trade(
                ts, log_side, log_msg,
                None, None, None, None, None, c,
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
            # 步骤 3.3: 检查并处理利润复投
            self._check_and_handle_reinvestment(c, ts)
            # 步骤 3.1: 检查并处理网格移动
            self._check_and_handle_grid_shift(h, l, c, ts, current_ma)

            # 步骤 3.2: 执行常规的买卖交易
            self._process_bar_trades(o, h, l, c, ts)

        # === 最终结算 ===
        final_equity = self.cash + sum(p['qty'] * df.iloc[-1]['close']
                                       for p in self.open_positions)
        return self.trades, self.realized_pnl, final_equity, self.total_fees, self.shift_count, self.open_positions


# # ==============================================================================
# # 5. 主程序/业务流程编排
# # ==============================================================================
# # ===== 主程序 (已修复返回值数量不匹配的错误) =====
# if __name__ == "__main__":
#     # ... (前面的 config 和数据加载部分无变动) ...
#     config = {
#         "symbol": "ETHUSDT",
#         "start_date": "2024-01-01",
#         "end_date": "2025-09-23",
#         "interval": "1m",
#         "ma_period": 720,
#         "capital": 10000,
#         "fee_rate": 0.00026,
#         "lower_bound": 2200,
#         "upper_bound": 4500,
#         "grid_n_range": [10]
#     }

#     # --- 1. 数据预加载与处理 ---
#     preload_start_date = datetime.strptime(
#         config["start_date"], "%Y-%m-%d") - timedelta(minutes=config["ma_period"], days=1)
#     preload_start_date_str = preload_start_date.strftime("%Y-%m-%d")

#     DATA_FILENAME = f"{config['symbol']}_{config['interval']}_{preload_start_date_str}_to_{config['end_date']}.csv"

#     if os.path.exists(DATA_FILENAME):
#         print(f"发现本地数据文件 '{DATA_FILENAME}'，正在加载...")
#         df_full = pd.read_csv(DATA_FILENAME)
#         if 'datetime' not in df_full.columns:
#             raise ValueError(f"CSV 文件 {DATA_FILENAME} 格式错误，缺少 datetime 列")

#         df_full['datetime'] = pd.to_datetime(df_full['datetime'])
#         print("数据加载完毕！")
#     else:
#         # ✅ 修改点：从调用 fetch_binance_klines 改为调用 load_from_sqlite
#         print(f"本地文件 '{DATA_FILENAME}' 不存在，尝试从数据库加载...")
#         # 假设你的数据库文件名为 'eth_data.db'
#         df_full = load_from_sqlite(
#             "eth_data.db",
#             config["symbol"],
#             preload_start_date_str,  # 使用预加载日期
#             config["end_date"]
#         )

#         if not df_full.empty:
#             df_full.to_csv(DATA_FILENAME, index=False)
#             print(f"数据已从数据库加载并缓存到 '{DATA_FILENAME}' 以便将来使用。")

#     if df_full.empty:
#         print("错误：未能获取K线数据，程序退出。")
#     else:
#         df_with_indicators = add_indicators(
#             df_full, period=config["ma_period"])
#         start_bound = pd.to_datetime(config["start_date"])
#         end_bound = pd.to_datetime(config["end_date"]) + pd.Timedelta(days=1)
#         df_backtest = df_with_indicators[(df_with_indicators['datetime'] >= start_bound) & (
#             df_with_indicators['datetime'] < end_bound)].copy()
#         df_backtest.reset_index(drop=True, inplace=True)
#         print(
#             f"指标预热完成！已截取出从 {config['start_date']} 开始的 {len(df_backtest)} 条数据用于回测。")

#         # --- 2. 自动化参数扫描 ---
#         results_list = []
#         output_filename = f"backtest_{config['symbol']}_full_report.xlsx"
#         try:
#             with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
#                 total_grids = len(config["grid_n_range"])
#                 for i, n_grids_value in enumerate(config["grid_n_range"], 1):
#                     print(
#                         f"--- 正在测试网格数量 (N_GRIDS) = {n_grids_value} ({i}/{total_grids}) ---")

#                     # ✅ 核心变化：创建 Trader 实例并运行回测
#                     trader = GridTrader(
#                         capital=config["capital"],
#                         fee_rate=config["fee_rate"],
#                         n_grids=n_grids_value,
#                         initial_lower=config["lower_bound"],
#                         initial_upper=config["upper_bound"],
#                         ma_period=config["ma_period"],
#                         verbose=True
#                     )

#                     # 运行回测，并获取结果
#                     trades, realized, final_equity, total_fees, shift_count, final_positions = trader.simulate(
#                         df_backtest)

#                     # ✅ 核心修正：增加两列
#                     trade_df_columns = [
#                         "time", "side", "level price", "linked_buy_price",
#                         "average cost", "trade_qty", "amount_usdt",
#                         "cash_balance", "total_qty", "profit",
#                         "profit_pool", "fee", "total_fee", "total_capital",
#                         "close_price", "grid_range", "positions", "levels_snapshot"
#                     ]
#                     trade_df = pd.DataFrame(trades, columns=trade_df_columns)
#                     sheet_name = f"Grid_{n_grids_value}_Details"
#                     trade_df.to_excel(
#                         writer, sheet_name=sheet_name, index=False)
#                     print(f"    -> 交易明细已准备写入工作表: {sheet_name}")

#                     total_pnl = final_equity - config["capital"]
#                     unrealized_pnl = total_pnl - realized
#                     # 【核心修正 #2】使用更稳妥的方式计算最终持仓数
#                     current_positions = len(final_positions)

#                     init_buy_trades_count = len(
#                         trade_df[trade_df['side'] == 'INIT_BUY'])
#                     buy_trades_count = len(trade_df[trade_df['side'].isin(
#                         ['BUY', 'REBUILD_BUY', 'REDIST_BUY', 'REDIST_BUY_LOW'])])
#                     sell_trades_count = len(
#                         trade_df[trade_df['side'] == 'SELL'])
#                     avg_profit_per_sell = realized / sell_trades_count if sell_trades_count > 0 else 0

#                     # 【核心修正 #2】在总结报告中加入 shift_count
#                     result_summary = {
#                         '网格数量': n_grids_value,  # <--- 修改表头
#                         '总盈亏(%)': total_pnl / config["capital"] * 100,
#                         '已实现盈亏': realized,
#                         '未实现盈亏': unrealized_pnl,
#                         '卖出次数': sell_trades_count,
#                         '单次均利': avg_profit_per_sell,
#                         '当前持仓': current_positions,
#                         '总手续费': total_fees,
#                         '移动次数': shift_count  # <--- 新增
#                     }
#                     results_list.append(result_summary)

#                 results_df = pd.DataFrame(
#                     results_list).set_index('网格数量')
#                 results_df.sort_values(
#                     by='总盈亏(%)', ascending=False, inplace=True)
#                 results_df.to_excel(
#                     writer, sheet_name='Summary', float_format='%.2f')
#                 print("\n--- 对比总结报告已准备写入工作表: Summary ---")

#             print(f"\n✅ 完整回测报告已成功保存到文件: {output_filename}")
#             # ==========================================================
#             # ===== 【核心修正】在这里对 DataFrame 进行格式化，以便打印 =====
#             # ==========================================================

#             df_to_print = results_df.copy()

#             # 定义每一列的格式化规则
#             formatters = {
#                 '总盈亏(%)':   "{:,.2f}".format,
#                 '已实现盈亏':   "{:,.2f}".format,
#                 '未实现盈亏':   "{:,.2f}".format,
#                 '卖出次数':    "{:d}".format,
#                 '单次均利':    "{:,.2f}".format,
#                 '当前持仓':    "{:d}".format,
#                 '总手续费':    "{:,.2f}".format,
#                 '移动次数':    "{:d}".format
#             }

#             # 应用格式化
#             for col, formatter in formatters.items():
#                 if col in df_to_print:
#                     df_to_print[col] = df_to_print[col].apply(formatter)

#             # 计算每列的最大宽度（表头 vs 内容）
#             col_widths = {}
#             for col in df_to_print.columns:
#                 max_content_len = df_to_print[col].astype(str).map(len).max()
#                 col_widths[col] = max(max_content_len, len(col))

#             # 格式化表头
#             header = "  ".join(
#                 col.ljust(col_widths[col]) for col in df_to_print.columns)
#             index_name = df_to_print.index.name or ""
#             header = index_name.ljust(
#                 len(str(df_to_print.index.max()))) + "  " + header

#             # 格式化行数据
#             rows = []
#             for idx, row in df_to_print.iterrows():
#                 idx_str = str(idx).ljust(len(str(df_to_print.index.max())))
#                 row_str = "  ".join(str(val).rjust(
#                     col_widths[col]) for col, val in row.items())
#                 rows.append(idx_str + "  " + row_str)

#             # 打印结果
#             print("\n" + "="*30 + " 不同网格数量参数回测对比报告 " + "="*30)
#             print(header)
#             for r in rows:
#                 print(r)
#             print("=" * len(header))

#         except Exception as e:
#             print(f"\n❌ 处理或保存文件时出错: {e}")
#             traceback.print_exc()

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

        if os.path.exists(data_filename):
            print(f"加载本地CSV缓存: '{data_filename}'")
            df_full = pd.read_csv(data_filename)
            df_full['datetime'] = pd.to_datetime(df_full['datetime'])
        else:
            print(f"CSV缓存不存在, 从SQLite数据库加载...")
            df_full = load_from_sqlite(
                "eth_data.db",  # 数据库名可以加入config
                config["symbol"],
                preload_start_date_str,
                config["end_date"]
            )
            if not df_full.empty:
                print(f"数据已加载并缓存到 '{data_filename}'")
                df_full.to_csv(data_filename, index=False)

        if df_full.empty:
            print("错误：未能获取任何K线数据。")
            return None

        # --- 2. 指标计算 ---
        ma_col = f"ma_{config['ma_period']}"
        if ma_col not in df_full.columns or df_full[ma_col].isnull().all():
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


def run_parameter_scan(config, df_backtest):
    """
    执行参数扫描回测，并生成Excel和控制台报告。
    """
    results_list = []
    output_filename = f"backtest_{config['symbol']}_report.xlsx"

    try:
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            for n_grids_value in config["grid_n_range"]:
                print(f"\n--- 正在测试网格数量 (N_GRIDS) = {n_grids_value} ---")

                trader = GridTrader(
                    capital=config["capital"],
                    fee_rate=config["fee_rate"],
                    n_grids=n_grids_value,
                    initial_lower=config["lower_bound"],
                    initial_upper=config["upper_bound"],
                    ma_period=config["ma_period"],
                    verbose=config.get("verbose", False)
                )

                trades, realized, final_equity, total_fees, shift_count, final_positions = trader.simulate(
                    df_backtest)

                # --- 结果处理与保存 ---
                trade_df = process_and_save_trades(
                    trades, writer, n_grids_value)

                summary = generate_summary(
                    config, n_grids_value, realized, final_equity,
                    total_fees, shift_count, final_positions, trade_df
                )
                results_list.append(summary)

            # --- 生成汇总报告 ---
            if results_list:
                results_df = pd.DataFrame(results_list).set_index('网格数量')
                results_df.sort_values(
                    by='总盈亏(%)', ascending=False, inplace=True)
                results_df.to_excel(
                    writer, sheet_name='Summary', float_format='%.2f')
                print("\n--- 汇总报告已准备写入: Summary ---")
                print_summary_report(results_df)

        print(f"\n✅ 完整回测报告已成功保存到: {output_filename}")

    except Exception as e:
        print(f"\n❌ 回测或报告生成阶段发生错误: {e}")
        traceback.print_exc()

# --- [新增] 更多的辅助函数，让 run_parameter_scan 更干净 ---


def process_and_save_trades(trades, writer, n_grids):
    trade_df_columns = [
        "time", "side", "level price", "linked_buy_price",
        "average cost", "trade_qty", "amount_usdt", "cash_balance", "total_qty", "profit",
        "profit_pool", "fee", "total_fee", "total_capital",
        "close_price", "grid_range", "positions", "levels_snapshot"
    ]
    trade_df = pd.DataFrame(trades, columns=trade_df_columns)
    sheet_name = f"Grid_{n_grids}_Details"
    trade_df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"    -> 交易明细已写入工作表: {sheet_name}")
    return trade_df


def generate_summary(config, n_grids, realized, final_equity, total_fees, shift_count, final_positions, trade_df):
    total_pnl = final_equity - config["capital"]
    unrealized_pnl = total_pnl - realized
    sell_trades_count = len(trade_df[trade_df['side'] == 'SELL'])
    avg_profit_per_sell = realized / sell_trades_count if sell_trades_count > 0 else 0

    return {
        '网格数量': n_grids,
        '总盈亏(%)': total_pnl / config["capital"] * 100,
        '已实现盈亏': realized,
        '未实现盈亏': unrealized_pnl,
        '卖出次数': sell_trades_count,
        '单次均利': avg_profit_per_sell,
        '当前持仓': len(final_positions),
        '总手续费': total_fees,
        '移动次数': shift_count
    }


def print_summary_report(results_df):
    df_to_print = results_df.copy()
    formatters = {
        '总盈亏(%)': "{:,.2f}".format, '已实现盈亏': "{:,.2f}".format,
        '未实现盈亏': "{:,.2f}".format, '卖出次数': "{:d}".format,
        '单次均利': "{:,.2f}".format, '当前持仓': "{:d}".format,
        '总手续费': "{:,.2f}".format, '移动次数': "{:d}".format
    }
    for col, formatter in formatters.items():
        if col in df_to_print:
            df_to_print[col] = df_to_print[col].apply(formatter)

    # ... (后续的打印格式化代码不变) ...
    print("\n" + "="*30 + " 不同网格数量参数回测对比报告 " + "="*30)
    print(df_to_print.to_string())  # 使用 pandas 自带的 to_string 更简单
    print("=" * 80)


def main():
    """
    主程序入口，负责编排整个回测流程。
    """
    config = {
        "symbol": "ETHUSDT",
        "start_date": "2021-01-01",
        "end_date": "2025-09-23",
        "interval": "1m",
        "ma_period": 720,
        "capital": 10000,
        "fee_rate": 0.00026,
        "lower_bound":  1272.43,
        "upper_bound": 4061.958499999999,
        "grid_n_range": [20],  # 可以测试多个参数
        "verbose": True  # 回测引擎是否打印详细日志
    }

    # 步骤 1: 准备数据
    df_for_backtest = setup_backtest_data(config)

    # 步骤 2: 执行回测与分析
    if df_for_backtest is not None and not df_for_backtest.empty:
        run_parameter_scan(config, df_for_backtest)
    else:
        print("数据准备失败，无法执行回测。")


if __name__ == "__main__":
    main()
