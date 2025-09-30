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
                 ma_period, reinvest_threshold=200, force_move_bars=360, verbose=False):
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
        self.trade_qty_per_grid = 0.0  # ✅ [新] 标准交易单位 (币数)

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
                   qty, amount_usdt, profit, close_price, positions_snapshot, levels_snapshot):
        """统一的交易日志记录函数"""
        total_qty_snapshot = sum(p['qty'] for p in self.open_positions)
        grid_range_str = f"{self.lower:.2f}-{self.upper:.2f}" if self.levels else "N/A"

        log_entry = (
            timestamp,          # 交易时间
            side,               # 买/卖/事件
            level_price,        # 网格价位
            linked_info,        # 买入时填 market@xx，卖出时填开仓价
            avg_cost,           # 平均成本
            qty,                # 成交数量
            amount_usdt,        # 成交金额 (不含手续费)
            self.cash,          # 当前现金余额
            total_qty_snapshot,  # 当前持仓总量
            profit,             # 卖出利润 (买入时为 None)
            grid_range_str,     # 网格区间
            close_price,        # 当前收盘价
            positions_snapshot,  # 持仓快照
            levels_snapshot     # 网格快照
        )
        self.trades.append(log_entry)

    def _execute_buy(self, level_price, buy_price, qty_to_buy, timestamp, positions_snapshot, levels_snapshot, close_price, side="BUY", modify_global_state=True):
        # nonlocal cash, total_fees, trades, open_positions, bought_levels
        if buy_price <= 0 or qty_to_buy <= 0:
            return None
        raw_cost = qty_to_buy*buy_price
        fee = raw_cost * self.fee_rate
        total_cost = raw_cost + fee

        if self.cash < total_cost:
            return None
        self.cash -= total_cost
        self.total_fees += fee

        new_position = {"price": level_price,
                        "qty": qty_to_buy, "cost": total_cost, "avg_cost": total_cost / qty_to_buy if qty_to_buy > 0 else 0}
        if modify_global_state:
            self.open_positions.append(new_position)
            self.bought_levels.add(level_price)

        self._log_trade(timestamp, side, level_price, f"market@{buy_price:.2f}",
                        new_position["avg_cost"], qty_to_buy, raw_cost,
                        None, close_price, positions_snapshot, levels_snapshot)
        return new_position

    def _execute_sell(self, position, sell_price, timestamp, positions_snapshot, levels_snapshot, close_price, profit_to_pool, side="SELL", modify_global_state=True):
        # nonlocal cash, realized_pnl, total_fees, trades, open_positions, bought_levels, highest_sell_level_watermark, profit_pool
        trade_qty = position["qty"]   # 本次卖出的数量
        proceeds = trade_qty * sell_price
        fee = proceeds * self.fee_rate
        self.total_fees += fee

        net_proceeds = proceeds - fee

        # 【修改】计算单笔利润
        single_profit = net_proceeds - position["cost"]
        if profit_to_pool:
            self.profit_pool += single_profit  # 把利润存入利润池
            self.cash += position["cost"]  # 只把本金归还给 cash
        else:
            self.cash += net_proceeds
        self.realized_pnl += single_profit

        # === 是否修改全局仓位 ===
        if modify_global_state:
            try:
                self.open_positions.remove(position)
            except ValueError:
                pass
            self.bought_levels.discard(position["price"])

        self.highest_sell_level_watermark = max(
            self.highest_sell_level_watermark, sell_price)

        self._log_trade(timestamp, side, round(sell_price, 2), position["price"],
                        position["avg_cost"], trade_qty, proceeds,
                        single_profit, close_price, positions_snapshot, levels_snapshot)
        return True

    def _redistribute_positions(self, current_price, timestamp, old_levels_snapshot, init_or_not):
        """
        (V6 - 数量本位 + 渐进式迁移)
        1. 根据总净值，计算出新的“标准交易单位”(trade_qty_per_grid)。
        2. 以此为标准，通过“渐进式”的迁移（优先利用旧仓位），完成对新持仓的部署。
        """
        # nonlocal open_positions, bought_levels, cash, total_fees, realized_pnl, levels, trade_qty_per_grid

        # === Step 1 & 2: 资产盘点并计算新的“标准交易单位” (与V5版完全相同) ===
        total_positions_value = sum(
            p['qty'] * current_price for p in self.open_positions)
        total_asset_value = self.cash + self.profit_pool + \
            total_positions_value  # ✅ 利润池在这里正式并入计算
        self.profit_pool = 0.0  # 盘点后立即清空，防止重复计算

        deploy_levels = {lv for lv in self.levels if lv >= current_price}
        reserve_levels = {lv for lv in self.levels if lv < current_price}
        num_deploy_grids = len(deploy_levels)

        # ... (计算 trade_qty_per_grid 的成本方程逻辑不变) ...
        cost_factor_for_deployment = current_price * (1 + self.fee_rate)
        cost_factor_for_reserves = sum(reserve_levels) * (1 + self.fee_rate)
        total_cost_factor = (
            num_deploy_grids * cost_factor_for_deployment) + cost_factor_for_reserves

        if total_cost_factor > 1e-9:
            self.trade_qty_per_grid = total_asset_value / total_cost_factor*0.999
        else:
            self.trade_qty_per_grid = 0

        if self.verbose:
            print(f"{timestamp} 🏦 宏观调仓: 总净值={total_asset_value:.2f}, "
                  f"新标准交易单位: {self.trade_qty_per_grid:.6f}")

        # === ✅ [核心修改] Step 3: "渐进式迁移" 执行逻辑 ===

        # 3.1 打包现有持仓，作为“可分配的资产池”
        survivors_pool = sorted(
            [{"qty": p["qty"], "price": p["price"], "avg_cost": p["avg_cost"]}
             for p in self.open_positions if p["qty"] > 1e-8],
            key=lambda x: x["avg_cost"]  # 按成本从低到高排序，优先保留低成本仓位
        )

        new_positions = []

        # 3.2 遍历所有【新的持仓目标格】，用“资产池”和“现金”去填满它们
        for lv in sorted(list(deploy_levels)):
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
                bought_position_part = self._execute_buy(
                    level_price=lv,
                    buy_price=current_price,
                    qty_to_buy=qty_needed,
                    timestamp=timestamp,
                    # 快照应该是当时的、正在构建中的新仓位列表
                    positions_snapshot=sorted(
                        [p['price'] for p in new_positions]),
                    levels_snapshot=self.levels,
                    close_price=current_price,
                    side=init_or_not,
                    modify_global_state=False
                )

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
            if self.verbose:
                print(f"调仓   -> 卖掉 {len(survivors_pool)} 个多余的旧仓位...")
            for sp in survivors_pool:
                dummy_position = {  # 构建一个临时的position对象用于卖出
                    "price": sp["price"], "qty": sp["qty"],
                    "cost": sp["qty"] * sp["avg_cost"], "avg_cost": sp["avg_cost"]
                }
                self._execute_sell(
                    position=dummy_position,
                    sell_price=current_price,
                    timestamp=timestamp,
                    # 这里传递旧的网格快照，以记录当时的环境
                    positions_snapshot=sorted(
                        # 传递卖出前的持仓快照
                        [(p['price'], p['qty']) for p in self.open_positions]),
                    levels_snapshot=old_levels_snapshot,
                    close_price=current_price,
                    profit_to_pool=False,  # 收益直接进现金池
                    side="REDIST_SELL_LEFTOVER",
                    modify_global_state=False
                )

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
                                              positions_snapshot=sorted(
                                                  # 卖出前一刻的快照
                                                  [pos['price'] for pos in self.open_positions if pos != p]),
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
                    self.bought_levels.add(lv)

    def _initial_setup(self, df):
        """处理初始建仓的私有方法"""
        if df.empty:
            return

        init_price = df.iloc[0]['close']
        self.highest_sell_level_watermark = init_price

        # 初始建仓也应该是一次宏观调仓，所以我们直接调用 redistribute_positions，它会计算出初始的 trade_qty_per_grid
        # 并根据初始价格部署仓位
        if not self.levels:
            self.levels, self.step = build_levels(
                self.initial_lower, self.initial_upper, self.n_grids)
            self.lower, self.upper = self.levels[0], self.levels[-1]

        self._redistribute_positions(
            init_price, df.iloc[0]['datetime'], old_levels_snapshot=self.levels, init_or_not="INIT_BUY")

        # 初始化MA参考点
        first_ma = df.iloc[0][f'ma_{self.ma_period}']
        self.reference_ma = first_ma if not pd.isna(first_ma) else init_price
        if self.verbose:
            print(
                f"{df.iloc[0]['datetime']} 📌 初始 MA 参考点设为: {self.reference_ma:.2f}")

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

            if self.verbose:
                print(
                    f"{ts} ▼ 网格移动并重分配: {old_levels[0]:.2f}-{old_levels[-1]:.2f} → {self.lower:.2f}-{self.upper:.2f}")

            self.highest_sell_level_watermark = c

            self._redistribute_positions(
                c, ts, old_levels, init_or_not="SHIFT BUY")

            self.shift_count += 1
            boundary_changed = True
            self.reference_ma = current_ma

            # 记录事件
            self._log_trade(timestamp=ts, side=f"SHIFT_{shift_direction}", level_price="Grid Shifted & Redistributed",
                            linked_info=None, avg_cost=None, qty=None, amount_usdt=None, profit=None, close_price=c, positions_snapshot=[(p['price'], p['qty']) for p in self.open_positions], levels_snapshot=self.levels)

        return boundary_changed

    def _check_and_handle_reinvestment(self, c, ts):
        """
        检查并处理利润复投的逻辑。
        """
        if self.profit_pool >= self.REINVESTMENT_THRESHOLD:
            reinvest_amount = self.profit_pool

            if self.verbose:
                print(f"{ts} 💰 利润复投事件: {reinvest_amount:.2f} USDT,触发全局资产重分配...")

            # 直接调用 redistribute_positions，它会自动把 profit_pool 的钱算进去
            self._redistribute_positions(
                c, ts, self.levels, init_or_not="REINVEST BUY")

            # 记录事件
            self._log_trade(ts, "REINVEST_REDIST", f"Profit reinvested: {reinvest_amount:.2f}",
                            None, None, None, None, None, c, positions_snapshot=[(p['price'], p['qty']) for p in self.open_positions], levels_snapshot=self.levels)

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
            # 步骤 3.1: 检查并处理网格移动
            self._check_and_handle_grid_shift(h, l, c, ts, current_ma)

            # 步骤 3.2: 执行常规的买卖交易
            self._process_bar_trades(o, h, l, c, ts)

            # 步骤 3.3: 检查并处理利润复投
            self._check_and_handle_reinvestment(c, ts)

        # === 最终结算 ===
        final_equity = self.cash + sum(p['qty'] * df.iloc[-1]['close']
                                       for p in self.open_positions)
        return self.trades, self.realized_pnl, final_equity, self.total_fees, self.shift_count, self.open_positions


# ==============================================================================
# 5. 主程序/业务流程编排
# ==============================================================================
# ===== 主程序 (已修复返回值数量不匹配的错误) =====
if __name__ == "__main__":
    # ... (前面的 config 和数据加载部分无变动) ...
    config = {
        "symbol": "ETHUSDT",
        "start_date": "2025-06-07",
        "end_date": "2025-09-23",
        "interval": "1m",
        "ma_period": 720,
        "capital": 10000,
        "fee_rate": 0.00026,
        "lower_bound": 2500,
        "upper_bound": 4500,
        "grid_n_range": [10]
    }

    # --- 1. 数据预加载与处理 ---
    preload_start_date = datetime.strptime(
        config["start_date"], "%Y-%m-%d") - timedelta(minutes=config["ma_period"], days=1)
    preload_start_date_str = preload_start_date.strftime("%Y-%m-%d")

    DATA_FILENAME = f"{config['symbol']}_{config['interval']}_{preload_start_date_str}_to_{config['end_date']}.csv"

    if os.path.exists(DATA_FILENAME):
        print(f"发现本地数据文件 '{DATA_FILENAME}'，正在加载...")
        df_full = pd.read_csv(DATA_FILENAME)
        if 'datetime' not in df_full.columns:
            raise ValueError(f"CSV 文件 {DATA_FILENAME} 格式错误，缺少 datetime 列")

        df_full['datetime'] = pd.to_datetime(df_full['datetime'])
        print("数据加载完毕！")
    else:
        # ✅ 修改点：从调用 fetch_binance_klines 改为调用 load_from_sqlite
        print(f"本地文件 '{DATA_FILENAME}' 不存在，尝试从数据库加载...")
        # 假设你的数据库文件名为 'eth_data.db'
        df_full = load_from_sqlite(
            "eth_data.db",
            config["symbol"],
            preload_start_date_str,  # 使用预加载日期
            config["end_date"]
        )

        if not df_full.empty:
            df_full.to_csv(DATA_FILENAME, index=False)
            print(f"数据已从数据库加载并缓存到 '{DATA_FILENAME}' 以便将来使用。")

    if df_full.empty:
        print("错误：未能获取K线数据，程序退出。")
    else:
        df_with_indicators = add_indicators(
            df_full, period=config["ma_period"])
        start_bound = pd.to_datetime(config["start_date"])
        end_bound = pd.to_datetime(config["end_date"]) + pd.Timedelta(days=1)
        df_backtest = df_with_indicators[(df_with_indicators['datetime'] >= start_bound) & (
            df_with_indicators['datetime'] < end_bound)].copy()
        df_backtest.reset_index(drop=True, inplace=True)
        print(
            f"指标预热完成！已截取出从 {config['start_date']} 开始的 {len(df_backtest)} 条数据用于回测。")

        # --- 2. 自动化参数扫描 ---
        results_list = []
        output_filename = f"backtest_{config['symbol']}_full_report.xlsx"
        try:
            with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                total_grids = len(config["grid_n_range"])
                for i, n_grids_value in enumerate(config["grid_n_range"], 1):
                    print(
                        f"--- 正在测试网格数量 (N_GRIDS) = {n_grids_value} ({i}/{total_grids}) ---")

                    # ✅ 核心变化：创建 Trader 实例并运行回测
                    trader = GridTrader(
                        capital=config["capital"],
                        fee_rate=config["fee_rate"],
                        n_grids=n_grids_value,
                        initial_lower=config["lower_bound"],
                        initial_upper=config["upper_bound"],
                        ma_period=config["ma_period"],
                        verbose=False
                    )

                    # 运行回测，并获取结果
                    trades, realized, final_equity, total_fees, shift_count, final_positions = trader.simulate(
                        df_backtest)

                    trade_df = pd.DataFrame(
                        trades, columns=["time", "side", "level price", "linked_buy_price", "average cost", "trade_qty", "amount_usdt", "cash_balance", "total_qty", "profit", "grid_range", "close_price", "positions", "levels_snapshot"])
                    sheet_name = f"Grid_{n_grids_value}_Details"
                    trade_df.to_excel(
                        writer, sheet_name=sheet_name, index=False)
                    print(f"    -> 交易明细已准备写入工作表: {sheet_name}")

                    total_pnl = final_equity - config["capital"]
                    unrealized_pnl = total_pnl - realized
                    # 【核心修正 #2】使用更稳妥的方式计算最终持仓数
                    current_positions = len(final_positions)

                    init_buy_trades_count = len(
                        trade_df[trade_df['side'] == 'INIT_BUY'])
                    buy_trades_count = len(trade_df[trade_df['side'].isin(
                        ['BUY', 'REBUILD_BUY', 'REDIST_BUY', 'REDIST_BUY_LOW'])])
                    sell_trades_count = len(
                        trade_df[trade_df['side'] == 'SELL'])
                    avg_profit_per_sell = realized / sell_trades_count if sell_trades_count > 0 else 0

                    # 【核心修正 #2】在总结报告中加入 shift_count
                    result_summary = {
                        '网格数量': n_grids_value,  # <--- 修改表头
                        '总盈亏(%)': total_pnl / config["capital"] * 100,
                        '已实现盈亏': realized,
                        '未实现盈亏': unrealized_pnl,
                        '卖出次数': sell_trades_count,
                        '单次均利': avg_profit_per_sell,
                        '当前持仓': current_positions,
                        '总手续费': total_fees,
                        '移动次数': shift_count  # <--- 新增
                    }
                    results_list.append(result_summary)

                results_df = pd.DataFrame(
                    results_list).set_index('网格数量')
                results_df.sort_values(
                    by='总盈亏(%)', ascending=False, inplace=True)
                results_df.to_excel(
                    writer, sheet_name='Summary', float_format='%.2f')
                print("\n--- 对比总结报告已准备写入工作表: Summary ---")

            print(f"\n✅ 完整回测报告已成功保存到文件: {output_filename}")
            # ==========================================================
            # ===== 【核心修正】在这里对 DataFrame 进行格式化，以便打印 =====
            # ==========================================================

            df_to_print = results_df.copy()

            # 定义每一列的格式化规则
            formatters = {
                '总盈亏(%)':   "{:,.2f}".format,
                '已实现盈亏':   "{:,.2f}".format,
                '未实现盈亏':   "{:,.2f}".format,
                '卖出次数':    "{:d}".format,
                '单次均利':    "{:,.2f}".format,
                '当前持仓':    "{:d}".format,
                '总手续费':    "{:,.2f}".format,
                '移动次数':    "{:d}".format
            }

            # 应用格式化
            for col, formatter in formatters.items():
                if col in df_to_print:
                    df_to_print[col] = df_to_print[col].apply(formatter)

            # 计算每列的最大宽度（表头 vs 内容）
            col_widths = {}
            for col in df_to_print.columns:
                max_content_len = df_to_print[col].astype(str).map(len).max()
                col_widths[col] = max(max_content_len, len(col))

            # 格式化表头
            header = "  ".join(
                col.ljust(col_widths[col]) for col in df_to_print.columns)
            index_name = df_to_print.index.name or ""
            header = index_name.ljust(
                len(str(df_to_print.index.max()))) + "  " + header

            # 格式化行数据
            rows = []
            for idx, row in df_to_print.iterrows():
                idx_str = str(idx).ljust(len(str(df_to_print.index.max())))
                row_str = "  ".join(str(val).rjust(
                    col_widths[col]) for col, val in row.items())
                rows.append(idx_str + "  " + row_str)

            # 打印结果
            print("\n" + "="*30 + " 不同网格数量参数回测对比报告 " + "="*30)
            print(header)
            for r in rows:
                print(r)
            print("=" * len(header))

        except Exception as e:
            print(f"\n❌ 处理或保存文件时出错: {e}")
            traceback.print_exc()
