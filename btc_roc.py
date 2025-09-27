import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import traceback

# ==============================================================================
# 1. 数据获取模块
# ==============================================================================


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

# ==============================================================================
# 2. 特征工程/指标计算模块
# ==============================================================================


def add_indicators(df, period=720):
    """
    为DataFrame添加技术指标：
    - MA: 移动平均线
    """
    print(f"正在计算 {period} 分钟的移动平均线...")
    df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
    print("指标计算完毕！")
    return df

# ==============================================================================
# 3. 策略配置模块
# ==============================================================================


# ==============================================================================
# 3. 策略配置模块 (已重构)
# ==============================================================================

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

# ==============================================================================
# 4. 回测引擎模块
# ==============================================================================


def simulate(df, initial_lower, initial_upper, n_grids, capital, fee_rate, ma_period, verbose=False):
    """
    动态网格（最终完美版）：
    - 【新增】在交易记录中加入了每一笔平仓交易的利润。
    - 确保所有模块都已达到最终的、最稳健的状态。
    """

    # --- 初始状态 ---
    lower, upper = initial_lower, initial_upper
    # 【核心修正 #1】调用新的 build_levels 并接收 step
    levels, step = build_levels(lower, upper, n_grids)
    if not levels:
        return [], 0.0, capital, 0.0, 0

    lower, upper = levels[0], levels[-1]
    cash = capital
    open_positions, bought_levels, trades = [], set(), []
    realized_pnl, total_fees, shift_count = 0.0, 0.0, 0
    reference_ma, reference_ma_initialized = 0.0, False
    # 【核心修正】引入“最高卖出水位线”
    highest_sell_level_watermark = 0.0

    if verbose:
        print("回测开始...")

    # ===== 内部辅助函数 (execute_sell 已改造) =====
    def execute_sell(position, sell_price, timestamp, levels_snapshot, close_price, side="SELL", modify_global_state=True):
        nonlocal cash, realized_pnl, total_fees, trades, open_positions, bought_levels, highest_sell_level_watermark
        trade_qty = position["qty"]   # 本次卖出的数量
        proceeds = trade_qty * sell_price
        fee = proceeds * fee_rate
        total_fees += fee
        net_proceeds = proceeds - fee
        cash += net_proceeds

        # 【修改】计算单笔利润
        single_profit = net_proceeds - position["cost"]
        realized_pnl += single_profit

        # === 是否修改全局仓位 ===
        if modify_global_state:
            try:
                open_positions.remove(position)
            except ValueError:
                pass
            bought_levels.discard(position["price"])

            # 【核心修正】更新水位线
        highest_sell_level_watermark = max(
            highest_sell_level_watermark, sell_price)

        positions_snapshot = sorted([p['price'] for p in open_positions])
        total_qty_snapshot = sum(p['qty'] for p in open_positions)
        cash_snapshot = cash

        trades.append((timestamp, side, round(
            sell_price, 2), position["price"],  position["avg_cost"], trade_qty, proceeds, cash_snapshot, total_qty_snapshot, single_profit, f"{lower:.2f}-{upper:.2f}", close_price, positions_snapshot, levels_snapshot))

        return True

    def execute_buy(level_price, buy_price, value_to_invest, timestamp, levels_snapshot, close_price, side="BUY", modify_global_state=True):
        nonlocal cash, total_fees, trades, open_positions, bought_levels
        if buy_price <= 0 or value_to_invest <= 0:
            return None
        qty_to_buy = value_to_invest / buy_price
        cost_before_fee = value_to_invest
        fee = cost_before_fee * fee_rate
        total_cost = cost_before_fee + fee
        if cash < total_cost:
            return None
        cash -= total_cost
        total_fees += fee
        new_position = {"price": level_price,
                        "qty": qty_to_buy, "cost": total_cost, "avg_cost": total_cost / qty_to_buy if qty_to_buy > 0 else 0}
        if modify_global_state:
            open_positions.append(new_position)
            bought_levels.add(level_price)

        positions_snapshot = sorted([p['price'] for p in open_positions])
        total_qty_snapshot = sum(p['qty'] for p in open_positions)
        cash_snapshot = cash

        # 为买入交易的 profit 列填充 None
        trades.append((timestamp, side, level_price,
                      f"market@{buy_price:.2f}", new_position["avg_cost"], qty_to_buy, cost_before_fee, cash_snapshot, total_qty_snapshot, None, f"{lower:.2f}-{upper:.2f}", close_price, positions_snapshot, levels_snapshot))
        return new_position

    def redistribute_positions(current_price, timestamp, old_levels_snapshot):
        """
        渐进式网格迁移（重写版）：
        - 汇总所有仓位 + 现金，计算每个新格子的目标资金。
        - 遗留仓位优先分配，不够用现金补。
        - 多余遗留仓位直接卖出换现金。
        """
        nonlocal open_positions, bought_levels, cash, total_fees, realized_pnl, levels

        # === Step 1: 汇总资产 ===
        total_asset_value = cash + \
            sum(p['qty'] * current_price for p in open_positions)

        effective_grids = max(len(levels), 1)  # 至少保证 1，避免除零
        value_per_grid = total_asset_value / effective_grids  # 每格目标资金
        qty_per_grid = value_per_grid / current_price if current_price > 0 else 0

        if verbose:
            print(
                f"区间移动   -> 总净值 {total_asset_value:.2f}, 每格目标资金 {value_per_grid:.2f}")

        # === Step 2: 把遗留仓位打包成一个“库存池” ===
        survivors_pool = []
        for p in open_positions:
            if p["qty"] > 1e-8:
                survivors_pool.append({
                    "qty": p["qty"],
                    "price": p["price"],
                    "avg_cost": p["avg_cost"] if p["qty"] > 0 else current_price
                })

        survivors_pool.sort(key=lambda x: x["avg_cost"])  # 按成本低优先消耗

        new_positions = []

        # === Step 3: 从上到下重新分配 ===
        for lv in sorted(levels, reverse=True):
            if lv <= current_price:
                continue

            qty_needed = qty_per_grid
            cost_from_survivors, qty_from_survivors = 0.0, 0.0

            # 先消耗遗留仓位
            while qty_needed > 1e-8 and survivors_pool:
                sp = survivors_pool[0]
                take_qty = min(sp["qty"], qty_needed)
                take_cost = take_qty * sp["avg_cost"]

                qty_from_survivors += take_qty
                cost_from_survivors += take_cost
                sp["qty"] -= take_qty
                qty_needed -= take_qty

                if sp["qty"] <= 1e-8:
                    survivors_pool.pop(0)

            # 如果遗留仓位不足 → 用现金买
            bought_position_part = None
            if qty_needed > 1e-8:
                value_to_invest = qty_needed * current_price
                bought_position_part = execute_buy(
                    lv, current_price, value_to_invest, timestamp, levels,
                    current_price, side="REDIST_BUY_PART", modify_global_state=False
                )

            final_qty = qty_from_survivors + \
                (bought_position_part['qty'] if bought_position_part else 0)
            final_cost = cost_from_survivors + \
                (bought_position_part['cost'] if bought_position_part else 0)

            if final_qty > 1e-8:
                final_avg_cost = final_cost / final_qty
                new_positions.append(
                    {"price": lv, "qty": final_qty, "cost": final_cost, "avg_cost": final_avg_cost})

        # === Step 4: 卖掉剩余的遗留仓位 ===
        if survivors_pool:
            if verbose:
                print(f"区间移动   -> 卖掉遗留 {len(survivors_pool)} 个仓位，换成现金")
            for sp in survivors_pool:
                dummy_position = {
                    "price": sp["price"],  # 遗留的网格
                    "qty": sp["qty"],
                    "cost": sp["qty"] * sp["avg_cost"],
                    "avg_cost": sp["avg_cost"]  # <=== 保留原始成本
                }
                execute_sell(
                    dummy_position,
                    current_price,
                    timestamp,
                    old_levels_snapshot,
                    current_price,
                    side="REDIST_SELL_LEFTOVER"
                )

        # === Step 5: 更新仓位 ===
        open_positions = new_positions
        bought_levels = {p["price"] for p in open_positions}

    # 【您的核心贡献】将常规交易逻辑完全封装
    def process_bar_trades(o, h, l, c, ts):
        nonlocal levels_sold_this_bar, bought_levels, open_positions, highest_sell_level_watermark, cash

        # 动态分配资金（仅用于买入）
        value_per_grid_now = cash / \
            max(len([lv for lv in levels if lv < c]),
                1) if len(levels) > 0 else 0

        # 提前排序仓位，减少循环中的开销
        sorted_positions = sorted(open_positions, key=lambda x: x['price'])

        # 分段遍历价格路径
        segments = [(o, h), (h, l), (l, c)]
        for seg_start, seg_end in segments:
            if seg_start == seg_end:
                continue

            # ========= 上涨段：检查卖出 =========
            if seg_start < seg_end:
                for p in sorted_positions:
                    if p not in open_positions:
                        continue  # 可能已被卖出，跳过

                    # --- 优先用 levels.index() 查找下一个格子 ---
                    next_level = None
                    try:
                        idx = levels.index(p['price'])
                        if idx + 1 < len(levels):
                            next_level = levels[idx + 1]
                    except ValueError:
                        # --- fallback：用 min() 查找更大的格子 ---
                        next_level = min(
                            (lv for lv in levels if lv > p['price']), default=None)

                    # 如果 next_level 被价格路径穿越 → 卖出
                    if next_level and seg_start < next_level <= seg_end:
                        if execute_sell(p, next_level, ts, levels, c):
                            levels_sold_this_bar.add(next_level)

            # ========= 下跌段：检查买入 =========
            else:
                touched = [lv for lv in levels if seg_end <= lv < seg_start]
                for lv in sorted(touched, reverse=True):
                    if lv in bought_levels or lv in levels_sold_this_bar or lv >= highest_sell_level_watermark:
                        continue
                    execute_buy(lv, lv, value_per_grid_now,
                                ts, levels, c, side="BUY")

    # --- 初始建仓 ---
    per_grid_capital_init = capital / len(levels) if len(levels) > 0 else 0
    if per_grid_capital_init > 0 and not df.empty:
        init_price = df.iloc[0]['close']
        highest_sell_level_watermark = init_price

        init_ts = df.iloc[0]['datetime']
        init_levels = [lv for lv in levels if lv >=
                       init_price]
        for lv in sorted(init_levels):
            execute_buy(lv, init_price, per_grid_capital_init,
                        init_ts, levels, init_price, side="INIT_BUY")

    # --- 主循环 ---
    ma_col_name = f'ma_{ma_period}'
    ma_series, open_series, high_series, low_series, close_series, ts_series = df[ma_col_name].to_numpy(
    ), df['open'].to_numpy(), df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy(), df['datetime'].to_numpy()

    if not df.empty:
        first_ma = df.iloc[0][ma_col_name]
        first_close = df.iloc[0]['close']
        reference_ma = first_ma if not pd.isna(first_ma) else first_close
        reference_ma_initialized = True
        if verbose:
            print(
                f"{df.iloc[0]['datetime']} 📌 初始 MA 参考点设为: {reference_ma:.2f}")
    for i in range(1, len(df)):
        o, h, l, c, ts = open_series[i], high_series[i], low_series[i], close_series[i], ts_series[i]

        current_ma = ma_series[i]

        breakout_buffer, boundary_changed, shift_direction = 0.01, False, None

        if reference_ma is not None and not pd.isna(current_ma):
            ma_roc_from_ref = (current_ma - reference_ma) / reference_ma
            if h > upper * (1 + breakout_buffer) and ma_roc_from_ref >= 0.005:
                shift_direction = "UP"
            elif l < lower * (1 - breakout_buffer) and ma_roc_from_ref <= -0.005:
                shift_direction = "DOWN"

        if shift_direction:
            old_levels = levels
            if shift_direction == "UP":
                target_lower, target_upper = lower * 1.01, upper * 1.01
            else:  # SHIFT_DOWN
                target_lower, target_upper = lower * 0.99, upper * 0.99

            # 【核心修正 #2】每次移动时，都重新计算 levels 和 step
            levels, step = build_levels(target_lower, target_upper, n_grids)

            if not levels:
                continue

            lower, upper = levels[0], levels[-1]
            if verbose:
                print(
                    f"{ts} ▼ 网格移动并重分配: {old_levels[0]:.2f}-{old_levels[-1]:.2f} → {lower:.2f}-{upper:.2f}")

            highest_sell_level_watermark = c

            redistribute_positions(c, ts, old_levels)
            shift_count += 1
            boundary_changed = True

            if boundary_changed:
                reference_ma = current_ma
                # some logs
                positions_snapshot = sorted(
                    [p['price'] for p in open_positions])
                total_qty_snapshot = sum(p['qty'] for p in open_positions)
                cash_snapshot = cash

                event_desc = "Grid Shifted & Redistributed"
                actual_range_str = f"{levels[0]:.2f}-{levels[-1]:.2f}" if levels else "N/A"
                # 【修改】为事件记录的 profit 列填充 None
                trades.append((ts, f"SHIFT_{shift_direction}",
                              event_desc, None, None, None, None, cash_snapshot, total_qty_snapshot, None, actual_range_str, c, positions_snapshot, levels))

        # 【核心修正】将常规交易逻辑的调用放在这里
        levels_sold_this_bar = set()
        process_bar_trades(o, h, l, c, ts)

    # === 最终结算 ===
    final_equity = cash + sum(p['qty'] * df.iloc[-1]['close']
                              for p in open_positions)
    return trades, realized_pnl, final_equity, total_fees, shift_count, open_positions


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
        "lower_bound": 2200,
        "upper_bound": 4000,
        "grid_n_range": [72]
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
        df_full = fetch_binance_klines(
            config["symbol"], config["interval"], preload_start_date_str, config["end_date"])
        if not df_full.empty:
            df_full.to_csv(DATA_FILENAME, index=False)
            print(f"数据已保存到 '{DATA_FILENAME}' 以便将来使用。")

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

                    # 【核心修正 #1】增加一个变量 shift_count 来接收第5个返回值
                    trades, realized, final_equity, total_fees, shift_count, final_positions = simulate(
                        df_backtest,
                        config["lower_bound"],
                        config["upper_bound"],
                        n_grids_value,  # <--- 传递网格数量
                        config["capital"],
                        config["fee_rate"],
                        config["ma_period"],
                        verbose=False
                    )

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
                        trade_df[trade_df['side'].str.contains('SELL')])
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
