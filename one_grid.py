import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# ===== 参数 (这里的值将作为默认值，但在自动化测试中会被覆盖) =====
CAPITAL = 10000
FEE_RATE = 0.00026
LOWER, UPPER = 3000, 5000
STEP = 45  # Default value
SYMBOL = "ETHUSDT"
INTERVAL = "1m"
DAYS = 60

DATA_FILENAME = f"{SYMBOL}_{INTERVAL}_{DAYS}d.csv"

# ===== 获取Binance 1分钟K线 =====


def fetch_binance_klines(symbol, interval, days):
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int(
        (datetime.now() - timedelta(days=days)).timestamp() * 1000)
    url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    data = []
    print(f"本地数据文件 '{DATA_FILENAME}' 不存在或需要更新。")
    print("正在从币安获取K线数据...")
    while start_time < end_time:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": start_time, "limit": limit}
        try:
            resp = requests.get(url, params=params, timeout=10).json()
            if not resp or "code" in resp:
                print(f"API错误: {resp}")
                break
            data.extend(resp)
            print(f"已获取 {len(data)} 条数据...")
            start_time = resp[-1][0] + 60_000
            if len(resp) < limit:
                break
        except requests.exceptions.RequestException as e:
            print(f"网络请求失败: {e}")
            break
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume",
                                     "close_time", "qav", "trades", "tb_base", "tb_quote", "ignore"])
    df = df[["open_time", "high", "low", "close"]].astype(float)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    print("数据获取完毕！")
    return df

# ===== 构造网格 =====


def build_levels(lower, upper, step):
    levels = []
    price = lower
    while price <= upper:
        levels.append(round(price, 2))
        price += step
    return levels

# ===== 网格回测 (增加 verbose 参数以控制打印) =====


def simulate(df, levels, capital, verbose=False):
    if not levels:
        return [], 0.0, capital, 0.0

    per_grid_capital = capital / len(levels)
    cash = capital
    open_positions = []
    bought_levels = set()
    trades = []
    realized_pnl = 0.0
    total_fees = 0.0
    step = levels[1] - levels[0] if len(levels) > 1 else 0

    if verbose:
        print("开始回测...")

    # --- 中位启动初始化 ---
    initial_price = df.iloc[0]['close']
    initial_ts = df.iloc[0]['datetime']
    initial_buy_levels = [lv for lv in levels if lv > initial_price]
    for lv in sorted(initial_buy_levels):
        if cash < per_grid_capital:
            break
        cost_before_fee = per_grid_capital
        qty = cost_before_fee / initial_price
        fee = cost_before_fee * FEE_RATE
        total_fees += fee
        total_cost = cost_before_fee + fee
        if cash < total_cost:
            continue
        cash -= total_cost
        new_position = {"price": lv, "qty": qty, "cost": total_cost}
        open_positions.append(new_position)
        bought_levels.add(lv)
        trades.append((initial_ts, "INIT_BUY", lv,
                      f"Market buy at {initial_price:.2f}", cost_before_fee))

    # --- 主循环 ---
    for i in range(1, len(df)):
        lo, hi, ts = df.iloc[i]['low'], df.iloc[i]['high'], df.iloc[i]['datetime']
        levels_sold_this_tick = set()
        sellable_positions = [
            p for p in open_positions if hi >= p['price'] + step]
        if sellable_positions:
            for position in sorted(sellable_positions, key=lambda p: p['price']):
                if position in open_positions:  # 确保仓位未被之前的卖单处理
                    buy_price = position['price']
                    sell_price = buy_price + step
                    proceeds = position['qty'] * sell_price
                    fee = proceeds * FEE_RATE
                    total_fees += fee
                    cash += (proceeds - fee)
                    realized_pnl += (proceeds - fee) - position['cost']
                    levels_sold_this_tick.add(round(sell_price, 2))
                    trades.append(
                        (ts, "SELL", round(sell_price, 2), buy_price, proceeds))
                    open_positions.remove(position)
                    if buy_price in bought_levels:
                        bought_levels.remove(buy_price)

        touched_levels = [lv for lv in levels if lo <= lv <= hi]
        if touched_levels:
            for lv in sorted(touched_levels, reverse=True):
                if lv not in bought_levels and lv not in levels_sold_this_tick and cash >= per_grid_capital:
                    cost_before_fee = per_grid_capital
                    qty = cost_before_fee / lv
                    fee = cost_before_fee * FEE_RATE
                    total_fees += fee
                    total_cost = cost_before_fee + fee
                    if cash < total_cost:
                        continue
                    cash -= total_cost
                    new_position = {"price": lv,
                                    "qty": qty, "cost": total_cost}
                    open_positions.append(new_position)
                    bought_levels.add(lv)
                    trades.append((ts, "BUY", lv, None, cost_before_fee))

    if verbose:
        print("回测结束。")
    final_equity = cash
    last_close_price = df.iloc[-1]['close']
    for position in open_positions:
        final_equity += position['qty'] * last_close_price

    return trades, realized_pnl, final_equity, total_fees


# ===== 主程序 (自动化参数扫描，并将所有结果和明细导出到Excel) =====
if __name__ == "__main__":

    # 1. 只加载一次数据
    if os.path.exists(DATA_FILENAME):
        print(f"发现本地数据文件 '{DATA_FILENAME}'，正在加载...")
        df = pd.read_csv(DATA_FILENAME)
        df['datetime'] = pd.to_datetime(df['datetime'])
        print("数据加载完毕！")
    else:
        df = fetch_binance_klines(SYMBOL, INTERVAL, DAYS)
        if not df.empty:
            df.to_csv(DATA_FILENAME, index=False)
            print(f"数据已保存到 '{DATA_FILENAME}' 以便将来使用。")

    if df.empty:
        print("错误：未能获取K线数据，程序退出。")
    else:
        # 2. 定义要测试的步长范围和结果存储
        step_range = range(20, 201, 5)
        results_list = []

        # ==========================================================
        # ===== 新增：创建 ExcelWriter 对象来管理多工作表写入 =====
        # ==========================================================
        output_filename = "backtest_full_report.xlsx"
        try:
            with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                # 3. 循环执行回测
                total_steps = len(step_range)
                for i, step_value in enumerate(step_range, 1):
                    print(
                        f"--- 正在测试步长 (STEP) = {step_value} ({i}/{total_steps}) ---")

                    levels = build_levels(LOWER, UPPER, step_value)
                    trades, realized, final_equity, total_fees = simulate(
                        df, levels, CAPITAL, verbose=False)

                    # 4. 创建当前步长的交易明细 DataFrame
                    trade_df = pd.DataFrame(
                        trades, columns=["time", "side", "price", "linked_buy_price", "amount_usdt"])

                    # ==========================================================
                    # ===== 新增：将当前步长的交易明细写入其专属的工作表 =====
                    # ==========================================================
                    sheet_name = f"Step_{step_value}_Details"
                    trade_df.to_excel(
                        writer, sheet_name=sheet_name, index=False)
                    print(f"    -> 交易明细已准备写入工作表: {sheet_name}")

                    # 5. 计算并存储核心汇总结果
                    total_pnl = final_equity - CAPITAL
                    unrealized_pnl = total_pnl - realized

                    init_buy_trades_count = len(
                        trade_df[trade_df['side'] == 'INIT_BUY'])
                    buy_trades_count = len(trade_df[trade_df['side'] == 'BUY'])
                    sell_trades_count = len(
                        trade_df[trade_df['side'] == 'SELL'])
                    current_positions = init_buy_trades_count + buy_trades_count - sell_trades_count
                    avg_profit_per_sell = realized / sell_trades_count if sell_trades_count > 0 else 0

                    result_summary = {
                        '步长(Step)': step_value,
                        '总盈亏(%)': total_pnl / CAPITAL * 100,
                        '已实现盈亏': realized,
                        '未实现盈亏': unrealized_pnl,
                        '卖出次数': sell_trades_count,
                        '单次均利': avg_profit_per_sell,
                        '当前持仓': current_positions,
                        '总手续费': total_fees,
                    }
                    results_list.append(result_summary)

                # 6. 创建并排序最终的对比报告DataFrame
                results_df = pd.DataFrame(results_list)
                results_df = results_df.set_index('步长(Step)')
                results_df.sort_values(
                    by='总盈亏(%)', ascending=False, inplace=True)

                # ==========================================================
                # ===== 新增：将最终的对比报告写入名为 "Summary" 的工作表 =====
                # ==========================================================
                results_df.to_excel(
                    writer, sheet_name='Summary', float_format='%.2f')
                print("\n--- 对比总结报告已准备写入工作表: Summary ---")

                # 'with' 语句结束时，writer 会自动保存并关闭文件

            print(f"\n✅ 完整回测报告已成功保存到文件: {output_filename}")

            # (可选) 在终端打印最终的总结报告
            pd.options.display.float_format = '{:,.2f}'.format
            print("\n" + "="*30 + " 不同步长参数回测对比报告 " + "="*30)
            print(results_df.to_string())
            print("=" * 82)

        except Exception as e:
            print(f"\n❌ 保存到 Excel 文件时出错: {e}")
            print("请确保您已正确安装 'openpyxl' 库 (运行: pip install openpyxl)")
