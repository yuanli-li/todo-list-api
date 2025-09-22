import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# ===== 参数 =====
CAPITAL = 10000
FEE_RATE = 0.00026
LOWER, UPPER = 3000, 5000
STEP = 200
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
    print(f"本地数据文件 '{DATA_FILENAME}' 不存在。")
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
# ===== 网格回测 (多仓位版本) - 最终修复版 =====
# ===== 网格回测 (多仓位版本) - 修正版“中位启动”初始化 =====


def simulate(df, levels, capital):
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

    # --- 【完全修正】正确的中位启动初始化逻辑 ---
    print("开始“中位启动”初始化...")
    if df.empty:
        print("错误：数据为空，无法进行初始化。")
        return [], 0.0, capital, 0.0

    # 1. 获取初始市价和时间戳
    initial_price = df.iloc[0]['close']
    initial_ts = df.iloc[0]['datetime']

    # 2. 识别所有高于初始价格的网格线。这些是我们未来要卖出的位置，所以现在必须买入资产。
    initial_buy_levels = [lv for lv in levels if lv > initial_price]
    print(
        f"初始价格: {initial_price:.2f} USDT。将在其上方的 {len(initial_buy_levels)} 个网格上批量建仓。")

    # 3. 循环执行批量建仓，买入未来要卖出的“库存”
    for lv in sorted(initial_buy_levels):  # 从低到高建仓
        # 检查是否有足够的资金用于建仓
        if cash < per_grid_capital:
            print(f"警告：初始建仓时资金不足，无法在 {lv} USDT 及更高价位建仓。")
            break

        # 【核心修正】我们用每格分配的资金 (per_grid_capital)，以当前市价 (initial_price) 买入资产
        cost_before_fee = per_grid_capital
        qty = cost_before_fee / initial_price  # <--- 在这里使用 initial_price

        fee = cost_before_fee * FEE_RATE
        total_fees += fee
        total_cost = cost_before_fee + fee

        if cash < total_cost:
            continue

        cash -= total_cost

        # 创建新仓位。注意：仓位的'price'仍然是其所属的网格线'lv'，这决定了它未来的卖出点。
        new_position = {"price": lv, "qty": qty, "cost": total_cost}
        open_positions.append(new_position)
        bought_levels.add(lv)

        # 记录交易。价格记录为'lv'，表示我们填充了哪个网格。
        trades.append((initial_ts, "INIT_BUY", lv,
                      f"Market buy at {initial_price:.2f}", cost_before_fee))

    print(f"初始化完成。共建仓 {len(open_positions)} 个。剩余现金: {cash:.2f} USDT。")
    print("开始回测...")

    # --- 主循环从第二根K线开始 ---
    for i in range(1, len(df)):
        lo, hi, ts = df.iloc[i]['low'], df.iloc[i]['high'], df.iloc[i]['datetime']

        levels_sold_this_tick = set()

        # --- 卖出逻辑 (无变动) ---
        sellable_positions = []
        for position in open_positions:
            sell_price = position['price'] + step
            if hi >= sell_price:
                sellable_positions.append(position)

        if sellable_positions:
            sorted_sellables = sorted(
                sellable_positions, key=lambda p: p['price'])
            for position in sorted_sellables:
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

        # --- 买入逻辑 (无变动) ---
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

    print("回测结束。")
    final_equity = cash
    last_close_price = df.iloc[-1]['close']
    for position in open_positions:
        final_equity += position['qty'] * last_close_price

    return trades, realized_pnl, final_equity, total_fees


# ===== 主程序 (已修改统计逻辑) =====
if __name__ == "__main__":

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
        levels = build_levels(LOWER, UPPER, STEP)
        trades, realized, final_equity, total_fees = simulate(
            df, levels, CAPITAL)

        trade_df = pd.DataFrame(
            trades, columns=["time", "side", "price", "linked_buy_price", "amount_usdt"])
        trade_df.to_csv("grid_trades_multi_position.csv", index=False)

        # ==========================================================
        # ===== 这就是您需要放置新代码的位置 =====
        # ==========================================================
        total_pnl = final_equity - CAPITAL
        unrealized_pnl = total_pnl - realized

        # 分别统计不同类型的交易
        init_buy_trades_count = len(trade_df[trade_df['side'] == 'INIT_BUY'])
        buy_trades_count = len(trade_df[trade_df['side'] == 'BUY'])
        sell_trades_count = len(trade_df[trade_df['side'] == 'SELL'])

        # 正确计算当前持仓数
        current_positions = init_buy_trades_count + buy_trades_count - sell_trades_count

        total_volume = trade_df['amount_usdt'].sum()
        avg_profit_per_sell = realized / sell_trades_count if sell_trades_count > 0 else 0

        # --- 修改并美化输出格式 ---
        print("\n" + "="*20 + " 多仓位网格回测结果 " + "="*20)
        print(f"回测标的: {SYMBOL}, 时间周期: {INTERVAL}")
        print(f"回测区间: {df.iloc[0]['datetime']} -> {df.iloc[-1]['datetime']}")
        print(f"网格参数: 范围 {LOWER}-{UPPER}, 步长 {STEP}, 共 {len(levels)} 格")

        print("\n" + "-"*15 + " 盈亏分析 (PNL Analysis) " + "-"*15)
        print(f"初始资金: {CAPITAL:12.2f} USDT")
        print(f"期末净值: {final_equity:12.2f} USDT")
        print(f"总 盈 亏: {total_pnl:12.2f} USDT ({total_pnl/CAPITAL:8.2%})")
        print(f"  - 已实现盈亏: {realized:12.2f} USDT")
        print(f"  - 未实现盈亏: {unrealized_pnl:12.2f} USDT")

        print("\n" + "-"*15 + " 交易统计 (Trade Stats) " + "-"*15)
        # 【修改】使用新的变量来打印
        print(f"总成交次数: {len(trades):<5}")
        print(f"  - 买入次数: {buy_trades_count:<5} (不含初始建仓)")
        print(f"  - 卖出次数: {sell_trades_count:<5}")
        print(f"当前持仓数: {current_positions:<5} (初始建仓: {init_buy_trades_count})")
        print("-" * 52)
        print(f"总交易量  : {total_volume:12.2f} USDT")
        print(f"总手续费  : {total_fees:12.2f} USDT")
        print(f"单次卖出平均利润: {avg_profit_per_sell:8.2f} USDT (已实现利润 / 卖出次数)")

        print("\n交易明细已保存到 grid_trades_multi_position.csv")
        print("=" * 62)
