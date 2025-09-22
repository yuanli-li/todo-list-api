import requests
import pandas as pd
from datetime import datetime, timedelta
import os  # 引入os模块来检查文件是否存在

# ===== 参数 =====
CAPITAL = 10000        # 初始资金 (USDT)
FEE_RATE = 0.00026      # 手续费 (单边)
LOWER, UPPER = 3000, 5000  # 您可以根据当前市场调整这个范围
STEP = 65              # 网格间距
SYMBOL = "ETHUSDT"
INTERVAL = "1m"
DAYS = 60

# 定义本地数据文件名
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

# ===== 网格回测 (多仓位版本) =====


def simulate(df, levels, capital):
    if not levels:
        print("错误：网格为空，无法回测。")
        return [], 0.0, capital

    per_grid_capital = capital / len(levels)
    cash = capital
    open_positions = []
    bought_levels = set()

    trades = []
    realized_pnl = 0.0

    step = levels[1] - levels[0] if len(levels) > 1 else 0

    print("开始回测...")
    for i in range(len(df)):
        lo, hi, ts = df.iloc[i]['low'], df.iloc[i]['high'], df.iloc[i]['datetime']

        # --- 卖出逻辑 ---
        for position in list(open_positions):
            buy_price = position['price']
            sell_price = buy_price + step

            if hi >= sell_price:
                proceeds = position['qty'] * sell_price
                fee = proceeds * FEE_RATE
                cash += (proceeds - fee)
                realized_pnl += (proceeds - fee) - position['cost']

                trades.append((ts, "SELL", round(sell_price, 2), buy_price))

                open_positions.remove(position)
                if buy_price in bought_levels:
                    bought_levels.remove(buy_price)

        # --- 买入逻辑 ---
        touched_levels = [lv for lv in levels if lo <= lv <= hi]
        for lv in touched_levels:
            if lv not in bought_levels and cash >= per_grid_capital:
                cost_before_fee = per_grid_capital
                qty = cost_before_fee / lv
                fee = cost_before_fee * FEE_RATE
                total_cost = cost_before_fee + fee

                if cash < total_cost:
                    continue

                cash -= total_cost

                new_position = {"price": lv, "qty": qty, "cost": total_cost}
                open_positions.append(new_position)
                bought_levels.add(lv)

                trades.append((ts, "BUY", lv, None))

    print("回测结束。")
    final_equity = cash
    last_close_price = df.iloc[-1]['close']
    for position in open_positions:
        final_equity += position['qty'] * last_close_price

    return trades, realized_pnl, final_equity


# ===== 主程序 (已修改，增加本地缓存功能) =====
if __name__ == "__main__":

    # 检查本地数据文件是否存在
    if os.path.exists(DATA_FILENAME):
        print(f"发现本地数据文件 '{DATA_FILENAME}'，正在加载...")
        df = pd.read_csv(DATA_FILENAME)
        # **重要**: CSV中的日期/时间列在读取时是字符串，需要转换回datetime对象
        df['datetime'] = pd.to_datetime(df['datetime'])
        print("数据加载完毕！")
    else:
        # 如果文件不存在，则从API获取并保存
        df = fetch_binance_klines(SYMBOL, INTERVAL, DAYS)
        if not df.empty:
            # 使用 index=False 避免将DataFrame的索引写入CSV文件
            df.to_csv(DATA_FILENAME, index=False)
            print(f"数据已保存到 '{DATA_FILENAME}' 以便将来使用。")

    # --- 后续逻辑不变 ---
    if df.empty:
        print("错误：未能获取K线数据，程序退出。")
    else:
        levels = build_levels(LOWER, UPPER, STEP)
        trades, realized, final_equity = simulate(df, levels, CAPITAL)

        trade_df = pd.DataFrame(
            trades, columns=["time", "side", "price", "linked_buy_price"])
        trade_df.to_csv("grid_trades_multi_position.csv", index=False)

        unrealized_pnl = final_equity - CAPITAL - realized
        total_pnl = final_equity - CAPITAL

        print("\n=== 多仓位网格回测结果 (Binance, 过去60天) ===")
        print(f"回测区间: {df.iloc[0]['datetime']} -> {df.iloc[-1]['datetime']}")
        print(f"网格范围: {LOWER} - {UPPER} USDT, 步长: {STEP}, 共 {len(levels)} 格")
        print("-" * 30)
        print(f"初始资金: {CAPITAL:.2f} USDT")
        print(f"期末净值: {final_equity:.2f} USDT")
        print(f"总盈亏  : {total_pnl:.2f} USDT ({total_pnl/CAPITAL:.2%})")
        print(f"  - 已实现盈亏: {realized:.2f} USDT")
        print(f"  - 未实现盈亏: {unrealized_pnl:.2f} USDT")
        print("-" * 30)
        print(f"总成交次数: {len(trades)}")
        print(f"买入次数  : {len(trade_df[trade_df['side'] == 'BUY'])}")
        print(f"卖出次数  : {len(trade_df[trade_df['side'] == 'SELL'])}")
        print(
            f"当前持仓数: {len(trade_df) - 2*len(trade_df[trade_df['side'] == 'SELL'])}")
        print("\n交易明细已保存到 grid_trades_multi_position.csv")
