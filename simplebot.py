# Author: Sabastian Highton

from alpaca_trade_api.rest import REST
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from datetime import datetime
from datetime import timedelta
from dotenv import load_dotenv
import os
import time
import warnings
import pandas as pd

# IDEAS:
# - Time the discrepency between "latest" calculation and final action taken

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

BASE_URL = "https://paper-api.alpaca.markets"

# get environment variables (Create a .env file with Alpaca Keys for new users - Sabastian)
try:
    load_dotenv('.env')
except:
    pass

try:
    load_dotenv('../.env')
except:
    pass

# Instantiate REST API Connection
api = REST(key_id=os.getenv('KEY_ID'), secret_key=os.getenv('SECRET_KEY'), base_url=BASE_URL)

SYMBOL = ['BTC/USD']
SYM = 'BTCUSD'
equity = float(TradingClient(os.getenv('KEY_ID'), os.getenv('SECRET_KEY')).get_account().equity)
SMA_FAST = 10
SMA_SLOW = 20
QTY_PER_TRADE = 0.5


def get_position(symbol):
    positions = api.list_positions()
    for p in positions:
        if p.symbol == symbol:
            return float(p.qty)
    return 0


def can_buy(symbol):
    val = get_position(symbol)
    snap = api.get_latest_crypto_quotes(SYMBOL)['BTC/USD'].ap
    if val > QTY_PER_TRADE:
        switch = (equity / (val + QTY_PER_TRADE)) > snap
    else:
        switch = equity > snap
    return switch


def can_sell(symbol):
    val = get_position(symbol)
    return val > QTY_PER_TRADE


# Returns a series with the moving average
def get_sma(series, periods):
    return series.rolling(periods).mean()


# Checks whether we should buy (fast ma > slow ma)
def get_signal(fast, slow):
    return fast[-1] > slow[-1]


def bollinger_bands(series: pd.Series, length: int = 20, *,
                    num_stds: tuple[float, ...] = (2, 0, -2), prefix: str = '') -> pd.DataFrame:
    # Ref: https://stackoverflow.com/a/74283044/
    rolling = series.rolling(length)
    b_band0 = rolling.mean()
    b_band_std = rolling.std(ddof=0)
    df = pd.DataFrame({f'{prefix}{num_std}': (b_band0 + (b_band_std * num_std)) for num_std in num_stds})
    return df


def over_bought_and_sold(the_bars, df):
    o_sold_recent = False
    o_bought_recent = False
    current_price = the_bars.close.values.tolist()[-1]

    if current_price < df[df.columns[2]].iloc[-1]:
        o_sold_recent = True
    if current_price > df[df.columns[0]].iloc[-1]:
        o_bought_recent = True

    return o_sold_recent, o_bought_recent


# Get up-to-date 1 minute data from Alpaca and add the moving averages
def get_bars(symbol):
    yesterday_ts = datetime.timestamp(datetime.strptime(datetime.now().strftime('%Y-%m-%d'), '%Y-%m-%d')) - 86400
    yesterday = datetime.fromtimestamp(yesterday_ts).strftime('%Y-%m-%d')

    crypto_bars = api.get_crypto_bars(symbol, TimeFrame.Minute, start=yesterday).df
    crypto_bars[f'sma_fast'] = get_sma(crypto_bars.close, SMA_FAST)
    crypto_bars[f'sma_slow'] = get_sma(crypto_bars.close, SMA_SLOW)
    return crypto_bars


def get_latest():
    _bars = get_bars(symbol=SYMBOL)
    _close = _bars.close.values.tolist()
    _latest = _close[-1]
    return _latest


def get_transactions():
    activities_df = pd.DataFrame(api.get_activities())
    transactions = []

    for i in range(len(activities_df)):
        transactions.append([activities_df.values[i][0].price, activities_df.values[i][0].side,
                             activities_df.values[i][0].qty, activities_df.values[i][0].transaction_time.value])

    transactions.reverse()

    return transactions


def get_active_positions():
    transactions = get_transactions()
    active_positions = []
    active_positions_costs = []
    buy = 0
    sell = 0
    been_a_week = False

    for i in range(len(transactions)):
        if transactions[i][1] == 'buy':
            buy += float(transactions[i][2])
            active_positions.append(transactions[i])
        if transactions[i][1] == 'sell':
            sell += float(transactions[i][2])

    qty_difference = buy - sell

    num_active_trades = int(qty_difference // QTY_PER_TRADE)

    if num_active_trades < 0:
        active_positions = active_positions[num_active_trades:]
    else:
        active_positions = active_positions[-num_active_trades:]

    if num_active_trades > 0:
        been_a_week = True if (datetime.now() - datetime.fromtimestamp((active_positions[-1][3] / 1000000000))) > \
                              timedelta(days=7) else False

    for i in range(len(active_positions)):
        active_positions_costs.append(float(active_positions[i][0]))

    return active_positions_costs, been_a_week


def run():
    no_action_count = 0
    transactions, no_trades_in_a_week = get_active_positions()

    while True:
        # Data collection
        bars = get_bars(symbol=SYMBOL)
        close = bars.close.values.tolist()
        latest = close[-1]

        if len(transactions) == 0:
            transactions.append(latest)

        if len(close) > 20:
            band_df = bollinger_bands(pd.Series(close))
            position = get_position(symbol=SYM)

            # Boolean values for conditions
            able_buy = can_buy(SYM)
            able_sell = can_sell(SYM)
            should_buy_sma = get_signal(bars.sma_fast, bars.sma_slow)
            o_sold, o_bought = over_bought_and_sold(bars, band_df)
            sell_high = latest > (transactions[-1] * 1.004)

            if ((((position >= 0) & able_buy) & should_buy_sma) & (o_sold | (o_bought == False))):
                print(f"\rPosition: {position} / Can Buy: {'T' if able_buy else 'F'} /"
                      f" Can Sell: {'T' if able_sell else 'F'} / SMA Buy: {'T' if should_buy_sma else 'F'}"
                      f" / Oversold: {'T' if o_sold else 'F'} / Sell High: {'T' if sell_high else 'F'}")
                api.submit_order(SYM, qty=QTY_PER_TRADE, side='buy', time_in_force="gtc")
                print(f'Symbol: {SYM} / Side: BUY / Quantity: {QTY_PER_TRADE}')
                latest = get_latest()
                transactions.append(latest)
                time.sleep(2)  # Give position time to update
                print(f"New Position: {get_position(symbol=SYM)}")
                print("*" * 20, 'buy\n')
                no_action_count = 0
                _, no_trades_in_a_week = get_active_positions()
            elif (((position >= 0) & no_trades_in_a_week) | ((((position >= 0) & able_sell) & (
                    should_buy_sma == False)) & sell_high) & (o_bought | (o_sold == False))):
                print(f"\rPosition: {position} / Can Buy: {'T' if able_buy else 'F'} /"
                      f" Can Sell: {'T' if able_sell else 'F'} / SMA Buy: {'T' if should_buy_sma else 'F'}"
                      f" / Overbought: {'T' if o_bought else 'F'} / Sell High: {'T' if sell_high else 'F'}"
                      f" / Trade Dump: {'T' if no_trades_in_a_week else 'F'}")
                if able_sell:
                    api.submit_order(SYM, qty=QTY_PER_TRADE, side='sell', time_in_force="gtc")
                else:
                    api.submit_order(SYM, qty=position, side='sell', time_in_force="gtc")
                print(f'Symbol: {SYM} / Side: SELL / Quantity: {QTY_PER_TRADE} / Last Buy: {transactions[-1]}'
                      f' / Latest Close {latest}')
                transactions.pop()
                # if len(transactions) == 0:
                #     latest = get_latest()
                #     transactions.append(latest)
                time.sleep(2)  # Give position time to update
                print(f"New Position: {get_position(symbol=SYM)}")
                print("*" * 20, 'sell\n')
                no_action_count = 0
                _, no_trades_in_a_week = get_active_positions()
            else:
                print(f"\rPosition: {position} / Can Buy: {'T' if able_buy else 'F'} /"
                      f" Can Sell: {'T' if able_sell else 'F'} / SMA Buy: {'T' if should_buy_sma else 'F'}"
                      f" / Overbought: {'T' if o_bought else 'F'} / Oversold: {'T' if o_sold else 'F'} /"
                      f" Sell High: {'T' if sell_high else 'F'}", end='')
                time.sleep(5)
                no_action_count += 1
                for i in range(50):
                    print('\r' + 'No action #' + str(no_action_count) + '. Seconds until next trade: ' +
                          str(50 - i), end='')
                    time.sleep(1)
                    i += 1
        else:
            print("Waiting for required data.")
            time.sleep(1200)
