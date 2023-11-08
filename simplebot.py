# Author: Marty MK (https://www.qmr.ai/cryptocurrency-trading-bot-with-alpaca-in-python/)
# Author: @huseinzol05 on GitHub
# Modified by Sabastian Highton
# Info: Evolution strategy agent (reinforcement learning)

from alpaca_trade_api.rest import REST
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import time
import types
import warnings
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

BASE_URL = "https://paper-api.alpaca.markets"
KEY_ID = 'PKNTDYKX9SW3RRWPBD0Z'
SECRET_KEY = '0HN9VFrPt0Gcpxkh0RMe6pCc9gA7vHAKHRaodfmQ'

# Instantiate REST API Connection
api = REST(key_id=KEY_ID, secret_key=SECRET_KEY, base_url=BASE_URL)

SYMBOL = ['BTC/USD']
SYM = 'BTCUSD'
starting_money = 100000
SMA_FAST = 12
SMA_SLOW = 24
QTY_PER_TRADE = 1

'''
# Description is given in the article
def get_pause():
    now = datetime.now()
    next_min = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    pause = math.ceil((next_min - now).seconds)
    print(f"Sleep for {pause}")
    return pause
'''


# Same as the function in the random version
def get_position(symbol):
    positions = api.list_positions()
    for p in positions:
        if p.symbol == symbol:
            return float(p.qty)
    return 0


def can_buy(symbol):
    val = get_position(symbol)
    snap = api.get_latest_crypto_quotes(SYMBOL)['BTC/USD'].ap
    if val > 0:
        switch = starting_money / (val + 1) > snap
    else:
        switch = starting_money > snap
    return switch


def can_sell(symbol):
    val = get_position(symbol)
    return val > QTY_PER_TRADE


# Returns a series with the moving average
def get_sma(series, periods):
    return series.rolling(periods).mean()


# Checks whether we should buy (fast ma > slow ma)
def get_signal(fast, slow):
    # print(f"Fast {fast[-1]}  /  Slow: {slow[-1]}")
    return fast[-1] > slow[-1]


def bollinger_bands(series: pd.Series, length: int = 20, *,
                    num_stds: tuple[float, ...] = (2, 0, -2), prefix: str = '') -> pd.DataFrame:
    # Ref: https://stackoverflow.com/a/74283044/
    rolling = series.rolling(length)
    bband0 = rolling.mean()
    bband_std = rolling.std(ddof=0)
    df = pd.DataFrame({f'{prefix}{num_std}': (bband0 + (bband_std * num_std)) for num_std in num_stds})
    sns.lineplot(df)
    plt.show()
    return df


# Get up-to-date 1 minute data from Alpaca and add the moving averages
def get_bars(symbol):
    yesterday_ts = datetime.timestamp(datetime.strptime(datetime.now().strftime('%Y-%m-%d'), '%Y-%m-%d')) - 86400
    yesterday = datetime.fromtimestamp(yesterday_ts).strftime('%Y-%m-%d')

    crypto_bars = api.get_crypto_bars(symbol, TimeFrame.Minute, start=yesterday).df
    crypto_bars[f'sma_fast'] = get_sma(crypto_bars.close, SMA_FAST)
    crypto_bars[f'sma_slow'] = get_sma(crypto_bars.close, SMA_SLOW)
    return crypto_bars


def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            name = val.__name__.split('.')[0]
        elif isinstance(val, type):
            name = val.__module__.split('.')[0]
        pkgs = {'PIL': 'Pillow', 'sklearn': 'scikit-learn'}
        if name in pkgs.keys():
            name = pkgs[name]
        yield name


no_action_count = 0

while True:
    # GET DATA
    bars = get_bars(symbol=SYMBOL)
    close = bars.close.values.tolist()
    # print(len(close))
    close_length = len(close) - 1

    # CHECK POSITIONS
    position = get_position(symbol=SYM)
    should_buy_sma = get_signal(bars.sma_fast, bars.sma_slow)
    able_buy = can_buy(SYM)
    able_sell = can_sell(SYM)
    # if ((((((position >= 0) & able_buy) & agent_good) & agent_buy) & should_buy_sma) & (agent_sell != True)):
    if (((position >= 0) & able_buy) & should_buy_sma):
        # print(f"\rPosition: {position} / Can Buy: {able_buy} / RL Buy: {agent_buy} / RL Sell: {agent_sell} / "
        # f"RL Good: {agent_good} / SMA Buy: {should_buy_sma}")
        print(f"\rPosition: {position} / Can Buy: {able_buy} / Can Sell: {able_sell} / SMA Buy: {should_buy_sma}")
        api.submit_order(SYM, qty=QTY_PER_TRADE, side='buy', time_in_force="gtc")
        print(f'Symbol: {SYM} / Side: BUY / Quantity: {QTY_PER_TRADE}')
        time.sleep(2)  # Give position time to update
        print(f"New Position: {get_position(symbol=SYM)}")
        print("*" * 20, 'buy\n')
        no_action_count = 0
    # elif ((((((position > 0) & able_sell) & agent_good) & (agent_buy != True)) & agent_sell) & (
    # should_buy_sma != True)):
    elif (((position >= 0) & able_sell) & (should_buy_sma != True)):
        # print(f"\rPosition: {position} / Can Buy: {able_buy} / RL Buy: {agent_buy} / RL Sell: {agent_sell} / "
        # f"RL Good: {agent_good} / SMA Buy: {should_buy_sma}")
        print(f"\rPosition: {position} / Can Buy: {able_buy} / Can Sell: {able_sell} / SMA Buy: {should_buy_sma}")
        api.submit_order(SYM, qty=QTY_PER_TRADE, side='sell', time_in_force="gtc")
        print(f'Symbol: {SYM} / Side: SELL / Quantity: {QTY_PER_TRADE}')
        time.sleep(2)  # Give position time to update
        print(f"New Position: {get_position(symbol=SYM)}")
        print("*" * 20, 'sell\n')
        no_action_count = 0
    else:
        # print(f"\rPosition: {position} / Can Buy: {able_buy} / RL Buy: {agent_buy} / RL Sell: {agent_sell} / "
        # f"RL Good: {agent_good} / SMA Buy: {should_buy_sma}", end='')
        print(f"\rPosition: {position} / Can Buy: {able_buy} / Can Sell: {able_sell} / "
              f"SMA Buy: {should_buy_sma}", end='')
        time.sleep(5)
        no_action_count += 1
        print('\r' + 'No action #' + str(no_action_count), end='')
        time.sleep(2)
        for i in range(50):
            print('\r' + str(i) + ' / 50', end='')
            time.sleep(1)
            i += 1
