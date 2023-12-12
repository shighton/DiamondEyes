# Author: Marty MK (https://www.qmr.ai/cryptocurrency-trading-bot-with-alpaca-in-python/)
# Author: @huseinzol05 on GitHub
# Modified by Sabastian Highton
# Info: Evolution strategy agent (reinforcement learning)

from alpaca_trade_api.rest import REST
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from datetime import datetime
from dotenv import load_dotenv
import os
import time
import types
import numpy as np
import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

BASE_URL = "https://paper-api.alpaca.markets"

load_dotenv('../.env')

# Instantiate REST API Connection
api = REST(key_id=os.getenv('KEY_ID'), secret_key=os.getenv('SECRET_KEY'), base_url=BASE_URL)

SYMBOL = ['BTC/USD']
SYM = 'BTCUSD'
equity = float(TradingClient(os.getenv('KEY_ID'), os.getenv('SECRET_KEY')).get_account().equity)
SMA_FAST = 12
SMA_SLOW = 24
QTY_PER_TRADE = 1


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
        switch = equity / (val + 1) > snap
    else:
        switch = equity > snap
    return switch


def can_sell(symbol):
    val = get_position(symbol)
    return val > QTY_PER_TRADE


# Get up-to-date 1 minute data from Alpaca and add the moving averages
def get_bars(symbol):
    yesterday_ts = datetime.timestamp(datetime.strptime(datetime.now().strftime('%Y-%m-%d'), '%Y-%m-%d')) - 86400
    yesterday = datetime.fromtimestamp(yesterday_ts).strftime('%Y-%m-%d')

    crypto_bars = api.get_crypto_bars(symbol, TimeFrame.Minute, start=yesterday).df
    return crypto_bars


def get_latest():
    _bars = get_bars(symbol=SYMBOL)
    _close = _bars.close.values.tolist()
    _latest = _close[-1]
    return _latest


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


def get_state(data, end_index, num_considered):
    start_index = end_index - num_considered + 1
    data_block = data[start_index: end_index + 1] if start_index >= 0 else -start_index * [data[0]] + \
                                                                           data[0: end_index + 1]
    result = []
    for i in range(num_considered - 1):
        result.append(data_block[i + 1] - data_block[i])
    return np.array([result])


def get_transactions():
    activities_df = pd.DataFrame(api.get_activities())
    transactions = []

    for i in range(len(activities_df)):
        transactions.append([activities_df.values[i][0].price, activities_df.values[i][0].side,
                             activities_df.values[i][0].qty])

    transactions.reverse()
    return transactions


def get_active_positions():
    transactions = get_transactions()
    active_positions = []
    active_positions_costs = []
    buy = 0
    sell = 0

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
        active_positions = []

    for i in range(len(active_positions)):
        active_positions_costs.append(float(active_positions[i][0]))

    return active_positions_costs


class DeepEvolutionStrategy:
    inputs = None

    def __init__(self, weights, reward_function, population_size, sigma, learning_rate):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):
        weights_population = []
        for index, i in enumerate(population):
            weight_offset = self.sigma * i
            weights_population.append(weights[index] + weight_offset)
        return weights_population

    def get_weights(self):
        return self.weights

    def train(self, epoch=100, print_every=1):
        last_time = time.time()
        for i in range(epoch):
            population = []
            rewards = np.zeros(self.population_size)
            for k in range(self.population_size):
                x_list = []
                for w in self.weights:
                    x_list.append(np.random.randn(*w.shape))
                population.append(x_list)
            for k in range(self.population_size):
                weights_population = self._get_weight_from_population(self.weights, population[k])
                rewards[k] = self.reward_function(weights_population)
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                        w
                        + self.learning_rate
                        / (self.population_size * self.sigma)
                        * np.dot(A.T, rewards).T
                )

            if i == epoch // 2:
                if self.reward_function(self.weights) < 0:
                    break

            print('\rEpoch ' + str(i) + ' out of ' + str(epoch) + ' reward: '
                  + '{0:.2f}'.format(self.reward_function(self.weights)), end='')
            if (i + 1) % print_every == 0:
                print(
                    '\riter %d. reward: %f'
                    % (i + 1, self.reward_function(self.weights)), end=''
                )


class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.randn(input_size, layer_size),
            np.random.randn(layer_size, output_size),
            np.random.randn(layer_size, 1),
            np.random.randn(1, layer_size),
        ]

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        decision = np.dot(feed, self.weights[1])
        buy = np.dot(feed, self.weights[2])
        return decision, buy

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class Agent:
    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(self, a_model, money, max_buy, max_sell):
        self.model = a_model
        self.initial_money = money
        self.max_buy = max_buy
        self.max_sell = max_sell
        self.es = DeepEvolutionStrategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )

    def act(self, sequence):
        decision, buy = self.model.predict(np.array(sequence))
        return np.argmax(decision[0]), int(buy[0])

    def get_reward(self, weights):
        money = self.initial_money
        start_money = money
        self.model.weights = weights
        state = get_state(close, 0, window_size + 1)
        inventory = []
        quantity = 0
        for t in range(0, close_length, step_size):
            action, buy = self.act(state)
            next_state = get_state(close, t + 1, window_size + 1)
            if action == 1 and money >= close[t]:
                if buy < 0:
                    buy = 1
                if buy > self.max_buy:
                    buy_units = self.max_buy
                else:
                    buy_units = buy
                total_buy = buy_units * close[t]
                money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units
            elif action == 2 and len(inventory) > 0:
                if quantity > self.max_sell:
                    sell_units = self.max_sell
                else:
                    sell_units = quantity
                quantity -= sell_units
                total_sell = sell_units * close[t]
                money += total_sell

            state = next_state
        return ((money - start_money) / start_money) * 100

    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every=checkpoint)

    def buy(self):
        state = get_state(close, 0, window_size + 1)
        money = self.initial_money
        start_money = money
        states_sell = []
        states_buy = []
        inventory = []
        quantity = 0
        buy_switch = False
        sell_switch = False
        for t in range(0, close_length, step_size):
            action, buy = self.act(state)
            next_state = get_state(close, t + 1, window_size + 1)
            if action == 1 and money >= close[t]:
                if buy < 0:
                    buy = 1
                if buy > self.max_buy:
                    buy_units = self.max_buy
                else:
                    buy_units = buy
                if close[t] in close[close_length - 2:]:
                    buy_switch = True
                total_buy = buy_units * close[t]
                money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units
                states_buy.append(t)
            elif action == 2 and len(inventory) > 0:
                bought_price = inventory.pop(0)
                if quantity > self.max_sell:
                    sell_units = self.max_sell
                else:
                    sell_units = quantity
                if sell_units < 1:
                    continue
                if close[t] in close[close_length - 2:]:
                    sell_switch = True
                quantity -= sell_units
                total_sell = sell_units * close[t]
                money += total_sell
                states_sell.append(t)
                try:
                    invest = ((total_sell - bought_price) / bought_price) * 100
                except:
                    invest = 0
            state = next_state

        invest = ((money - start_money) / start_money) * 100
        return buy_switch, sell_switch, invest > 0


window_size = 10
step_size = 1

bars = get_bars(symbol=SYMBOL)
close = bars.close.values.tolist()
close_length = len(close) - 1
no_action_count = 0
transactions = get_active_positions()

while True:
    bars = get_bars(symbol=SYMBOL)
    close = bars.close.values.tolist()
    close_length = len(close) - 1
    latest = close[-1]

    if len(transactions) == 0:
        transactions.append(latest)

    position = get_position(symbol=SYM)

    model = Model(input_size=window_size, layer_size=500, output_size=3)
    agent = Agent(a_model=model, money=equity, max_buy=1, max_sell=1)
    agent.fit(iterations=50, checkpoint=50)

    able_buy = can_buy(SYM)
    able_sell = can_sell(SYM)
    agent_buy, agent_sell, agent_good = agent.buy()

    if ((((position >= 0) & able_buy) & agent_good) & agent_buy):
        print(f"\rPosition: {position} / Can Buy: {'T' if able_buy else 'F'} /"
              f" Can Sell: {'T' if able_sell else 'F'}")
        api.submit_order(SYM, qty=QTY_PER_TRADE, side='buy', time_in_force="gtc")
        print(f'Symbol: {SYM} / Side: BUY / Quantity: {QTY_PER_TRADE}')
        latest = get_latest()
        transactions.append(latest)
        time.sleep(2)  # Give position time to update
        print(f"New Position: {get_position(symbol=SYM)}")
        print("*" * 20, 'buy\n')
        no_action_count = 0
    elif ((((((position >= 0) & able_sell)) & agent_good) & (agent_buy != True)) & agent_sell):
        print(f"\rPosition: {position} / Can Buy: {'T' if able_buy else 'F'} /"
              f" Can Sell: {'T' if able_sell else 'F'}")
        api.submit_order(SYM, qty=QTY_PER_TRADE, side='sell', time_in_force="gtc")
        print(f'Symbol: {SYM} / Side: SELL / Quantity: {QTY_PER_TRADE}')
        transactions.pop()
        if len(transactions) == 0:
            latest = get_latest()
            transactions.append(latest)
        time.sleep(2)  # Give position time to update
        print(f"New Position: {get_position(symbol=SYM)}")
        print("*" * 20, 'sell\n')
        no_action_count = 0
    else:
        print(f"\rPosition: {position} / Can Buy: {'T' if able_buy else 'F'} /"
              f" Can Sell: {'T' if able_sell else 'F'}", end='')
        time.sleep(5)
        no_action_count += 1
        for i in range(50):
            print('\r' + 'No action #' + str(no_action_count) + '. Seconds until next trade: ' +
                  str(50 - i), end='')
            time.sleep(1)
            i += 1

