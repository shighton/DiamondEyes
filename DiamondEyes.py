import csv

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import requests
from matplotlib import rcParams

RAPID_API_KEY = "4b6872356fmsh4de9af3f8449e91p1640e8jsn691817a923ae"
RAPID_API_HOST = "coinranking1.p.rapidapi.com"

# BTC
coin_id = "Qwsogvtv82FCd"
inputData = {}


def fetchStockData(coin_uuid):
    response = requests.request("GET", "https://api.coinranking.com/v2/coin/" + coin_uuid,
                                headers={'X-RapidAPI-Key': RAPID_API_KEY,
                                         'X-RapidAPI-Host': RAPID_API_HOST})

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Error")
        return None


def parseTimestamp(input_data):
    timestampList = [input_data["data"]["coin"]["priceAt"]]
    calendarTime = []
    for ts in timestampList:
        dt = datetime.fromtimestamp(ts)
        calendarTime.append(dt.strftime("%H:%M-%m/%d/%Y"))
    return calendarTime


def parseValues(input_data):
    valueList = [input_data["data"]["coin"]["price"],
                 input_data["data"]["coin"]["24hVolume"],
                 input_data["data"]["coin"]["change"]]
    return valueList


def run():
    if inputData is not None:
        coin_data = fetchStockData(coin_id)
        inputData["Timestamp"] = parseTimestamp(coin_data)
        inputData["Value"] = parseValues(coin_data)
        with open('history.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(inputData["Timestamp"] + inputData["Value"])


run()

df = pd.read_csv('history.csv')

sns.set(style='darkgrid')
rcParams['figure.figsize'] = 13, 5
rcParams['figure.subplot.bottom'] = 0.2

ax = sns.lineplot(x='Timestamp', y='Value', dashes=False, markers=True, data=df, sort=False)
ax.set_title('Coin: ' + 'BTC')
plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize='xx-small')
plt.show()
