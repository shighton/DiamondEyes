# Author: Sabastian Highton
# Info: Used for crypto API calling and storing relevant information into csv file

import csv
from datetime import datetime
from dotenv import load_dotenv
import os
import requests
import re

RAPID_API_HOST = "coinranking1.p.rapidapi.com"

# BTC
coin_id = "Qwsogvtv82FCd"
inputData = {}

load_dotenv('../.env')

def fetchStockData(coin_uuid):
    response = requests.request("GET", "https://api.coinranking.com/v2/coin/" + coin_uuid,
                                headers={'X-RapidAPI-Key': os.getenv('RAPID_API_KEY'),
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


# For use on past data
# Add the time changes between each bitcoin info check to the history.csv file
def addTimesChanges():
    calendarTimeList = []
    f = open("history.csv", "r")
    for line in f:
        match = re.match(r'\d+:\d+-\d+/\d+/\d+', line)
        if match:
            calendarTimeList.append(datetime.strptime(match.group(), '%H:%M-%m/%d/%Y'))
    timestamps = []
    for time in calendarTimeList:
        ts = datetime.timestamp(time)
        timestamps.append(ts)
    times_between = []
    for i in range(len(timestamps)):
        if i == 0:
            temp = timestamps[i]
            times_between.append(0)
        elif i == 1:
            times_between.append(timestamps[i] - temp)
        else:
            times_between.append(timestamps[i] - timestamps[i - 1])
    fread = open("history.csv", "r")
    save = fread.read().split()
    file = open("history.csv", "w")
    i = -1
    for line in save:
        if i == -1:
            file.write(line + '\n')
        elif i == 0:
            file.write(line.split()[0] + ",0" + '\n')
        else:
            file.write(line.split()[0] + "," + str(int(times_between[i])) + '\n')
        i += 1


# For use on new data
# Add the time change between each bitcoin info check to the history.csv file
def parseTimeChange():
    fread = open("history.csv", "r")
    fread_split = fread.read().split()
    second_last_date = datetime.timestamp(datetime.strptime(fread_split[-2].split(',')[0], '%H:%M-%m/%d/%Y'))
    last_date = datetime.timestamp(datetime.strptime(fread_split[-1].split(',')[0], '%H:%M-%m/%d/%Y'))
    time_between = last_date - second_last_date
    file = open("history.csv", "w")
    i = 0
    for line in fread_split:
        if fread_split[i] != fread_split[-1]:
            file.write(line + '\n')
        else:
            file.write(line.split()[0] + "," + str(int(time_between)) + '\n')
        i += 1


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
    parseTimeChange()


# run()
