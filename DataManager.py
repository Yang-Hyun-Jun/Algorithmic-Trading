import pandas as pd
import numpy as np

Features_Raw = ["Open", "High", "Low", "Close", "Volume"]
Features_Raw2 = ["Open", "High", "Low", "Close", "Trend"]

def load_data(path, date_start=None, date_end=None):
    data = pd.read_csv(path, thousands=",", converters={"Date": lambda x: str(x)})
    date_start = data["Date"].iloc[0] if date_start == None else date_start
    date_end = data["Date"].iloc[-1] if date_end == None else date_end

    date_start = date_start.replace("-", "")
    date_end = date_end.replace("-", "")
    data["Date"] = data["Date"].str.replace("-","")
    data = data[(data["Date"] >= date_start) & (data["Date"] <= date_end)]
    data = data.set_index("Date")
    data = data.dropna()
    return data.loc[:,Features_Raw]


def get_moving_average(data, w):
    data["Close_ma{}".format(w)] = data["Close"].rolling(w).mean()
    return data


def get_moving_trend(data, w=20, v=5):
    data = get_moving_average(data, w=w)
    data["Trend"] = np.zeros(len(data))-1

    index = list(range(len(data)))
    del index[:w + v]
    for i in index:
        values = []
        for k in range(v + 1):
            values.append(data["Close_ma{}".format(w)].iloc[i - k])
        values_up = values.copy()
        values_up.sort()
        values_down = values.copy()
        values_down.sort(reverse=True)

        if values_down == values:
            data["Trend"].copy()[i] = 0  # Up-Trend
        elif values_up == values:
            data["Trend"].copy()[i] = 1  # Down-Trend
        else:
            data["Trend"].copy()[i] = 2  # Side-Trend
    return data.loc[:, Features_Raw2].iloc[w+v:]

def get_ulbv_vector(data):
    data["Upper"] = np.zeros(len(data))
    data["Body"] = np.zeros(len(data))
    data["Lower"] = np.zeros(len(data))
    data["Direc"] = np.zeros(len(data))

    data["Body"] = data["High"] - data["Low"]
    data["Upper"] = (data["High"] - data.loc[:, ["Open", "Close"]].max(axis=1)) / data["Body"]
    data["Lower"] = (data.loc[:, ["Open", "Close"]].min(axis=1) - data["Low"]) / data["Body"]
    data["Direc"] = (data["Open"] - data["Close"]) / abs(data["Open"] - data["Close"])
    data = data.fillna(0)
    return data


if __name__ == "__main__":
    path = "/Users/macbook/Desktop/OHLCV_data/ALL_OHLCV/005930"
    data = load_data(path=path)
    data = get_moving_trend(data)
    data = get_ulbv_vector(data)
    print(data)


