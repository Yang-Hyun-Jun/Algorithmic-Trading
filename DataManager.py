import pandas as pd

Features_Raw = ["Open", "High", "Low", "Close"]

def load_data(path, date_start, date_end):
    data = pd.read_csv(path, thousands=",", converters={"Date": lambda x: str(x)})

    date_start = date_start.replace("-", "")
    date_end = date_end.replace("-", "")
    data["Date"] = data["Date"].str.replace("-","")
    data = data[(data["Date"] >= date_start) & (data["Date"] <= date_end)]
    data = data.set_index("Date")
    data = data.dropna()
    return data.loc[:,Features_Raw]

if __name__ == "__main__":
    path = "/Users/macbook/Desktop/OHLCV_data/KOSPI_OHLCV/005930" #삼성전자
    data = load_data(path=path, date_start="20170101", date_end="20170131")
    print(data)
