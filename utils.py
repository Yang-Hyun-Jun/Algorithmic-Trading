import numpy as np

stock_code = "005930"
start_date = "20140101"
end_date = "20170131"
Base_DIR = "/Users/macbook/Desktop/OHLCV_data/ALL_OHLCV"
SAVE_DIR = "/Users/macbook/Desktop/Save Results"

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1. / (1. + np.exp(-x))

def exp(x):
    return np.exp(x)