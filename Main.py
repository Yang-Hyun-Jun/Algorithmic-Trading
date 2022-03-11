import os
import argparse
import DataManager as data_manager
import Visualizer
import utils
import torch
import numpy as np
import pandas as pd

from Learner import DQNLearner
from Metrics import Metrics

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--stock_code", nargs="+")
  parser.add_argument("--lr", type=float, default=1e-6)
  parser.add_argument("--discount_factor", type=float, default=0.7)
  parser.add_argument("--num_episode", type=int, default=50)
  parser.add_argument("--balance", type=int, default=1000000)
  parser.add_argument("--batch_size", type=int, default=30)
  parser.add_argument("--memory_size", type=int, default=50)

  parser.add_argument("--start_date", default="20100101")
  parser.add_argument("--end_date", default="20180101")
  args = parser.parse_args()

#유틸 저장
utils.Base_DIR = "/Users/macbook/Desktop/OHLCV_data/ALL_OHLCV"
utils.SAVE_DIR = "/Users/macbook/Desktop/Save Results"
utils.stock_code = args.stock_code[0]
utils.start_date = args.start_date
utils.end_date = args.end_date

# 학습 데이터 준비
data_path = os.path.join(utils.Base_DIR, utils.stock_code)
training_data = data_manager.load_data(path=data_path, date_start=utils.start_date, date_end=utils.end_date)
training_data = data_manager.get_moving_trend(training_data, w=20, v=5)
training_data = data_manager.get_ulbv_vector(training_data)

# 최소/최대 투자 가격 설정
min_trading_price = max(int(100000 / training_data.iloc[-1]["Close"]), 1) * training_data.iloc[-1]["Close"]
max_trading_price = max(int(1000000 / training_data.iloc[-1]["Close"]), 1) * training_data.iloc[-1]["Close"]

# 파라미터 설정
params = {"lr":args.lr,
            "training_data": training_data, "discount_factor":args.discount_factor,
            "min_trading_price": min_trading_price, "max_trading_price": max_trading_price,
            "batch_size":args.batch_size, "memory_size":args.memory_size}

# 학습 수행
learner = DQNLearner(**params)
learner.run(num_episode=args.num_episode, balance=args.balance)
learner.save_model(utils.SAVE_DIR + "/Models" + "/MLPFeatureExtractor.pth")