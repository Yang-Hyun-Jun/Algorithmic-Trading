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
  parser.add_argument("--balance", type=int, default=1000)
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
min_trading_price = max(int(100 / training_data.iloc[-1]["Close"]), 1) * training_data.iloc[-1]["Close"]
max_trading_price = max(int(1000 / training_data.iloc[-1]["Close"]), 1) * training_data.iloc[-1]["Close"]

# 파라미터 설정
params = {"lr":args.lr,
            "training_data": training_data, "discount_factor":args.discount_factor,
            "min_trading_price": min_trading_price, "max_trading_price": max_trading_price,
            "batch_size":args.batch_size, "memory_size":args.memory_size}

# 학습 수행
learner = DQNLearner(**params)
learner.run(num_episode=args.num_episode, balance=args.balance)
learner.save_model(utils.SAVE_DIR + "/Models" + "/MLPFeatureExtractor.pth")


#Test
# start_date = "20170101"
# end_date = "20180824"
# test_data_path = utils.Base_DIR + "/" + args.stock_code[0]
# test_data = data_manager.load_data(test_data_path, date_start=start_date, date_end=end_date)
# test_data = data_manager.get_moving_trend(test_data, w=20, v=7)
# test_data = data_manager.get_ulbv_vector(test_data)
#
# # Parameter
# min_trading_price = max(int(100 / test_data.iloc[-1]["Close"]), 1) * test_data.iloc[-1]["Close"]
# max_trading_price = max(int(1000 / test_data.iloc[-1]["Close"]), 1) * test_data.iloc[-1]["Close"]
#
# # #Test
# metrics = Metrics()
# learner.agent.set_balance(1000)
# learner.agent.reset()
# learner.agent.environment.reset()
#
# actions = [None] * len(test_data)
# state = learner.agent.environment.observe()
# samples = pd.DataFrame(
#     {"state": [], "action": [], "confidence": [], "next_state": [], "reward": [], "done": [], "exp": []})
#
# steps_done = 0
# while True:
#     state = torch.tensor(state).float().view(1, -1)
#     action, confidence, exp = learner.agent.get_action(state)
#     next_state, reward, done = learner.agent.step(action, confidence)
#     steps_done += 1
#
#     samples.loc[steps_done, "state"] = np.array(state[0][3])
#     samples.loc[steps_done, "action"] = action
#     samples.loc[steps_done, "confidence"] = np.array(confidence)
#     samples.loc[steps_done, "next_state"] = next_state[3]
#     samples.loc[steps_done, "reward"] = reward
#     samples.loc[steps_done, "done"] = done
#     samples.loc[steps_done, "exp"] = exp
#
#     state = next_state
#
#     metrics.portfolio_values.append(learner.agent.portfolio_value)
#     metrics.profitlosses.append(learner.agent.profitloss)
#     if done:
#         break
#
# samples.to_csv(utils.SAVE_DIR + "/Metrics" + "/samples_test")
# Vsave_path1 = utils.SAVE_DIR + "/Metrics" + "/Close Price Curve_test"
# Vsave_path2 = utils.SAVE_DIR + "/Metrics" + "/Portfolio Value Curve_test"
# Vsave_path3 = utils.SAVE_DIR + "/Metrics" + "/Daily Return Curve_test"
# Vsave_path4 = utils.SAVE_DIR + "/Metrics" + "/Profitloss Curve_test"
#
# Msave_path1 = utils.SAVE_DIR + "/Metrics" + "/Portfolio Value_test"
# Msave_path2 = utils.SAVE_DIR + "/Metrics" + "/Daily Return_test"
# Msave_path3 = utils.SAVE_DIR + "/Metrics" + "/Profitloss_test"
#
# metrics.get_portfolio_values(save_path=Msave_path1)
# metrics.get_daily_returns(save_path=Msave_path2)
# metrics.get_profitlosses(save_path=Msave_path3)
#
# Visualizer.get_close_price_curve(utils.stock_code, start_date, end_date, save_path=Vsave_path1)
# Visualizer.get_portfolio_value_curve(metrics.portfolio_values, save_path=Vsave_path2)
# Visualizer.get_daily_return_curve(metrics.daily_returns, save_path=Vsave_path3)
# Visualizer.get_profitloss_curve(metrics.profitlosses, save_path=Vsave_path4)
# Visualizer.get_action_and_candle(test_data.iloc[200:500], samples["action"].iloc[200:500], samples["exp"].iloc[200:500])
