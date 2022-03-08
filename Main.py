import os
import argparse
import utils
import DataManager as data_manager

from Learner import DQNLearner

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--stock_code", nargs="+")
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--discount_factor", type=float, default=0.7)
  parser.add_argument("--num_episode", type=int, default=100)
  parser.add_argument("--balance", type=int, default=1000000)
  parser.add_argument("--batch_size", type=int, default=30)
  parser.add_argument("--memory_size", type=int, default=50)

  parser.add_argument("--start_date", default="20180101")
  parser.add_argument("--end_date", default="20200824")
  args = parser.parse_args()

for stock_code in args.stock_code:
    #유틸 저장
    utils.BASE_DIR = "/Users/macbook/Desktop/OHLCV_data/ALL_OHLCV"
    utils.SAVE_DIR = "/Users/macbook/Desktop/Save Results"
    utils.stock_code = args.stock_code[0]
    utils.start_date = args.start_date
    utils.end_date = args.end_date

    # 학습 데이터 준비
    data_path = os.path.join(utils.BASE_DIR, utils.stock_code)
    training_data = data_manager.load_data(path=data_path, date_start=utils.start_date, date_end=utils.end_date)

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
    # learner.save_model(utils.SAVE_DIR + "/Models" + "/MLPFeatureExtractor")

