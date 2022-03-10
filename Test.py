import DataManager
import Visualizer
import utils
import torch
import numpy as np
import pandas as pd
from Metrics import Metrics
from Environment import Environment
from Agent import Agent
from FeatureExtractor import MLPEncoder
from FeatureExtractor import MLPDecoder
from FeatureExtractor import MLPAutoEncoder

if __name__ == "__main__":
    test_data_path = utils.Base_DIR + "/0010"
    test_data = DataManager.load_data(test_data_path, date_start="20180101")
    start_date = "20180101"
    end_date = "20200824"
    #Test Model load
    encoder = MLPEncoder(input_dim=test_data.shape[1], num_classes=50)
    policy_decoder = MLPDecoder(num_classes=50, output_dim=3)
    target_decoder = MLPDecoder(num_classes=50, output_dim=3)
    policy_net = MLPAutoEncoder(encoder, policy_decoder)
    target_net = MLPAutoEncoder(encoder, target_decoder)

    model_path = utils.SAVE_DIR + "/Models" + "/MLPFeatureExtractor"
    policy_net.load_state_dict(torch.load(model_path))
    target_net.load_state_dict(policy_net.state_dict())

    #Parameter
    eps_start = 0.0
    lr = 0.0
    discount_factor = 0.0
    min_trading_price = max(int(100 / test_data.iloc[-1]["Close"]), 1) * test_data.iloc[-1]["Close"]
    max_trading_price = max(int(1000 / test_data.iloc[-1]["Close"]), 1) * test_data.iloc[-1]["Close"]

    #Agent
    environment = Environment(chart_data=test_data)
    agent = Agent(environment=environment,
                  policy_net=policy_net, target_net=target_net,
                  epsilon=eps_start, lr=lr, discount_factor=discount_factor,
                  min_trading_price=min_trading_price,
                  max_trading_price=max_trading_price)

    agent.policy_net.eval()
    agent.target_net.eval()

    # #Test
    metrics = Metrics()
    agent.set_balance(1000)
    agent.reset()
    agent.environment.reset()

    actions = [None] * len(test_data)
    state = agent.environment.observe()
    samples = pd.DataFrame(
        {"state": [], "action": [], "confidence": [], "next_state": [], "reward": [], "done": []})

    steps_done = 0
    while True:
        state = torch.tensor(state).float().view(1,-1)
        action, confidence = agent.get_action(state)
        next_state, reward, done = agent.step(action, confidence)
        steps_done += 1

        samples.loc[steps_done, "state"] = np.array(state[0][-1])
        samples.loc[steps_done, "action"] = action
        samples.loc[steps_done, "confidence"] = np.array(confidence)
        samples.loc[steps_done, "next_state"] = next_state[-1]
        samples.loc[steps_done, "reward"] = reward
        samples.loc[steps_done, "done"] = done

        state = next_state

        metrics.portfolio_values.append(agent.portfolio_value)
        metrics.profitlosses.append(agent.profitloss)
        if done:
            break

    samples.to_csv(utils.SAVE_DIR + "/Metrics" + "/samples_test")
    Vsave_path1 = utils.SAVE_DIR + "/Metrics" + "/Close Price Curve_test"
    Vsave_path2 = utils.SAVE_DIR + "/Metrics" + "/Portfolio Value Curve_test"
    Vsave_path3 = utils.SAVE_DIR + "/Metrics" + "/Daily Return Curve_test"
    Vsave_path4 = utils.SAVE_DIR + "/Metrics" + "/Profitloss Curve_test"

    Msave_path1 = utils.SAVE_DIR + "/Metrics" + "/Portfolio Value_test"
    Msave_path2 = utils.SAVE_DIR + "/Metrics" + "/Daily Return_test"
    Msave_path3 = utils.SAVE_DIR + "/Metrics" + "/Profitloss_test"

    metrics.get_portfolio_values(save_path=Msave_path1)
    metrics.get_daily_returns(save_path=Msave_path2)
    metrics.get_profitlosses(save_path=Msave_path3)

    Visualizer.get_close_price_curve(utils.stock_code, start_date, end_date, save_path=Vsave_path1)
    Visualizer.get_portfolio_value_curve(metrics.portfolio_values, save_path=Vsave_path2)
    Visualizer.get_daily_return_curve(metrics.daily_returns, save_path=Vsave_path3)
    Visualizer.get_profitloss_curve(metrics.profitlosses, save_path=Vsave_path4)
    Visualizer.get_action_and_candle(test_data.iloc[:150], samples["action"].iloc[:150])
    # # metrics.reset()




