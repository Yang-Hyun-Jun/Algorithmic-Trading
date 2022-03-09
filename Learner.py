import torch
import utils
import Visualizer
import pandas as pd
import numpy as np

from Environment import Environment
from Agent import Agent
from ReplayMemory import ReplayMemory
from FeatureExtractor import MLPEncoder
from FeatureExtractor import MLPDecoder
from FeatureExtractor import MLPAutoEncoder
from Metrics import Metrics

class DQNLearner:

    target_update_interval = 5
    print_every = 10

    def __init__(self, lr, discount_factor=0.7,
                 eps_start=0.9, eps_end=0.05, eps_decay=500, training_data=None,
                 min_trading_price=None, max_trading_price=None,
                 batch_size=30, memory_size=50):

        assert min_trading_price > 0
        assert max_trading_price > 0
        assert max_trading_price >= min_trading_price

        self.environment = Environment(training_data)
        self.memory = ReplayMemory(max_size=memory_size)
        self.training_data = training_data
        self.batch_size = batch_size

        encoder = MLPEncoder(input_dim=training_data.shape[1], num_classes=50)
        policy_decoder = MLPDecoder(num_classes=50, output_dim=3)
        target_decoder = MLPDecoder(num_classes=50, output_dim=3)
        policy_net = MLPAutoEncoder(encoder, policy_decoder)
        target_net = MLPAutoEncoder(encoder, target_decoder)

        # 에이전트
        self.agent = Agent(environment=self.environment,
                           policy_net=policy_net, target_net=target_net,
                           epsilon=eps_start, lr=lr, discount_factor=discount_factor,
                           min_trading_price=min_trading_price,
                           max_trading_price=max_trading_price)

        # 에포크 관련 정보
        self.itr_cnt = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay


    def reset(self):
        self.environment.reset()
        self.agent.reset()
        self.itr_cnt = 0


    def prepare_training_inputs(self, sampled_exps):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for sampled_exp in sampled_exps:
            states.append(sampled_exp[0])
            actions.append(sampled_exp[1])
            rewards.append(sampled_exp[2])
            next_states.append(sampled_exp[3])
            dones.append(sampled_exp[4])

        states = torch.cat(states, dim=0).float()
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0).float()
        next_states = torch.cat(next_states, dim=0).float()
        dones = torch.cat(dones, dim=0).float()
        return states, actions, rewards, next_states, dones


    def run(self, num_episode=100, balance=1000000):
        samples = pd.DataFrame({"state":[], "action":[], "confidence":[], "next_state":[], "reward":[], "done":[], "epi":[]})
        self.agent.set_balance(balance)
        self.agent.policy_net.load_state_dict(self.agent.target_net.state_dict())
        metrics = Metrics()
        steps_done = 0

        for episode in range(num_episode):
            self.reset()

            cum_r = 0
            state = self.environment.observe()
            while True:
                #epsilon decaying
                self.agent.epsilon = \
                    self.eps_end + (self.eps_start - self.eps_end) * utils.exp(-steps_done / self.eps_decay)

                state = torch.tensor(state).float().view(1, -1)
                action, confidence = self.agent.get_action(state)
                next_state, reward, done = self.agent.step(action, confidence)

                #sample DataFrame 저장
                samples.loc[steps_done, "state"] = np.array(state[0][-1])
                samples.loc[steps_done, "action"] = action
                samples.loc[steps_done, "confidence"] = np.array(confidence)
                samples.loc[steps_done, "next_state"] = np.array(next_state[-1])
                samples.loc[steps_done, "reward"] = reward
                samples.loc[steps_done, "done"] = done
                samples.loc[steps_done, "epi"] = episode

                steps_done += 1
                experience = (state,
                              torch.tensor(action).view(1, -1),
                              torch.tensor(reward).view(1, -1),
                              torch.tensor(next_state).float().view(1, -1),
                              torch.tensor(done).view(1, -1))

                self.memory.push(experience)
                self.itr_cnt += 1
                cum_r += reward
                state = next_state

                #학습
                if len(self.memory) >= self.batch_size:
                    sampled_exps = self.memory.sample(self.batch_size)
                    sampled_exps = self.prepare_training_inputs(sampled_exps)
                    self.agent.update(*sampled_exps)

                #metrics 마지 episode 대해서만
                if episode == range(num_episode)[-1]:
                    metrics.portfolio_values.append(self.agent.portfolio_value)
                    metrics.profitlosses.append(self.agent.profitloss)

                if done:
                    break

            if episode == range(num_episode)[-1]:

                #metric 계산과 저장
                metrics.get_daily_returns()
                metrics.get_profitlosses()
                metrics.get_portfolio_values()

                #계산한 metric 시각화와 저장
                Visualizer.get_close_price_curve(utils.stock_code, utils.start_date, utils.end_date)
                Visualizer.get_portfolio_value_curve(metrics.portfolio_values)
                Visualizer.get_profitloss_curve(metrics.profitlosses)
                Visualizer.get_daily_return_curve(metrics.daily_returns)

            if episode % DQNLearner.target_update_interval == 0:
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())

            if episode % DQNLearner.print_every == 0:
                print("episode: {} | cum_r:{}".format(episode, cum_r))

        samples.to_csv(utils.SAVE_DIR + "/samples")

    def save_model(self, path):
        torch.save(self.agent.policy_net.state_dict(), path)

