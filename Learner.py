import torch
import utils
from tqdm import tqdm
from Environment import Environment
from Agent import Agent
from ReplayMemory import ReplayMemory
from FeatureExtractor import MLPEncoder
from FeatureExtractor import MLPDecoder
from FeatureExtractor import MLPAutoEncoder
from Metrics import Metrics

class DQNLearner:

    sampling_only_until = 64
    target_update_interval = 5
    print_every = 20

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


    def run(self, num_epoches=100, balance=1000000):
        self.agent.set_balance(balance)
        self.agent.policy_net.load_state_dict(self.agent.target_net.state_dict())

        for epoch in tqdm(range(num_epoches)):
            self.reset()
            # epsilon decaying
            self.agent.epsilon = \
                self.eps_end + (self.eps_start - self.eps_end)*utils.exp(-(epoch/self.eps_decay))

            cum_r = 0
            state = self.environment.observe()
            while True:

                state = torch.tensor(state).float().view(1, -1)
                action, confidence = self.agent.get_action(state)
                next_state, reward, done = self.agent.step(action, confidence)

                experience = (state,
                              torch.tensor(action).view(1, -1),
                              torch.tensor(reward).view(1, -1),
                              torch.tensor(next_state).float().view(1, -1),
                              torch.tensor(done).view(1, -1))

                self.memory.push(experience)
                self.itr_cnt += 1
                cum_r += reward
                state = next_state
                if done:
                    break


            if len(self.memory) >= DQNLearner.sampling_only_until:
                sampled_exps = self.memory.sample(self.batch_size)
                sampled_exps = self.prepare_training_inputs(sampled_exps)
                self.agent.update(*sampled_exps)

            if epoch % DQNLearner.target_update_interval == 0:
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())

            if epoch % DQNLearner.print_every == 0:
                print("cum_r:{}".format(cum_r))


    def save_model(self):
        pass

#디버깅
if __name__ == "__main__":
    import DataManager

    path = "/Users/macbook/Desktop/OHLCV_data/KOSPI_OHLCV/005930"  # 삼성전자
    training_data = DataManager.load_data(path=path, date_start="20100101", date_end="20170131")
    min_trading_price = max(int(100000 / training_data.iloc[-1]["Close"]), 1) * training_data.iloc[-1]["Close"]
    max_trading_price = max(int(1000000 / training_data.iloc[-1]["Close"]), 1) * training_data.iloc[-1]["Close"]

    learner = DQNLearner(lr=1e-4, discount_factor=0.7,
                         training_data=training_data,
                         min_trading_price=min_trading_price,
                         max_trading_price=max_trading_price,
                         batch_size=30, memory_size=50)

    learner.run(num_epoches=100, balance=1000000)

    learner.agent.set_balance(1000000)
    learner.reset()
    state = learner.environment.observe()
    learner.agent.epsilon = learner.epsilon * (1.0 - (float(1) / (3 - 1)))

    learner.agent.target_net.load_state_dict(learner.agent.policy_net.state_dict())
    #한 에피소드만 수집
    while True:
        state = torch.tensor(state).float().view(1, -1)
        action, confidence = learner.agent.get_action(state)
        next_state, reward, done = learner.agent.step(action, confidence)

        experience = (state,
                      torch.tensor(action).view(1, -1),
                      torch.tensor(reward).float().view(1, -1),
                      torch.tensor(next_state).float().view(1, -1),
                      torch.tensor(done).view(1, -1))

        learner.memory.push(experience)
        learner.itr_cnt += 1
        state = next_state
        print(learner.agent.profitloss)
        print(learner.agent.portfolio_value)
        if done:
            break

    sampled_exps = learner.memory.sample(10)
    sampled_exps = learner.prepare_training_inputs(sampled_exps)
    # learner.agent.update(*sampled_exps)
