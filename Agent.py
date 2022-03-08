import numpy as np
import torch
import torch.nn as nn
import utils

class Agent(nn.Module):
    # 주식 보유 비율, 현재 손익, 평균 매수 단가 대비 등락률
    STATE_DIM = 3
    # TRADING_CHARGE = 0.00015
    # TRADING_TEX = 0.0025
    TRADING_CHARGE = 0.0
    TRADING_TEX = 0.0

    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2

    ACTIONS = ["BUY", "SELL", "HOLD"]
    NUM_ACTIONS = len(ACTIONS)

    def __init__(self, environment, policy_net, target_net, lr, discount_factor, epsilon,
                 min_trading_price, max_trading_price):

        super().__init__()

        self.environment = environment
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

        #가치함수(Feature Extraction 포함) 설정
        self.policy_net = policy_net
        self.target_net = target_net

        self.lr = lr
        self.epsilon = epsilon
        self.discount_factor = discount_factor

        self.opt = torch.optim.Adam(params=self.policy_net.parameters(), lr=lr)
        self.criteria = nn.SmoothL1Loss()

        self.initial_balance = 0
        self.balance = 0
        self.num_stocks = 0
        self.portfolio_value = 0

        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.profitloss = 0  # 수익률


    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.profitloss = 0


    def set_balance(self, balance):
        self.initial_balance = balance


    def get_action(self, state):
        confidence = 0

        with torch.no_grad():
            self.policy_net.eval()
            q_value = self.policy_net(state)
            self.policy_net.train()
        prob = np.random.uniform(low=0.0, high=1.0, size=1)

        if prob <= self.epsilon:
            action = np.random.choice(range(Agent.NUM_ACTIONS))
            confidence = np.array(0.5)
        else:
            action = np.array(q_value.argmax(dim=-1))
            confidence = utils.sigmoid(q_value[0][action]) #sigmoid
        return int(action), confidence


    def validate_action(self, action):
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE):
                return False

        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                return False
        return True


    def decide_trading_unit(self, confidence):
        #confidence에 따라 거래 단위 결정
        added_trading_price = \
            max(min(int(confidence * (self.max_trading_price - self.min_trading_price)),
                    self.max_trading_price - self.min_trading_price), 0)

        trading_price = self.min_trading_price + added_trading_price
        return max(int(trading_price / self.environment.get_price()), 1)


    def get_reward(self, p1, p2, action):
        if p1 == None:
            return None
        else:
            if self.num_stocks >0:
                own = True
            else:
                own = False

            #Transaction cost
            if action == 0:
                TC = Agent.TRADING_CHARGE
            elif action == 0:
                TC = Agent.TRADING_CHARGE + Agent.TRADING_TEX
            else:
                TC = 0

            if action == 0 or (action==2 and own):
                reward = ( ((1-TC)**2)*(p2/p1)-1 )*100
            elif action == 1 or (action==2 and not own):
                reward = ( ((1-TC)**2)*(p1/p2)-1 )*100
            return reward


    def step(self, action, confidence):
        if self.validate_action(action) != True:
            action = Agent.ACTION_HOLD

        p1_price = self.environment.get_price()

        # 매수
        if action == Agent.ACTION_BUY:
            trading_unit = self.decide_trading_unit(confidence)
            balance = (self.balance - p1_price * (1 + self.TRADING_CHARGE) * trading_unit)

            #돈 부족 한 경우
            if balance < 0:
                trading_unit = min(
                    int(self.balance / (p1_price * (1 + self.TRADING_CHARGE))),
                    int(self.max_trading_price / p1_price))

            # 수수료 적용하여 총 매수 금액 산정
            invest_amount = p1_price * (1 + self.TRADING_CHARGE) * trading_unit
            if invest_amount != 0:
                self.balance -= invest_amount
                self.num_stocks += trading_unit
                self.num_buy += 1

        elif action == Agent.ACTION_SELL:
            trading_unit = self.decide_trading_unit(confidence)
            trading_unit = min(trading_unit, self.num_stocks)

            invest_amount = p1_price * (1 - (self.TRADING_TEX + self.TRADING_CHARGE)) * trading_unit
            if invest_amount != 0:
                self.num_stocks -= trading_unit
                self.balance += invest_amount
                self.num_sell += 1

        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1

        self.portfolio_value = self.balance + p1_price * self.num_stocks
        self.profitloss = (self.portfolio_value / self.initial_balance) - 1

        #다음 상태로 진행
        next_state = self.environment.observe()
        p2_price = self.environment.get_price()
        reward = self.get_reward(p1=p1_price, p2=p2_price, action=action)

        if len(self.environment.chart_data) <= self.environment.idx + 1:
            done = 1
        else:
            done = 0

        return next_state, reward, done


    def update(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state

        with torch.no_grad():
            self.target_net.eval()
            q_value = self.target_net(ns)
            self.target_net.train()
            q_max = q_value.amax(dim=-1).view(-1, 1)
            target = r + self.discount_factor * q_max * (1-done)

        self.policy_net.eval()
        q = self.policy_net(s).gather(1, a)
        self.policy_net.train()
        self.loss = self.criteria(q, target)

        self.opt.zero_grad()
        self.loss.backward()
        self.opt.step()


#디버깅
if __name__ == "__main__":
    import torch
    from Environment import Environment
    from Agent import Agent
    from FeatureExtractor import MLPEncoder
    from FeatureExtractor import MLPDecoder
    from FeatureExtractor import MLPAutoEncoder
    import DataManager

    path = "/Users/macbook/Desktop/OHLCV_data/KOSPI_OHLCV/005930"  # 삼성전자
    training_data = DataManager.load_data(path=path, date_start="20170101", date_end="20170131")
    min_trading_price = max(int(100000 / training_data.iloc[-1]["Close"]), 1) * training_data.iloc[-1]["Close"]
    max_trading_price = max(int(1000000 / training_data.iloc[-1]["Close"]), 1) * training_data.iloc[-1]["Close"]

    environment = Environment(chart_data=training_data)

    encoder = MLPEncoder(input_dim=4, num_classes=50)
    policy_decoder = MLPDecoder(num_classes=50, output_dim=3)
    target_decoder = MLPDecoder(num_classes=50, output_dim=3)
    policy_net = MLPAutoEncoder(encoder, policy_decoder)
    target_net = MLPAutoEncoder(encoder, target_decoder)

    # 에이전트
    agent = Agent(environment=environment,
                  policy_net=policy_net, target_net=target_net,
                  epsilon=0.2, lr=0., discount_factor=0.7,
                  min_trading_price=min_trading_price,
                  max_trading_price=max_trading_price)

    agent.reset()
    agent.environment.reset()
    agent.set_balance(10000000)
    state = environment.observe()
    state = torch.tensor(state).float().view(1, -1)

    #액션 겟
    action, confidence = agent.get_action(state)

    #한 스텝 진행
    next_state, reward, done = agent.step(action, confidence)
    experience = (state,
                  torch.tensor(action).view(1,-1),
                  torch.tensor(reward).float().view(1,-1),
                  torch.tensor(next_state).float().view(1,-1),
                  torch.tensor(done).view(1,-1))
    print(experience)
    agent.update(*experience)









