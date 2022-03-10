class Environment:
    PRICE_IDX = 3  # 종가의 위치

    def __init__(self, chart_data=None):
        self.chart_data = chart_data
        self.observation = None
        self.idx = - 1
        # self.done = False

    def reset(self):
        self.observation = None
        self.idx = - 1
        # self.done = False

    def observe(self):
        if len(self.chart_data)-1 >= self.idx:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation.tolist()
        else:
            return None

    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None