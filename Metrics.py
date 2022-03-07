import numpy as np

class Metrics:
    def __init__(self):
        self.portfolio_values = [] #portfolio_value는 첫 step 진행 이후부터 저장
        self.profitlosses = []
        self.daily_returns = []
        self.total_return = None
        self.volatility = None

    """
    The sequence of daily returns 
    """
    def get_daily_returns(self):
        assert len(self.portfolio_values) > 1
        for i in range(len(self.portfolio_values)-1):
            t1_step_pv = self.portfolio_values[i]
            t2_step_pv = self.portfolio_values[i+1]
            daily_return = (t2_step_pv - t1_step_pv)/t1_step_pv
            self.daily_returns.append(daily_return)
        return self.daily_returns

    """
    The ratio of the capital growth during testing
    adn training time
    """
    def get_total_return(self):
        assert len(self.portfolio_values) > 1
        self.total_return = \
            (self.portfolio_values[-1]-self.portfolio_values[0])/self.portfolio_values[0]
        return self.total_return

    """
    The volatility of the daily returns that tells us 
    about the financial risk level of the trading rules
    """
    def get_volatility(self):
        daily_returns = self.daily_returns
        return np.std(daily_returns)


