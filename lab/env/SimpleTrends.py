import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class SimpleTrends(gym.Env):
	metadata = {"render_modes": ["human"]}

	def __init__(self):
		super(SimpleTrends, self).__init__()
		# Parent attributes
		# 0: Hold, 1: Buy, 2: Sell
		self.action_space = spaces.Discrete(3)
		# 0: trend-down, 1: trend-up, 2: chop
		self.observation_space = spaces.MultiDiscrete([3, 2], dtype=np.int32)
		# Custom attributes
		self.state = None
		self.max_steps = 365
		self.current_step = 0
		self.funds = 0
		self.bought_count = 0
		self.init_options = None
		# Protected
		self._prices = None
		self._short_sma = None
		self._long_sma = None

	def reset(self, seed=None, options=None):
		# todo: dict.update()
		if options is None:
			options = {
				"start_price": 100,
				"num_days": 60,
				"mu": 0.0005,
				"sigma": 0.01,
				"short_sma": 12,
				"long_sma": 60,
				"funds": 10_000.0
			}

		super().reset(seed=seed)
		self.init_options = options
		self.funds = options["funds"]

		# Generate historical data
		self._prices = self._init_gen_prices(**options)
		self._short_sma = self._init_calc_sma(self._prices, window=options["short_sma"])
		self._long_sma = self._init_calc_sma(self._prices, window=options["long_sma"])

		self.state = self._determine_state()
		self.current_step = 0
		return self.state, self._get_info()

	def step(self, action):
		current_price = self._prices[-1]
		if action == 0:  # 0: Hold
			pass
		elif action == 1:  # 1: Buy
			self.bought_count += self.funds // current_price
			self.funds -= current_price * self.bought_count
		elif action == 2:  # 1: Sell
			self.funds += current_price * self.bought_count
			self.bought_count = 0
		
		self._prices = np.append(self._prices, self._gen_next_price(current_price, **self.init_options))
		self._short_sma = np.append(self._short_sma, self._calc_new_sma(self._prices, self.init_options["short_sma"]))
		self._long_sma = np.append(self._long_sma, self._calc_new_sma(self._prices, self.init_options["long_sma"]))
		
		self.state = self._determine_state()
		reward = self._calc_reward()
		terminated = self.current_step >= self.max_steps
		self.current_step += 1
		
		info = self._get_info()

		return self.state, reward, terminated, False, info

	def render(self):
		print(f"{self.funds=}, {self.bought_count=}, {self._prices[-1]=}")

	def close(self):
		pass

	def _get_info(self):
		return {"action_mask": np.array([True, self.funds > self._prices[-1], self.bought_count > 1])}
	
	def _determine_state(self):
		trend = 2
		if self._short_sma[-1] < self._long_sma[-1]:
			trend = 0  # 0: trend-down
		elif self._short_sma[-1] >= self._long_sma[-1]:
			trend = 1  # 1: trend-up
		is_bought = 1 if self.bought_count > 0 else 0
		return np.array([trend, is_bought])
	
	def _calc_reward(self):
		return 0 if self.bought_count <= 0 else -1 if self._prices[-2] - self._prices[-1] <= 0 else 1

	@staticmethod
	def _gen_returns_t(num_days=5, mu=0.0005, sigma=0.01, df=5):
		returns = np.random.standard_t(df=df, size=num_days)
		returns = mu + sigma * returns
		return returns

	@staticmethod
	def _init_gen_prices(start_price=100.0, num_days=5, mu=0.0005, sigma=0.01, df=5, *args, **kwargs):
		log_returns = SimpleTrends._gen_returns_t(num_days=num_days, mu=mu, sigma=sigma, df=df)
		log_prices = np.cumsum(log_returns)
		return start_price * np.exp(log_prices)

	@staticmethod
	def _init_calc_sma(prices: np.ndarray, window: int = 5):
		return np.concatenate([
			[np.nan] * (window - 1),
			np.convolve(prices, np.ones(window) / window, mode='valid')
		])

	@staticmethod
	def _gen_next_price(previous_price=100.0, mu=0.0005, sigma=0.01, df=5, *args, **kwargs):
		log_return_next = SimpleTrends._gen_returns_t(num_days=1, mu=mu, sigma=sigma, df=df)
		next_price = previous_price * np.exp(log_return_next)
		return next_price[0]

	@staticmethod
	def _calc_new_sma(prices: np.ndarray, window: int = 5, *args, **kwargs):
		return np.mean(prices[-(window - 1):])


if __name__ == "__main__":
	env = SimpleTrends()
	env.reset()

	prices = env._init_gen_prices(num_days=3)
	sma = SimpleTrends._init_calc_sma(prices, window=3)
	# sma_l = SimpleTrends._init_calc_sma(prices, window=6)
	print(f"{prices=}")
	print(f"{sma=}")
	new_prices = np.append(prices, env._gen_next_price(prices[-1]))
	new_sma = np.append(sma, SimpleTrends._calc_new_sma(new_prices, window=3))
	print(f"{new_prices=}")
	print(f"{new_sma=}")

# days = np.arange(len(prices))
# # Plot
# plt.figure(figsize=(8, 4))
# plt.plot(days, prices, marker='.', label='Price')
# plt.plot(days, sma, linestyle='-', label='12-day Moving Avg')
# plt.plot(days, sma_l, linestyle='-', label='60-day Moving Avg')
# 
# plt.title('Stock Price and Moving Average')
# plt.xlabel('Day')
# plt.ylabel('Price')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
