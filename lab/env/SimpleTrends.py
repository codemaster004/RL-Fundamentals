from gymnasium import spaces
import numpy as np

from .GenerativeEnv import GenerativeEnv


class SimpleTrends(GenerativeEnv):
	metadata = {"render_modes": ["human"]}

	def __init__(self, simulations_length=356):
		super(SimpleTrends, self).__init__()
		# Parent attributes
		# 0: Hold, 1: Buy, 2: Sell
		self.action_space = spaces.Discrete(3)
		# 0: trend-down, 1: trend-up, 2: chop
		self.observation_space = spaces.MultiDiscrete([3, 2], dtype=np.int32)
		# Protected
		self._short_sma = None
		self._long_sma = None

	def reset(self, seed=None, options=None):
		if options is None:
			options = {
				"short_sma": 12,
				"long_sma": 60,
			}

		super().reset(seed=seed, options=options)

		# Generate historical data
		self._short_sma = self._init_calc_sma(self._prices, window=options["short_sma"])
		self._long_sma = self._init_calc_sma(self._prices, window=options["long_sma"])

		self.state = self._determine_state()
		return self.state, self._get_info()

	def step(self, action):
		current_price = float(self._prices[-1])
		if action == 0:  # 0: Hold
			pass
		elif action == 1:  # 1: Buy
			self.shares_count += self.funds // current_price
			self.funds -= current_price * self.shares_count
		elif action == 2:  # 1: Sell
			self.funds += current_price * self.shares_count
			self.shares_count = 0
		
		super().step(action=action)
		
		# Update to Simple Moving Avg
		self._short_sma = np.append(self._short_sma, self._calc_new_sma(self._prices, self.init_options["short_sma"]))
		self._long_sma = np.append(self._long_sma, self._calc_new_sma(self._prices, self.init_options["long_sma"]))
		
		self.state = self._determine_state()
		self.info = self._get_info()

		return self.state, self.reward, self.terminated, self.truncated, self.info

	def render(self):
		print(f"{self.funds=}, {self.shares_count=}, {self._prices[-1]=}")

	def close(self):
		pass

	def _get_info(self):
		return {"action_mask": np.array([True, self.funds > self._prices[-1], self.shares_count > 1])}
	
	def _determine_state(self):
		trend = 2
		if self._short_sma[-1] < self._long_sma[-1]:
			trend = 0  # 0: trend-down
		elif self._short_sma[-1] >= self._long_sma[-1]:
			trend = 1  # 1: trend-up
		is_bought = 1 if self.shares_count > 0 else 0
		return np.array([trend, is_bought])
	
	@staticmethod
	def _init_calc_sma(prices: np.ndarray, window: int = 5):
		return np.concatenate([
			[np.nan] * (window - 1),
			np.convolve(prices, np.ones(window) / window, mode='valid')
		])

	@staticmethod
	def _calc_new_sma(prices: np.ndarray, window: int = 5, *args, **kwargs):
		return np.mean(prices[-(window - 1):])
