import gymnasium as gym
import numpy as np


class GenerativeEnv(gym.Env):
	metadata = {"render_modes": ["human"]}
	
	def __init__(self, simulations_length=356):
		super(GenerativeEnv, self).__init__()

		# Custom attributes
		self.max_steps = simulations_length
		self.current_step = 0
		
		self.funds = 0
		self.shares_count = 0
		self.init_options = {
			"start_price": 100,
			"funds": 10_000.0,
			"init_days": 60,
			"mu": 0.0005,
			"sigma": 0.01,
		}
		
		# Env RL attributes
		self.state = None
		self.reward = 0.0
		self.terminated = False
		self.truncated = False
		self.info = None
		
		# Protected
		self._prices = None
	
	def reset(self, seed=None, options=None):
		if options is None:
			options = {}
		super().reset(seed=seed)

		self.init_options.update(options)
		self.funds = self.init_options["funds"]
		
		self.current_step = 0
		
		# Generate historical data
		self._prices = self._init_gen_prices(**options)
	
	def step(self, action):
		self._prices = np.append(self._prices, self._gen_next_price(self._prices[-1], **self.init_options))
		self.current_step += 1

		self.terminated = self._is_terminated()
		self.truncated = self._is_truncated()
		self.reward = self._calc_reward()
	
	def _determine_state(self):
		raise NotImplementedError  # Must be overwritten
	
	def _get_info(self):
		raise NotImplementedError  # Must be overwritten

	def _calc_reward(self):
		if self.truncated:
			return -100
		if self.shares_count <= 0:
			return 0
		return -1 if self._prices[-2] - self._prices[-1] <= 0 else 1

	def _is_terminated(self):
		return self.current_step >= self.max_steps
	
	def _is_truncated(self):
		return self._prices[-1] <= 0

	@staticmethod
	def _init_gen_prices(start_price=100.0, num_days=5, mu=0.0005, sigma=0.01, df=5, *args, **kwargs):
		log_returns = GenerativeEnv._gen_returns_t(num_days=num_days, mu=mu, sigma=sigma, df=df)
		log_prices = np.cumsum(log_returns)
		return start_price * np.exp(log_prices)

	@staticmethod
	def _gen_returns_t(num_days=5, mu=0.0005, sigma=0.01, df=5):
		returns = np.random.standard_t(df=df, size=num_days)
		returns = mu + sigma * returns
		return returns

	@staticmethod
	def _gen_next_price(previous_price=100.0, mu=0.0005, sigma=0.01, df=5, *args, **kwargs):
		log_return_next = GenerativeEnv._gen_returns_t(num_days=1, mu=mu, sigma=sigma, df=df)
		next_price = previous_price * np.exp(log_return_next)
		return next_price[0]
