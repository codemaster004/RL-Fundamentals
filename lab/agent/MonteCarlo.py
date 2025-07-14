import numpy as np


class MCBuffer:
	def __init__(self):
		self.states = None
		self.actions = None
		self.rewards = None

		self.reset()

	def reset(self):
		self.states = []
		self.actions = []
		self.rewards = []

	def add(self, state, action, reward):
		self.states.append(state)
		self.actions.append(action)
		self.rewards.append(reward)

	def get(self):
		return self.states, self.actions, self.rewards


class MonteCarloAgent:
	def __init__(self, state_dim, action_dim, options=None):
		self.state_dim = state_dim
		self.action_dim = action_dim

		self.q_table = np.zeros((self.state_dim, self.action_dim))  # todo: this will work only when state_dim is an int

	def select_action(self, state, epsilon=0.1):
		if np.random.random() < epsilon:
			action = np.random.randint(self.action_dim)
		else:
			action = np.argmax(self.q_table[state])

		return action

	def update(self, states, actions, rewards, alpha):
		for s, a, r in zip(states, actions, rewards):
			self.q_table[s, a] += alpha * (r - self.q_table[s, a])

	def train(self, env, episodes=10_000, discounting=0.9, learning_rate=0.01):
		for episode in range(episodes):
			print("Episode: ", episode)
			state, _ = env.reset()

			buffer = MCBuffer()

			done = False
			while not done:
				action = self.select_action(state)

				next_state, reward, terminated, truncated, _ = env.step(action)
				buffer.add(state, action, reward)
				state = next_state

				done = terminated or truncated

			states, actions, rewards = buffer.get()
			rewards = self._calc_cumsum_rewards(rewards, discounting)
			self.update(states, actions, rewards, learning_rate)

	@staticmethod
	def _calc_cumsum_rewards(rewards, lam):
		# Calculating rewards from final reward with discounting lam
		running_sum = 0.0
		for t in reversed(range(len(rewards))):
			running_sum = rewards[t] + lam * running_sum
			rewards[t] = running_sum
		return rewards


if __name__ == '__main__':
	from lab.env.SimpleTrends import SimpleTrends

	env = SimpleTrends()
	agent = MonteCarloAgent(state_dim=3, action_dim=3)
	agent.train(env)
	print(agent.q_table)
