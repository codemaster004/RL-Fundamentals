from os.path import join as pjoin

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

		self.q_table = np.zeros((*self.state_dim, self.action_dim))
		self.buffer = MCBuffer()

	def select_action(self, state, mask, epsilon=0.1):
		allowed_indices = np.where(mask)[0]
		if np.random.random() < epsilon:
			action = np.random.choice(allowed_indices)
		else:
			masked_q_values = self.q_table[(*state,)][mask]
			action = allowed_indices[np.argmax(masked_q_values)]

		return action

	def update(self, states, actions, rewards, alpha):
		for s, a, r in zip(states, actions, rewards):
			self.q_table[(*s, a)] += alpha * (r - self.q_table[(*s, a)])

	def train(self, env, episodes=10_000, discounting=0.9, learning_rate=0.01, epsilon=0.3, trajectories_per_update=1):
		print('Starting Monte Carlo Training...')
		for episode in range(episodes):
			self.buffer.reset()
			# Collect Experience
			for _ in range(trajectories_per_update):
				self._run_episode(env, epsilon)

			states, actions, rewards = self.buffer.get()
			rewards = self._calc_cumsum_rewards(rewards, discounting)
			# Update Q-table
			self.update(states, actions, rewards, learning_rate)
			# Logging
			if (episode + 1) % 100 == 0:
				print("Episode: ", episode + 1)
	
	def save(self, path='.', filename='MC-Agent.npy'):
		np.save(pjoin(path, filename), self.q_table)
	
	def load(self, path='.', filename='MC-Agent.npy'):
		self.q_table = np.load(pjoin(path, filename))
	
	def _run_episode(self, env, epsilon):
		state, info = env.reset()
		mask = info['action_mask']

		done = False
		while not done:
			action = self.select_action(state, mask, epsilon=epsilon)

			next_state, reward, terminated, truncated, info = env.step(action)
			mask = info['action_mask']
			self.buffer.add(state, action, reward)
			state = next_state

			done = terminated or truncated
	
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
	agent = MonteCarloAgent(state_dim=(3, 2), action_dim=3)
	agent.train(env, episodes=5000, trajectories_per_update=5)
	agent.save(path='saves/')
	print(agent.q_table)
