from lab.env.SimpleTrends import SimpleTrends
from lab.agent.MonteCarlo import MonteCarloAgent


def test_agent(env, agent):
	state, info = env.reset()
	mask = info['action_mask']

	done = False
	while not done:
		action = agent.select_action(state, mask, epsilon=0)

		state, reward, terminated, truncated, info = env.step(action)
		mask = info['action_mask']
		done = terminated or truncated


if __name__ == '__main__':
	env = SimpleTrends()
	agent = MonteCarloAgent(env.observation_space, env.action_space)
	agent.load(path='saves')
	test_agent(env, agent)
