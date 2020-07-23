from agent import AgentWithMemory
from GridEnvironment import SingleCellEnvironment
from game import InfiniteGame

import pickle
from os import path

reward = 100
action_cost = 90
dir_name = '/Users/mariano/developing/results/infinite_game/mutualism/1_25500_100_90'

if __name__ == "__main__":
	with open(path.join(dir_name, 'agents.pkl'), 'rb') as f:
		agents = pickle.load(f)

	env = SingleCellEnvironment()
	game = InfiniteGame(env, agents, "indirect", reward=reward, action_cost=action_cost)
	game.umpire.compute_distance_agents(agents)
	for agent in agents:
		print(agent.architecture.language_matrix)
