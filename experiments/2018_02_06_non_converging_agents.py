from agent import AgentWithMemory, HolisticCognitiveArchitecture
from GridEnvironment import SingleCellEnvironment
from game import InfiniteGame
import pickle
from os import path



dir_name = '/Users/mariano/developing/results/infinite_game/altruistic/15_tests/09_500000_100_90'
reward = 100
action_cost = 90
n_interactions = 1000

if __name__ == "__main__":

	with open(path.join(dir_name,'agents.pkl'), 'rb') as f:
		agents = pickle.load(f)
	with open(path.join(dir_name, 'umpire.pkl'), 'rb') as f:
		umpire = pickle.load(f)

	env = SingleCellEnvironment()
	game = InfiniteGame(env, agents, "indirect", reward=reward, action_cost=action_cost)
	current_interaction = 0
	while current_interaction <= n_interactions:
		if game.status.new_interaction:
			current_interaction += 1

		game.consume_time_step()

