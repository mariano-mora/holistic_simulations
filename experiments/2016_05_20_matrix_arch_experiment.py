from agent import AgentWithStrategy
from game import GameResult, CooperationGame
from game_utils import get_new_environment
from strategies import StrategyTypes
import cPickle as pickle
"""
	Experiment of cooperate game.
	Agents keep a matrix representation of association between symbols and meanings
	Word order is fixed: ACTION-COLOR-SHAPE-DIRECTION
"""

num_agents = 10
env_size = 8
num_objects = 60
num_games = 50
max_attempts = 20

if __name__ == "__main__":

	results = []
	for i in range(max_attempts):
		print("group: ", i
		agents = [AgentWithStrategy(StrategyTypes.MIXED, architecture='matrix', max_tries=i+1) for i in range(num_agents)]
		grid_env = get_new_environment(env_size, num_objects)
		game = CooperationGame(grid_env, agents, "cooperation-game-test", umpire="fixed_word_order")
		is_game_finished = False
		time_steps = []
		for j in range(num_games):
			print("game: ", j
			num_timesteps = 0
			while not is_game_finished:
				is_game_finished = game.consume_time_step()
				num_timesteps += 1
			time_steps.append(num_timesteps)
			game.set_new_environment(get_new_environment(env_size, num_objects))
			is_game_finished = False
		results.append((game.umpire, time_steps, agents))
	with open('/Users/mariano/developing/repos/phdnotebooks/results/20_groups_umpires_fixed_order.pkl', 'w') as f:
		pickle.dump(results, f)