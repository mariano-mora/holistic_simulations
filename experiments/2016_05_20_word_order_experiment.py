from agent import AgentWithStrategy
from game import GameResult, CooperationGameWithWordOrder
from game_utils import get_new_environment
from strategies import StrategyTypes
import cPickle as pickle

num_agents = 10
env_size = 8
num_objects = 70
max_attempts = 5
num_games = 5

if __name__ == "__main__":

	# agents = [AgentWithStrategy(StrategyTypes.MIXED, architecture='matrix_word_order', max_tries=5) for i in range(num_agents)]
	# grid_env = GridEnvironment(env_size, env_size)
	# RandomInitializer(num_objects=num_objects).init_environments(grid_env)
	# game = CooperationGameWithWordOrder(grid_env, agents, "One_game_test")
	# is_game_finished = False
	# while not is_game_finished:
	# 	is_game_finished = game.consume_time_step()
	results = []
	for i in range(max_attempts):
		agents = [AgentWithStrategy(StrategyTypes.MIXED, architecture='matrix_variable_word_order', max_tries=i+1) for i in range(num_agents)]
		grid_env = get_new_environment(env_size, num_objects)
		game = CooperationGameWithWordOrder(grid_env, agents, "cooperation-game-test")
		is_game_finished = False
		time_steps = []
		for j in range(num_games):
			num_timesteps = 0
			while not is_game_finished:
				is_game_finished = game.consume_time_step()
				num_timesteps += 1
			time_steps.append(num_timesteps)
			game.set_new_environment(get_new_environment(env_size, num_objects))
			is_game_finished = False
		results.append((game.umpire, time_steps))
	with open('/Users/mariano/developing/repos/phdnotebooks/results/5_groups_umpires_variable_order.pkl', 'w') as f:
		pickle.dump(results, f)