from agent import AgentWithStrategy
from game import CooperationGame
from strategies import StrategyTypes
import pickle as pickle
from game_utils import get_new_environment

num_agents = 5
env_size = 8
num_objects = 40
num_games = 50
max_attempts = 20


if __name__ == "__main__":
	results = []
	for i in range(max_attempts):
		print(i)
		agents = [AgentWithStrategy(StrategyTypes.MIXED, architecture='multiple', max_tries=i+1, creator='uniform_random') for i in range(num_agents)]
		grid_env = get_new_environment(env_size, num_objects)
		game = CooperationGame(grid_env, agents, "cooperation-game-test")
		is_game_finished = False
		time_steps = []
		for j in range(num_games):
			print(j
			num_timesteps = 0
			while not is_game_finished:
				is_game_finished = game.consume_time_step()
				num_timesteps += 1
			time_steps.append(num_timesteps)
			game.set_new_environment(get_new_environment(env_size, num_objects))
			is_game_finished = False
		results.append((game.umpire, time_steps, agents))
	with open('/Users/mariano/developing/repos/phdnotebooks/results/20_groups_umpires_holophrastic.pkl', 'w') as f:
		pickle.dump(results, f)

	# number_tests = 10
	# total_tests = []

	# for i in reversed(xrange(max_attempts)):
	# 	games_played = 1
	# 	agents = [AgentWithStrategy(StrategyTypes.MIXED, architecture='multiple', max_tries=i+1, creator='uniform_random') for i in range(num_agents)]
	# 	grid_env = get_new_environment(env_size, num_objects)
	# 	game = CooperationGame(grid_env, agents, "cooperation-game-test")
	# 	is_game_finished = False
	# 	time_steps = []
	# 	while not game.umpire.grammar_agreement_point:
	# 		while not is_game_finished:
	# 			is_game_finished = game.consume_time_step()
	# 			if game.umpire.grammar_agreement_point:
	# 				print("finally", game.umpire.grammar_agreement_point
	# 				filename = '/Users/mariano/developing/repos/phdnotebooks/results/20_groups_agreement_point_{0}.txt'.format(i)
	# 				with open(filename, 'w') as f:
	# 					f.write("%d" % game.umpire.grammar_agreement_point)
	# 				break
	# 		game.set_new_environment(get_new_environment(env_size, num_objects))
	# 		games_played += 1
	# 		if games_played % 20 == 0:
	# 			print(games_played
	# 		is_game_finished = False
	# 	print(games_played
