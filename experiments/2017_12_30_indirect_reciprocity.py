from agent import AgentWithMemory, HolisticCognitiveArchitecture
from strategies import StrategyTypes
from GridEnvironment import SingleCellEnvironment
from umpire import InfiniteGameUmpire, SingleActionWordTrackUmpire
from environment import REDUCED_DESTINATIONS, REDUCED_OBJECTS
from game import InfiniteGame
import pickle
from os import path, mkdir
from numpy import random
import logging
import argparse
from scipy.stats import mannwhitneyu, shapiro
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
# handler = logging.FileHandler('/Users/mariano/developing/results/infinite_game/altruistic/info.log')
handler = logging.FileHandler('info.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)


n_agents = 10
n_interactions = 1000000
current_interaction = 0
reward = 100
# action_rate = 0.70
store_distance_interval = 100
consistency_interval = 20
initial_period = 40
store_dir = '/Users/mariano/developing/simulation_results/infinite_game/altruistic/parameters/learning/learning_rate'
# store_dir = '/home/mariano/repos/research_sketches/game_results/infinite_game/altruistic/parameters/var_action_coord'
# number_of_tests = 30
# learning_rate = .1
delta_learning_rate = .1
delta_cost = 0.1
max_cost = 1.20
# coord_rate = 1.15
delta_coord_rate = 0.05
max_coord_rate = 1.2
max_learning_rate = 1.2


def store_umpire_and_agents(umpire_, agents_, directory_name):
	if not path.exists(directory_name):
		mkdir(directory_name)
	with open(path.join(directory_name, 'umpire.pkl'), 'wb') as f:
		pickle.dump(umpire_, f)
	with open(path.join(directory_name, 'agents.pkl'), 'wb') as f:
		pickle.dump(agents_, f)


def make_parameter_dir(parameter_value):
	dir_name_ = path.join(store_dir, '{0:.2f}'.format(parameter_value))
	if not path.exists(dir_name_):
		mkdir(dir_name_)
	return dir_name_


def make_parameter_subdir(parameter_value, parent_dir):
	dir_name_ =  path.join(parent_dir, '{0:.2f}'.format(parameter_value))
	if not path.exists(dir_name_):
		mkdir(dir_name_)
	return dir_name_

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-c', type=float, default=0.25)
	parser.add_argument('-a', type=float, default=0.4)
	parser.add_argument('-f', type=int, default=1)
	parser.add_argument('-n', type=int, default=30)
	parser.add_argument('-l', type=float, default=0.10)
	args = parser.parse_args()
	coord_rate = args.c
	action_rate = args.a
	first_test = args.f
	number_of_tests = args.n
	learning_rate = args.l
	print(args)
	logger.info("starting test")
	language_matrix_rows = language_matrix_cols = len(REDUCED_OBJECTS)*len(REDUCED_DESTINATIONS)

	# while action_rate <= max_cost:
	action_dir = make_parameter_dir(action_rate)
	action_cost = reward*action_rate
		# while coord_rate <= max_coord_rate:
	coord_dir = make_parameter_subdir(coord_rate, action_dir)
	coord_cost = action_cost*coord_rate
	for test_number in range(first_test, number_of_tests+1):
		random.seed(test_number)
		current_interaction = 0
		interaction = n_interactions
		agents_ = [AgentWithMemory(StrategyTypes.ALTRUISTIC, id=i+1, architecture=HolisticCognitiveArchitecture(language_matrix_rows, language_matrix_cols, delta=learning_rate)) for i in range(n_agents)]
		env = SingleCellEnvironment()
		umpire = SingleActionWordTrackUmpire(env)
		game = InfiniteGame(env, agents_, "indirect", umpire=umpire, reward=reward, action_cost=action_cost, coordination_cost=coord_cost)

		while current_interaction <= n_interactions:
			if game.status.new_interaction:
				current_interaction += 1
				interaction = current_interaction
					# game.umpire.store_language_matrices(game.agents)
				game.umpire.compute_distance_agents(game.agents)
				if current_interaction % consistency_interval == 0:
					game.umpire.compute_mean_fitness(game.agents)
					game.umpire.track_consistency(game.agents)
					# game.umpire.compute_mean_fitness(game.agents)
					# game.umpire.store_agents_stats(game.agents)
				if current_interaction > initial_period and current_interaction % store_distance_interval == 0:
					print(current_interaction)
					if game.umpire.is_game_finished(env):
						logger.info("exiting at interaction {0} ".format(current_interaction))
						break
			game.consume_time_step()
		dir_name = path.join(coord_dir, '{0:02d}_{1}'.format(test_number, interaction))
		store_umpire_and_agents(game.umpire, game.agents, dir_name)
	# coord_rate += delta_coord_rate
# coord_rate = 0.05
# action_rate += delta_cost
