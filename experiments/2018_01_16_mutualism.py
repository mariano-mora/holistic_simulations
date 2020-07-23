from agent import AgentWithMemory, HolisticCognitiveArchitecture
from strategies import MutualisticStrategy
from GridEnvironment import SingleCellEnvironment
from umpire import SingleActionWordTrackUmpire
from environment import REDUCED_DESTINATIONS, REDUCED_OBJECTS
from game import InfiniteGame
import pickle
from os import path, mkdir, makedirs
from numpy import random
import argparse

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('info.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)


n_agents = 10
n_interactions = 150000
current_interaction = 0
reward = 100
store_distance_interval = 100
consistency_interval = 20
window_size = 1
# store_dir = '/Volumes/MARIANO/phd_research/simulation_results/mutualism/parameters/single_var_action_coord'
store_dir = '/Users/mariano/developing/simulation_results/infinite_game/mutualism/agent_stats'
# store_dir = '/home/mariano/repos/research_sketches/game_results/infinite_game/mutualism/parameters/var_action_coord'
# learning_rate = .1
delta_learning_rate = .1
max_learning_rate = 1.2
delta_cost = .1
max_cost = 1.2
delta_coord_rate = 0.1
max_coord_rate = 1.2


def store_umpire_and_agents(umpire_, agents_, directory_name):
	if not path.exists(directory_name):
		mkdir(directory_name)
	with open(path.join(directory_name, 'umpire.pkl'), 'wb') as f:
		pickle.dump(umpire_, f)
	with open(path.join(directory_name, 'agents.pkl'), 'wb') as f:
		pickle.dump(agents_, f)


def make_parameter_dir(parameter_value_1, parameter_value_2 ):
	dir_name_ = path.join(store_dir, '{0:.2f}/{1:.2f}'.format(parameter_value_1, parameter_value_2))
	if not path.exists(dir_name_):
		makedirs(dir_name_)
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
	parser.add_argument('-l', type=float, default=.1)
	args = parser.parse_args()
	coord_rate = args.c
	action_rate = args.a
	learning_rate = args.l
	first_test = args.f
	number_of_tests = args.n
	print(args)
	language_matrix_rows = language_matrix_cols = len(REDUCED_OBJECTS) * len(REDUCED_DESTINATIONS)
	test_number = 1
	# while action_rate <= max_cost:
	learning_dir = make_parameter_dir(action_rate, coord_rate)
	action_cost = reward * action_rate
	# while coord_rate <= max_coord_rate:
	# 	coord_dir = make_parameter_subdir(coord_rate, action_dir)
	coord_cost = action_cost * coord_rate
	# for test_number in range(first_test, number_of_tests+1):
	current_interaction = 0
	seed = int(action_cost+coord_cost + test_number)
	random.seed(seed)
	agents = [AgentWithMemory(MutualisticStrategy(), window_size=window_size, architecture=HolisticCognitiveArchitecture(language_matrix_rows,language_matrix_cols, delta=learning_rate), id=i+1) for i in range(n_agents)]
	env = SingleCellEnvironment()
	umpire = SingleActionWordTrackUmpire(env)
	game = InfiniteGame(env, agents, "indirect", umpire=umpire, reward=reward, action_cost=action_cost, coordination_cost=coord_cost)

	while current_interaction <= n_interactions:
		interaction = current_interaction
		if game.status.new_interaction:
			current_interaction += 1
			game.umpire.compute_distance_agents(game.agents)
			if current_interaction % consistency_interval == 0:
				game.umpire.compute_mean_fitness(game.agents)
				# game.umpire.track_consistency(game.agents)
				game.umpire.store_agents_stats(game.agents)
			if current_interaction > n_agents and current_interaction % store_distance_interval == 0:
				print(current_interaction)
				if game.umpire.is_game_finished(env):
					interaction = current_interaction
					logger.info("exiting at interaction {0} ".format(current_interaction))
					break
		game.consume_time_step()
	dir_name = path.join(learning_dir, '{0:02d}_{1}'.format(test_number, interaction))
	store_umpire_and_agents(game.umpire, game.agents, dir_name)
# 	coord_rate += delta_coord_rate
# coord_rate = 0.05
# action_rate += delta_cost
