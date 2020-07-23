from agent import AgentWithMemory, HolisticCognitiveArchitecture
from strategies import StrategyTypes, MutualisticStrategy
from GridEnvironment import SingleCellEnvironment
from umpire import PopulationGameUmpire
from environment import REDUCED_DESTINATIONS, REDUCED_OBJECTS
from game import InfiniteGame
import pickle
from os import path, mkdir
from math import floor, factorial
from random import shuffle
import numpy as np
import argparse
import logging
import datetime




logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
# handler = logging.FileHandler('/Users/mariano/developing/results/infinite_game/altruistic/info.log')
handler = logging.FileHandler('info.log')
handler.setLevel(logging.INFO)

store_distance_interval = consistency_distance_interval = 40
initial_period = 40
store_dir = '/homes/mmm31/developing/game_results/infinite_game/population_game_holistic'
# store_dir = '/Users/mariano/developing/simulation_results/infinite_game/population_game/fitness/'
# store_dir = '/home/mariano/repos/research_sketches/game_results/infinite_game/population_game/parameters/var_action_coord'
# store_dir = '/Volumes/MARIANO/phd_research/simulation_results/population_game/parameters/var_action_coord'
n_interactions = 150000

# Parameters

n_agents = 100
potential_pairings = factorial(n_agents)/(factorial(n_agents-2)*2)
reward = 100
learning_rate = .1
delta_learning_rate = .05
delta_coord_rate = 0.05
max_coord_rate = 1.2
max_learning_rate = 0.9
window_size = 1
rate_altruistic = 0.05
delta_rate_alt = 0.1
max_rate_altruistic = 0.99
rate_imitators = 0.1
n_imitators = int(np.floor(n_agents * rate_imitators))


def store_umpire_and_agents(umpire, agents, directory_name):
	if not path.exists(directory_name):
		mkdir(directory_name)
	with open(path.join(directory_name, 'umpire.pkl'), 'wb') as f:
		pickle.dump(umpire, f)
	# with open(path.join(directory_name, 'agents.pkl'), 'wb') as f:
	# 	pickle.dump(agents, f)


def make_parameter_dir(store_dir_, parameter_value):
	dir_name_ = path.join(store_dir_, '{0:.2f}'.format(parameter_value))
	if not path.exists(dir_name_):
		mkdir(dir_name_)
	return dir_name_


def imitate_process(agents, umpire):
	n_changes_ = 0
	for i in range(n_imitators):
		imitator, model = np.random.choice(agents, 2)
		if imitator.strategy.type == model.strategy.type:
			continue
		if model.fitness > imitator.fitness:
			imitator.strategy = model.strategy
			imitator.record_change()
			umpire.record_change(imitator.id)
			n_changes_ += 1

	return n_changes_

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', type=float, default=0.05)
	parser.add_argument('-a', type=float, default=0.05)
	parser.add_argument('-r', type=float, default=0.5)
	parser.add_argument('-n', type=int, default=1)
	parser.add_argument('-f', type=int, default=1)
	args = parser.parse_args()
	coord_rate = args.c
	rate_altruistic = args.r
	action_rate= args.a
	number_of_tests = args.n
	first_test = args.f
	print(args)

	logger.info("starting test")
	language_matrix_rows = language_matrix_cols = len(REDUCED_OBJECTS)*len(REDUCED_DESTINATIONS)

	# while rate_altruistic <= max_rate_altruistic:
	# rate_altruistic = 0.95
	action_cost = reward*action_rate
	coord_cost = action_cost*coord_rate
	cost_dir = make_parameter_dir(store_dir, action_rate)
	coord_dir = make_parameter_dir(cost_dir, coord_rate)
	# while rate_altruistic <= max_rate_altruistic:
	parameter_dir = make_parameter_dir(coord_dir, rate_altruistic)
	for test_number in range(first_test, number_of_tests+1):
		print(datetime.datetime.now())
		n_alt_agents = floor(n_agents * rate_altruistic)
		alt_agents = [AgentWithMemory(StrategyTypes.ALTRUISTIC, id=i+1, architecture=HolisticCognitiveArchitecture(language_matrix_rows, language_matrix_cols, delta=learning_rate), window_size=window_size) for i in range(n_alt_agents)]
		mut_agents = [AgentWithMemory(MutualisticStrategy(), id=i+n_alt_agents+1, architecture=HolisticCognitiveArchitecture(language_matrix_rows, language_matrix_cols, delta=learning_rate), window_size=window_size) for i in range(n_agents-n_alt_agents)]
		agents_ = alt_agents + mut_agents
		assert len(agents_) == n_agents
		shuffle(agents_)
		np.random.seed(n_alt_agents)
		env = SingleCellEnvironment()
		umpire = PopulationGameUmpire(env, action_cost, coord_cost, store_distance_interval)
		game = InfiniteGame(env, agents_, "indirect", umpire=umpire, reward=reward, action_cost=action_cost,
		                    coordination_cost=coord_cost)

		current_interaction = 0
		while current_interaction <= n_interactions:
			if game.status.new_interaction:
				if current_interaction % store_distance_interval == 0:
					n_changes = imitate_process(game.agents, game.umpire)
					game.umpire.track_ratios(game.agents)
					game.umpire.track_changes(n_changes)
					game.umpire.compute_distance_agents(game.agents)
					game.umpire.compute_mean_fitness(game.agents)
				if current_interaction % consistency_distance_interval == 0:
					game.umpire.sample_consistency(game.agents, potential_pairings)

				# game.umpire.store_agents_stats(game.agents)
				if current_interaction > n_agents:
					if game.umpire.is_game_finished(n_agents):
						interaction = current_interaction
						logger.info("exiting at interaction {0} ".format(current_interaction))
						print("--------storing umpire action cost: {0}, coord cost: {1}, rate: {2}, test number: {3}".format(action_cost, coord_cost, rate_altruistic, test_number))
						break
				current_interaction += 1
			game.consume_time_step()
		dir_name = path.join(parameter_dir, '{0:02d}_{1}'.format(test_number, current_interaction))
		store_umpire_and_agents(game.umpire, game.agents, dir_name)
	# rate_altruistic += delta_rate_alt
		# first_test = 1
