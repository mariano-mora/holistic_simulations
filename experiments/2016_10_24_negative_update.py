from agent import AgentWithStrategy, AgentKnowledge, HolisticCognitiveArchitecture, reward_agents
from game import CooperationHolisticGameWithTeachersAndLearners, Role, HolisticGameUmpire
from strategies import StrategyTypes
from game_utils import get_non_zero_permutation_matrix
from game_utils import store_files
import argparse
from GridEnvironment import GridEnvironment
from environment import RandomInitializer, GOALS
from numpy import arange

import signal
import sys


num_games = 10
env_size = 8
num_objects = 120
min_tries = 1
max_tries = 12
reward = 1000
num_tries = 7
min_neg_beta = 0.1
max_neg_beta = 0.8
beta_step = -0.05



def create_game(neg_beta):
	env = GridEnvironment(env_size, env_size, goals=GOALS, symbols=AgentKnowledge.holistic_symbols)
	RandomInitializer(num_objects=num_objects, reduced=True).init_environments(env)
	learners = [AgentWithStrategy(StrategyTypes.MIXED,
	                              architecture=HolisticCognitiveArchitecture(len(AgentKnowledge.holistic_symbols), len(env.goals), neg_beta=neg_beta, random=True),
	                              role=Role.LISTENER,
	                              max_tries=num_tries,
	                              failure_alignment=True)
		for i in range(num_agents)]
	teacher = AgentWithStrategy(StrategyTypes.MIXED, architecture=HolisticCognitiveArchitecture(len(AgentKnowledge.holistic_symbols), len(env.goals)), role=Role.LISTENER)
	teacher.architecture.language_matrix = get_non_zero_permutation_matrix(len(AgentKnowledge.holistic_symbols), len(env.goals))
	name = "teachers_learners_game_holistic_"+str(num_tries)
	game = CooperationHolisticGameWithTeachersAndLearners(env, [teacher], learners, name, umpire=HolisticGameUmpire(env, len(env.goals)))
	return game


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--d', default='/home/mariano/repos/research_sketches/game_results/')
	parser.add_argument('--n', type=int, default=10)
	args = parser.parse_args()
	directory = args.d
	num_agents = args.n
	winners = []

	def signal_handler(signal, frame):
		print('You pressed Ctrl+C - or killed me with -2')
		learners = game.learners
		umpires = game.umpire
		aligned = "false_aligned"
		file_name = "holistic_teachers_{0}_sweeping_neg_beta.pkl".format(aligned)
		store_files(umpires, learners, winners, file_name, directory)
		sys.exit(0)

	for beta in arange(max_neg_beta, min_neg_beta, beta_step):
		print(beta)
		game = create_game(beta)
		language_matrix = game.teachers[0].architecture.language_matrix
		game.umpire.compute_distance_learners(game.learners, language_matrix)
		for i in range(num_games):
			is_game_finished = False
			while not is_game_finished:
				is_game_finished = game.consume_time_step()
				if is_game_finished:
					print("GAME FINISHED ",i)
					winner_tries = game.learners[0].strategy.max_tries
					winners.append(winner_tries)
					reward_agents(game.learners, reward)
					break
			game.umpire.compute_distance_learners(game.learners, language_matrix)
			game.umpire.compute_mean_fitness(game.learners)
			env = GridEnvironment(env_size, env_size, goals=GOALS, symbols=AgentKnowledge.holistic_symbols)
			RandomInitializer(num_objects=num_objects, reduced=True).init_environments(env)
			game.reset_game(env)
		umpires = game.umpire
		learners = game.learners
		aligned = "false_aligned"
		file_name = "holistic_teachers_{0}_sweeping_neg_beta_{1}.pkl".format(aligned, beta)
		store_files(umpires, learners, winners, file_name, directory)

