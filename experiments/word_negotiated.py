from scripts.agent import AgentWithStrategy, AgentKnowledge, StochasticMatrixCognitiveArchitecture, reward_agents
from scripts.game import CooperationGame, HolisticGameUmpire
from scripts.game_utils import get_new_environments, store_files
from scripts.alignment import *
from scripts.strategies import StrategyTypes
from scripts.environment import meanings, REDUCED_GOALS
from scripts.umpire import Distances
import argparse

import signal
import sys




num_agents = 10
env_size = 8
num_objects = 120
min_tries = 1
max_tries = 12
reward = 1000



def create_games(failure_alignment, num_agents, align_strat):
	games = []
	envs = get_new_environments(max_tries, env_size, num_objects, goals=REDUCED_GOALS)
	for tries in range(min_tries, max_tries+1):
		env = envs[tries-1]
		agents = [AgentWithStrategy(StrategyTypes.MIXED,
		                                architecture=StochasticMatrixCognitiveArchitecture(len(AgentKnowledge.symbols),
		                                                                                   len(meanings),
		                                                                                   random=True,
		                                                                                   align_strat=align_strat),
		                                max_tries=tries,
		                                failure_alignment=failure_alignment)
			for i in range(num_agents)]
		name = "negotiated_game_word_"+str(tries)
		is_chosen = True if tries == 7 else False
		game = CooperationGame(env, agents, name, umpire=HolisticGameUmpire(env, len(env.goals), store_mean=True), print_learner=is_chosen)
		games.append(game)
	return games


def main(n_iterations, failure_alignment, alignment_strategy, num_games, directory):
	for ite in range(n_iterations):
		print("ITERATION: ", ite
		winners = []
		alignment = alignment_strategy() if alignment_strategy else None
		games = create_games(failure_alignment, num_agents, alignment)
		computed_games = 0

		# def signal_handler(signal, frame):
		# 	print('You pressed Ctrl+C - or killed me with -2'
		# 	agents = [game.agents for game in games]
		# 	umpires = [game.umpire for game in games]
		# 	aligned = "negative" if failure_alignment else "positive"
		# 	align_strat = games[0].agents[0].architecture.align_strat.__class__.__name__ if failure_alignment else ""
		# 	file_name = "word/negotiated_{0}_{1}_{2}_games_{3}_agents.pkl".format(aligned, align_strat, computed_games, num_agents)
		# 	store_files(umpires, agents, winners, file_name, directory)
		# 	sys.exit(0)
		#
		# signal.signal(signal.SIGINT, signal_handler)

		for k, game_ in enumerate(games):
			game_.umpire.compute_distance_agents(game_.agents)
			game_.umpire.compute_distance_agents(game_.agents, measure=Distances.HELLINGER)
			game_.umpire.compute_mean_fitness(game_.agents)

		is_game_finished = False
		for i in range(num_games):
			while not is_game_finished:
				for game in games:
					is_game_finished = game.consume_time_step()
					if is_game_finished:
						winners.append(game.agents[0].strategy.max_tries)
						reward_agents(game.agents, reward)
						computed_games += 1
						break
			print("GAME FINISHED ", i

			envs = get_new_environments(max_tries, env_size, num_objects)

			for k, game_ in enumerate(games):
				game_.umpire.compute_distance_agents(game_.agents)
				game_.umpire.compute_distance_agents(game_.agents, measure=Distances.HELLINGER)
				game_.umpire.compute_mean_fitness(game_.agents)
				game_.reset_game(envs[k])
			is_game_finished = False

		umpires = [game.umpire for game in games]
		learners = [game.agents for game in games]

		aligned = "negative" if failure_alignment else "positive"
		align_strat = games[0].agents[0].architecture.align_strat.__class__.__name__ if failure_alignment else ""
		file_name = "word/negotiated_{0}_{1}_{2}_games_{3}.pkl".format(aligned, align_strat, num_games, str(ite))
		store_files(umpires, learners, winners, file_name, directory)
