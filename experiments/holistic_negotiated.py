from scripts.agent import AgentWithStrategy, AgentKnowledge, HolisticCognitiveArchitecture, reward_agents
from scripts.game import CooperationGame, HolisticGameUmpire
from scripts.game_utils import get_new_environments, store_files
from scripts.strategies import StrategyTypes
from scripts.umpire import Distances
import argparse

import signal
import sys



env_size = 8
num_agents = 10
num_objects = 120
min_tries = 1
max_tries = 12
reward = 1000
agent_to_store = 7


def create_games(failure_alignment, num_agents):
	games = []
	envs = get_new_environments(max_tries, env_size, num_objects, symbols=AgentKnowledge.holistic_symbols)
	for tries in range(min_tries, max_tries+1):
		env = envs[tries-1]
		agents = [AgentWithStrategy(StrategyTypes.MIXED,
		                            architecture=HolisticCognitiveArchitecture(len(AgentKnowledge.holistic_symbols), len(env.goals), random=True),
		                            max_tries=tries,
		                            failure_alignment=failure_alignment)
									for i in range(num_agents)]
		name = "teachers_learners_game_holistic_"+str(tries)
		# print_learner = True if tries == agent_to_store else False
		game = CooperationGame(env, agents, name, umpire=HolisticGameUmpire(env, len(env.goals), store_mean=True))
		games.append(game)
	return games


def main(n_iterations, failure_alignment, alignment_strategy, num_games, directory):
	for ite in range(n_iterations):
		winners = []
		games = create_games(failure_alignment, num_agents)
		is_game_finished = False
		computed_games = 0

		# def signal_handler(signal, frame):
		# 	print('You pressed Ctrl+C - or killed me with -2'
		# 	agents = [game.agents for game in games]
		# 	umpires = [game.umpire for game in games]
		# 	aligned = "negative" if failure_alignment else "positive"
		# 	file_name = "holistic/negotiated_{0}_{1}_games_{2}_agents.pkl".format(aligned, computed_games, num_agents)
		# 	store_files(umpires, agents, winners, file_name, directory)
		# 	sys.exit(0)

		# signal.signal(signal.SIGINT, signal_handler)

		for k, game_ in enumerate(games):
			game_.umpire.compute_distance_agents(game_.agents)
			game_.umpire.compute_distance_agents(game_.agents, measure=Distances.HELLINGER)
			game_.umpire.compute_mean_fitness(game_.agents)

		for i in range(num_games):
			while not is_game_finished:
				for game in games:
					is_game_finished = game.consume_time_step()
					if is_game_finished:
						winners.append(game.agents[0].strategy.max_tries)
						reward_agents(game.agents, reward)
						computed_games += 1
						break
			print("GAME FINISHED ", i)
			envs = get_new_environments(max_tries, env_size, num_objects, symbols=AgentKnowledge.holistic_symbols)
			for k, game_ in enumerate(games):
				game_.umpire.compute_distance_agents(game_.agents)
				game_.umpire.compute_distance_agents(game_.agents, measure=Distances.HELLINGER)
				game_.umpire.compute_mean_fitness(game_.agents)
				game_.reset_game(envs[k])
			is_game_finished = False

		agents = [game.agents for game in games]
		umpires = [game.umpire for game in games]
		aligned = "negative" if failure_alignment else "positive"
		file_name = "holistic/negotiated_{0}_{1}_games_{2}.pkl".format(aligned, computed_games, str(ite))

		store_files(umpires, agents, winners, file_name, directory)