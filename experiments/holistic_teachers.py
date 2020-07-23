## An experiment for determining the effect of cooperation in agents learning single utterances for each action
## Includes an experiment where the system is underdetermined: there are elements that have same color or shape


from scripts.agent import AgentWithStrategy, AgentKnowledge, HolisticCognitiveArchitecture, reward_agents
from scripts.game import CooperationHolisticGameWithTeachersAndLearners, Role, HolisticGameUmpire
from scripts.umpire import Distances
from scripts.strategies import StrategyTypes
from scripts.game_utils import get_new_environments, get_non_zero_permutation_matrix
from scripts.game_utils import store_files, store_teachers


import signal
import sys

num_agents = 10
env_size = 8
num_objects = 120
max_tries = 12
reward = 1000


def create_games(failure_alignment, num_agents):
	games = []
	envs = get_new_environments(max_tries, env_size, num_objects, symbols=AgentKnowledge.holistic_symbols)
	for tries in range(1, max_tries+1):
		env = envs[tries-1]
		learners = [AgentWithStrategy(StrategyTypes.MIXED,
		                              architecture=HolisticCognitiveArchitecture(len(AgentKnowledge.holistic_symbols), len(env.goals), random=True),
		                              role=Role.LISTENER,
		                              max_tries=tries,
		                              failure_alignment=failure_alignment)
			for i in range(num_agents)]
		teacher = AgentWithStrategy(StrategyTypes.MIXED, architecture=HolisticCognitiveArchitecture(len(AgentKnowledge.holistic_symbols), len(env.goals)), role=Role.LISTENER)
		teacher.architecture.language_matrix = get_non_zero_permutation_matrix(len(AgentKnowledge.holistic_symbols), len(env.goals))
		name = "teachers_learners_game_holistic_"+str(tries)
		# is_chosen = True if tries == 7 else False
		is_chosen = False
		game = CooperationHolisticGameWithTeachersAndLearners(env, [teacher], learners, name, umpire=HolisticGameUmpire(env, len(env.goals), store_mean=True), print_learner=is_chosen)
		games.append(game)
	return games

def main(n_iterations, failure_alignment, alignment_strategy, num_games, directory):
	for ite in range(n_iterations):
		winners = []
		games = create_games(failure_alignment, num_agents)

		for k, game_ in enumerate(games):
			language_matrix = game_.teachers[0].architecture.language_matrix
			game_.umpire.compute_distance_learners(game_.learners, language_matrix)
			game_.umpire.compute_distance_agents(game_.learners, measure=Distances.JENSEN_SHANNON)
			game_.umpire.compute_distance_agents(game_.learners, measure=Distances.HELLINGER)

		is_game_finished = False
		computed_games = 0

		# def signal_handler(signal, frame):
		# 	print('You pressed Ctrl+C - or killed me with -2'
		# 	learners = [game.learners for game in games]
		# 	umpires = [game.umpire for game in games]
		# 	aligned = "negative" if failure_alignment else "positive"
		# 	file_name = "holistic_teachers_{0}_{1}_games_{2}.pkl".format(aligned, computed_games, str(ite))
		# 	store_files(umpires, learners, winners, file_name, directory)
		# 	sys.exit(0)

		for i in range(num_games):
			while not is_game_finished:
				for game in games:
					is_game_finished = game.consume_time_step()
					if is_game_finished:
						winner_tries = game.learners[0].strategy.max_tries
						winners.append(winner_tries)
						reward_agents(game.learners, reward)
						computed_games += 1
						break
			envs = get_new_environments(max_tries, env_size, num_objects, symbols=AgentKnowledge.holistic_symbols)
			print("GAME FINISHED ", i
			for k, game_ in enumerate(games):
				language_matrix = game_.teachers[0].architecture.language_matrix
				game_.umpire.compute_distance_learners(game_.learners, language_matrix)
				game_.umpire.compute_distance_agents(game_.learners, measure=Distances.JENSEN_SHANNON)
				game_.umpire.compute_distance_agents(game_.learners, measure=Distances.HELLINGER)
				game_.umpire.compute_mean_fitness(game_.learners)
				game_.reset_game(envs[k])
			is_game_finished = False

		umpires = [game.umpire for game in games]
		learners = [game.learners for game in games]
		teachers = [game.teachers[0] for game in games]
		aligned = "negative" if failure_alignment else "positive"
		file_name = "holistic/teachers_{0}_{1}_games_{2}.pkl".format(aligned, num_games, str(ite))
		store_files(umpires, learners, winners, file_name, directory)
		teachers_filename = '/holistic/teachers_teachers_{0}_{1}_games{2}.pkl'.format(aligned, num_games, str(ite))
		store_teachers(teachers, teachers_filename, directory)








