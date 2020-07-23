from scripts.agent import AgentWithStrategy, AgentKnowledge, Matrix3DFixedWordOrderArchitecture, reward_agents
from scripts.game import CooperationGameWithTeachersAndLearners
from scripts.umpire import Distances, FixedWordOrderGameUmpire
from scripts.game_utils import get_new_environments, store_files, get_fixed_position_3D_permutation_matrix
from scripts.strategies import StrategyTypes
from scripts.environment import meanings, REDUCED_GOALS, Categories
import argparse
import signal
import sys


num_games = 150
env_size = 8
num_objects = 120
min_tries = 1
max_tries = 12
reward = 1000


def create_games(failure_alignment, num_agents):
	games = []
	envs = get_new_environments(max_tries, env_size, num_objects, goals=REDUCED_GOALS, symbols=AgentKnowledge.symbols)
	row_size = len(AgentKnowledge.symbols)
	col_size = len(meanings)
	n_positions = Categories.get_number_of_categories()
	for tries in range(min_tries, max_tries+1):
		env = envs[tries-1]
		learners = [AgentWithStrategy(StrategyTypes.MIXED,
		                              architecture=Matrix3DFixedWordOrderArchitecture(row_size, col_size, n_positions),
		                              max_tries=tries,
		                              failure_alignment=failure_alignment)
			for i in range(num_agents)]
		teacher = AgentWithStrategy(StrategyTypes.MIXED, architecture=Matrix3DFixedWordOrderArchitecture(row_size, col_size, n_positions), max_tries=tries)
		teacher.architecture.language_matrix = get_fixed_position_3D_permutation_matrix(row_size, col_size, n_positions)
		name = "negotiated_game_position_"+str(tries)
		game = CooperationGameWithTeachersAndLearners(env, [teacher], learners, name, umpire=FixedWordOrderGameUmpire(env, store_mean=True))
		games.append(game)
	return games


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--d', default='/home/mariano/repos/research_sketches/game_results/')
	parser.add_argument('--a', action='store_true')
	parser.add_argument('--n', type=int, default=10)

	args = parser.parse_args()
	directory = args.d
	failure_alignment = args.a
	num_agents = args.n

	winners = []
	games = create_games(failure_alignment, num_agents)
	computed_games = 0

	def signal_handler(signal, frame):
		print('You pressed Ctrl+C - or killed me with -2'
		agents = [game.agents for game in games]
		umpires = [game.umpire for game in games]
		aligned = "negative" if failure_alignment else "positive"
		file_name = "position/teachers_{0}_{1}_games_{2}_agents.pkl".format(aligned, computed_games, num_agents)
		store_files(umpires, agents, winners, file_name, directory)
		sys.exit(0)

	signal.signal(signal.SIGINT, signal_handler)

	for k, game_ in enumerate(games):
		game_.umpire.compute_distance_learners(game_.learners, game_.teachers[0].architecture.language_matrix)
		game_.umpire.compute_distance_agents(game_.learners, measure=Distances.JENSEN_SHANNON)
		game_.umpire.compute_mean_fitness(game_.agents)
		# game_.umpire.compute_distance_agents(game_.learners, measure=Distances.HELLINGER)

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
			game_.umpire.compute_distance_learners(game_.learners, game_.teachers[0].architecture.language_matrix)
			game_.umpire.compute_distance_agents(game_.learners, measure=Distances.JENSEN_SHANNON)
			# game_.umpire.compute_distance_agents(game_.learners, measure=Distances.HELLINGER)
			game_.umpire.compute_mean_fitness(game_.agents)
			game_.reset_game(envs[k])
		is_game_finished = False

	umpires = [game.umpire for game in games]
	learners = [game.learners for game in games]

	aligned = "negative" if failure_alignment else "positive"
	file_name = "position/teachers_{0}_{1}_games_{2}_agents.pkl".format(aligned, computed_games, num_agents)

	store_files(umpires, learners, winners, file_name, directory)
