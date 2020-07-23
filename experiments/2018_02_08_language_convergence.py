from agent import AgentWithMemory, HolisticCognitiveArchitecture, AgentKnowledge
from strategies import StrategyTypes
from GridEnvironment import SingleActionEnvironment
from sequence import Sequence
from game import InfiniteGame
from umpire import SingleActionWordTrackUmpire
import pickle
from os import path, mkdir
from numpy import argwhere, argmax, zeros, array, full, ones
from random import choice
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('/Users/mariano/developing/results/infinite_game/altruistic/info.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)


n_agents = 10
n_interactions = 2000000
current_interaction = 0
reward = 100
action_cost = 90
store_distance_interval = 100
store_dir = '/Users/mariano/developing/results/infinite_game/altruistic/single_action/agent_stats'
number_of_tests = 1
goal_index = 0


class SingleActionHolisticArchitecture(HolisticCognitiveArchitecture):
	def __init__(self, n_rows, n_columns):
		super(SingleActionHolisticArchitecture, self).__init__(n_rows, n_columns)
		self.language_matrix = full((n_rows, n_columns), 1. / n_rows, dtype=float)
		# self.language_matrix = zeros((n_rows,n_columns), dtype=int)

	def react_to_success(self, status, env):
		goal_index_, symbol_index_ = self.get_goal_and_symbol_index(status, env)
		# self.language_matrix[symbol_index_,goal_index_] += 1
		column = self.language_matrix[:,goal_index_]
		mask = ones(column.shape, dtype=bool)
		mask[symbol_index_] = 0
		column[mask] *= self.neg_alpha
		column[symbol_index_] *= self.pos_alpha
		column[column < self.minimal] = self.minimal
		column[column > self.maximal] = self.maximal
		column /= column.sum()

	def get_sequence(self, goal, env):
		goal_index_= env.goals.index(goal)
		column = self.language_matrix[:, goal_index_]
		maxis = argwhere(column==column.max())
		symbol_index_ = choice(maxis)[0]
		return Sequence([env.symbols[symbol_index_]])

	def choose_goal_to_perform(self, status, env):
		goals = env.goals
		symbol_index = env.symbols.index(status.sequence.symbols[0])
		row = self.interaction_matrix[symbol_index, :]
		max_arg = argmax(row)
		return goals[max_arg]


# class WordTrackUmpire(InfiniteGameUmpire):
# 	def __init__(self, env):
# 		super(WordTrackUmpire, self).__init__(env)
# 		self.words = []
# 		self.word_counts = []
#
# 	def track_word(self, status):
# 		self.words.append(self.env.symbols.index(status.sequence.symbols[0]))
#
# 	def on_new_interaction(self, status):
# 		self.attempts_per_interaction.append(status.number_of_attempts)
# 		self.interaction_index += 1
# 		self.interaction_agents.append((status.speaker.id, status.listener.id))
# 		self.goals_chosen.append(self.env.goals.index(status.goal))
# 		self.track_word(status)
#
# 	# The game is over when all agents have a unique maximum value and it's the same index
# 	def is_game_finished(self, agents):
# 		maxis = []
# 		for agent in agents:
# 			column_ = agent.architecture.language_matrix[:,goal_index]
# 			agent_max = argwhere(column_ == column_.max())
# 			if agent_max.shape[0] != 1:
# 				return False
# 			maxis.append(agent_max[0])
# 		return all(maxis == maxis[0])
#
# 	def get_word_count(self, agents):
# 		sum_ = array([agent.architecture.language_matrix for agent in agents])
# 		self.word_counts.append(sum_.sum(axis=0))

def store_umpire_and_agents(umpire, agents, directory_name):
	if not path.exists(directory_name):
		mkdir(directory_name)
	with open(path.join(directory_name, 'umpire.pkl'), 'wb') as f:
		pickle.dump(umpire, f)
	with open(path.join(directory_name, 'agents.pkl'), 'wb') as f:
		pickle.dump(agents, f)


if __name__ == "__main__":
	logger.info("starting test")
	language_matrix_rows = len(AgentKnowledge.reduced_holistic_symbols)
	language_matrix_cols = 1
	for test_number in range(1, number_of_tests + 1):
		print(test_number)
		current_interaction = 0
		interaction = n_interactions
		logger.info("test number {0}".format(test_number))
		agents_ = [AgentWithMemory(StrategyTypes.ALTRUISTIC, id=i+1, architecture=SingleActionHolisticArchitecture(language_matrix_rows, language_matrix_cols)) for i in range(n_agents)]
		env = SingleActionEnvironment(goal_index)
		umpire = SingleActionWordTrackUmpire(env, goal_index)
		game = InfiniteGame(env, agents_, "indirect", reward=reward, action_cost=action_cost, umpire=umpire)
		while current_interaction <= n_interactions:
			if game.status.new_interaction:
				current_interaction += 1
				game.umpire.compute_distance_agents(game.agents)
				game.umpire.compute_mean_fitness(game.agents)
				game.umpire.get_word_count(game.agents)
				game.umpire.store_agents_stats(game.agents)
				if current_interaction % n_agents == 0:
					if game.umpire.is_game_finished(agents_):
						interaction = current_interaction
						logger.info("exiting at interaction {0} ".format(current_interaction))
						break
			game.consume_time_step()
		agent_directory = path.join(store_dir, "{0}_agents".format(n_agents))
		if not path.exists(agent_directory):
			mkdir(agent_directory)
		dir_name = path.join(agent_directory, '{0:02d}_{1}_{2}_{3}'.format(test_number, interaction, reward, action_cost))
		store_umpire_and_agents(game.umpire, game.agents, dir_name)
