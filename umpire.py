from measures import compute_agents_jensen_shannon, compute_hellinger, compute_agents_jensen_shannon_3d, consistent_pairings
from strategies import StrategyTypes
import numpy as np
from scipy.stats import entropy
from itertools import combinations
import functools
import collections
from scipy.special import comb
# from game_utils import save_agents_matrix


class Distances:
	ENTROPY = 0
	NORM = 1
	JENSEN_SHANNON = 2
	HELLINGER = 3

class GameUmpire(object):
	"""
		A class to keep record of game
	"""

	def __init__(self):
		self.success_counter = 0
		self.number_of_iterations = 0
		self.interaction_index = 0
		self.game_outcomes = []
		self.successes = []
		self.attempts = 0

	def on_success(self, status):
		self.success_counter += 1
		self.umpire.successes.append((self.interaction_index, status.number_of_attempts))

	def on_failure(self):
		self.success_counter = 0

	def restart_umpire(self):
		self.success_counter = 0
		self.number_of_iterations = 0
		self.interaction_index = 0
		self.successes = []


class GameUmpireTrackingSuccesses(GameUmpire):
	"""
		A class that also keeps track of number of successes per interaction in an iterated game
	"""

	def __init__(self, n_interactions):
		super(GameUmpireTrackingSuccesses, self).__init__()
		self.successes_per_interaction = np.zeros(n_interactions)

	def on_success(self):
		self.successes_per_interaction[self.interaction_index] += 1  # self.successes_per_interaction[self.interaction_index] + 1

	def on_failure(self):
		pass

	"""
		Method to keep track of the life-span of the words created and what meanings they are associated to.
		Should be implemented by each sub-class of umpire depending on the agent's cognitive architecture
	"""
	def keep_track(self, agents, goal, sequence):
		pass


class CooperationGameUmpire(GameUmpire):
	def __init__(self, env, store_mean=False):
		super(CooperationGameUmpire, self).__init__()
		self.attempts_per_interaction = []
		self.grammar_agreement_point = None
		self.env = env
		self.distances = {}
		self.mean_fitness = []
		self.mean_altruistic_fitness = []
		self.mean_mut_fitness = []
		self.store_mean = store_mean
		self.track = []
		self.interaction_agents = []
		self.goals_chosen = []
		self.consistency = []


	def on_success(self, status):
		self.successes.append((self.interaction_index, status.number_of_attempts))
		self.success_counter += 1
		self.track.append((1, status.number_of_attempts))

	def on_failure(self, status):
		self.track.append((0, status.number_of_attempts))

	def on_new_interaction(self, status):
		self.attempts_per_interaction.append(status.number_of_attempts)
		self.interaction_index += 1
		self.interaction_agents.append((status.speaker.id, status.listener.id))
		self.goals_chosen.append(self.env.goals.index(status.goal))

	def on_new_word(self, symbols, meanings):
		pass

	def is_game_finished(self):
		# first check if there are elements outside the four corners
		for cell in self.env.cells.values():
			if cell.index not in self.env.destination_cells.values():
				if cell.contained:
					return False
		# then make sure that the corner cells include only the right objects
		for object, cell_index in self.env.destination_cells.iteritems():
			cell = self.env.select_cell(cell_index)
			for element in cell.contained:
				if not element.compare(object):
					return False
		return True

	def compute_mean_fitness(self, agents):
		self.mean_fitness.append(np.mean([agent.fitness for agent in agents]))
		self.mean_altruistic_fitness.append(np.mean([agent.fitness for agent in agents if agent.strategy.type == StrategyTypes.ALTRUISTIC]))
		self.mean_mut_fitness.append(np.mean([agent.fitness for agent in agents if agent.strategy.type == StrategyTypes.MUTUALISTIC]))

	def compute_distance_learners(self, agents, matrix, measure=Distances.ENTROPY):
		distances = None
		if measure not in self.distances:
			self.distances[measure] = []
		if measure == Distances.ENTROPY:
			distances = [self.compute_entropy(agent, matrix) for agent in agents]
		elif measure == Distances.NORM:
			distances = [self.compute_norm(agent, matrix) for agent in agents]
		if self.store_mean:
			self.distances[measure].append((np.mean(distances), np.std(distances)))
		else:
			self.distances[measure].append(distances)

	def compute_distance_agents(self, agents, measure=Distances.JENSEN_SHANNON):
		dists = None
		if measure not in self.distances:
			self.distances[measure] = []
		if measure == Distances.ENTROPY or measure == Distances.NORM:
			raise Exception("Cannot use these measures as we have no 'true' distribution")
		if measure == Distances.JENSEN_SHANNON:
			dists = compute_agents_jensen_shannon(agents)
			dists = np.sqrt(dists)
			if np.isnan(dists):
				if self.distances and all(np.isnan(self.distances[measure][-10:])):
					raise ValueError('Too many nans')
		elif measure == Distances.HELLINGER:
			combs = combinations(agents, 2)
			dists = [compute_hellinger(comb[0].architecture.language_matrix, comb[1].architecture.language_matrix) for comb in combs]
		if self.store_mean:
			self.distances[measure].append((np.mean(dists), np.std(dists)))
		else:
			self.distances[measure].append(dists)

	def compare_agents_grammar(self, agents):
		agent1 = agents[0]
		for agent in agents[1:]:
			if not agent.has_same_model(agent1):
				return False
		return True

	def keep_track(self, goal, sequence, agents):
		pass

	def get_symbol_and_goal_index(self, status):
		symbol_index = self.env.symbols.index(status.sequence.symbols[0])
		goal_index = self.env.goals.index(status.goal)
		return symbol_index, goal_index

	def get_maximum(self, agent, symbol_index):
		row = agent.architecture.language_matrix[symbol_index, :]
		max_set = np.argwhere(row==row.max())
		if max_set.shape[0] == 1:
			return np.argmax(row)
		return None

	# def sample_consistency_old_way(self, agents, n_potential_pairings):
	# 	agent_pairings = combinations(agents, 2)
	# 	self.consistency_old.append(consistent_pairings(len(self.env.symbols), agent_pairings, n_potential_pairings))

	def sample_consistency(self, agents, potential_pairings):
		sum = 0
		n_signals = len(self.env.symbols)
		for signal in range(n_signals):
			freq_signal = [x for x in map(functools.partial(self.get_maximum, symbol_index=signal), agents) if x is not None]
			count = collections.Counter(freq_signal)
			counts = np.array([comb(x, 2) for x in count.values() if x>1])
			sum += counts.sum()
		sum /= n_signals
		sum /= potential_pairings
		self.consistency.append(sum)

	def are_agents_consistent(self, status):
		speaker_ = status.speaker
		listener_ = status.listener
		signal_index = self.env.symbols.index(status.sequence.symbols[0])
		row_speaker = speaker_.architecture.language_matrix[signal_index, :]
		row_listener = listener_.architecture.language_matrix[signal_index, :]
		max_1 = np.argwhere(row_speaker == row_speaker.max())
		max_2 = np.argwhere(row_listener == row_listener.max())
		if max_1.shape[0] == 1 and max_2.shape[0] == 1 and max_1 == max_2:
			return True
		return False


class HolisticGameUmpire(CooperationGameUmpire):

	def __init__(self, env, n_goals, store_mean=False):
		super(HolisticGameUmpire, self).__init__(env, store_mean)
		self.goal_track = np.zeros((n_goals), dtype=int)

	def on_new_interaction(self, status):
		self.attempts_per_interaction.append(status.number_of_attempts)
		self.interaction_index += 1

	def keep_track(self, goal, sequence, agents):
		goal_index = self.env.goals.index(goal)
		self.goal_track[goal_index] += 1

	def compute_norm(self, agent, matrix):
		diff = np.linalg.norm(agent.architecture.language_matrix - matrix)
		return diff

	def compute_entropy(self, agent, matrix):
		S = entropy(matrix, agent.architecture.language_matrix)
		return S.sum()


class FixedWordOrderGameUmpire(CooperationGameUmpire):

	def __init__(self, env, store_mean=False):
		super(FixedWordOrderGameUmpire, self).__init__(env, store_mean)
		# self.language_connection_matrix = np.zeros((len(AgentKnowledge.symbols), len(meanings)), dtype=int)
		self.word_lifespan = {}

	def keep_track(self, goal, sequence, agents):
		pass
		# symbols = get_symbol_indices(sequence)
		# goals = get_goal_indices(goal)
		# for index, symbol_index in enumerate(symbols):
		# 	goal_index = goals[index]
		# 	counter = sum(1 for agent in agents if agent.has_symbol_meaning_connection(symbol_index, goal_index))
		# 	if counter == 1:  # this means that only the speaker has it and it has just created it
		# 		created = self.word_lifespan.get((symbol_index, goal_index), None)
		# 		if not created:
		# 			self.word_lifespan[(symbol_index, goal_index)] = (self.interaction_index, )
		# 	self.language_connection_matrix[symbol_index, goal_index] = counter
		# self.check_lifespan(agents)
		# if not self.grammar_agreement_point:
		# 	if self.compare_agents_grammar(agents):
		# 		self.grammar_agreement_point = self.interaction_index

	def check_lifespan(self, agents):
		for k,v in self.word_lifespan.iteritems():
			if len(v) == 1:
				if sum(1 for agent in agents if agent.has_symbol_meaning_connection(k[0], k[1])) == 0:
					self.word_lifespan[k] = v + (self.interaction_index,)

	def compute_entropy(self, agent, mat_p):
		mat_q = agent.architecture.language_matrix
		S = 0.
		for i in range(mat_p.shape[0]):
			for j in range(mat_p.shape[1]):
				S += entropy(mat_p[i,j,:], mat_q[i,j,:])
		return S

	def compute_distance_agents(self, agents, measure=Distances.JENSEN_SHANNON):
		dists = None
		if measure not in self.distances:
			self.distances[measure] = []
		if measure == Distances.ENTROPY or measure == Distances.NORM:
			raise Exception("Cannot use these measures as we have no 'true' distribution")
		if measure == Distances.JENSEN_SHANNON:
			dists = compute_agents_jensen_shannon_3d(agents)
			dists = np.sqrt(dists)
		elif measure == Distances.HELLINGER:
			combs = combinations(agents, 2)
			dists = [compute_hellinger(comb[0].architecture.language_matrix, comb[1].architecture.language_matrix) for comb in combs]
		if self.store_mean:
			self.distances[measure].append((np.mean(dists), np.std(dists)))
		else:
			self.distances[measure].append(dists)


class VariableWordOrderUmpire(CooperationGameUmpire):
	def __init__(self):
		super(VariableWordOrderUmpire, self).__init__()
		self.distances = []

	def compare_agents_grammar(self, agents):
		distances = []
		top = (len(agents)/2)+1
		for i in range(top):
			current = agents[i].architecture.language_matrix
			current_measure = len(np.where(current != 0)[0])
			for j, learner in enumerate(agents[i+1:], start=i+1):
				new = learner.architecture.language_matrix
				equal = current == new
				new_measure = len(np.where(new != 0)[0])
				total_measure = current_measure + new_measure
				if total_measure == 0:
					total_measure = 1
				difference = len(np.where(equal==False)[0])
				distance = float(difference)/total_measure
				distances.append(distance)
		self.distances.append(distances)


class StochasticVariableWordOrderUmpire(VariableWordOrderUmpire):
	def __init__(self):
		super(StochasticVariableWordOrderUmpire, self).__init__()

	def compare_agents_grammar_to_teachers(self, agents, matrix):
		distances = [self.compute_distance(agent, matrix) for agent in agents]
		self.distances.append(distances)

	def compare_agents_grammar(self, agents):
		pass

	def compute_distance(self, agent, matrix):
		diff = np.linalg.norm(agent.architecture.language_matrix - matrix)
		return diff


class InfiniteGameUmpire(CooperationGameUmpire):

	def __init__(self, env):
		super(InfiniteGameUmpire, self).__init__(env)

	def is_game_finished(self, env):
		if not self.distances[Distances.JENSEN_SHANNON]:
			return False
		return self.distances[Distances.JENSEN_SHANNON][-1] == 0.0

	def track_measures(self):
		pass



class WordTrackUmpire(InfiniteGameUmpire):
	def __init__(self, env):
		super(WordTrackUmpire, self).__init__(env)
		self.words = []
		self.word_counts = []

	def track_word(self, status):
		self.words.append((self.env.goals.index(status.goal), self.env.symbols.index(status.sequence.symbols[0])))

	def on_new_interaction(self, status):
		self.interaction_index += 1
		self.interaction_agents.append((status.speaker.id, status.listener.id))
		if status.goal:
			self.goals_chosen.append(self.env.goals.index(status.goal))
			self.track_word(status)

	def get_word_count(self, agents):
		sum_ = np.array([agent.architecture.language_matrix for agent in agents])
		self.word_counts.append(sum_.sum(axis=0))


class SingleActionWordTrackUmpire(WordTrackUmpire):
	def __init__(self, env, goal_index=0):
		super(SingleActionWordTrackUmpire, self).__init__(env)
		self.goal_index = goal_index
		self.words = []
		self.word_counts = []
		self.agent = 1
		self.single_agent_stats = dict(zip(range(len(self.env.goals)), [list() for i in range(len(self.env.goals))]))
		self.global_agent_stats = dict(zip(range(len(self.env.goals)), [list() for i in range(len(self.env.goals))]))
		self.language_matrices = []


	def store_language_matrices(self, agents):
		self.language_matrices.append([agent.architecture.language_matrix for agent in agents])


	def store_agents_stats(self, agents):
		for index, goal in enumerate(self.env.goals):
			self.single_agent_stats[index].append(np.copy(agents[self.agent].architecture.language_matrix[:, index]))
			sum_ = np.array([agent.architecture.language_matrix[:, index] for agent in agents])
			self.global_agent_stats[index].append(sum_.sum(axis=0)/len(agents))


class PopulationGameUmpire(WordTrackUmpire):
	def __init__(self, env, action_cost, coord_cost, interval):
		super(PopulationGameUmpire, self).__init__(env)
		self.action_cost = action_cost
		self.coord_cost = coord_cost
		self.interval = interval
		self.ratios = []
		self.changes = []
		self.changed_ids = []
		self.interaction_agents = []

	def on_new_interaction(self, status):
		self.interaction_index += 1
		self.interaction_agents.append((status.speaker.id_strategy, status.listener.id_strategy))
		last_goal = self.env.goals.index(status.goal) if status.goal else None
		self.goals_chosen.append(last_goal)
		self.update_cons_matrix = True


	def track_ratios(self, agents):
		ratio = sum(1 for agent in agents if agent.strategy.type == StrategyTypes.ALTRUISTIC)
		self.ratios.append(ratio)

	def track_changes(self, n_changes):
		self.changes.append(n_changes)

	def record_change(self, agent_id):
		self.changed_ids.append(agent_id)

	def is_game_finished(self, n_agents_):
		return self.consistency[-1] == 1.0 and (self.ratios[-1] == n_agents_ or self.ratios[-1]==0)
		# return self.ratios[-1] == n_agents_ or self.ratios[-1] == 0
