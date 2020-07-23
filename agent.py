# from network import Network
from sequence import creators
from environment import DESTINATIONS,is_cell_in_grid, meanings, meanings_dict, Categories, categories_columns, columns_categories, get_symbol_indices, get_goal_indices, REDUCED_GOALS
from alignment import DefaultWordAlignment
from random import choice
from numpy import zeros, nonzero, argwhere, where, full, copy, argmax, argmin, ones, unravel_index, argsort, all, squeeze
from numpy.random import shuffle, uniform
from strategies import strategies, MutualisticStrategy, StrategyTypes
from sequence import Sequence
from itertools import product
from collections import deque, defaultdict
import inspect


__all__ = ['Role', 'Agent', 'CognitiveArchitecture', 'AgentWithNetwork', 'AgentWithStrategy']


class Role:
	SPEAKER = 0
	LISTENER = 1
	names = {SPEAKER: 'SPEAKER', LISTENER: 'LISTENER'}


class AgentKnowledge:
	symbols = ['pro', 'pre', 'ante', 'ing', 'ung', 'ion', 'ba', 'beta', 'boto', 'sasa']
	reduced = ['bli', 'blo', 'ama', 'ita', 'don', 'go', 'du']
	holistic_symbols = symbols + reduced
	reduced_holistic_symbols = reduced +  ['kap']
	english_symbols = meanings



class Agent(object):
	"""
		An agent participates in a game. Its cognitive architecture will hopefully evolve (and will hopefully deserve such
		a pretentious name).

	"""

	def __init__(self, role=None, architecture='simple', creator='literal', failure_alignment=False, id=0):
		if isinstance(architecture, CognitiveArchitecture):
			self.architecture = architecture
		elif inspect.isclass(architecture) and issubclass(architecture, CognitiveArchitecture):
			self.architecture = architecture()
		elif architecture in architectures:
			self.architecture = architectures[architecture]()
		else:
			raise Exception("Agent needs a valid cognitive architecture")
		self.role = role
		self.sequence_creator = creators[creator]()
		self.fitness = 0
		self.num_interactions = 0
		self.failure_alignment = failure_alignment
		self.id = id

	def print_role(self):
		print(Role.names[self.role])

	def set_role(self, role):
		self.role = role

	def add_interaction(self):
		self.num_interactions += 1

	def pay_cost(self, amount):
		self.fitness -= amount

	def receive_reward(self, amount):
		self.fitness += amount

	def add(self, goal, sequence):
		self.architecture.add(goal, sequence)

	def knows_sequence(self, sequence):
		return self.architecture.is_sequence_contained(sequence)

	def get_goal_from_sequence(self, sequence):
		return self.architecture.get_goal_from_sequence(sequence)

	def create_new_sequence(self, length):
		"""
			create a sequence of at least 1 and at most max_length random symbols
			It's best to do this in the agent, because we want each agent to have a different strategy
			for creating sequences
		"""
		return self.sequence_creator.create_sequence(AgentKnowledge.symbols, length)

	def knows_goal(self, goal):
		return self.architecture.is_goal_contained(goal)

	def get_goals(self, goal):
		return [g for g in self.architecture.dictionary.keys() if g.is_equal(goal)]

	def choose_goal_to_perform(self, status, env):
		return self.architecture.choose_goal_to_perform(status, env)

	def get_sequence(self, goal, env):
		seq = self.architecture.get_sequence(goal, env)
		if seq and type(seq) == list:
			seq = self.choose_sequence(seq)
		return seq

	# for now we can just choose at random
	def choose_sequence(self, seqs):
		return choice(seqs)

	def has_symbol_meaning_connection(self, symbol_index, meaning_index):
		return self.architecture.has_symbol_meaning_connection(symbol_index, meaning_index)

	def has_same_model(self, agent):
		return self.architecture.is_same_model(agent.architecture)

	def react_to_failure(self, status, env):
		if self.failure_alignment:
			self.architecture.react_to_failure(status, env)

	def react_to_success(self, status, env):
		self.architecture.react_to_success(status, env)

	def print_model(self):
		for sequence in self.architecture.dictionary:
			print("sequence: {0}, outcome: \'{1}\'".format(sequence.symbols, self.architecture.dictionary[sequence]))


class AgentWithStrategy(Agent):
	def __init__(self, strategy_type, role=None, architecture='simple', creator='literal', max_tries=None, failure_alignment=False, id=0):
		super(AgentWithStrategy, self).__init__(role=role, architecture=architecture, creator=creator, failure_alignment=failure_alignment, id=id)
		if isinstance(strategy_type, MutualisticStrategy):
			self.strategy = strategy_type
		elif strategy_type == StrategyTypes.ALTRUISTIC:
			self.strategy = strategies[strategy_type]()
		elif strategy_type in strategies:
			self.strategy = strategies[strategy_type](max_tries)
		else:
			raise Exception("Not a valid strategy")
		self.id_strategy = (self.id, self.strategy.type)

	def choose_feasible_goal(self, current_cell, attempted_goal, strategy='random', num_colums=8, num_rows=8):
		return self.architecture.choose_feasible_goal(current_cell, attempted_goal)

	def should_repeat(self, status):
		return self.strategy.should_repeat(status.attempts)

	def check_feasibility_of_goal_direction(self, current_index, goal, n_columns, n_rows):
		return self.architecture.check_feasibility_of_goal_direction(self, current_index, goal, n_columns, n_rows)

	def get_sequence_and_meanings(self, goal, env):
		return self.architecture.get_sequence(goal, env)

	def add_word_position(self, symbols_meanings):
		self.architecture.add_word_position(symbols_meanings)

	def prepare_for_interaction(self, status, env):
		self.architecture.prepare_for_interaction(status, env)


class AgentWithMemory(AgentWithStrategy):
	def __init__(self, strategy_type, role=None, architecture='simple', creator='literal', window_size=5, failure_alignment=False, id=0):
		super(AgentWithMemory, self).__init__(role=role, strategy_type=strategy_type, architecture=architecture, creator=creator, failure_alignment=failure_alignment, id=id)
		# self.cost_window = deque(zeros(window_size))
		self.cost_memory = [0]*window_size
		self.fitness_memory = []
		self.n_changes = 0
		self.changes = []

	def record_interaction_cost(self, amount):
		self.cost_memory.append(amount)
		# self.cost_window.popleft()
		# self.cost_window.append(amount)

	def should_repeat(self, reward, coord_cost, status):
		return self.strategy.should_repeat(reward, coord_cost, status, self)

	def record_fitness(self, index):
		self.fitness_memory.append((index, self.fitness))

	def record_change(self):
		self.n_changes += 1
		self.changes.append(self.strategy.name)


def reward_agents(agents, reward):
	for agent in agents:
		agent.receive_reward(reward)


class CognitiveArchitecture(object):
	"""
		An architecture contains an agent\'s representation of the environment
	"""

	def __init__(self, delta=.1):
		self.dictionary = {}
		self.delta = delta
		self.pos_alpha = 1. + self.delta
		self.neg_alpha = 1. - self.delta
		self.threshold = 1.e-20

	@staticmethod
	def init_matrix_(n_rows, n_columns):
		mat = uniform(size=(n_rows, n_columns))
		for i in range(mat.shape[0]):
			mat[i, :] /= mat[i, :].sum()
		return mat

	def add(self, goal, sequence):
		self.dictionary[goal] = sequence

	def get_outcome(self, sequence):
		return self.dictionary[sequence]

	def get_sequence(self, goal):
		return self.dictionary.get(goal, None)

	def get_goal_from_sequence(self, sequence):
		return [k for k, v in self.dictionary.iteritems() if v.symbols == sequence.symbols]

	def is_sequence_contained(self, sequence):
		any(v.symbols == sequence.symbols for v in self.dictionary.values())

	def is_goal_contained(self, goal):
		return goal in self.architecture.dictionary.keys()

	def has_symbol_meaning_connection(self, symbol_index, meaning_index):
		return False

	def choose_goal_to_perform(self, status, env):
		knows = self.is_sequence_contained(status.sequence)
		goal = None
		if knows:
			goals = self.get_goal_from_sequence(status.sequence)
			if goals:
				for possible_goal in goals:
					if self.check_feasibility_of_goal_direction(status.index, possible_goal, env.num_columns, env.num_rows):
						goal = possible_goal
						continue
		if not goal:
			goal = self.choose_feasible_goal(env, status)
		return goal

	def choose_feasible_goal(self, env, status, strategy='random'):
		possible_goals = self.get_possible_goals(env, status, env.goals)
		if strategy == 'random':
			return choice(possible_goals)
		elif strategy == 'max':
			return None
		else:
			return None

	def get_goal_from_meanings(self, columns, env, status):
		possible_goals = env.goals
		column_meanings = [meanings[column] if column is not None else None for column in columns]
		for cat, meaning in enumerate(column_meanings):
			if meaning:
				possible_goals = [goal for goal in possible_goals if goal.compare_by_category(cat, meaning)]
		if len(possible_goals) == 0 and column_meanings[0]=="DO_NOTHING":
			return env.goals[-1]
		if len(possible_goals) > 1:
			return choice(self.choose_feasible_goal(env, status, possible_goals))
		return possible_goals[0]

	def get_possible_goals(self, env, status, goals):
		current_cell = env.select_cell(status.index)
		possible_cells = self.select_feasible_cells(current_cell.index, env.num_columns, env.num_rows)
		possible_directions = (goal for goal in goals if goal.direction in possible_cells)
		possible_goals = [goal for goal in possible_directions for element in current_cell.contained if
							element.compare(goal.element) and goal not in status.goals_performed]
		if env.goals[-1] not in status.goals_performed:
			possible_goals.append(env.goals[-1])
		return possible_goals

	@staticmethod
	def select_feasible_cells(current_cell, num_columns, num_rows):
		return [dest for dest, ix in DESTINATIONS.iteritems() if
					is_cell_in_grid(current_cell[0] + ix[0], current_cell[1] + ix[1], num_columns, num_rows)]

	def check_feasibility_of_direction(self, direction, env, status):
		dest = DESTINATIONS.get(direction, None)
		if not dest:
			return True
		cell_index = status.index
		return is_cell_in_grid(cell_index[0] + dest[0], cell_index[1] + dest[1], env.num_columns, env.num_rows)

	def check_feasibility_of_goal_direction(self, current_index, goal, n_columns, n_rows):
		if not goal.direction:
			return True
		destination = DESTINATIONS.get(goal.direction, None)
		if not destination:
			return True
		x,y = (current_index[0] + destination[0], current_index[1] + destination[1])
		return is_cell_in_grid(x , y, n_columns, n_rows)

	def check_feasibility_of_goal(self, goal, status, env):
		if not goal.element:
			return True
		element = goal.element
		if (
					not self.check_feasibility_of_goal_direction(status.index, goal, env.num_columns, env.num_rows) or
					not self.check_feasibility_of_meaning(element.color.category, element.color.value, env, status) or
					not self.check_feasibility_of_meaning(element.shape.category, element.shape.value, env, status)
		):
			return False
		return True

	def is_same_model(self, other):
		if len(self.dictionary) != len(other.dictionary):
			return False
		diff = set(self.dictionary.keys()) - set(other.dictionary.keys())
		if len(diff) != 0:
			return False
		for key in self.dictionary:
			if self.dictionary[key] != other.dictionary[key]:
				return False
		return True

	def check_feasibility_of_meaning(self, category, meaning, env, status):
		if category == Categories.COLOR or category == Categories.SHAPE:
			contained = env.cells[status.index].contained
			return any(element for element in contained if element.has_property(category, meaning))
		elif category == Categories.DIRECTION:
			return self.check_feasibility_of_direction(meaning, env, status)
		else:
			return True

	def react_to_failure(self, status, env):
		pass

	def react_to_success(self, status, env):
		if status.goal in self.dictionary:
			del self.dictionary[status.goal]
		self.add(status.goal, status.sequence)

	def prepare_for_interaction(self, status, env):
		pass


class HolisticCognitiveArchitecture(CognitiveArchitecture):

	def __init__(self, n_rows, n_columns, random=False, delta=.1):
		super(HolisticCognitiveArchitecture, self).__init__(delta=delta)
		self.language_matrix = full((n_rows, n_columns), 1./n_columns, dtype=float) if not random else self.init_matrix_(n_rows, n_columns)
		self.interaction_matrix = None
		self.init = .00001
		self.minimal = self.init if n_columns == 1 else self.init/(n_columns - 1)
		self.maximal = 1 - self.init

	def get_sequence(self, goal, env):
		goal_index = env.goals.index(goal)
		column = self.language_matrix[:, goal_index]
		max_set = argwhere(column == column.max())
		symbol_index = choice(squeeze(max_set)) if len(max_set) > 1 else argmax(column)
		return Sequence([env.symbols[symbol_index]])

	## we choose a symbol at random. If it's already the greatest value of some other meaning, then we discard it and choose another
	def choose_new_symbol(self):
		indices = range(self.language_matrix.shape[0])
		columns = range(self.language_matrix.shape[1])
		shuffle(columns)
		index = None
		while indices:
			index = choice(indices)
			is_chosen = True
			for i in columns:
				if index == self.language_matrix[:,i].argmax():
					indices.remove(index)
					is_chosen = False
					break
			if is_chosen:
				break
		if not index:
			print("THIS SHOULD NOT HAPPEN!!!")
			index = choice(range(self.language_matrix.shape[0]))
		return index

	def prepare_for_interaction(self, status, env):
		self.interaction_matrix = copy(self.language_matrix)
		# goals = env.goals
		# for i, goal in enumerate(goals):
		# 	feasible = self.check_feasibility_of_goal(goal, status, env)
		# 	if not feasible:
		# 		self.interaction_matrix[:, i] = 0.

	def choose_goal_to_perform(self, status, env):
		goals = env.goals
		symbol_index = env.symbols.index(status.sequence.symbols[0])
		row = self.interaction_matrix[symbol_index, :]
		for goal in status.goals_performed:
			row[goals.index(goal)] = 0.
		active = where(row != 0.)[0]
		if not active.size:
			raise Exception("Have tried out all possibilities")
		max_arg = argmax(row[active])
		if any(row[active]==row[active[max_arg]]):
			max_s = argwhere(row == row[active[max_arg]])
			choose = choice(max_s)[0]
			return goals[choose]
		return goals[max_arg]


	def get_goal_and_symbol_index(self, status, env):
		symbol = status.sequence.symbols[0]
		return env.goals.index(status.goal), env.symbols.index(symbol)

	def react_to_success(self, status, env):
		goal_index, symbol_index = self.get_goal_and_symbol_index(status, env)
		# column = self.language_matrix[:,goal_index]
		# mask = ones(column.shape, dtype=bool)
		# mask[symbol_index] = 0
		# column[mask] *= self.neg_alpha
		# column[symbol_index] *= self.pos_alpha
		# column[column < self.minimal] = self.minimal
		# column[column > self.maximal] = self.maximal
		# column /= column.sum()
		row = self.language_matrix[symbol_index, :]
		mask = ones(row.shape, dtype=bool)
		mask[goal_index] = 0
		row[mask] *= self.neg_alpha
		row[goal_index] *= self.pos_alpha
		row[row < self.minimal] = self.minimal
		row[row > self.maximal] = self.maximal
		row /= row.sum()

	def react_to_failure(self, status, env):
		goal_index = env.goals.index(status.goals_performed[-1])
		symbol = status.sequence.symbols[0]
		symbol_index = env.symbols.index(symbol)
		# row = self.language_matrix[symbol_index, :]
		# row[goal_index] *= self.neg_beta
		# row[row < self.minimal] = self.minimal
		# row /= row.sum()
		column = self.language_matrix[goal_index, :]
		column[symbol_index] *= self.neg_beta
		row = self.language_matrix[symbol_index, :]
		row[row < self.minimal] = self.minimal
		row /= row.sum()



class MultipleValueCognitiveArchitecture(CognitiveArchitecture):
	"""
		An architecture in which an agent might have more than one sequence for the same goal
	"""
	def __init__(self):
		super(MultipleValueCognitiveArchitecture, self).__init__()

	def add(self, goal, sequence):
		values = self.dictionary.get(goal, None)
		values = [] if not values else values
		values.append(sequence)
		self.dictionary[goal] = values

	def get_sequence(self, goal):
		return self.dictionary.get(goal, None)

	# we might have associated the same sequence to more than one goal
	def get_goal_from_sequence(self, sequence):
		return [goal for goal, seq in self.dictionary.iteritems() for v in seq if v.compare(sequence)]

	def is_sequence_contained(self, sequence):
		return any(seq for seq in self.dictionary.values() for v in seq if v.compare(sequence))



class StochasticMatrixCognitiveArchitecture(CognitiveArchitecture):

	class InteractionMemory:
		def __init__(self):
			self.cats_searched = dict(zip(range(len(categories_columns)), [0]*len(categories_columns)))
			self.cats_columns = None
			self.flat_cats = None
			self.symbols = None
			self.columns = None

	def __init__(self, n_symbols, n_meanings, random=False, align_strat=None):
		super(StochasticMatrixCognitiveArchitecture, self).__init__()
		self.random = random
		self.language_matrix = self.initialise_matrix(n_symbols, n_meanings)
		self.interaction_matrix = None
		self.init = .00001
		self.minimal = self.init/(n_meanings - 1)
		self.interaction_memory = self.InteractionMemory()
		self.maximal = 1 - self.init
		self.neg_beta = 1. - .4
		self.align_strat = align_strat if align_strat else DefaultWordAlignment()

	def initialise_matrix(self, n_rows, n_columns):
		return full((n_rows, n_columns), 1./n_rows, dtype=float) if not self.random else self.init_matrix_(n_rows, n_columns)

	def get_sequence(self, goal, env):
		goal_indices = get_goal_indices(goal)
		symbols = []
		for index in goal_indices:
			column = self.language_matrix[:, index]
			if any(column != column[0]):
				symbol_index = column.argmax()
			else:
				symbol_index = self.choose_new_symbol()
			symbols.append(AgentKnowledge.symbols[symbol_index])
		return Sequence(symbols)

	## we choose a symbol at random. If it's already the greatest value of some other meaning, then we discard it and choose another
	def choose_new_symbol(self):
		indices = range(self.language_matrix.shape[0])
		columns = range(self.language_matrix.shape[1])
		shuffle(columns)
		index = None
		while indices:
			index = choice(indices)
			is_chosen = True
			for i in columns:
				if index == self.language_matrix[:,i].argmax():
					indices.remove(index)
					is_chosen = False
					break
			if is_chosen:
				break
		if not index:
			print("THIS SHOULD NOT HAPPEN!!!")
			index = choice(range(self.language_matrix.shape[0]))
		return index

	def react_to_success(self, status, env):
		symbol_indices = get_symbol_indices(status.sequence, env.symbols)
		goal_indices = get_goal_indices(status.goal)
		mask = ones(self.language_matrix.shape[1], dtype=bool)
		mask[symbol_indices] = 0
		for goal in goal_indices:
			colum = self.language_matrix[:, goal]
			colum[mask] *= self.neg_alpha
			colum[~mask] *= self.pos_alpha
		for symbol in symbol_indices:
			row = self.language_matrix[symbol,:]
			row[row < self.minimal] = self.minimal
			row[row > self.maximal] = self.maximal
			row /= row.sum()

	def react_to_failure(self, status, env):
		self.align_strat.negative_align(self, status, env)

	def prepare_for_interaction(self, status, env):
		self.interaction_memory = self.InteractionMemory()
		self.interaction_matrix = copy(self.language_matrix)
		for i in range(self.interaction_matrix.shape[1]):
			meaning = meanings[i]
			category = meanings_dict[meaning]
			if not self.check_feasibility_of_meaning(category, meaning, env, status):
				self.interaction_matrix[:, i] = 0.

	def choose_goal_to_perform(self, status, env):
		symbol_indices = get_symbol_indices(status.sequence, env.symbols)
		is_first_search = False
		if not self.interaction_memory.cats_columns:
			self.interaction_memory.cats_columns = self.divide_matrix_in_categories(symbol_indices)
		if not self.interaction_memory.flat_cats:
			self.interaction_memory.flat_cats = [cat.flatten().argsort()[::-1] for cat in self.interaction_memory.cats_columns]
			is_first_search = True
		symbols = [] if is_first_search else self.interaction_memory.symbols
		columns = [] if is_first_search else self.interaction_memory.columns
		if is_first_search:
			for i, cat in enumerate(self.interaction_memory.cats_columns):
				index_to_search = self.interaction_memory.cats_searched[i]
				self.interaction_memory.cats_searched[i] += 1
				row, column = unravel_index(self.interaction_memory.flat_cats[i][index_to_search], cat.shape)
				real_row = symbol_indices[row]
				symbols.append(real_row)
				real_column = categories_columns[i][column]
				columns.append(real_column)
			## we need to make sure there are no repeated symbols
			# self.eliminate_repeated(symbols, columns, symbol_indices)
		else:
			values = [self.interaction_matrix[symbols[i], columns[i]] for i in range(len(symbols))]
			sorted_values = list(argsort(values))
			is_chosen = False
			while not is_chosen and sorted_values:
				minimum = sorted_values.pop(0)
				is_chosen = self.check_available_values(minimum)
			if is_chosen:
				next_row, next_column = self.find_next_greatest(minimum)
				if next_row is not None:
					real_row = symbol_indices[next_row]
					real_column = categories_columns[minimum][next_column]
					symbols[minimum] = real_row
					columns[minimum] = real_column
			else:
				minimum = argmin(values)
				symbols[minimum] = None
				columns[minimum] = None
		self.interaction_memory.symbols = symbols
		self.interaction_memory.columns = columns
		return self.get_goal_from_meanings(columns, env, status)

	def check_available_values(self, index):
		to_search = self.interaction_memory.cats_searched[index]
		if self.interaction_memory.flat_cats[index].shape[0] <= to_search:
			return False
		next_index = self.interaction_memory.flat_cats[index][to_search]
		cat = self.interaction_memory.cats_columns[index]
		row, column = unravel_index(self.interaction_memory.flat_cats[index][to_search], cat.shape)
		next_value = cat[row, column]
		if next_value == 0.:
			return False
		return True

	def divide_matrix_in_categories(self, symbol_indices):
		cats = []
		for cat in categories_columns.keys():
			start = categories_columns[cat][0]
			end = categories_columns[cat][-1]+1
			cats.append(self.interaction_matrix[symbol_indices, start:end])
		return cats

	def find_next_greatest(self, index):
		to_search = self.interaction_memory.cats_searched[index]
		self.interaction_memory.cats_searched[index] += 1
		if self.interaction_memory.flat_cats[index].shape[0] <= to_search:
			return None, None
		next_row, next_column = unravel_index(self.interaction_memory.flat_cats[index][to_search], self.interaction_memory.cats_columns[index].shape)
		return next_row, next_column

	def find_repeated(self, symbols):
		D = defaultdict(list)
		for i,item in enumerate(symbols):
			D[item].append(i)
		return {k:v for k,v in D.items() if len(v)>1}

	def eliminate_repeated(self, symbols, columns, symbol_indices):
		repeated_symbols = self.find_repeated(symbols)
		while repeated_symbols:
			for k, v in repeated_symbols.iteritems():
				values = [self.interaction_matrix[k, columns[value]] for value in v]
				max_value = argmax(values)
				rest = v[:max_value]+ v[max_value+1:]
				for i, r in enumerate(rest):
					index = r
					real_row, real_column = self.choose_next_value(index, symbol_indices)
					if not real_row:
						if i < len(rest)-1:
							pass
						else:
							index = v[max_value]
							real_row, real_column = self.choose_next_value(index, symbol_indices)
					symbols[index] = real_row
					columns[index] = real_column
			repeated_symbols = self.find_repeated(symbols)

	def is_applicable(self, row, column):
		return self.interaction_matrix[row, column] != 0.

	def choose_next_value(self, index, symbol_indices):
		next_row, next_column = self.find_next_greatest(index)
		real_row = symbol_indices[next_row]
		real_column = categories_columns[index][next_column]
		if not self.is_applicable(real_row, real_column):
			return None, None
		return real_row, real_column


class Matrix3DFixedWordOrderArchitecture(StochasticMatrixCognitiveArchitecture):

	class InteractionMemory:
		def __init__(self):
			self.cats_searched = dict(zip(range(len(categories_columns)), [0]*len(categories_columns)))
			self.positions = None
			self.ordered_positions = None
			self.cats = None
			self.columns = None
			self.conflicts = {}

	def __init__(self, n_rows, n_columns, n_positions, align_strat=None):
		super(Matrix3DFixedWordOrderArchitecture, self).__init__(n_rows, n_columns)
		self.language_matrix = self.init_language_matrix_(n_rows, n_columns, n_positions)
		self.interaction_memory = self.InteractionMemory()
		self.align_strat = align_strat

	def init_language_matrix_(self, n_rows, n_columns, n_positions):
		mat = uniform(size=(n_rows, n_columns,n_positions))
		for i in range(mat.shape[0]):
			for j in range(mat.shape[1]):
				mat[i, j, :] /= mat[i, j, :].sum()
		return mat

	def get_sequence(self, goal, env):
		goal_indices = get_goal_indices(goal)
		symbols = [None] * len(goal_indices)
		positions = []
		for index in goal_indices:
			pos_found = False
			search_index = 0
			column = self.language_matrix[:, index, :]
			sorted_column = column.flatten().argsort()[::-1]
			while not pos_found:
				sym_index, pos = unravel_index(sorted_column[search_index], column.shape)
				if pos not in positions:
					positions.append(pos)
					position = pos if pos < len(goal_indices) else 0
					symbols[position] = sym_index
					pos_found = True
				else:
					contained_sym = symbols[pos]
					contained_value = column[contained_sym, pos]
					new_value = column[sym_index, pos]
					if new_value > contained_value:
						symbols[pos] = sym_index
					search_index += 1
		symbs = [env.symbols[symbol] for symbol in symbols]
		return Sequence(symbs)

	def prepare_for_interaction(self, status, env):
		self.interaction_memory = self.InteractionMemory()
		self.interaction_matrix = copy(self.language_matrix)
		for i in range(self.interaction_matrix.shape[1]):
			meaning = meanings[i]
			category = meanings_dict[meaning]
			if not self.check_feasibility_of_meaning(category, meaning, env, status):
				self.interaction_matrix[:, i, :] = 0.

	def choose_goal_to_perform(self, status, env):
		symbol_indices = get_symbol_indices(status.sequence, env.symbols)
		is_first_search = False
		if not self.interaction_memory.positions:
			self.interaction_memory.positions = [self.interaction_matrix[index, :, i] for i, index in enumerate(symbol_indices)]
		if not self.interaction_memory.ordered_positions:
			self.interaction_memory.ordered_positions = [row.argsort()[::-1] for row in self.interaction_memory.positions]
			is_first_search = True
		columns = [] if is_first_search else self.interaction_memory.columns
		if is_first_search:
			for i, pos in enumerate(self.interaction_memory.ordered_positions):
				index_to_search = self.interaction_memory.cats_searched[i]
				self.interaction_memory.cats_searched[i] += 1
				columns.append(pos[index_to_search])
			self.interaction_memory.cats = [columns_categories[column] for column in columns]
			repeated = self.find_repeated(self.interaction_memory.cats, columns)
			if repeated:
				for cat, indexes in repeated.iteritems():
					sorted_cats = sorted(indexes, key=lambda index: self.interaction_memory.positions[index[0]][self.interaction_memory.cats_searched[cat]-1],
					                     reverse=True)
					self.interaction_memory.conflicts[cat] = sorted_cats[1:]
					for item in self.interaction_memory.conflicts[cat]:
						columns[item[0]] = None
		else:
			if self.interaction_memory.conflicts:
				k = self.interaction_memory.conflicts.keys()[0]
				conflicted = [item for item in self.interaction_memory.columns if item is not None and columns_categories[item] == k]
				if len(conflicted) > 1:
					raise Exception("There should not be two values of the same category after purging")
				conf_index = columns.index(conflicted[0])
				columns[conf_index] = None
				index, column = self.interaction_memory.conflicts[k].pop()
				columns[index] = column
				if not self.interaction_memory.conflicts[k]:
					del self.interaction_memory.conflicts[k]
			# else:

		self.interaction_memory.columns = columns
		goal = self.try_to_guess(columns, env, status)
		if goal:
			return goal
		else:
			return self.choose_feasible_goal(env, status)

	def try_to_guess(self, columns, env, status):
		possible_goals = self.get_possible_goals(env, status, env.goals)
		column_meanings = [(columns_categories[column], meanings[column]) for column in columns if column is not None]
		for cat, meaning in column_meanings:
			if meaning:
				possible_goals = [goal for goal in possible_goals if goal.compare_by_category(cat, meaning)]
		if len(possible_goals) == 0 and column_meanings[0][1]=="DO_NOTHING":
			return env.goals[-1]
		elif len(possible_goals) == 1:
			return possible_goals[0]
		elif len(possible_goals) > 1:
			return choice(possible_goals)

	def find_repeated(self, cats, columns):
		D = defaultdict(list)
		for i,item in enumerate(cats):
			D[item].append((i, columns[i]))
		return {k:v for k,v in D.items() if len(v)>1}

	#we know the position of the symbol, but not the exact meaning, so we increase the value of all possible four meanings in that position
	def react_to_success(self, status, env):
		symbol_indices = get_symbol_indices(status.sequence, env.symbols)
		goal_indices = get_goal_indices(status.goal)
		for position, symbol in enumerate(symbol_indices):
			for goal_index in goal_indices:
				column = self.language_matrix[symbol, goal_index, :]
				mask = ones(column.shape[0], dtype=bool)
				mask[position] = 0
				column[mask] *= self.neg_alpha
				column[position] *= self.pos_alpha
				column[column < self.minimal] = self.minimal
				column[column > self.maximal] = self.maximal
				column /= column.sum()

	def react_to_failure(self, status, env):
		if self.align_strat:
			self.align_strat(self, status, env)
		else:
			symbol_indices = get_symbol_indices(status.sequence, env.symbols)
			goal_indices = get_goal_indices(status.goals_performed[-1])
			for position, symbol in enumerate(symbol_indices):
				# column = self.interaction_memory.columns[position]
				for goal_index in goal_indices:
					column = self.language_matrix[symbol, goal_index, :]
					column[position] *= self.neg_beta
					column /= column.sum()
				# column = self.language_matrix[symbol, :, position]
				# mask = ones((self.language_matrix.shape[1], self.language_matrix.shape[2]), dtype=bool)
				# mask[goal_indices] = 0
				# if column:
				# 	row = self.language_matrix[symbol, column, :]
				# 	row[position] *= self.neg_beta
				# 	row[row < self.minimal] = self.minimal
				# 	row /= row.sum()




class MatrixVariableWordOrderArchitecture(CognitiveArchitecture):
	"""
		2-D matrix in which we store the sequence position (1,2,3,4)
	"""
	def __init__(self):
		super(MatrixVariableWordOrderArchitecture, self).__init__()
		categories = Categories.get_categories()
		self.interaction_memory = dict(zip([cat[1] for cat in categories], [[] for i in range(len(categories))])) # a dictionary to keep track of the different interpretations the agent has tried in each interaction
		self.combinations = []
		self.known_categories = []

	def get_sequence(self, goal):
		goal_indices = get_goal_indices(goal)
		shuffle(goal_indices)    # this is to make sure that the order is not determined by categories
		symbols = [0]*len(goal_indices)
		indices_without_meaning = []
		for index in goal_indices:
			column = self.language_matrix[:, index]
			symbol_indices = list(nonzero(column)[0])
			if symbol_indices:
				symbol_was_positioned = False
				while symbol_indices:
					chosen = choice(symbol_indices)   # if there is some symbol associated to this meaning we choose one (if there is only one symbol then it will always be chosen)
					symbol_indices.remove(chosen)
					position = column[chosen]         # the matrix stores the position of this symbol (starting with 1, so we don't confuse it with no value)
					if symbols[position-1] == 0:
						symbols[position-1] = (AgentKnowledge.symbols[chosen], meanings[index])
						symbol_was_positioned = True
						break
				if not symbol_was_positioned:
					indices_without_meaning.append((AgentKnowledge.symbols[chosen], index))
			else:
				indices_without_meaning.append((choice(AgentKnowledge.symbols), index))
		non_occupied_indices = [idx for idx in range(len(symbols)) if symbols[idx] == 0]
		assert len(non_occupied_indices) == len(indices_without_meaning), "THIS SHOULD NOT HAPPEN"
		for i, index in enumerate(non_occupied_indices):
			symbols[index] = (indices_without_meaning[i][0], meanings[indices_without_meaning[i][1]])
		return symbols

	def add_word_position(self, symbols_meanings):
		for i, symbol_meaning in enumerate(symbols_meanings):
			symbol = AgentKnowledge.symbols.index(symbol_meaning[0])
			meaning = meanings.index(symbol_meaning[1])
			self.language_matrix[symbol, meaning] = i+1

	def build_combinations(self, status):
		symbol_indices = get_symbol_indices(status.sequence)
		for position, symbol_index in enumerate(symbol_indices):
			position += 1
			row = self.language_matrix[symbol_index, :]
			meaning_indices = nonzero(row)[0]
			possible_meanings = (meaning_index for meaning_index in meaning_indices if self.language_matrix[symbol_index,meaning_index] == position)
			for pos_meaning in possible_meanings:
				self.store_possible_meaning(pos_meaning)
		self.known_categories = [(k,v) for k,v in self.interaction_memory.iteritems() if v]
		if self.known_categories:
			return list(product(*(t[1] for t in self.known_categories)))

	def store_possible_meaning(self, meaning):
		category = meanings_dict[meanings[meaning]]
		if meaning not in self.interaction_memory[category]:
			self.interaction_memory[category].append(meaning)

	def choose_goal_to_perform(self, status, env):
		if status.number_of_attempts == 1:
			self.combinations = self.build_combinations(status)
		if self.combinations:
			possible_goals = self.get_possible_goals(env, status, env.goals)
			while self.combinations:
				combination = self.combinations.pop()
				goal = self.get_goal_from_combination(combination, possible_goals)
				if goal:
					return goal
		return self.choose_feasible_goal(env, status)

	def get_goal_from_combination(self, combination, possible_goals):
		for i, comb in enumerate(combination):
			category = self.known_categories[i][0]
			meaning = meanings[comb]
			possible_goals = self.filter_goal_by_category(possible_goals, category,meaning)
			if not possible_goals:
				return None
		if possible_goals:
			return choice(possible_goals)

	def check_feasibility_of_goal(self, goal, status, env):
		if not self.check_feasibility_of_goal_direction(status.index, goal, env.num_columns, env.num_rows):
			return False
		return True

	def choose_from_occupied_categories(self, memorised_categories, possible_goals, status):
		for k in memorised_categories:
			chosen_meaning = choice(self.interaction_memory[k])
			possible_goals = self.filter_goal_by_category(possible_goals, k, meanings[chosen_meaning])
			self.interaction_memory[k].remove(chosen_meaning)
			possible_goals = [goal for goal in possible_goals if goal not in status.goals_performed]
			if possible_goals:
				return choice(possible_goals)
		return None
	"""
	 This can be done in different ways. Check game_spec_7
	    - we delete all connections where the same word is used in the same position
	"""
	def react_to_success(self, status, env):
		for position, symbol_meaning in enumerate(status.sequence_tuples):
			position += 1
			symbol_index = AgentKnowledge.symbols.index(symbol_meaning[0])
			meaning_index = meanings.index(symbol_meaning[1])
			indices = where(self.language_matrix[symbol_index,:] == position)
			self.language_matrix[symbol_index, indices] = 0
			self.language_matrix[symbol_index, meaning_index] = position

	def prepare_for_interaction(self, status, env):
		categories = Categories.get_categories()
		self.interaction_memory = dict(zip([cat[1] for cat in categories], [[] for i in range(len(categories))]))
		self.combinations = []


class CountingVariableWordOrderArchitecture(MatrixVariableWordOrderArchitecture):
	def __init__(self):
		super(CountingVariableWordOrderArchitecture, self).__init__()
		self.language_matrix = zeros((len(AgentKnowledge.symbols), len(meanings), Categories.get_number_of_categories()), dtype=float)

	def build_combinations(self, status):
		symbol_indices = get_symbol_indices(status.sequence)
		for position, symbol_index in enumerate(symbol_indices):
			meanings = self.language_matrix[symbol_index, :, position]
			if nonzero(meanings)[0].size > 0:
				sorted_meanings = meanings.argsort()[::-1]
				non = nonzero(meanings[sorted_meanings])
				sorted_meanings = sorted_meanings[:len(non[0])]
				for meaning in sorted_meanings:
					self.store_possible_meaning(meaning)
		self.known_categories = [(k,v) for k,v in self.interaction_memory.iteritems() if v]
		if self.known_categories:
			return deque(product(*(t[1] for t in self.known_categories)))

	def react_to_success(self, status, env):
		symbol_indices = get_symbol_indices(status.sequence)
		meaning_indices = get_goal_indices(status.goal)
		for position, symbol_index in enumerate(symbol_indices):
			row = self.language_matrix[symbol_index, :, position]
			mask = zeros(row.shape, dtype=bool)
			mask[meaning_indices] = True
			row[~mask] -= 1
			row[mask] += 1
			row[row < 0] = 0


class StochasticVariableWordOrderArchitecture(CountingVariableWordOrderArchitecture):

	def __init__(self, delta=0.5, memory=2):
		super(StochasticVariableWordOrderArchitecture, self).__init__()
		row_size = len(AgentKnowledge.symbols)
		col_size = len(meanings)
		pos_size = Categories.get_number_of_categories()
		init_weight = 1./(row_size*pos_size)
		self.language_matrix = full((row_size, col_size, pos_size), init_weight, dtype=float)
		self.pos_alpha = 1. + delta
		self.neg_alpha = 1. - delta
		self.memory_capacity = memory
		self.threshold = 0.001

	def build_combinations(self, status):
		used = {}
		symbol_indices = get_symbol_indices(status.sequence)
		for position, symbol_index in enumerate(symbol_indices):
			used[position] = []
			means = self.language_matrix[symbol_index, :, position]
			sorted_meanings = means.argsort()[::-1]
			for meaning in sorted_meanings:
				category = meanings_dict[meanings[meaning]]
				if meaning not in used[position]:
					self.interaction_memory[category].append((meaning, means[meaning]))
					used[position].append(meaning)
		for k,v in self.interaction_memory.iteritems():
			v.sort(key=lambda t: t[1], reverse=True)
			indices = list(set(value[0] for value in v))
			self.interaction_memory[k] = indices[:self.memory_capacity]
		self.known_categories = [(k,v) for k,v in self.interaction_memory.iteritems() if v]
		if self.known_categories:
			return deque(product(*(t[1] for t in self.known_categories)))

	def react_to_success(self, status, env):
		symbol_indices = get_symbol_indices(status.sequence, env.symbols)
		meaning_indices = get_goal_indices(status.goal)
		for position, symbol_index in enumerate(symbol_indices):
			row = self.language_matrix[symbol_index, :, position]
			mask = zeros(row.shape, dtype=bool)
			mask[meaning_indices] = True
			row[~mask] *= self.neg_alpha
			row[mask] *= self.pos_alpha
			row[row < self.threshold] = 0
			self.language_matrix[symbol_index, :, position] = row/row.sum()


class SingleActionHolisticArchitecture(HolisticCognitiveArchitecture):
	def __init__(self, n_rows, n_columns):
		super(SingleActionHolisticArchitecture, self).__init__(n_rows, n_columns)
		# self.language_matrix = full((n_rows, n_columns), 1. / n_rows, dtype=float)
		self.language_matrix = zeros((n_rows,n_columns), dtype=int)

	def react_to_success(self, status, env):
		goal_index_, symbol_index_ = self.get_goal_and_symbol_index(status, env)
		self.language_matrix[symbol_index_,goal_index_] += 1

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


def choose_and_remove(list_):
	chosen = choice(list_)
	list_.remove(chosen)
	return chosen

architectures = {'simple': CognitiveArchitecture,
                 'holistic':HolisticCognitiveArchitecture,
                 'multiple': MultipleValueCognitiveArchitecture,
                 'matrix_variable_word_order':MatrixVariableWordOrderArchitecture,
                 'matrix_3D_fixed_word_order':Matrix3DFixedWordOrderArchitecture,
                 'counting_variable_word_order':CountingVariableWordOrderArchitecture,
				'stochastic_variable_word_order':StochasticVariableWordOrderArchitecture
                 }
