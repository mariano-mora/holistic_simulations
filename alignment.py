from environment import get_goal_indices, get_symbol_indices
from random import choice
from numpy import ones

class DefaultWordAlignment:
	@staticmethod
	def negative_align(architecture, status, env):
		symbol_indices = get_symbol_indices(status.sequence, env.symbols)
		# symbol_indices = architecture.interaction_memory.symbols
		goal_indices = get_goal_indices(status.goals_performed[-1])
		mask = ones(architecture.language_matrix.shape[1], dtype=bool)
		mask[symbol_indices] = 0
		for goal in goal_indices:
			colum = architecture.language_matrix[:, goal]
			colum[~mask] *= architecture.neg_beta
		for symbol in symbol_indices:
			row = architecture.language_matrix[symbol, :]
			# row[~mask] *= architecture.neg_beta
			row[row < architecture.minimal] = architecture.minimal
			row /= row.sum()

class RandomWordAlignment:
	@staticmethod
	def negative_align(architecture, status, env):
		symbol_indices = get_symbol_indices(status.sequence, env.symbols)
		goal_indices = get_goal_indices(status.goals_performed[-1])
		index = choice(symbol_indices)
		row = architecture.language_matrix[index, :]
		row[choice(goal_indices)] *= architecture.neg_beta
		row[row < architecture.minimal] = architecture.minimal
		row /= row.sum()

class MinimumWordAlignment:
	@staticmethod
	def negative_align(architecture, status, env):
		symbol_indices = get_symbol_indices(status.sequence, env.symbols)
		goal_indices = get_goal_indices(status.goals_performed[-1])
		test = architecture.language_matrix[symbol_indices, goal_indices]
		# for symbol in symbol_indices:
		min_ = test.argmin()
		row = architecture.language_matrix[min_,:]
		row[min_] *= architecture.neg_beta
		row[row < architecture.minimal] = architecture.minimal
		row /= row.sum()


alignment_strategies = {'default':DefaultWordAlignment, 'random':RandomWordAlignment, 'minimum':MinimumWordAlignment}