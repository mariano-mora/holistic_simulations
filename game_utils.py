""" Game utility functions """

from GridEnvironment import GridEnvironment
from environment import RandomInitializer, meanings, meanings_dict, GOALS, columns_categories
# from game import CooperationGame, GraphicalGameWithStatus
from agent import AgentKnowledge
import numpy as np
import pickle
# import matplotlib.pyplot as plt


def setup_environments(size, num_objects):
	grid_env_1 = GridEnvironment(size, size)
	grid_env_2 = GridEnvironment(size, size)
	RandomInitializer(num_objects=num_objects).init_environments(grid_env_1, grid_env_2)
	return grid_env_1, grid_env_2


def get_new_environment(size, num_objects):
	grid_env = GridEnvironment(size, size)
	RandomInitializer(num_objects=num_objects).init_environments(grid_env)
	return grid_env


def get_new_environments(n_environments, size, num_objects, goals=GOALS, symbols=AgentKnowledge.symbols):
	envs = [GridEnvironment(size, size, goals=goals, symbols=symbols) for i in range(n_environments)]
	RandomInitializer(num_objects=num_objects, reduced=True).init_environments(envs)
	return envs


def get_new_reduced_environment(size, num_objects):
	grid_env = GridEnvironment(size, size)
	RandomInitializer(num_objects=num_objects, reduced=True).init_environments(grid_env)
	return grid_env

# def setup_game_with_agents(size, num_objects, agents1, agents2):
# 	env_1, env_2 = setup_environments(size, num_objects)
# 	cooperative_game = CooperationGame(env_1, agents1, "cooperative_10_agents_1")
# 	non_cooperative_game = CooperationGame(env_2, agents2, "non_cooperative_10_agents_1")
# 	return cooperative_game, non_cooperative_game


# def setup_graphic_game_with_agents(size, num_objects, agents1, agents2):
# 	env_1, env_2 = setup_environments(size, num_objects)
# 	graphic_game1 = GraphicalGameWithStatus(env_1, agents1, "cooperative_10_agents_1")
# 	graphic_game2 = GraphicalGameWithStatus(env_2, agents2, "non_cooperative_10_agents_1")
# 	return graphic_game1, graphic_game2



def get_permutation_matrix(row_size, column_size, dtype=int):
	fixed_matrix = np.zeros((row_size, column_size), dtype=dtype)
	symbols = np.random.permutation(row_size)
	meanings = np.random.permutation(column_size)
	for i in range(fixed_matrix.shape[0]):
		fixed_matrix[symbols[i], meanings[i]] = dtype(1)
	return fixed_matrix

def get_non_zero_permutation_matrix(row_size, column_size, init_value=.00001):
	init_non_zero = 1. - init_value
	init_zero = (init_value)/(column_size - 1)
	perm_matrix = get_permutation_matrix(row_size, column_size, dtype=float)
	perm_matrix[perm_matrix != 1.] = init_zero
	perm_matrix[perm_matrix == 1.] = init_non_zero
#	assert(perm_matrix[0,:].sum() == 1.)
	return perm_matrix

def get_stochastic_permutation_matrix(row_size, column_size, pos_size):
	fixed_matrix =  np.zeros((row_size, column_size, pos_size), dtype=float)
	symbols_perm = np.random.permutation(row_size)
	meanings_perm = np.random.permutation(column_size)
	for i in range(fixed_matrix.shape[0]):
		meaning = meanings_perm[i]
		category = meanings_dict[meanings[meaning]]
		fixed_matrix[symbols_perm[i], meaning, category] = 1
	return fixed_matrix

def get_fixed_position_3D_permutation_matrix(row_size, column_size, position_size, init_value=.00001, dytpe=float):
	perm_matrix = get_permutation_matrix(row_size, column_size)
	d3_perm = np.zeros((row_size, column_size, position_size))
	init_non_zero = 1. - init_value
	init_zero = init_value/((column_size*position_size)-1)
	for i in range(row_size):
		row = perm_matrix[i, :]
		column = np.nonzero(row)[0][0]
		position = columns_categories[column]
		d3_perm[i, column, position] = init_non_zero
		d3_perm[d3_perm==0.] = init_zero
	return d3_perm

def store_files(umpires, agents, winners, file_name, directory, learners=False):
	with open("{0}agents/{1}".format(directory, file_name), 'w') as f:
		pickle.dump(agents, f)
	with open("{0}umpires/{1}".format(directory, file_name), 'w') as g:
		pickle.dump(umpires, g)
	with open("{0}winners/{1}".format(directory, file_name), 'w') as h:
		pickle.dump(winners, h)

def store_teachers(teachers, file_name, directory):
	with open("{0}agents/{1}".format(directory, file_name), 'w') as f:
		pickle.dump(teachers, f)


def get_distance(umpires, distance):
	return [[d[0] for d in umpire.distances[distance]] for umpire in umpires]

def get_std(umpires, distance):
	return [[d[1] for d in umpire.distances[distance]] for umpire in umpires]


# def save_matrix_heat_map(matrix, x_labels, y_labels, name, save=False):
# 	fig, ax = plt.subplots()
# 	fig = plt.gcf()
# 	fig.set_size_inches(8, 11)
# 	heatmap = plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.coolwarm)
# 	cbar = fig.colorbar(heatmap, ticks=[0, 1], shrink=0.57)
# 	ax.set_frame_on(False)
#
# 	# put the major ticks at the middle of each cell
# 	ax.set_yticks(np.arange(len(y_labels)) + 0.1, minor=False)
# 	ax.set_xticks(np.arange(len(x_labels)) + 0.1, minor=False)
#
# 	# want a more natural, table-like display
# 	ax.invert_yaxis()
# 	ax.xaxis.tick_top()
# 	ax.set_xticklabels(x_labels, minor=False)
# 	ax.set_yticklabels(y_labels, minor=False)
# 	ax = plt.gca()
# 	for t in ax.xaxis.get_major_ticks():
# 		t.tick1On = False
# 		t.tick2On = False
# 	for t in ax.yaxis.get_major_ticks():
# 		t.tick1On = False
# 		t.tick2On = False
# 	plt.xticks(rotation=90)
# 	if save:
# 		fig.savefig(name)
# 	plt.close(fig)

def save_agents_matrix(agents, filename):
	with open(filename, 'wb') as f:
		pickle.dump([agent.architecture.language_matrix for agent in agents], f)