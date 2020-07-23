import math
from itertools import chain, combinations
from os import path, listdir
import pickle
from numpy import array, argwhere, ones, linspace, stack, any, full

''' Math utility functions '''


def manhattan_distance(index, dest):
	""" index and destinations are tuples  """
	assert len(index) == len(dest)
	return sum([abs(pair[0]-pair[1]) for pair in zip(dest,index)])


def calculate_cardinal_neighbors(x, y, num_rows, num_columns):
	X = num_columns -1
	Y = num_rows-1
	return [(x,y1) for y1 in [y-1,y+1] if -1 < y <= Y and (0 <= y1 <= Y)] + \
	[(x1,y) for x1 in [x-1,x+1] if -1 < x <= X and (0 <= x1 <= X)]


def power_set(iterable, include_zero=False):
	#  "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
	assert type(iterable) == list
	start = 0 if include_zero else 1
	return chain.from_iterable(combinations(iterable, r) for r in range(start, len(iterable)+1))


def logToConsole(console, msg):
	if type(msg) == 'list':
		msg = " ".join(msg)
	console.write(msg)
	console.write("\n")


def get_list_of_agents(direc):
	with open(path.join(direc, 'agents.pkl'), 'rb') as f:
		agents = pickle.load(f)
	return agents


def get_umpire(direc):
	with open(path.join(direc, 'umpire.pkl'), 'rb') as f:
		umpire = pickle.load(f)
	return umpire


def get_umpires(directory):
	umpires = {}
	direc_list = sorted(listdir(directory))
	for direc_ in direc_list:
		full_path = path.join(directory, direc_)
		if not path.isdir(full_path):
			continue
		batch = sorted(listdir(full_path))
		umpires[direc_] = [get_umpire(path.join(full_path, dir_)) for dir_ in batch if path.isdir(path.join(full_path, dir_))]
	return umpires


def get_agents(directory):
	agents = {}
	direc_list = sorted(listdir(directory))
	for direc_ in direc_list:
		full_path = path.join(directory, direc_)
		if not path.isdir(full_path):
			continue
		batch = sorted(listdir(full_path))
		agents[direc_] = [get_list_of_agents(path.join(full_path, dir_)) for dir_ in batch if path.isdir(path.join(full_path, dir_))]
	return agents


def get_consistency_interaction(consistency_array):
	return argwhere(consistency_array == 1)[0][0] if any(consistency_array==1) else consistency_array.shape[0]


def get_consistencies(parameter_dict, n_pairings, n_rows):
	interactions = {}
	for param_value, umpires in parameter_dict.items():
		cons = [(array(umpire.consistency) / n_pairings) / n_rows for umpire in umpires]
		n_interactions = [get_consistency_interaction(inter) for inter in cons]
		interactions[param_value] = n_interactions
	return interactions


def normalise_consistency(cons_array, pairings_,n_rows_):
	return (array(cons_array)/pairings_)/n_rows_


def get_linear_values(consistency_array, cons_interval_):
	full_ = get_consistency_interaction(consistency_array)
	size = full_ * cons_interval_
	linear_interpol = ones(size+(cons_interval_),dtype=float)
	linear_interpol[cons_interval_::cons_interval_]=consistency_array[:full_]
	linear_interpol[-cons_interval_:] = consistency_array[-1]
	previous = 0
	for index, value in enumerate(consistency_array[:full_]):
		interp = linspace(previous, value, cons_interval_)
		linear_interpol[index*cons_interval_:(index+1)*cons_interval_] = interp
		previous = value
	return linear_interpol


def interpolate_array(obs_array, sample_interval):
	obs_inter = full(len(obs_array * sample_interval), obs_array[-1], dtype=float)
	obs_inter[::sample_interval] = obs_array
	previous = obs_array[0]
	index = 0
	for obs in obs_array[1:]:
		start_ = index * sample_interval
		finish_ = start_ + sample_interval
		linear_ = linspace(previous, obs, sample_interval + 1)
		obs_inter[start_ + 1:finish_] = linear_[1:-1]
		index += 1
		previous = obs
	return obs_inter


def padded(arr, size):
	tmp = ones(size, dtype=float)
	tmp[:arr.shape[0]] = arr
	return tmp


def get_stacked_padded(arrays_):
	max_val_ = max(arrays_, key=len).shape[0]
	padded_ = [padded(arr_, max_val_) for arr_ in arrays_]
	return stack(padded_)


def get_dir_hierarchy_tuples(action_parent):
	dir_list_ = []
	for action_dir_ in sorted(listdir(action_parent)):
		action_path_ = path.join(action_parent, action_dir_)
		if not path.isdir(action_path_):
			continue
		for coord_dir_ in sorted(listdir(action_path_)):
			coord_path_ = path.join(action_path_, coord_dir_)
			if not path.isdir(coord_path_):
				continue
			for rate_dir_ in sorted(listdir(coord_path_)):
				rate_path_ = path.join(coord_path_, rate_dir_)
				if not path.isdir(rate_path_):
					continue
				dir_list_.append((action_dir_, coord_dir_, rate_dir_))
	return dir_list_


def get_rate_dir(store_dir, dir_tuple):
	return '{0}{1}/{2}/{3}'.format(store_dir, dir_tuple[0], dir_tuple[1], dir_tuple[2])


def get_full_path(test_dir, rate_dir, target='umpire.pkl'):
	return path.join(path.join(test_dir, rate_dir), target)


def get_single_slice(rate_, parent_dir, target='umpire'):
	target_ = 'umpire.pkl' if target=='umpire' else 'agents.pkl'
	slice_ = {}
	dir_targets = []
	test_dir = get_rate_dir(parent_dir, rate_)
	print(test_dir)
	rate_dirs=sorted(listdir(test_dir))
	for rate_dir in rate_dirs:
		slice_fp  = get_full_path(test_dir, rate_dir, target=target_)
		with open(slice_fp, 'rb') as f:
			dir_targets.append(pickle.load(f))
	slice_[rate_] = dir_targets
	return slice_