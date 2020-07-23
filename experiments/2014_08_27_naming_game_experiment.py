import time
import cPickle as pickle
import argparse

import numpy as np

from experiments.naming_game_script import *


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0))
        return ret
    return wrap



def iterations_against_number_of_agents(n_of_runs):
    iterations = np.zeros(n_of_runs, dtype=np.int32)
    for i in xrange(n_of_runs):
        experiment = NamingGameExperiment(n_agents=(i+2))
        umpire = experiment.carry_out_experiment()
        iterations[i] = umpire.number_of_iterations
    return iterations


@timing
def timed_iterations_against_number_of_agents(n_of_runs):
	return iterations_against_number_of_agents(n_of_runs)


def averaged_iterations(max_number_of_agents, n_to_average):
	"""
		run each game n_to_average number of times, store the result and return the average
	"""
	iterations = np.zeros(max_number_of_agents, dtype=np.int32)
	average_per_iteration = np.zeros(n_to_average, dtype=np.int32)
	for i in xrange(max_number_of_agents):
		average_per_iteration.fill(0)
		for j in xrange(n_to_average):
			experiment = NamingGameExperiment(n_agents=(i+2))
			umpire = experiment.carry_out_experiment()
			average_per_iteration[j] = umpire.number_of_iterations
		iterations[i] = average_per_iteration.mean()
	return iterations



@timing
def timed_averaged_iterations(max_number_of_agents, n_to_average=5):
	return averaged_iterations(max_number_of_agents, n_to_average)



#SEQUENCES

def variable_sequences(max_number_of_sequences):
	iterations = np.zeros(max_number_of_sequences, dtype=np.int32)
	for i in xrange(max_number_of_sequences):
		experiment = NamingGameExperiment(n_sequences=i+1)
		umpire = experiment.carry_out_experiment()
		iterations[i] = umpire.number_of_iterations
	return iterations


#TODO: return an array with the vectors to be averaged and use pandas
def averaged_variable_sequences(max_number_of_sequences, n_to_average=10):
	iterations = np.zeros(max_number_of_sequences, dtype=np.int32)
	average_per_iteration = np.zeros(n_to_average, dtype=np.int32)
	for i in xrange(max_number_of_sequences):
		average_per_iteration.fill(0)
		for j in xrange(n_to_average):
			experiment = NamingGameExperiment(n_sequences=i+1)
			umpire = experiment.carry_out_experiment()
			average_per_iteration[j] = umpire.number_of_iterations
		iterations[i] = average_per_iteration.mean()
	return iterations



#TIME-STEP APPROACH: here we are interested in the number of successes and failures at each interaction
# this would help us learn a distribution

def success_counter_experiment(n_runs, length_of_game):
	
	""" 
		experiment that runs n_runs number of games and keeps track of the number of successes at each interaction
		This is done by a subclass of GameUmpire that holds a vector of success counter at each interaction.
		:param n_runs: number of times to run each game.
		:param length_of_game: number of interactions in each game.

		:returns: the vector of successes. Each :math:`i` in the vector is an interaction having been repeated 'n_runs'
					number of times.
	"""

	umpire = GameUmpireTrackingSuccesses(length_of_game)
	
	for j in xrange(n_runs):
		experiment = NamingGameExperiment(n_interactions=length_of_game, umpire=umpire)
		experiment.carry_out_experiment()
		umpire.restart_umpire()


	return umpire.successes_per_interaction



def pickle_array(array, function_name):
	filename = "arrays/{0}_array_{1}.pkl".format(function_name, len(array))
	with open(filename, "w") as file:
		pickle.dump(array, file)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--runs', help='number of runs')
	parser.add_argument('--function', help='which function to run')
	parser.add_argument('--length', help='length of game')

	args = parser.parse_args()

	n_runs = 50 if not args.runs else int(args.runs)
	game_length = 1000 if not args.length else int(args.length)

	function = 'timed_iterations_against_number_of_agents' if not args.function else args.function

	function_map = {'timed_iterations_against_number_of_agents':"iterations",
					'timed_averaged_iterations':'averaged_iterations',
					'variable_sequences':'iterations_sequences',
					'averaged_variable_sequences':'averaged_iterations_sequences',
					'success_counter_experiment':'success_counter'}



	method = locals()[function]
	if function == 'success_counter_experiment':
		results_array = method(n_runs, game_length)
	else:
		results_array = method(n_runs)

	if results_array is not None:
		pickle_array(results_array, function)


