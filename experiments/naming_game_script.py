from game import *
from agent import *
from sequence import Sequence
import numpy as np
import time as tp
import argparse





class NamingGameExperiment:

	def __init__(self, n_agents=20, n_sequences=10, seq_length=3, n_interactions=None, stability=15, umpire=None):

		self.number_of_agents = n_agents
		self.number_of_sequences = n_sequences
		self.seq_length = seq_length
		self.number_of_interactions = n_interactions
		self.stability_check = stability
		self.agents = []
		self.symbols = ['red','orange','yellow','green','blue','indigo', 'violet','black']
		self.outcomes = ['run','hide','ignore']
		self.sequences = []
		self.umpire = umpire if umpire is not None else GameUmpire()



	def setup_game(self):
		'''
			Sets up a list of sequences. This represents the environment: the representations that will
			 be learned by the agents.
		'''
		# the environment
		for i in range(self.number_of_sequences):
			self.sequences.append(Sequence(symbols=self.get_random_symbols()))

		# the players
		for i in range(self.number_of_agents):
			self.agents.append(Agent())


	def play_game(self):

		if self.number_of_interactions is None:
			self.run_game_until_stable()
		else:
			self.iterate_game()


	def run_game_until_stable(self):
		is_stable = False
		while not is_stable:
			self.play_out_interaction()
			if self.test_stability():
				is_stable = True
			self.umpire.number_of_interactions = self.umpire.number_of_interactions + 1
			self.umpire.interaction_index = self.umpire.interaction_index + 1
	
	


	def test_stability(self):
		"""
			We check stability by counting the number of times the games has been successful in a row.
			If the game has been successful a fixed number of times in a row we can then compare the agents
			to check that they all have the same architecture
		"""
		if not self.umpire.success_counter > self.stability_check:
			return False
		else:
			for i in range(1, self.number_of_agents):
				if not self.agents[i].has_same_model(self.agents[0]):
					self.umpire.success_counter = 0
					return False
			return True



	def game_is_succesful(self):
		self.umpire.on_success()


	def game_is_failure(self):
		self.umpire.on_failure()

	def iterate_game(self):
		for iter in xrange(self.number_of_interactions):
			self.play_out_interaction()
			self.umpire.interaction_index = self.umpire.interaction_index + 1


	def play_out_interaction(self):
		agent_1, agent_2 = self.choose_players()
		#we need to make sure both players are not the same
		while agent_1 == agent_2:
			agent_1, agent_2 = self.choose_players()

		# each player takes either the role of the speaker or of the listener
		agent_1.set_role(Role.SPEAKER)
		agent_2.set_role(Role.LISTENER)

		sequence = self.sequences[np.random.randint(0,high=len(self.sequences), size=1)]
		if not agent_1.knows_sequence(sequence):
			outcome = self.select_outcome()
			agent_1.add(sequence, outcome)
			agent_2.add(sequence, outcome)
			self.game_is_failure()
		else:
			if agent_2.knows_sequence(sequence):
				if agent_1.get_outcome(sequence) != agent_2.get_outcome(sequence):
					agent_2.set_outcome(sequence, agent_1.get_outcome(sequence))
					self.game_is_failure()
				else: 
					self.game_is_succesful()
			else:
				agent_2.add(sequence, agent_1.get_outcome(sequence))
				self.game_is_failure()


	def choose_players(self):
		indices = np.random.randint(0, self.number_of_agents, 2)
		agent_1 = self.agents[indices[0]]
		agent_2 = self.agents[indices[1]]	
		return agent_1, agent_2


	def setup_random(self):
		return np.random.RandomState(np.int(np.floor(tp.time())))

	def create_sequences(self):
		for i in range(self.number_of_sequences):
			random_symbols = self.get_random_symbols()
			self.sequences.append(Sequence(symbols=random_symbols))


	def get_random_symbols(self):
		indices = np.random.randint(0, high=len(self.symbols), size=self.seq_length)
		random_symbols = []
		for index in indices:
			random_symbols.append(self.symbols[index])

		return random_symbols

	def select_outcome(self):
		return self.outcomes[np.random.randint(0,high=len(self.outcomes), size=1)]


	def carry_out_experiment(self):
		"""  Utility method to run experiment from the class  """
		self.setup_game()
		self.play_game()
		# return self.umpire




if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--agents', help='number of agents')
	parser.add_argument('--seqs', help='number of sequences')
	parser.add_argument('--seq_length', help='length of sequences')
	parser.add_argument('--runs', help='number of iterations')
	parser.add_argument('--stable', help='number of straight successes which indicate stability')
	args = parser.parse_args()
	number_of_agents = 20 if not args.agents else int(args.agents) 
	number_of_sequences = 10 if not args.seqs else int(args.seqs)
	seq_length = 3 if not args.seq_length else int(args.seq_length)
	number_of_interactions = None if not args.runs else int(args.runs)
	stability_check = 15 if not args.stable else int(args.stable)

	experiment = NamingGameExperiment(number_of_agents, number_of_sequences, \
		seq_length, number_of_interactions, stability_check)
	experiment.setup_game()
	experiment.play_game()
