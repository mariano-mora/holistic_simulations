from abc import abstractmethod, ABCMeta
# import random
from numpy import random
from sequence import Sequence
from umpire import *
from action import move_element
from agent import Role, AgentKnowledge
from utils import logToConsole
from environment import REDUCED_GOALS, meanings
import pickle
# from game_utils import save_matrix_heat_map

__all__ = ['GameOutcome',
           'Game', 'SequenceGame', 'NamingGame',
           'GameUmpire', 'GameUmpireTrackingSuccesses',
           'GraphicalGame']


goals = [" ".join(goal.as_symbols()) for goal in REDUCED_GOALS[:-1]]
goals.append("DO NOTHING")


class GameOutcome:
	"""
		Enum for game outcomes
	"""
	failure = 0
	success = 1


class Game(object):
	"""
		A base class for all games
	"""
	__metaclass__ = ABCMeta

	def __init__(self, agents):
		self.agents = agents
		self.coordination_cost = 1

	def choose_agents(self):
		agents = random.sample(self.agents, 2)
		agents[0].set_role(Role.SPEAKER)
		agents[1].set_role(Role.LISTENER)
		agents[0].add_interaction()
		agents[1].add_interaction()

		return tuple(agents)

	def interaction_cost(self, speaker, listener):
		speaker.pay_cost(self.coordination_cost)
		listener.pay_cost(self.coordination_cost)


	# this needs to go with args
	@abstractmethod
	def play_game(self):
		return


class SequenceGame(Game):
	"""  A game that uses sequences of symbols. Added functionality: 
			- max length of sequence
	"""
	max_length = 5

	def __init__(self):
		pass

	@classmethod
	def set_max_length(cls, max_length):
		cls.max_length = max_length


class NamingGame(Game):
	"""
		First game: players agree on the outcome of a sequence
	"""

	def __init__(self):
		pass

	def play_game(self):
		sequence = self.create_sequence()

	def create_sequence(self):
		return Sequence()


"""
	GAMES
"""
class GraphicalGame(Game):
	"""
		A class that holds a representation of the environment, agents and a callback function which comes from a strategy
	"""

	def __init__(self, grid_widget, callback_func, agents, name):
		super(GraphicalGame, self).__init__(agents)
		self.grid_widget = grid_widget
		self.console = self.grid_widget.console
		self.grid = self.grid_widget.grid
		self.env = self.grid_widget.grid.grid_env
		self.f = callback_func
		self.status = InteractionStatus()
		self.umpire = CooperationGameUmpire()
		self.game_name = name

	def timer_callback(self):
		if self.status.new_interaction:
			self.umpire.attempts_per_interaction.append(self.status.number_of_attempts)
			if self.umpire.is_game_finished(self.env):
				self.store_umpire()
			self.status.reset_status()
			self.console.clear()
			self.umpire.interaction_index += 1
			self.status.speaker, self.status.listener = self.choose_agents()
		success = self.f(self)
		if success:
			self.umpire.successes.append((self.umpire.interaction_index,self.status.number_of_attempts))
		self.grid.animate()
		return True

	def play_game(self):
		return None

	def update_status(self):
		self.status.index, cell = self.env.select_non_empty_cell()
		if self.status.selected_element:
			self.status.selected_element.is_selected = False
		# self.status.selected_element = state.select_closest_element()
		self.status.selected_element = self.env.select_element(cell)
		if self.status.selected_element:
			self.status.selected_element.is_selected = True
			self.stage = InteractionStage.ELEMENT_SELECTED
		self.status.destination = self.env.select_destination(self.status.index, self.status.selected_element)
		if self.status.destination:
			contained = self.env.cells[self.status.destination].contained
			self.status.destination_elements = len(
				filter(lambda el: el.compare(self.status.selected_element), contained))
		else:
			self.status.destination_elements = None
		self.status.new_interaction = False

	def store_umpire(self):
		fileName = '/Users/mariano/developing/repos/research_sketches/game_results/' + self.game_name + '.pkl'
		with open(fileName, 'w') as f:
			pickle.dump(self.umpire, f)

	def log_to_console(self, text):
		logToConsole(self.console, text)



class CooperationGame(Game):
	def __init__(self, env, agents, name, umpire="coordination", store_umpire=False, store_directory=None, print_learner=False):
		super(CooperationGame, self).__init__(agents)
		self.env = env
		self.status = InteractionStatus()
		if type(umpire) == str and umpire in umpires:
			self.umpire = umpires[umpire](env)
		else:
			self.umpire = umpire
		self.game_name = name
		self.should_store_umpire = store_umpire
		self.store_directory = store_directory if store_directory else '/Users/mariano/developing/repos/research_sketches/game_results/'
		self.print_learner = print_learner
		self.chosen_agents = (self.agents[0], self.agents[1]) if print_learner else None

	def set_new_environment(self, env):
		self.env = env

	def play_out_interaction(self):
		if self.status.new_interaction:
			self.status.reset_status()
			if not self.at_new_interaction():
				return False
		success = self.perform_interaction()
		return False

	def consume_time_step(self):
		return self.play_out_interaction()

	def choose_agents(self):
		agents = random.choice(self.agents,2, replace=False)
		agents[0].set_role(Role.SPEAKER)
		agents[1].set_role(Role.LISTENER)
		agents[0].add_interaction()
		agents[1].add_interaction()
		return tuple(agents)

	def at_new_interaction(self):
		status = self.status
		self.update_status()
		status.listener.prepare_for_interaction(status, self.env)
		status.goal = self.env.produce_interaction_goal(status.selected_element, status.index, status.destination)
		sequence = status.speaker.get_sequence(status.goal, self.env)
		status.sequence = status.speaker.create_new_sequence(4) if not sequence else sequence
		status.speaker.add(status.goal, status.sequence)
		self.umpire.on_new_interaction(self.status)

	def at_element_selected(self):
		status = self.status
		status.number_of_attempts += 1
		status.stage = InteractionStage.PERFORM_ACTION
		return False

	def at_perform_action(self):
		status = self.status
		listener = status.listener
		env = self.env
		goal = listener.choose_goal_to_perform(status, env)
		if not goal:
			print("NO GOAL!!!")
			return False
		if goal.direction:
			dest = env.get_destination(goal.direction)
			status.performed_destination = (status.index[0] + dest[0], status.index[1] + dest[1])
			if status.performed_destination[0] == -1 or status.performed_destination[0] == 8 or status.performed_destination[1] == -1 or status.performed_destination[1] == 8:
				return False
		else:
			status.performed_destination = None
		element = None
		state = env.select_cell(status.index)
		if goal.element:
			elements = [el for el in state.contained if el.compare(goal.element)]
			element = elements[0] if elements else None
		try:
			dest = env.select_cell(status.performed_destination) if status.performed_destination else None
		except Exception as e:
			print(e)
		goal.f(element, state, dest)
		status.performed_element = element
		status.goals_performed.append(goal)
		status.stage = InteractionStage.ACTION_PERFORMED
		return False

	def at_action_performed(self):
		status = self.status
		if status.goal.is_equal(status.goals_performed[-1]):
			status.listener.react_to_success(status, self.env)
			status.speaker.react_to_success(status, self.env)
			status.new_interaction = True
			status.stage = InteractionStage.NEW_INTERACTION
			self.at_interaction_end()
			return True
		else:
			status.listener.react_to_failure(status, self.env)
			if status.performed_element and status.performed_destination:
				move_element(status.performed_element, self.env.select_cell(status.performed_destination), self.env.select_cell(status.index))
			if status.listener.should_repeat(status.number_of_attempts):
				status.stage = InteractionStage.ELEMENT_SELECTED
			else:
				status.stage = InteractionStage.NEW_INTERACTION
				status.new_interaction = True
				self.at_interaction_end()
			return False

	def at_interaction_end(self):
		pass

	def perform_interaction(self):
		status = self.status
		if status.stage == InteractionStage.NEW_INTERACTION:
			self.at_new_interaction()
			self.umpire.keep_track(status.goal, status.sequence, self.agents)
		if status.stage == InteractionStage.ELEMENT_SELECTED:
			return self.at_element_selected()
		elif status.stage == InteractionStage.PERFORM_ACTION:
			return self.at_perform_action()
		elif status.stage == InteractionStage.ACTION_PERFORMED:
			return self.at_action_performed()

	def play_game(self):
		pass

	def update_status(self):
		self.status.index, cell = self.env.select_non_empty_cell()
		if self.status.selected_element:
			self.status.selected_element.is_selected = False
		self.status.selected_element = self.env.select_element(cell)
		if self.status.selected_element:
			self.status.selected_element.is_selected = True
			self.status.stage = InteractionStage.ELEMENT_SELECTED
		self.status.destination = self.env.select_destination(self.status.index, self.status.selected_element)
		if self.status.destination:
			contained = self.env.cells[self.status.destination].contained
			self.status.destination_elements = len(
				filter(lambda el: el.compare(self.status.selected_element), contained))
		else:
			self.status.destination_elements = None
		self.status.new_interaction = False

	def on_success(self):
		self.umpire.on_success(self.status)

	def store_umpire(self):
		fileName = self.store_directory + self.game_name + '.pkl'
		with open(fileName, 'w') as f:
			pickle.dump(self.umpire, f)

	def reset_game(self, env):
		self.status.reset_status()
		self.status.stage = InteractionStage.NEW_INTERACTION
		self.status.new_interaction = True
		self.env = env

class CooperationGameWithWordOrder(CooperationGame):

	def __init__(self, env, agents, name):
		super(CooperationGameWithWordOrder, self).__init__(env, agents, name, umpire='variable_word_order')

	def at_new_interaction(self):
		status = self.status
		self.update_status()
		status.goal = self.env.produce_interaction_goal(status.selected_element, status.index, status.destination)
		status.sequence_tuples = status.speaker.get_sequence_and_meanings(status.goal)
		status.sequence = status.speaker.create_new_sequence(4) if not status.sequence_tuples else Sequence([value[0] for value in status.sequence_tuples])
		status.speaker.add_word_position(status.sequence_tuples)
		self.umpire.on_new_interaction(self.status)


class CooperationGameWithTeachersAndLearners(CooperationGame):

	def __init__(self, env, teachers, learners, name, umpire='coordination', print_learner=False):
		super(CooperationGameWithTeachersAndLearners, self).__init__(env, teachers+learners, name, umpire=umpire)
		self.teachers = teachers
		self.learners = learners
		self.print_learner = print_learner
		self.chosen_learner = self.learners[0] if print_learner else None

	def choose_agents(self):
		speaker = random.choice(self.teachers, 1)[0]
		listener = random.choice(self.learners, 1)[0]
		speaker.add_interaction()
		listener.add_interaction()
		# if self.print_learner:
			# if listener == self.chosen_learner:
				# num_tries = listener.strategy.max_tries
				# file_name = '/Users/mariano/Documents/PhD/writings/stage2/img/matrices/agent_{0}_matrix_{1}.pdf'.format(str(num_tries), str(listener.num_interactions))
				# save_matrix_heat_map(listener.architecture.language_matrix, goals, AgentKnowledge.holistic_symbols, file_name)
		return (speaker, listener)

	def at_new_interaction(self):
		status = self.status
		self.update_status()
		self.status.listener.prepare_for_interaction(status, self.env)
		status.goal = self.env.produce_interaction_goal(status.selected_element, status.index, status.destination)
		sequence = status.speaker.get_sequence_and_meanings(status.goal, self.env)
		status.sequence = status.speaker.create_new_sequence(4) if not sequence else sequence
		self.umpire.on_new_interaction(self.status)

	def at_action_performed(self):
		status = self.status
		if status.goal.is_equal(status.goals_performed[-1]):
			self.umpire.on_success(status)
			status.listener.react_to_success(status, self.env)
			status.new_interaction = True
			status.stage = InteractionStage.NEW_INTERACTION
			return True
		else:
			self.umpire.on_failure()
			status.listener.react_to_failure(status, self.env)
			if status.performed_element and status.performed_destination:
				move_element(status.performed_element, self.env.select_cell(status.performed_destination), self.env.select_cell(status.index))
			if status.listener.should_repeat(status.number_of_attempts):
				status.stage = InteractionStage.ELEMENT_SELECTED
			else:
				status.stage = InteractionStage.NEW_INTERACTION
				status.new_interaction = True
			return False

class CooperationHolisticGameWithTeachersAndLearners(CooperationGameWithTeachersAndLearners):
	def __init__(self,env, teachers, learners, name, umpire='coordination', print_learner=False):
		super(CooperationHolisticGameWithTeachersAndLearners, self).__init__(env, teachers, learners, name, umpire=umpire, print_learner=print_learner)

	def at_new_interaction(self):
		status = self.status
		self.update_status()
		status.listener.prepare_for_interaction(status, self.env)
		status.goal = self.env.produce_interaction_goal(status.selected_element, status.index, status.destination)
		status.sequence = status.speaker.get_sequence_and_meanings(status.goal, self.env)
		if not status.sequence:
			status.sequence = status.speaker.create_new_sequence(1)
		self.umpire.on_new_interaction(self.status)

class GraphicalGameWithStatus(GraphicalGame):

	def __init__(self, grid_widget, agents, name):
		self.agents = agents
		self.grid_widget = grid_widget
		self.console = self.grid_widget.console
		self.grid = self.grid_widget.grid
		self.env = self.grid_widget.grid.grid_env
		self.game = CooperationGameWithWordOrder(self.env, self.agents, name)
		self.game_name = name

	def timer_callback(self):
		success = self.game.play_out_interaction()
		self.log_status()
		self.grid.animate()
		return success

	def log_status(self):
		status = self.game.status
		if status.stage == InteractionStage.ELEMENT_SELECTED:
			self.log_to_console("Sequence: \"" + status.sequence.as_string() + "\" for goal: " + " ".join((tup[1] for tup in status.sequence_tuples)))


class InfiniteGame(CooperationGame):
	def __init__(self, env, agents, name, reward=100, action_cost=90, coordination_cost=0.05, umpire="infinite"):
		super(InfiniteGame, self).__init__(env, agents, name, umpire)
		self.reward = reward
		self.action_cost = action_cost
		self.coordination_cost = coordination_cost

	def at_new_interaction(self):
		status = self.status
		self.status.speaker, self.status.listener = self.choose_agents()
		if not status.listener.should_repeat(self.reward, self.coordination_cost, status):
			self.umpire.on_new_interaction(self.status)
			status.listener.react_to_failure(status, self.env)
			self.on_failure()
			self.umpire.on_failure(status)
			status.stage = InteractionStage.NEW_INTERACTION
			self.at_interaction_end()
			status.new_interaction = True
			return False
		self.status.listener.prepare_for_interaction(status, self.env)
		self.status.selected_element = self.env.select_element()
		self.status.destination = self.env.select_destination()
		self.status.stage = InteractionStage.ELEMENT_SELECTED
		status.goal = self.env.produce_interaction_goal(status)
		sequence = status.speaker.get_sequence(status.goal, self.env)
		status.sequence = status.speaker.create_new_sequence(4) if not sequence else sequence
		status.speaker.add(status.goal, status.sequence)
		self.umpire.on_new_interaction(self.status)
		return True

	def at_perform_action(self):
		''' We don't need to perform the action, only to know if it's the right one'''
		status = self.status
		goal_ = status.listener.choose_goal_to_perform(status, self.env)
		if not goal_:
			raise Exception('NO GOAL!')
		status.goals_performed.append(goal_)
		status.stage = InteractionStage.ACTION_PERFORMED
		return False

	def at_action_performed(self):
		status = self.status
		if status.goal.is_equal(status.goals_performed[-1]):
			status.listener.react_to_success(status, self.env)
			# status.speaker.react_to_success(status, self.env)
			status.new_interaction = True
			status.stage = InteractionStage.NEW_INTERACTION
			self.on_success()
			self.at_interaction_end()
			return True
		else:
			status.listener.react_to_failure(status, self.env)
			if status.listener.should_repeat(self.reward, self.coordination_cost, status):
				status.stage = InteractionStage.ELEMENT_SELECTED
			else:
				self.on_failure()
				self.umpire.on_failure(status)
				status.stage = InteractionStage.NEW_INTERACTION
				self.at_interaction_end()
				status.new_interaction = True
			return False

	def at_interaction_end(self):
		self.umpire.attempts_per_interaction.append(self.status.number_of_attempts)
		self.status.listener.record_fitness(self.umpire.interaction_index)
		self.status.speaker.record_fitness(self.umpire.interaction_index)

	def on_success(self):
		self.reward_success()
		self.umpire.on_success(self.status)

	def on_failure(self):
		coord_cost_ = self.status.number_of_attempts*self.coordination_cost
		amount = self.reward-(self.action_cost+coord_cost_)
		self.status.speaker.receive_reward(amount)
		self.status.listener.pay_cost(coord_cost_)
		self.status.speaker.record_interaction_cost(self.action_cost+coord_cost_)
		self.status.listener.record_interaction_cost(coord_cost_)

	def reward_success(self):
		coord_cost_ = self.status.number_of_attempts*self.coordination_cost
		total_cost = (self.action_cost/2)+coord_cost_
		amount = self.reward-(total_cost)
		self.status.listener.receive_reward(amount)
		self.status.speaker.receive_reward(amount)
		self.status.speaker.record_interaction_cost(total_cost)
		self.status.listener.record_interaction_cost(total_cost)



"""
	Game state recorder
"""
class InteractionStatus:
	"""
		A class that holds the elements needed during the interaction between agents.
		We need to store it since the interactions are repeated by cooperating agents
	"""

	def __init__(self):
		self.selected_element = None
		self.index = None
		self.destination = None
		self.destination_elements = None
		self.new_interaction = True
		self.goal = None
		self.goals_performed = []
		self.sequence_tuples = None
		self.performed_destination = None
		self.performed_element = None
		self.sequence = None
		self.speaker = None
		self.listener = None
		self.stage = InteractionStage.NEW_INTERACTION
		self.number_of_attempts = 0
		self.number_of_combinations = 0

	def reset_status(self):
		self.goal = None
		self.goals_performed = []
		self.sequence_tuples = None
		self.performed_destination = None
		self.performed_element = None
		self.number_of_attempts = 0
		self.number_of_combinations = 0
		self.new_interaction = False


class InteractionStage:
	NEW_INTERACTION = 0
	ELEMENT_SELECTED = 1
	PERFORM_ACTION = 2
	ACTION_PERFORMED = 3
	WRONG_ACTION = 4


class GameResult:
	def __init__(self, winner, time_steps):
		self.winner = winner
		self.time_steps = time_steps


umpires = {"coordination":CooperationGameUmpire,
			"fixed_word_order":FixedWordOrderGameUmpire,
			"variable_word_order":VariableWordOrderUmpire,
			"stochastic_word_order":StochasticVariableWordOrderUmpire,
			"infinite":InfiniteGameUmpire}