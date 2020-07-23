# from openGL.grid import Grid
from agent import AgentKnowledge
from environment import Cell, DESTINATIONS, REDUCED_DESTINATIONS, GOALS, Goal, REDUCED_OBJECTS, REDUCED_GOALS
from utils import manhattan_distance, calculate_cardinal_neighbors


import numpy as np
import random


class SingleCellEnvironment(object):
	def __init__(self):
		self.cell = Cell(0)
		self.cell.contained = REDUCED_OBJECTS
		self.goals = REDUCED_GOALS
		self.symbols = AgentKnowledge.reduced_holistic_symbols

	def select_element(self):
		return random.choice(self.cell.contained)

	def select_destination(self):
		return random.choice(REDUCED_DESTINATIONS)

	def produce_interaction_goal(self, status):
		goal = Goal("MOVE", status.selected_element, status.destination)
		goals_ = list(filter(lambda g: g.is_equal(goal), self.goals))
		assert len(goals_) == 1
		return goals_[0]


class SingleActionEnvironment(SingleCellEnvironment):

	def __init__(self, goal_index):
		self.cell = Cell(0)
		self.goal = REDUCED_GOALS[goal_index]
		self.goals = [self.goal]
		self.cell.contained = self.goal.element
		self.symbols = AgentKnowledge.reduced_holistic_symbols

	def select_element(self):
		return self.cell.contained

	def select_destination(self):
		return self.goal.direction

	def produce_interaction_goal(self, status):
		return self.goal


class GridEnvironment(object):
	"""
		An environment shaped like a grid
		We store:
			- a grid of dimensions N X M
			- a dictionary between indices and states
	"""
	def __init__(self, numberOfRows=None, numberOfColumns=None, width=300, height=300, cardinal=True, goals=GOALS, symbols=AgentKnowledge.symbols):
		assert numberOfRows and numberOfColumns, "please initialize number of rows and columns"
		self.num_rows = numberOfRows
		self.num_columns = numberOfColumns
		self.cells = {}
		for index in ((x,y) for x in range(numberOfColumns) for y in range(numberOfRows)):
			self.cells[index] = Cell(index)
		self.grid_matrix = np.zeros((numberOfRows,numberOfColumns), dtype=np.int16)
		self.corner_cells = [(0,0),(0, self.num_rows-1),(self.num_columns-1,0),(self.num_columns-1, self.num_rows-1)]
		self.cardinal = cardinal
		self.goals = goals
		self.symbols = symbols
		self.destination_cells = None

	def select_cell(self, index):
		return self.cells[index]

	def compute_distances(self, index):
		distances = {}
		for item, destination in self.destination_cells.items():
			distances[(item.shape, item.color)] = manhattan_distance(destination, index)
		return distances

	def init_cell_distances(self):
		for index,cell in self.cells.items():
			cell.store_distances(sorted(self.compute_distances(index).items(), key=lambda x: x[1]))

	def check_goal(self):
		""" check if all the items are in the destination cell """
		for index, state in self.cells.items():
			for item in state.contained:
				if index != self.destination_cells[item]:
					return False
		return True

	def select_non_empty_cell(self):
		non_empty = dict((i, state) for i, state in self.cells.items() if state.contained)
		index = random.choice(non_empty.keys())
		state = self.cells[index]
		return index, state

	def get_destination_cell(self, object):
		for element, destination in self.destination_cells.iteritems():
			if element.compare(object):
				return destination

	def select_element(self, cell):
		contained = list(cell.contained)
		is_selected = False
		selected = None
		while not is_selected:
			if not contained:
				return
			selected = random.choice(contained)
			destination = self.get_destination_cell(selected)
			distance = manhattan_distance(cell.index, destination)
			if distance != 0:
				is_selected = True
			else:
				contained.remove(selected)
		return selected

	def select_destination(self, index, object_to_move):
		"""
		:type object_to_move: element
		"""
		if not object_to_move:
			return None
		neighbors = self.cardinal_neighbors(index) if self.cardinal else self.neighbors(index)
		properties = (object_to_move.shape, object_to_move.color)
		min_value = self.cells[index].distances[properties]
		min_index = index
		for i in neighbors:
			value = self.cells[i].distances[properties]
			if value < min_value:
				min_index = i
				min_value = value
		if min_index == index:
			return None
		else:
			return min_index

	def get_destination(self, dest_name):
		return DESTINATIONS[dest_name]

	def neighbors(self, index):
		x, y = index
		X = self.num_columns -1
		Y = self.num_rows-1
		return [(x2, y2) for x2 in range(x-1, x+2) for y2 in range(y-1, y+2) if (-1 < x <= X and -1 < y <= Y and \
			(x != x2 or y != y2) and (0 <= x2 <= X) and (0 <= y2 <= Y))]

	def cardinal_neighbors(self, index):
		x, y = index
		return calculate_cardinal_neighbors(x, y, self.num_rows, self.num_columns)


	"""
		For now we let the goal be provided to the speaker by the environment. In the future the speaker should use its cognitive
		abilities to deduce the goal
	"""
	def produce_interaction_goal(self, element, origin, destination):
		if not element or not destination:
			return self.goals[-1]
		direction = (destination[0] - origin[0], destination[1] - origin[1])
		dest = [dest for dest, dir in DESTINATIONS.iteritems() if dir == direction]
		dest = dest[0]
		goal = Goal("MOVE", element, dest)
		goals_ = filter(lambda g: g.is_equal(goal), self.goals)
		assert len(goals_) == 1
		return goals_[0]

