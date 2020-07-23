import random
import numpy as np
from itertools import product
from action import move_element, do_nothing
import inspect

class Categories:
	ACTION = 0
	COLOR = 1
	SHAPE = 2
	DIRECTION = 3

	@classmethod
	def get_categories(cls):
		non_semantic = dir(type('dummy', (object,), {}))
		semantic = [cat for cat in inspect.getmembers(cls) if cat[0] not in non_semantic and not inspect.ismethod(cat[1])]
		return semantic

	@classmethod
	def get_number_of_categories(cls):
		return len(cls.get_categories())


class Property:
	def __init__(self, value, category):
		self.value = value
		self.category = category


class Element:
	def __init__(self, shape, color):
		self.shape = shape
		self.color = color
		self.is_selected = False

	def compare(self, other):
		return self.shape == other.shape and self.color == other.color

	def has_property(self, category, value):
		if category == Categories.SHAPE:
			return self.shape.value == value
		elif category == Categories.COLOR:
			return self.color.value == value

class Goal:
	def __init__(self, action, element, direction, f=move_element):
		self.action = action
		self.element = element
		self.direction = direction
		self.f = f

	def is_equal(self, goal):
		if not self.element or not goal.element:
			return self.action == goal.action
		return self.action == goal.action and self.element.compare(goal.element) and self.direction == goal.direction

	def as_symbols(self):
		return [self.action, None, None, None] if not self.element or not self.direction else [self.action,
																								self.element.color.value,
																								self.element.shape.value,
																								self.direction]

	def as_string(self):
		return " ".join(self.as_symbols())

	def compare_by_category(self, category, meaning):
		if category == Categories.ACTION:
			return self.action == meaning
		elif category == Categories.SHAPE and self.element:
			return self.element.shape.value == meaning
		elif category == Categories.COLOR and self.element:
			return self.element.color.value == meaning
		elif category == Categories.DIRECTION and self.direction:
			return self.direction == meaning
		else:
			return False

categories_dict = {Categories.ACTION: ['MOVE', 'DO_NOTHING'],
					Categories.COLOR: ['green', 'red'],
                    Categories.SHAPE: ['square', 'circle'],
					Categories.DIRECTION: ['UP', 'DOWN', 'LEFT', 'RIGHT']}

def get_meanings_dict():
	meanings_dict = {}
	for key, value in categories_dict.items():
		for v in value:
			meanings_dict[v] = key
	return meanings_dict


meanings_dict = get_meanings_dict()
meanings = sorted(meanings_dict.keys(), key=lambda meaning: meanings_dict[meaning])


def get_symbol_indices(sequence, symbols):
	return [symbols.index(symbol) for symbol in sequence]


def get_goal_indices(goal):
	return [meanings.index(symbol) for symbol in goal.as_symbols() if symbol in meanings]


def get_column_categories():
	cols_dict = {}
	for i, meaning in enumerate(meanings):
		cols_dict[i] = meanings_dict[meaning]
	return cols_dict

def get_categories_columns():
	cats = {}
	for k,v in categories_dict.items():
		cats[k] = sorted([meanings.index(value) for value in v])
	return cats

columns_categories = get_column_categories()
categories_columns = get_categories_columns()

SQUARE = Property('square', Categories.SHAPE)
CIRCLE = Property('circle', Categories.SHAPE)
RED = Property('red', Categories.COLOR)
GREEN = Property('green', Categories.COLOR)
BLUE = Property('blue', Categories.COLOR)
BLACK = Property('black', Categories.COLOR)

RED_SQUARE = Element(SQUARE, RED)
RED_CIRCLE = Element(CIRCLE, RED)
GREEN_SQUARE = Element(SQUARE, GREEN)
GREEN_CIRCLE = Element(CIRCLE, GREEN)
BLUE_CIRCLE = Element(CIRCLE, BLUE)
BLACK_CIRCLE = Element(CIRCLE, BLACK)

PROPERTIES = [SQUARE, CIRCLE, RED, GREEN, BLUE, BLACK]
REDUCED_PROPERTIES = [SQUARE, CIRCLE, RED, GREEN]

SHAPES_FOR_FACTORY = [prop for prop in PROPERTIES if prop.category == Categories.SHAPE]
REDUCED_SHAPES_FOR_FACTORY = [prop for prop in REDUCED_PROPERTIES if prop.category == Categories.SHAPE]
COLORS_FOR_FACTORY = {SQUARE: [RED, GREEN], CIRCLE: [BLUE, BLACK]}
REDUCED_COLORS_FOR_FACTORY = [prop for prop in REDUCED_PROPERTIES if prop.category == Categories.COLOR]
OBJECTS = [RED_SQUARE, GREEN_SQUARE, BLUE_CIRCLE, BLACK_CIRCLE]
REDUCED_OBJECTS = [RED_SQUARE, GREEN_SQUARE, RED_CIRCLE, GREEN_CIRCLE]
COLORS = {"green": (0.1, 1.0, 0.1), 'red': (1.0, 0.0, 0.0), 'blue': (0.0, 0.0, 1.0), 'black': (0.0, 0.0, 0.0)}
ACTIONS = ["MOVE", "DO_NOTHING"]

DESTINATIONS = {"RIGHT":(1, 0), "LEFT":(-1, 0), "UP":(0, -1), "DOWN":(0, 1)}
REDUCED_DESTINATIONS = ["RIGHT", "LEFT"]


def get_goals():
	prod = product([ACTIONS[0]], OBJECTS, DESTINATIONS.keys())
	goals = [Goal(*p) for p in prod]
	goals.append(Goal(ACTIONS[1], None, None, f=do_nothing))
	return goals


def get_reduced_goals():
	prod = product([ACTIONS[0]], REDUCED_OBJECTS, REDUCED_DESTINATIONS)
	goals = [Goal(*p) for p in prod]
	# goals.append(Goal(ACTIONS[1], None, None, f=do_nothing))
	return goals


GOALS = get_goals()
REDUCED_GOALS = get_reduced_goals()


def random_object_factory():
	shape = random.choice(SHAPES_FOR_FACTORY)
	color = random.choice(COLORS_FOR_FACTORY[shape])
	return shape, color


def reduced_object_factory():
	return random.choice(SHAPES_FOR_FACTORY), random.choice(REDUCED_COLORS_FOR_FACTORY)


def is_cell_in_grid(cell_index_x, cell_index_y, num_columns, num_rows):
	return 0 <= cell_index_x < num_rows and 0 <= cell_index_y < num_columns


class RandomInitializer:
	"""
		An initializer to assign randomly objects to states.
			- p can be used to reduce the number of objects (and thus the complexity of the environment)
			- also, a distribution can be used to sample from
	"""

	def __init__(self, distribution=None, num_objects=None, max_elements_cell=4, max_objects=None, reduced=False, corners=False):
		if distribution:
			self.distribution = distribution
		self.num_objects = num_objects
		self.max_objects = max_objects
		self.max_elements_cell = max_elements_cell
		self.reduced = reduced
		self.corners = corners

	def init_environments(self, env_list):
		env = env_list[0]

	def init_environments(self, *environments):
		if type(environments[0]) == list:
			environments = environments[0]
		first_env = environments[0]
		number_of_cells = len(first_env.cells)
		max_number_of_objects = number_of_cells * self.max_elements_cell if not self.max_objects else self.max_objects
		num_of_objects = np.random.randint(max_number_of_objects) if not self.num_objects else self.num_objects
		assert (num_of_objects < len(first_env.cells.keys() * self.max_elements_cell))
		for i in range(num_of_objects):
			ix = random.choice(first_env.cells.keys())
			cell = first_env.select_cell(ix)
			while len(cell.contained) >= self.max_elements_cell:
				ix = random.choice(first_env.cells.keys())
				cell = first_env.select_cell(ix)
			shape, color = random_object_factory() if not self.reduced else reduced_object_factory()
			for env in environments:
				cell = env.select_cell(ix)
				cell.add_object(Element(shape, color))
		for env in environments:
			if self.reduced:
				env.goals = REDUCED_GOALS
			self.init_env_destination_cells(env, self.corners)
			env.init_cell_distances()

	def init_env_destination_cells(self, env, corners=False):
		objects = OBJECTS if not self.reduced else REDUCED_OBJECTS
		cells = random.sample(env.cells.keys(), 4) if not corners else env.corner_cells
		env.destination_cells = dict(zip(objects, cells))

'''
	The environment elements:
		- a grid of cells
		- a cells can contain objects
		- a matrix of indices assigned to cells (for look-up)
'''


class Cell:
	"""
		a state in a grid. It makes up a grid environment (represented by a matrix) and can contain objects
		We store the distance between this state and each object's destination state
	"""

	def __init__(self, index):
		self.index = index
		self.contained = []

	def add_object(self, thing):
		self.contained.append(thing)

	def remove_object(self, thing):
		self.contained.remove(thing)

	def store_distances(self, distances):
		"""
			stores a list of tuples with the object and the manhattan distance to its destination
		"""
		assert distances
		self.distances = dict(distances)
		self.ordered_objects = [k for k, v in distances if v]  # we only want the objects that are not already in the destination
		self.ordered_distances = [v for k, v in distances]

	def select_closest_element(self):
		for element in self.ordered_objects:
			for contained in self.contained:
				if contained.shape == element[0] and contained.color == element[1]:
					return contained
