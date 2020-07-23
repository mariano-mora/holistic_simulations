from environment import Element, Property, Cell
from GridEnvironment import GridEnvironment
from action import Action
from CognitiveNetwork import Node
import types
from CognitiveOperation import CognitiveOperation


"""
	Game types enum
"""


class GameTypes:
	ENVIRONMENT = GridEnvironment
	#TUPLE = types.Tuple
	#LIST = types.Iterator
	# TUPLE = types.TupleType
	# LIST = types.ListType
	NODE = Node
	COGNITIVE_FUNCTION = CognitiveOperation
	ELEMENT = Element
	PROPERTY = Property
	ACTION = Action
	STATE = Cell


"""
	The global type dictionary
		@key: the function registered
		@value: the list of types it accepts
"""
COGNITIVE_TYPE_DICT = {}

"""
	The action type dict: every action registers what kind of arguments it accepts
"""
ACTION_TYPE_DICT = {}

"""
	register types for a function. Every cognitive function should register
	which types it accepts.
	Only functions that take all the types of arguments passes will be executed

	@type_register(GameTypes.ELEMENT, GameTypes.Property)
	example:
	def element_has_property(*args):
		return arg[0].shape.value == arg[1].value or arg[0].color.value == arg[1].value

	- this example assumes the arguments are in order, which is unsafe
	- we could do kargs or type checking inside the function, which is kind of against the point

	We store the wrapped function in the dictionary because it is the wrapped function that we want
	called. We check if the types of the arguments are correct: if they are, we call the wrapped and
	return whatever was returned by the cognitive function.
"""


def type_register(*type_args):
	def wrap(f):
		def wrapped_f(*args):
			types_ = COGNITIVE_TYPE_DICT[wrapped_f]
			if len(args) != len(types_):
				return
			for arg in args:
				if type(arg) not in types and not any(isinstance(arg, clazz) for clazz in types_):
					return
			return f(*args)

		contained = [k.name for k in COGNITIVE_TYPE_DICT.keys() if k.name == f.__name__]
		if not contained:  # register this wrapper in the type_dictionary
			wrapped_f.__name__ = wrapped_f.name = f.__name__
			COGNITIVE_TYPE_DICT[wrapped_f] = type_args
		return wrapped_f

	return wrap



"""
	a decorator for a function that specified the types accepted by the function but:
		- does not require the number of arguments to be the same as the parameters passed
		- stores the variables in a tuple of valid arguments and then passes them to the function
"""


def loose_type_register(*type_args):
	def wrap(f):
		def wrapped_f(*args):
			valid_args = ()
			types = COGNITIVE_TYPE_DICT[wrapped_f]
			for arg in args:
				if type(arg) in types or any(isinstance(arg, clazz) for clazz in types):
					valid_args = valid_args + (arg,)
			if not valid_args:
				return
			return f(*valid_args)

		contained = [k.name for k in COGNITIVE_TYPE_DICT.keys() if k.name == f.__name__]
		if not contained:  # register this wrapper in the type_dictionary
			wrapped_f.__name__ = wrapped_f.name = f.__name__
			COGNITIVE_TYPE_DICT[wrapped_f] = type_args
		return wrapped_f
	return wrap



"""
	This decorator does not carry out type checking, only registers the function and types,
	it is up to the node to decide when to call the function
"""


def register_op(*type_args):
	def wrap(f):
		def wrapped_f(args):
			return f(args)
		contained = [k.name for k in COGNITIVE_TYPE_DICT.keys() if k.name == f.__name__]
		if not contained:  # register this wrapper in the type_dictionary
			wrapped_f.__name__ = wrapped_f.name = f.__name__
			COGNITIVE_TYPE_DICT[wrapped_f] = type_args
		return wrapped_f
	return wrap


""" utility functions """


#   By the time this function is called the all args are wrapped as variables,
#   so we need to take their value
def get_type(game_type, args):
	return (item for item in args if isinstance(item, game_type))


# composes funtions to form a new function
def pipeline_func(data, fns):
	return reduce(lambda a, x: x(a), fns, data)