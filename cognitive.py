from game_typing import (type_register, loose_type_register, register_op, GameTypes, get_type)
from utils import calculate_cardinal_neighbors

"""
Cognitive operations used by agents. All functions are registered in
typing.type_dict when this module is loaded
"""


@register_op(GameTypes.STATE)
def get_context(args):
	if len(args) > 1:
		return
	return args[0].value.contained


@register_op(GameTypes.ENVIRONMENT, GameTypes.STATE)
def get_neighbors(args):
	env_gen = get_type(GameTypes.ENVIRONMENT, args)
	state_gen = get_type(GameTypes.STATE, args)
	env = next(env_gen, None)
	state = next(state_gen, None)
	if not env or not state:
		return
	return calculate_cardinal_neighbors(state.value.index[0], state.value.index[1], env.value.num_rows, env.value.num_columns)


@register_op(GameTypes.LIST, GameTypes.PROPERTY)
def filter_color(args):
	context = get_type(GameTypes.LIST, args)
	elements = get_type(GameTypes.ELEMENT, context)
	props = get_type(GameTypes.PROPERTY, args)
	prop = next(props, None)
	if not prop:
		return None
	return [element for element in elements if element.color == prop]


@register_op(GameTypes.ELEMENT, GameTypes.PROPERTY)
def match_element_color(args):
	elems = get_type(GameTypes.ELEMENT, args)
	props = get_type(GameTypes.PROPERTY, args)
	prop = next(props, None)
	elem = next(elems, None)
	if not prop or not elem:
		return None
	return elem.color == prop

"""
	TODO: This operation takes a list, so the elements have to be extracted from the list.
		This should be done by another cognitive operation which is then bound to this
"""
@register_op(GameTypes.LIST, GameTypes.PROPERTY)
def filter_shape(args):
	context = get_type(GameTypes.LIST, args)
	elements = get_type(GameTypes.ELEMENT, context)
	props = get_type(GameTypes.PROPERTY, args)
	prop = next(props, None)
	if not prop:
		return None
	return [element for element in elements if element.shape == prop]



@register_op(GameTypes.LIST)
def extract_properties(args):
	elem = get_type(GameTypes.ELEMENT, args)
	for element in elem:
		print(element)


@register_op(GameTypes.COGNITIVE_FUNCTION, GameTypes.LIST, GameTypes.PROPERTY)
def filter_property(args):
	f_gen = get_type(GameTypes.COGNITIVE_FUNCTION, args)
	f = next(f_gen, None)
	function_arguments = get_type(GameTypes.COGNITIVE_FUNCTION, args)
	f_args = [arg for arg in function_arguments]
	if not f or not f_args:
		return
	elements = f(function_arguments)
	return elements


@register_op(GameTypes.ACTION)
def select_action(args):
	act = get_type(GameTypes.ACTION, args)


@register_op(GameTypes.LIST, GameTypes.ENVIRONMENT)
def select_destination(args):
	pass




"""
	Semantic
"""

