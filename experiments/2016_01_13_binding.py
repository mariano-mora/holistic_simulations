from igraph import Graph
from CognitiveNetwork import Node, Variable, add_node_to_graph
from CognitiveOperation import CognitiveOperation
from typing import (register_op, GameTypes, get_type)
from typing import COGNITIVE_TYPE_DICT
from environment import RandomInitializer, PROPERTIES
from GridEnvironment import GridEnvironment


@register_op(GameTypes.STATE)
def get_context(args):
	if len(args) > 1:
		return
	return args[0].value.contained


@register_op(GameTypes.LIST, GameTypes.PROPERTY)
def filter_color(args):
	ret = []
	elements = extract_elements(args)
	if not elements:
		return None
	props = [item for item in args if isinstance(item.value, GameTypes.PROPERTY)]
	for prop in props:
		returned = [element for element in elements if element.color == prop.value]
		if returned:
			ret.append(returned)
	return ret

@register_op(GameTypes.LIST)
def unique_entity(args):
	pass


""" binds semantic entities to network operations """
def bind(args):
	pass


def extract_elements(args):
	elements = []
	for arg in args:
		if type(arg.value) == GameTypes.LIST:
			elements = [item for item in arg.value if isinstance(item, GameTypes.ELEMENT)]
	return elements


def check_node(target_node, source_node, var, graph):
	if not var:
		return
	if target_node.take_args(var.value) and target_node.can_apply_operation():
		if target_node.do_operation():
			target_copy = add_node_to_graph(target_node, graph)
			bind_nodes(source_node, target_copy, graph)
			source_node.returned.bind(target_copy)


def check_graph(graph):
	for v in graph.vs:
		node = v['node']
		if node.returned:
			print(node.returned

def bind_nodes(source_node, target_node, graph):
	source = graph.vs['node' == source_node]
	target = graph.vs['node' == target_node]
	graph.add_edge(source, target)





if __name__ == "__main__":
	##
	#   the idea of a global pool of variables, see http://localhost:8000/journal_13_January_2016.html
	##
	global variables
	global graphs
	variables = []
	graphs = []
	env = GridEnvironment(8, 8)
	RandomInitializer().init_environments(env)
	# initialiser.init_environments(env)
	# env.init_environment(RandomInitializer())
	index, state = env.select_non_empty_cell()
	operations = sorted([CognitiveOperation(typ.name, typ, *val) for typ, val in COGNITIVE_TYPE_DICT.iteritems()], key=lambda operation: operation.cost)
	env_properties = [Variable(prop) for prop in PROPERTIES]
	variables.append(Variable(state))

	for operation in operations:
		operation_vars = operation.select_args(variables)
		if operation_vars:
			print(variables, operation.name
			node = Node(operation)
			returned = operation.do_operation(operation_vars)
			if returned:
				graph = Graph(directed=True)
				returned_var = Variable(returned)
				node.add_returned(returned_var)
				variables.append(returned_var)
				graph.add_vertex(node=node, name=node.name)
				graphs.append(graph)

	assert(graphs)
	graphs = sorted(graphs, key=lambda gr: gr.vs[0]['node'].operation.cost)
	for graph in graphs:
		for graph_node in graph.vs:
			current_node = graph_node['node']
			for op in operations:
				if op == current_node.operation:
					continue
				operation_vars = operation.select_args(variables)
				if not operation_vars:
					continue
				# does this operation use the returned value? if so, bind both operations
				if current_node.returned in operation_vars:
					target_node = Node(op)
					bind_nodes(current_node, target_node, graph)
					ret = op.do_operation(operation_vars)
					print(ret


