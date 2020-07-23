from igraph import Graph
from CognitiveNetwork import Node, Variable
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



def unpack_list(list_item):
	return (item.value for item in list_item.value)


@register_op(GameTypes.ELEMENT)
def bind_to_category(args):
	pass

""" extract elements should be a cognitive operation """
def extract_elements(args):
	elements = []
	for arg in args:
		if type(arg.value) == GameTypes.LIST:
			elements = [item for item in arg.value if isinstance(item, GameTypes.ELEMENT)]
	return elements


def add_node_to_graph(node, graph):
	node_copy = Node.copy_node(node)
	graph.add_vertex(node=node_copy, name=node_copy.name)
	return node_copy


def bind_nodes(source_node, target_node, graph):
	source = graph.vs['node' == source_node]
	target = graph.vs['node' == target_node]
	graph.add_edge(source, target)

""" check if the operation accepts the variable, and do operation if it does """
def check_node(target_node, source_node, var, graph):
	if not var:
		return
	if target_node.take_args(var.value) and target_node.can_apply_operation():
		if target_node.do_operation():
			target_copy = add_node_to_graph(target_node, graph)
			bind_nodes(source_node, target_copy, graph)
			source_node.returned.bind(target_copy)

"""
	to check graph we see whether all the returned values of all the nodes are bound.
		if we have a returned value that is not bound we assume that is a leaf and move backwards
"""
def check_graph(graph):
	for v in graph.vs:
		if v['node'].returned:
			print(returned


if __name__ == "__main__":
	global nodes
	variables = []
	env = GridEnvironment(8, 8)
	env.init_environment(RandomInitializer())
	index, state = env.select_non_empty_cell()
	variables.append(env)
	variables.append(state)
	nodes = [Node(CognitiveOperation(typ.name, typ, *val)) for typ, val in COGNITIVE_TYPE_DICT.iteritems()]
	env_properties = [Variable(prop) for prop in PROPERTIES]
	graphs = []

	##
	#  we sort the nodes according to their cost here, that it, after new costs have been assigned,
	#   at the beginning of each iteration
	#   initialize the graphs starting with whichever node responds to the first variables
	##
	nodes = sorted(nodes, key=lambda node: node.cost)
	for var in variables:
		for node in nodes:
			if node.take_args(var) and node.can_apply_operation():
				if node.do_operation():
					g = Graph(directed=True)
					add_node_to_graph(node, g)
					graphs.append(g)

	##
	#  graphs are sorted according to the cost of their first node
	##
	if graphs:
		graphs = sorted(graphs, key=lambda gr: gr.vs[0]['node'].cost)
		for graph in graphs:
			for graph_node in graph.vs:
				current_node = graph_node['node']
				returned = current_node.returned
				for n in nodes:
					if n == current_node:
						continue
					check_node(n, current_node, returned, graph)
					##
					#  see http://localhost:8000/journal_30_November_2015.html
					##
					for prop in env_properties:
						check_node(n, current_node, prop, graph)
			check_graph(graph)

