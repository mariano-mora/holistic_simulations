__author__ = 'mariano'

from igraph import Graph
from CognitiveNetwork import Node
from cognitive import CognitiveOperation
from typing import COGNITIVE_TYPE_DICT
from environment import RandomInitializer, PROPERTIES
from GridEnvironment import GridEnvironment


def check_knowledge(graph):
	for property in PROPERTIES:
		for vertex in graph.vs:
			node = vertex['node']
			if node.take_args(property) and node.can_apply_operation():
				if node.do_operation():
					print("node {0} used property {1}".format(node, property)


def search_for_operations_from_returned(root_node, graph):
	for returned in root_node.returned:
		for n in nodes:
			if n == root_node:
				continue
			if n.take_args(returned.value) and n.can_apply_operation():
				if n.do_operation():
					graph.add_vertex(n)
					source = graph.vs['node'==root_node]
					target = graph.vs['node'==n]
					graph.add_edge(source, target)
					returned.bind(n)




if __name__ == "__main__":
	global nodes, properties
	vars = []
	env = GridEnvironment(8, 8)
	env.init_environment(RandomInitializer())
	index, state = env.select_non_empty_cell()
	vars.append(env)
	vars.append(state)
	## initialize cognitive operations
	nodes = [Node(CognitiveOperation(typ.name, typ, *val)) for typ, val in COGNITIVE_TYPE_DICT.iteritems()]
	graphs = []
	for var in vars:
		for node in nodes:
			if node.take_args(var) and node.can_apply_operation():
				if node.do_operation():
					g = Graph(directed=True)
					g.add_vertex(node=node, name=node.name)
					check_knowledge(g)
					graphs.append(g)
					search_for_operations_from_returned(node, g)


	for g in graphs:
		for v in g.vs:
			print(v['name']

	for node in nodes:
		print(node.name, ":==== ", node.get_unhandled()