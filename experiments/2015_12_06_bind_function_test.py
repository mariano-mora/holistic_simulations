
from igraph import Graph
from CognitiveNetwork import Node
from CognitiveOperation import CognitiveOperation
from typing import (type_register, loose_type_register, register_op, GameTypes, get_type)
from typing import COGNITIVE_TYPE_DICT




@register_op(GameTypes.LIST, GameTypes.PROPERTY)
def filter_color(args):
	ret = []
	for arg in args:
		if type(arg.value)==GameTypes.LIST:
			elements = [item for item in arg.value if isinstance(item, GameTypes.ELEMENT)]
	if not elements:
		return None
	props = (item for item in args if isinstance(item.value, GameTypes.PROPERTY))
	for prop in props:
		returned = [element for element in elements if element.color == prop.value]
		if returned:
			ret.append(returned)
	return ret


@register_op(GameTypes.LIST, GameTypes.PROPERTY)
def select_category(args):
	pass

@register_op(GameTypes.NODE)
def select_function(args):
	nodes = filter(lambda item: isinstance(item.value, GameTypes.NODE), args)
	if len(nodes) == 1:
		node = nodes[0].value
		if node.take_args(args):
			node.do_operation()
	else:
		graphs = [build_graph(node) for node in nodes]
		for graph in graphs:
			for v in graph.vs:
				print(v['node'].name


def build_graph(node):
	g = Graph()
	g.add_vertex(node=node, name=node.name)
	return g

if __name__ == "__main__":
	nodes = [Node(CognitiveOperation(typ.name, typ, *val)) for typ, val in COGNITIVE_TYPE_DICT.iteritems()]
	for node in nodes:
		for n in nodes:
			if node == n:
				continue
			if n.take_args(node):
				n.do_operation()