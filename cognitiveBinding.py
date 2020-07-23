from igraph import Graph
from CognitiveNetwork import Node
import cognitive as cog # needs to be imported for the cognitive functions to register
from game_typing import COGNITIVE_TYPE_DICT
from environment import RandomInitializer, PROPERTIES
from GridEnvironment import GridEnvironment


#
# Initial variables are:
#   - the environment
#   - a chosen state
#   - the perception properties that we assume the agent is able to distinguish
#
def get_initial_variables():
	var = [prop for prop in PROPERTIES]
	env = GridEnvironment(8, 8)
	env.init_environment(RandomInitializer())
	index, state = env.select_non_empty_cell()
	var.append(state)
	var.append(env)
	return var

if __name__ == "__main__":

	g = Graph()

	for typ, val in COGNITIVE_TYPE_DICT.iteritems():
		operation = cog.CognitiveOperation(typ.name, typ, *val)
		g.add_vertex(name=operation.name, node=Node(operation))

	variables = get_initial_variables()

	# FIRST PASS: PERCOLATE POWERSET OF ALL VARIABLES THROUGH THE FUNCTIONS
	for var in variables:
		for v in g.vs:
			node = v['node']
			if node.take_args(var) and node.can_apply_operation():
				has_returned = node.do_operation()

	# SECOND PASS: ITERATE THROUGH ALL THE NODES AND BIND VARIABLES
	for v in g.vs:
		returned = v['node'].get_returned()
		for ret in returned:
			for node in g.vs:
				if not node == v:
					other_n = node['node']
					if other_n.take_args(ret.value) and other_n.can_apply_operation():
						if other_n.do_operation():
							ret.bind(other_n)

	# check returned values
	for v in g.vs:
		n = v['node']
		print(n.name, "returned: ", [ret.value for ret in n.get_returned()], "unhandled: ",[unh.value for unh in n.get_unhandled()])




