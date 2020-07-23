
"""
	Utility functions
"""

def add_node_to_graph(node, graph):
	node_copy = Node.copy_node(node)
	graph.add_vertex(node=node_copy, name=node_copy.name)
	return node_copy


class Variable:
	""" Wrapper around a value which lets us flag it as bound or unbound """
	id = 0

	def __init__(self, value, handled=False):
		self.value = value
		self.bound = False
		self.handled = handled
		self.nodes = []
		self.id = Variable.assign_id()

	def bind(self, node):
		self.bound = True
		self.nodes.append(node)
		print("added node============", self.nodes)

	@classmethod
	def assign_id(cls):
		cls.id += 1
		return cls.id

class Node:
	"""
		Object to encapsulate everything required by a cognitive operation and store at as a network element:
			- The operation itself
			- Parameters taken by the operation
			- Bind element: variable
		Details of this implementation can be found in the journal of October 11th: http://localhost:8000/journal_11_October_2015.html
	"""
	def __init__(self, operation):
		self.operation = operation
		self.name = operation.name
		self.vars = []
		self.returned = None

	def _is_variable_contained(self, variable):
		for var in self.vars:
			if id(variable) == id(var):
				return True
		return False

	def add_variable(self, value):
		variable = Variable(value)
		if not self._is_variable_contained(variable):
			self.vars.append(variable)

	## This could be problematic. We should probably make a copy of each operation for each graph, so that each graph can have its own
	## return values
	def add_returned(self, var):
		if not isinstance(var, Variable):
			var = Variable(var)
		self.returned = var

	def get_unbound_variables(self):
		return [var for var in self.vars if not var.bound]

	def get_bound_variables(self):
		return [var for var in self.returned if var.bound]

	def get_unhandled(self):
		return [var for var in self.vars if not var.handled]

	def get_returned(self):
		return self.returned

	""" Does the operation accept the argument? If not, then don't do the operation """
	def take_args(self, arg):
		if self.operation.discriminate_arg(arg):
			if arg not in self.vars:
				self.add_variable(arg)
				return True
		return False

	# this needs to change
	def can_apply_operation(self):
		return len(self.get_unhandled()) >= self.operation.n_args

	@staticmethod
	def flag_handled(var):
		var.handled = True
		return var

	@classmethod
	def copy_node(cls, node):
		node_copy = cls(node.operation)
		node_copy.returned = node.returned
		return node_copy

	def do_operation(self):
		unhandled = self.get_unhandled()
		ret = self.operation.do_operation(unhandled)
		if ret:
			map(Node.flag_handled, unhandled)
			self.add_returned(ret)
			return True
		return False
