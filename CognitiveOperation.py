

class CognitiveOperation:
	def __init__(self, name, f, *args):
		self.name = name
		self.f = f
		self.args = args
		self.n_args = len(args)
		self.cost = 1.0

	"""
		the idea is that each operation is going to select the variables it requires as they become available
		variables are wrapped in a Variable class, so we only want to find the type of the variable's value
	"""
	def select_args(self, args):
		return [variable for variable in args if self.discriminate_arg(variable.value)]

	def discriminate_arg(self, arg):
		return type(arg) in self.args or any(isinstance(arg, clazz) for clazz in self.args)

	def do_operation(self, vars):
		return self.f(vars)

