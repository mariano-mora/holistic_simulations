

class Action:
	""" An action carried out by an agent. Some agents are capable of carrying out certain actions. 
		Actions are defined by:
			- prerequisites: what the agent needs to carry them out
			- they have consequences
			- the consequences should be measurable in terms of the goals
			- this makes them depend entirely on the environment
	"""
	def __init__(self, f):
		self.f = f

	def perform(self, *args):
		self.f(*args)

	def result(self, *args):
		pass



def move_element(element, origin, dest):
	if not dest or not element:
		return
	origin.remove_object(element)
	try:
		dest.add_object(element)
	except Exception as e:
		print(e)

def do_nothing(element, origin, dest):
	pass

