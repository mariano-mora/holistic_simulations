from PyQt4 import QtGui, QtCore
from openGL.grid import GridWithConsole
from GridEnvironment import GridEnvironment, logToConsole
from environment import RandomInitializer
from agent import AgentWithNetwork, Role
from igraph import Graph
from coordinate_game_Qt import select_object, select_destination, move_object


class CooperateStrategy:
	@staticmethod
	def callback(grid_console, agent1, agent2):
		env = grid_console.grid.grid_env
		index, element, destination = get_goal(env)
		sequence = agent1.produce_sequence_from_goal(index, element, destination)
		logToConsole(grid_console.console, sequence)
		action, decoded_element, decoded_destination = agent2.decode_sequence(sequence, env, index)
		if decoded_element and decoded_destination:
			decoded_element.is_selected = True
			agent2.perform_action(action, env, decoded_element, index, decoded_destination)
		if action == "DO_NOTHING":
			logToConsole(grid_console.console, "DOING NOTHING")
		else:
			logToConsole(grid_console.console,
			             "Performing action {0} on element of shape {1} and color {2} from {3} to destination {4}"
			             .format(action, decoded_element.shape.value, decoded_element.color.value, index,
			                     decoded_destination))
		grid_console.grid.animate()
		return decoded_element
	# agent1.checkActionResult()


class NonCooperateStrategy:
	@staticmethod
	def callback(grid_console, agent1, agent2):
		env = grid_console.grid.grid_env
		index, element, destination = get_goal(env)
		sequence = agent1.produce_sequence_from_goal(index, element, destination)
		logToConsole(grid_console.console, sequence)
		action, decoded_element, decoded_destination = agent2.decode_sequence(sequence, env, index)
		if decoded_element and decoded_destination:
			decoded_element.is_selected = True
			agent2.perform_action(action, env, decoded_element, index, decoded_destination)
		if action == "DO_NOTHING":
			logToConsole(grid_console.console, "DOING NOTHING")
		else:
			logToConsole(grid_console.console,
			             "Performing action {0} on element of shape {1} and color {2} from {3} to destination {4}"
			             .format(action, decoded_element.shape.value, decoded_element.color.value, index,
			                     decoded_destination))
		grid_console.grid.animate()
		return decoded_element



def get_goal(grid_env):
	index, state = grid_env.select_non_empty_cell()
	element = state.select_closest_element()
	destination = select_destination(index, element, grid_env, grid_env.cardinal_neighbors(index))
	return index, element, destination


class AgentWithCognitiveNetwork(AgentWithNetwork):
	def __init__(self, role=None):
		super(AgentWithCognitiveNetwork, self).__init__(role=role)
		self.cognitive_network = Graph()

	def produce_sequence_from_goal(self, index, element, destination):
		sequence = []
		if not destination:
			sequence.append("DO_NOTHING")
			return sequence
		sequence.append("MOVE")
		sequence.append(element.color.value)
		sequence.append(element.shape.value)
		sequence.append("from {0}".format(index))
		sequence.append("to")
		sequence.append(destination)
		return sequence

	def perform_action(self, action, env, element, index, destination):
		if action != "DO_NOTHING":
			self.move_element(env, element, index, destination)

	def decode_sequence(self, sequence, env, index):
		if sequence[0] == "DO_NOTHING":
			return sequence[0], None, None
		action = "MOVE"
		elem = self.select_object(sequence, env, index)
		return action, elem, sequence[-1]

	def select_object(self, sequence, env, index):
		state = env.select_cell(index)
		elements = [elem for elem in state.contained if
		            elem.color.value == sequence[1] and elem.shape.value == sequence[2]]
		return elements[0]

	def move_element(self, env, element, index, destination):
		origin_state = env.select_cell(index)
		dest_state = env.select_cell(destination)
		origin_state.contained.remove(element)
		dest_state.contained.append(element)


class GraphicalGame:
	"""
        A class that holds a representation of the environment, agents and a callback function which comes
        from a strategy
    """

	def __init__(self, grid_console, callback_func, agent1, agent2):
		self.grid_console = grid_console
		self.f = callback_func
		self.agent1 = agent1
		self.agent2 = agent2
		self.selected_element = None


	def timer_callback(self):
		if self.selected_element:
			self.selected_element.is_selected = False
		self.selected_element = self.f(self.grid_console, self.agent1, self.agent2)


def game_callback(grid_console):
	move_object(grid_console.grid.grid_env, grid_console.console)
	grid_console.grid.animate()


def make_two_game_widget(grid1, grid2):
	gridWidget1 = GridWithConsole(label="COOPERATING", width=500, height=500)
	gridWidget1.grid.set_grid(grid1)
	gridWidget2 = GridWithConsole(label="NON-COOPERATING", width=500, height=500)
	gridWidget2.grid.set_grid(grid2)
	container = QtGui.QFrame()
	gridLayout = QtGui.QGridLayout()
	gridLayout.setSpacing(5)
	gridLayout.addWidget(gridWidget1, 1, 1, 2, 1)
	gridLayout.addWidget(gridWidget2, 1, 2, 2, 1)
	container.setLayout(gridLayout)
	return container, gridWidget1, gridWidget2


def make_buttons_container(buttons):
	button_container = QtGui.QFrame()
	vertical_layout = QtGui.QVBoxLayout()
	for button in buttons:
		vertical_layout.addWidget(button)
	button_container.setGeometry(QtCore.QRect(0, 100, 200, 100))
	button_container.setLayout(vertical_layout)
	vertical_layout.insertStretch(-1, 300)
	return button_container


def make_buttons():
	btn1 = QtGui.QPushButton('start game')
	btn2 = QtGui.QPushButton('stop game')
	QtCore.QObject.connect(btn1, QtCore.SIGNAL("clicked()"), start_game)
	QtCore.QObject.connect(btn2, QtCore.SIGNAL("clicked()"), stop_game)
	return btn1, btn2


@QtCore.pyqtSlot()
def start_game():
	timer.start(interval * 1000)


@QtCore.pyqtSlot()
def stop_game():
	timer.stop()


def setup_window(gridEnv1, gridEnv2):
	window = QtGui.QMainWindow()
	window.setGeometry(300, 300, 1000, 600)
	gamesContainer, gridWidget_1, gridWidget_2 = make_two_game_widget(gridEnv1, gridEnv2)
	buttonsContainer = make_buttons_container(make_buttons())
	mainWindowLayout = QtGui.QHBoxLayout()
	mainWindowLayout.addWidget(buttonsContainer)
	mainWidget = QtGui.QWidget()
	mainWindowLayout.addWidget(gamesContainer)
	mainWidget.setLayout(mainWindowLayout)
	window.setCentralWidget(mainWidget)
	return window, gridWidget_1, gridWidget_2


if __name__ == "__main__":
	grid_env_1 = GridEnvironment(8, 8)
	grid_env_2 = GridEnvironment(8, 8)
	initialiser = RandomInitializer(max_objects=30)
	initialiser.init_environments(grid_env_1, grid_env_2)
	global timer, interval, start_new_interaction
	timer = QtCore.QTimer()
	interval = 2.5
	start_new_interaction = True
	app = QtGui.QApplication([])
	window, grid_widget_1, grid_widget_2 = setup_window(grid_env_1, grid_env_2)
	window.show()
	# COOPERATIVE GAME
	cooperative_agent1 = AgentWithCognitiveNetwork(role=Role.SPEAKER)
	cooperative_agent2 = AgentWithCognitiveNetwork(role=Role.LISTENER)
	cooperative_game = GraphicalGame(grid_widget_1, CooperateStrategy.callback, cooperative_agent1, cooperative_agent2)
	# NON-COOPERATIVE GAME
	non_cooperative_agent1 = AgentWithCognitiveNetwork(role=Role.SPEAKER)
	non_cooperative_agent2 = AgentWithCognitiveNetwork(role=Role.LISTENER)
	non_cooperative_game = GraphicalGame(grid_widget_2, NonCooperateStrategy.callback, non_cooperative_agent1,
	                                     non_cooperative_agent2)
	timer.timeout.connect(cooperative_game.timer_callback)
	timer.timeout.connect(non_cooperative_game.timer_callback)
	## Start the Qt event
	app.exec_()
