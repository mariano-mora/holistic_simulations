from PyQt4.QtOpenGL import QGLWidget
from PyQt4 import QtGui, QtCore, QtOpenGL
from coordinate_game_Qt import *
from environment import RandomInitializer
from GridEnvironment import GridEnvironment
from openGL.grid import GridWithConsole
from agent import AgentWithNetwork
from igraph import Graph
from coordinate_game_Qt import select_object, select_destination



def logToConsole(console, msg):
    if type(msg) == 'list':
        msg = " ".join(msg)
    console.write(msg)
    console.write("\n")

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
        sequence.append(destination)
        return sequence

    def perform_action(self, action, env, element, index, destination):
        if action!= "DO_NOTHING":
            self.move_element(env, element, index, destination)

    def decode_sequence(self, sequence, env, index):
        if sequence[0]=="DO_NOTHING":
            return sequence[0], None, None
        action = "MOVE"
        elem = self.select_object(sequence, env, index)
        return action, elem, sequence[3]

    def select_object(self, sequence, env, index):
        state = env.select_cell(index)
        element_color = [elem for elem in state.contained if elem.color.value == sequence[1]]
        element_shape = [elem for elem in state.contained if elem.shape.value == sequence[2]]
        print(element_color, element_shape
        return element_color[0]

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

    def timer_callback(self):
        self.f(self.grid_console, self.agent1, self.agent2)

class CooperateStrategy:

    @staticmethod
    def callback(grid_console, agent1, agent2):
        env = grid_console.grid.grid_env
        if start_new_interaction:
            global_index, global_element, global_destination = get_goal(env)
            global_state = env.select_cell(global_destination).contained
            global_state.append(global_element)
            global_sequence = agent1.produce_sequence_from_goal(global_index, global_element, global_destination)
        logToConsole(grid_console.console, global_sequence)
        action, element, destination = agent2.decode_sequence(global_sequence, env, global_index)
        agent2.perform_action(action, env, element, global_index, destination)
        if action == "DO_NOTHING":
            logToConsole(grid_console.console, "DOING NOTHING")
        else:
            logToConsole(grid_console.console, "Performing action {0} on element of shape {1} and color {2} to destination {3}"
                     .format(action, element.shape.value, element.color.value, destination))
        grid_console.grid.animate()
        # agent1.checkActionResult()





class NonCooperateStrategy:

    @staticmethod
    def callback(grid_console, agent1, agent2):
        env = grid_console.grid.grid_env
        index, element, destination = get_goal(env)
        sequence = agent1.produce_sequence_from_goal(index, element, destination)
        logToConsole(grid_console.console, sequence)



def get_goal(gridEnv):
    index, state = gridEnv.select_non_empty_cell()
    element = select_object(state)
    destination = select_destination(index, element, grid_env)
    return index, element, destination

def game_callback(grid_console):
    move_object(grid_console.grid.grid_env, grid_console.console)
    grid_console.grid.animate()


def make_two_game_widget(grid):
    grid_widget_1 = GridWithConsole(label="COOPERATING", width=500, height=500)
    grid_widget_1.grid.set_grid(grid)
    grid_widget_2 = GridWithConsole(label="NON-COOPERATING", width=500, height=500)
    grid_widget_2.grid.set_grid(grid)
    container = QtGui.QFrame()
    gridLayout = QtGui.QGridLayout()
    gridLayout.setSpacing(5)
    gridLayout.addWidget(grid_widget_1, 1, 1, 2, 1)
    gridLayout.addWidget(grid_widget_2, 1, 2, 2, 1)
    container.setLayout(gridLayout)
    return container, grid_widget_1, grid_widget_2

def make_buttons_container(buttons):
    button_container = QtGui.QFrame()
    vertical_layout = QtGui.QVBoxLayout()
    for button in buttons:
        vertical_layout.addWidget(button)
    button_container.setGeometry(QtCore.QRect(0,100, 200, 100))
    button_container.setLayout(vertical_layout)
    vertical_layout.insertStretch(-1, 300)
    return button_container


@QtCore.pyqtSlot()
def start_game():
    timer.start(2.5*1000)

@QtCore.pyqtSlot()
def stop_game():
    timer.stop()

def make_buttons():
    btn1 = QtGui.QPushButton('start game')
    btn2 = QtGui.QPushButton('stop game')
    QtCore.QObject.connect(btn1, QtCore.SIGNAL("clicked()"), start_game)
    QtCore.QObject.connect(btn2, QtCore.SIGNAL("clicked()"), stop_game)

    return btn1, btn2



if __name__ == "__main__":
    grid_env = GridEnvironment(8,8)
    initialiser = RandomInitializer()
    initialiser.init_environments(grid_env)
    global timer, start_new_interaction, global_index, global_destination, global_goal, global_element, global_sequence
    timer = QtCore.QTimer()
    start_new_interaction = True
    global_index = None
    global_destination = None
    global_goal = None
    global_element = None
    global_sequence = None
    global_state = None
    app = QtGui.QApplication([])
    window = QtGui.QMainWindow()
    window.setGeometry(300, 300, 1000, 600)
    gamesContainer, gridWidget1, gridWidget2 = make_two_game_widget(grid_env)
    buttonsContainer = make_buttons_container(make_buttons())
    mainWindowLayout = QtGui.QHBoxLayout()
    mainWindowLayout.addWidget(buttonsContainer)
    mainWidget = QtGui.QWidget()
    mainWindowLayout.addWidget(gamesContainer)
    mainWidget.setLayout(mainWindowLayout)
    window.setCentralWidget(mainWidget)

    window.show()
    # COOPERATIVE GAME
    cooperative_agent1 = AgentWithCognitiveNetwork(role=Role.SPEAKER)
    cooperative_agent2 = AgentWithCognitiveNetwork(role=Role.LISTENER)
    cooperative_game = GraphicalGame(gridWidget1, CooperateStrategy.callback, cooperative_agent1, cooperative_agent2)
    # NON-COOPERATIVE GAME
    non_cooperative_agent1 = AgentWithCognitiveNetwork(role=Role.SPEAKER)
    non_cooperative_agent2 = AgentWithCognitiveNetwork(role=Role.LISTENER)
    non_cooperative_game = GraphicalGame(gridWidget2, NonCooperateStrategy.callback, non_cooperative_agent1, non_cooperative_agent2)
    timer.timeout.connect(cooperative_game.timer_callback)
    timer.timeout.connect(non_cooperative_game.timer_callback)
    ## Start the Qt event
    app.exec_()

