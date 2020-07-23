from agent import AgentWithStrategy
from strategies import StrategyTypes
from game_utils import setup_environments
from game import GraphicalGameWithStatus
from openGL.double_window import DoubleWindow
from PyQt4 import QtGui

if __name__ == "__main__":

	grid_env_1, grid_env_2 = setup_environments(8, 50)
	app = QtGui.QApplication([])
	name1 = "NON_EXHAUSTIVE_10_AGENTS_WORD_ORDER"
	name2 = "EXHAUSTIVE_10_AGENTS_WORD_ORDER"
	window = DoubleWindow(grid_env_1, grid_env_2, label1=name1, label2=name2, interval=0.5)

	agents1 = [AgentWithStrategy(StrategyTypes.NON_EXHAUSTIVE, architecture='matrix_word_order') for i in range(10)]
	agents2 = [AgentWithStrategy(StrategyTypes.EXHAUSTIVE, architecture='matrix_word_order') for i in range(10)]
	game_1 = GraphicalGameWithStatus(window.gridWidget_1, agents1, name1)
	game_2 = GraphicalGameWithStatus(window.gridWidget_2, agents2, name2)

	window.timer.timeout.connect(game_1.timer_callback)
	window.timer.timeout.connect(game_2.timer_callback)

	window.show()
	app.exec_()
