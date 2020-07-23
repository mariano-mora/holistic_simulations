from PyQt4 import QtGui
from scripts.openGL.double_window import DoubleWindow
from scripts.agent import AgentWithStrategy, AgentKnowledge, HolisticCognitiveArchitecture
from scripts.game import CooperationHolisticGameWithTeachersAndLearners, Role, HolisticGameUmpire
from scripts.strategies import StrategyTypes
from scripts.game_utils import get_permutation_matrix
from scripts.graphical import GraphicalEnvironment, GraphicalGameWrapper
from scripts.environment import RandomInitializer


num_agents = 20
num_games = 500
env_size = 8
num_objects = 120
max_tries = 12


if __name__=="__main__":
	grid_env_1 = GraphicalEnvironment(8, 8)
	grid_env_2 = GraphicalEnvironment(8, 8)
	initializer = RandomInitializer(num_objects=100, reduced=True, corners=True, max_elements_cell=2)
	initializer.init_environments([grid_env_1, grid_env_2])

	app = QtGui.QApplication([])
	window = DoubleWindow(grid_env_1, grid_env_2, interval=1.5, label1="TRYING 2 TIMES", label2="TRYING 4 TIMES")

	number_agents = 10
	learners1 = [AgentWithStrategy(StrategyTypes.MIXED, architecture=HolisticCognitiveArchitecture(len(AgentKnowledge.symbols), len(grid_env_1.goals)), role=Role.LISTENER, max_tries=2)
			for i in range(num_agents)]
	learners2 = [AgentWithStrategy(StrategyTypes.MIXED, architecture=HolisticCognitiveArchitecture(len(AgentKnowledge.symbols), len(grid_env_2.goals)), role=Role.LISTENER, max_tries=4)
			for i in range(num_agents)]
	teachers = [AgentWithStrategy(StrategyTypes.MIXED, architecture=HolisticCognitiveArchitecture(len(AgentKnowledge.symbols), len(grid_env_1.goals)), role=Role.LISTENER)
            for i in range(2)]
	for teacher in teachers:
		teacher.architecture.language_matrix = get_permutation_matrix(len(AgentKnowledge.symbols), len(grid_env_1.goals), float)
	name = "teachers_learners_game_holistic"
	game1 = CooperationHolisticGameWithTeachersAndLearners(grid_env_1, [teachers[0]],learners1, name, umpire=HolisticGameUmpire(grid_env_1, len(grid_env_1.goals)))
	game2 = CooperationHolisticGameWithTeachersAndLearners(grid_env_2, [teachers[1]], learners2, name, umpire=HolisticGameUmpire(grid_env_2, len(grid_env_2.goals)))
	graphical1 = GraphicalGameWrapper(game1, window.gridWidget_1)
	graphical2 = GraphicalGameWrapper(game2, window.gridWidget_2)

	window.timer.timeout.connect(graphical1.timer_callback)
	window.timer.timeout.connect(graphical2.timer_callback)

	window.show()
	app.exec_()

