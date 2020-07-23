from PyQt4 import QtGui
from agent import AgentWithStrategy
from GridEnvironment import GridEnvironment
from environment import RandomInitializer
from game import GraphicalGame, InteractionStage
from utils import logToConsole
from strategies import StrategyTypes
from openGL.double_window import DoubleWindow
from action import move_element



def perform_interaction(game, cooperates):
	env = game.grid.grid_env
	status = game.status
	speaker = status.speaker
	listener = status.listener
	if status.new_interaction:
		game.update_status()
		status.goal = env.produce_interaction_goal(status.selected_element, status.index, status.destination)
		sequence = speaker.get_sequence(status.goal)
		status.sequence = speaker.create_new_sequence(status.goal.as_symbols()) if not sequence else sequence
		speaker.add(status.goal, status.sequence)
	if status.stage == InteractionStage.ELEMENT_SELECTED:
		status.number_of_attempts += 1
		logToConsole(game.console, status.sequence.symbols)
		status.stage = InteractionStage.PERFORM_ACTION
		return False
	elif status.stage == InteractionStage.PERFORM_ACTION:
		knows = listener.knows_sequence(status.sequence)
		if knows:
			goals = listener.get_goal_from_sequence(status.sequence)
			goal = goals[0]
		else:
			goal = listener.choose_feasible_goal(env.select_cell(status.index), status.goal_performed)
		if goal.direction:
			dest = env.get_destination(goal.direction)
			status.performed_destination = (status.index[0] + dest[0], status.index[1] + dest[1])
		else:
			status.performed_destination = None
		element = None
		state = env.select_cell(status.index)
		if goal.element:
			elements = [el for el in state.contained if el.compare(goal.element)]
			element = elements[0] if elements else None
		dest  =  env.select_cell(status.performed_destination) if status.performed_destination else None
		goal.f(element, state, dest)
		status.performed_element = element
		if goal.element:
			msg = "performing action {0}, element {1} {2} to cell {3}".format(goal.action, goal.element.color.value, goal.element.shape.value, goal.direction)
		else:
			msg = "performing action {0}".format(goal.action)
		logToConsole(game.console, msg)
		status.goal_performed.append(goal)
		status.stage = InteractionStage.ACTION_PERFORMED
		return False
	elif status.stage == InteractionStage.ACTION_PERFORMED:
		if status.goal.is_equal(status.goal_performed[-1]):
			listener.react_to_success(status)
			speaker.react_to_success(status)
			status.new_interaction = True
			logToConsole(game.console, "Interaction was successful!")
			status.stage = InteractionStage.ELEMENT_SELECTED
			return True
		else:
			if status.performed_element and status.performed_destination:
				move_element(status.performed_element, env.select_cell(status.performed_destination), env.select_cell(status.index))
			logToConsole(game.console, "Interaction was not successful!")
			status.stage = InteractionStage.ELEMENT_SELECTED
			status.new_interaction = not cooperates
			return False


def coop_interaction_cb(game):
	return perform_interaction(game, True)

def non_coop_interaction_cb(game):
	return perform_interaction(game, False)


if __name__ == "__main__":
	grid_env_1 = GridEnvironment(8, 8)
	grid_env_2 = GridEnvironment(8, 8)
	initializer = RandomInitializer(num_objects=100)
	initializer.init_environments(grid_env_1, grid_env_2)

	app = QtGui.QApplication([])
	window = DoubleWindow(grid_env_1, grid_env_2, interval=2.5)

	number_agents = 10
	cooperative_agents = [AgentWithStrategy(StrategyTypes.EXHAUSTIVE) for i in range(number_agents)]
	cooperative_game = GraphicalGame(window.gridWidget_1, coop_interaction_cb, cooperative_agents, "cooperative_10_agents_1")
	non_cooperative_agents = [AgentWithStrategy(StrategyTypes.NON_EXHAUSTIVE) for i in range(number_agents)]
	non_cooperative_game = GraphicalGame(window.gridWidget_2, non_coop_interaction_cb, non_cooperative_agents, "non_cooperative_10_agents_1")

	window.timer.timeout.connect(cooperative_game.timer_callback)
	window.timer.timeout.connect(non_cooperative_game.timer_callback)

	window.show()
	app.exec_()
