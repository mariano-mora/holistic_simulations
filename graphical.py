from GridEnvironment import GridEnvironment
from openGL.grid import Grid
from environment import GOALS
from utils import logToConsole
from game import InteractionStage

class GraphicalEnvironment(GridEnvironment):

	def __init__(self, numberOfRows=None, numberOfColumns=None, width=300, height=300, cardinal=True, goals=GOALS):
		super(GraphicalEnvironment, self).__init__(numberOfRows, numberOfColumns, width, height, cardinal, goals)
		self.grid = Grid(numberOfRows,numberOfColumns, width, height)



class GraphicalGameWrapper:

	def __init__(self, game, grid_widget):
		self.game = game
		self.grid_widget = grid_widget
		self.was_successful = False

	def timer_callback(self):
		success = self.game.play_out_interaction()
		if self.game.status.goals_performed:
			self.was_successful = self.game.status.goal == self.game.status.goals_performed[-1]
		mesg = self.get_message()
		if mesg:
			logToConsole(self.grid_widget.console, mesg)
		self.grid_widget.grid.animate()

	def get_message(self):
		status = self.game.status
		if status.stage == InteractionStage.NEW_INTERACTION:
			self.grid_widget.console.clear()
			msg = "INTERACTION WAS SUCCESSFUL" if self.was_successful else "INTERACTION WAS NOT SUCCESSFUL"
			self.was_successful = False
			return msg
		elif status.stage == InteractionStage.PERFORM_ACTION:
			return "GOAL IS: {0}, SYMBOL IS: {1}".format(status.goal.as_symbols(), status.sequence.as_string())
		elif status.stage == InteractionStage.ACTION_PERFORMED:
			return "ACTION PERFORMED: {0}".format(status.goals_performed[-1].as_symbols())
