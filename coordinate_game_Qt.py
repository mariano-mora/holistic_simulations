from PyQt4 import QtGui, QtCore, QtOpenGL
from OpenGL.GL import *
import numpy.random as rdn
import numpy as np
import math
import sys
from time import sleep
from io import StringIO
from environment import RandomInitializer
from GridEnvironment import GridEnvironment
from openGL.grid import GridWidget, QDbgConsole
from agent import AgentWithNetwork, Role
# from cognitive import select_object, select_destination



WIDTH = 300
HEIGHT = 300

"""
Possible actions in this environment:
    - do nothing. If the agents find themselves at a state in which it is against the goals to perform an action, 
    then the speaker must be able to signal to do nothing
    - select object
    - pick up object
    - move object
"""


# function to run Qt application from ipython

def create_window(window):
	"""Create a Qt window in Python, or interactively in IPython with Qt GUI
	event loop integration.
	"""
	app_created = False
	app = QtCore.QCoreApplication.instance()
	if app is None:
		app = QtGui.QApplication(sys.argv)
		app_created = True
	app.references = set()
	app.references.add(window)
	window.show()
	if app_created:
		app.exec_()
	return window


def log_to_console(*args):
	for item in args:
		console.write(item)


class TestWindow(QtGui.QMainWindow):
	def __init__(self, grid, console, timer, f):
		super(TestWindow, self).__init__()

		self.GLwidget = GridWidget(self, width=800, height=600)
		self.GLwidget.set_grid(grid)
		self.console = console
		self.timer = timer
		gridLayout = QtGui.QGridLayout()
		gridLayout.setSpacing(5)
		gridLayout.addWidget(self.GLwidget, 1, 1, 2, 1)
		gridLayout.addWidget(self.console, 3, 1, 2, 1)
		self.setGeometry(
			300, 300, self.GLwidget.width, self.GLwidget.height)
		self.setWindowTitle('Coordination Game')
		self.centralWidget = QtGui.QWidget()
		self.centralWidget.setLayout(gridLayout)
		self.setCentralWidget(self.centralWidget)
		self.callbackFunction = f

	def set_timer(self, timer):
		"""  we need the timer so we can stop it when closing the widget """
		self.timer = timer

	def closeEvent(self, event):
		self.timer.stop()
		QtGui.QMainWindow.closeEvent(self, event)

	def callback(self):
		self.callbackFunction(self)


def move_object(environment, console):
	index, state = environment.select_non_empty_cell()
	toMove = select_object(state)
	console.write('selected object %s at index %s \n' % ((toMove.color.value + '_' + toMove.shape.value), (index),))
	destIndex = select_destination(index, toMove, environment)
	if not destIndex:
		return
	state.remove_object(toMove)
	environment.states[destIndex].add_object(toMove)
	# console.write(u'from index %s to index %s \n' % ((index),(destIndex),))


def game_callback(window):
	if window.GLwidget.grid_env.check_goal():
		window.console.write("GOALS ARE ACCOMPLISHED")
		window.timer.stop()
	else:
		move_object(window.GLwidget.grid_env, window.console)
		# game.play_game()
		window.GLwidget.animate()


def select_object(state):
	obj = state.select_closest_element()
	assert obj, "This state should not be empty"
	return obj


def select_destination(index, objectToMove, env, neighbors=None):
	if not objectToMove:
		return None
	if not neighbors:
		neighbors = env.neighbors(index)
	shape = objectToMove.shape
	color = objectToMove.color
	min_value = env.cells[neighbors[0]].distances[(shape, color)]
	min_index = 0
	for i in neighbors:
		value = env.cells[i].distances[(shape, color)]
		if value < min_value:
			min_index = i
	if min_index == index or min_index == 0:
		return None
	else:
		return min_index
