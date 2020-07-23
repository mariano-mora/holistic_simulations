from PyQt4.QtOpenGL import QGLWidget
from PyQt4 import QtGui, QtCore, QtOpenGL
from coordinate_game_Qt import *
from environment import RandomInitializer
from GridEnvironment import GridEnvironment
from grid import GridWithConsole


class GraphicalGame:
    def __init__(self, grid_console, callback_func):
        self.grid_console = grid_console
        self.f = callback_func

    def timer_callback(self):
        self.f(self.grid_console)

class CooperateStrategy:

    @staticmethod
    def callback(grid_console):
        print('cooperating')
        print(grid_console)


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
    initaliser = RandomInitializer()
    initaliser.init_environments(grid_env)
    # grid_env.init_environment(RandomInitializer())
    global timer
    timer = QtCore.QTimer()
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

    cooperative_game = GraphicalGame(gridWidget1, CooperateStrategy.callback)
    non_cooperative_game = GraphicalGame(gridWidget2, game_callback)
    timer.timeout.connect(cooperative_game.timer_callback)
    timer.timeout.connect(non_cooperative_game.timer_callback)
    ## Start the Qt event
    app.exec_()

