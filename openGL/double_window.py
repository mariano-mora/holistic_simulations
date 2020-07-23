from PyQt4 import QtGui, QtCore
from scripts.openGL.grid import GridWithConsole


class DoubleWindow(QtGui.QMainWindow):

	def __init__(self, grid1, grid2, label1="COOPERATING", label2="NON-COOPERATING",  interval=2.5):
		super(DoubleWindow, self).__init__()
		self.timer = QtCore.QTimer()
		self.grid_1 = grid1
		self.grid_2 = grid2
		self.gridWidget_1 = None
		self.gridWidget_2 = None
		self.interval = interval
		self.label1 = label1
		self.label2 = label2
		self.setup_window()

	def make_two_game_widget(self):
		gridWidget1 = GridWithConsole(label=self.label1, width=500, height=500)
		gridWidget1.grid.set_grid(self.grid_1)
		gridWidget2 = GridWithConsole(label=self.label2, width=500, height=500)
		gridWidget2.grid.set_grid(self.grid_2)
		container = QtGui.QFrame()
		gridLayout = QtGui.QGridLayout()
		gridLayout.setSpacing(5)
		gridLayout.addWidget(gridWidget1, 1, 1, 2, 1)
		gridLayout.addWidget(gridWidget2, 1, 2, 2, 1)
		container.setLayout(gridLayout)
		return container, gridWidget1, gridWidget2

	def make_buttons_container(self, buttons):
		button_container = QtGui.QFrame()
		vertical_layout = QtGui.QVBoxLayout()
		for button in buttons:
			vertical_layout.addWidget(button)
		button_container.setGeometry(QtCore.QRect(0, 100, 200, 100))
		button_container.setLayout(vertical_layout)
		vertical_layout.insertStretch(-1, 300)
		return button_container

	def make_buttons(self):
		btn1 = QtGui.QPushButton('start game')
		btn2 = QtGui.QPushButton('stop game')
		QtCore.QObject.connect(btn1, QtCore.SIGNAL("clicked()"), self.start_game)
		QtCore.QObject.connect(btn2, QtCore.SIGNAL("clicked()"), self.stop_game)
		return btn1, btn2

	@QtCore.pyqtSlot()
	def start_game(self):
		self.timer.start(self.interval * 1000)

	@QtCore.pyqtSlot()
	def stop_game(self):
		self.timer.stop()

	def setup_window(self):
		self.setGeometry(300, 300, 1000, 600)
		gamesContainer, self.gridWidget_1, self.gridWidget_2 = self.make_two_game_widget()
		buttonsContainer = self.make_buttons_container(self.make_buttons())
		mainWindowLayout = QtGui.QHBoxLayout()
		mainWindowLayout.addWidget(buttonsContainer)
		mainWidget = QtGui.QWidget()
		mainWindowLayout.addWidget(gamesContainer)
		mainWidget.setLayout(mainWindowLayout)
		self.setCentralWidget(mainWidget)


