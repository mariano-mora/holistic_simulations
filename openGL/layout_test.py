

if __name__ == "__main__":
    from PyQt4 import QtGui  # (the example applies equally well to PySide)
    import pyqtgraph as pg
    from double_game_window import GridWithConsole
    from coordinate_game_Qt import QDbgConsole, GridWidget
    from GridEnvironment import GridEnvironment
    from environment import RandomInitializer
    ## Always start by initializing Qt (only once per application)
    app = QtGui.QApplication([])

    ## Define a top-level widget to hold everything
    w = QtGui.QWidget()

    # w = GridWithConsole(width=500, height=500)
    ## Create some widgets to be placed inside
    btn = QtGui.QPushButton('press me')
    # text = QtGui.QLineEdit('enter text')
    listw = QtGui.QListWidget()
    text = QDbgConsole()
    grid_env = GridEnvironment(6,6)
    grid_env.init_environment(RandomInitializer())
    plot = GridWidget(width=500, height=400)
    plot.set_grid(grid_env)
    # w.grid.set_grid(grid_env)
    ## Create a grid layout to manage the widgets size and position
    layout = QtGui.QGridLayout()
    w.setLayout(layout)

    # Add widgets to the layout in their proper positions
    layout.addWidget(btn, 0, 0)   # button goes in upper-left
    layout.addWidget(text,  1, 0, 1, 1)   # text edit goes in middle-left
    # layout.addWidget(listw, 2, 2, 1)  # list widget goes in bottom-left
    layout.addWidget(plot, 0, 1, 1, 1)  # plot goes on right side, spanning 3 rows

    ## Display the widget as a new window
    w.show()

    ## Start the Qt event loop
    app.exec_()
