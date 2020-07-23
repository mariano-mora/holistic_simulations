from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PyQt4.QtOpenGL import QGLWidget
from PyQt4 import QtGui, QtCore
from io import StringIO
import math
from scripts.environment import OBJECTS, COLORS

'''
	Drawing utility functions
'''


def set_color(color):
    r, g, b = COLORS[color]
    glColor3f(r, g, b)


def draw_selected(x,y):
    glColor3f(0.0,0.0,1.0)
    glLineWidth(1)
    offset = 10
    glBegin(GL_LINE_LOOP)
    glVertex2f( x+offset, y+offset )
    glVertex2f( x+offset, y-offset )
    glVertex2f( x-offset, y-offset )
    glVertex2f( x-offset, y+offset )
    glEnd()

def draw_box(color, x, y, side=20):
    halfSide = side / 2
    set_color(color)
    glBegin(GL_QUADS)
    glVertex3f(x + halfSide, y - halfSide, 0)
    glVertex3f(x + halfSide, y + halfSide, 0)
    glVertex3f(x - halfSide, y + halfSide, 0)
    glVertex3f(x - halfSide, y - halfSide, 0)
    glEnd()


def draw_ball(color, x, y, radius=10):
    radius = radius - 7
    smoothness = 10
    set_color(color)
    glBegin(GL_TRIANGLE_FAN)
    for i in range(0, smoothness):
        angle = i * math.pi * 2.0 / smoothness
        glVertex2f(x + radius * math.cos(angle), y + radius * math.sin(angle))
    glEnd()


def draw_thing(thing, x, y, size=None):
    if thing.is_selected:
        draw_selected(x,y)
    callback = FUNCTION_OBJECTS[thing.shape.value]
    color = thing.color.value
    if size:
        callback(color, x, y, size)
    else:
        callback(color, x, y)


FUNCTIONS = [(draw_box, 'box'), (draw_box, 'green'), (draw_ball, 'blue'), (draw_ball, 'black')]
FUNCTION_OBJECTS = {'square':draw_box, 'circle':draw_ball}
# FUNCTION_OBJECTS = dict(zip(OBJECTS, FUNCTIONS))


class Grid:
    def __init__(self, n_rows, n_columns, width, height):
        self.n_columns = n_columns
        self.n_rows = n_rows
        self.width = width
        self.height = height
        self.cell_width = math.ceil((width) / self.n_columns)
        self.cell_height = math.ceil((height) / self.n_rows)

    def resize(self, width, height):
        self.width = width
        self.height = height
        self.cell_width = math.ceil((width) / self.n_columns)
        self.cell_height = math.ceil((height) / self.n_rows)

    def draw_grid(self):
        glLineWidth(1.5)
        glColor3f(0.0, 0.0, 0.0)
        glBegin(GL_LINES)
        for i in range(1, self.n_rows):
            for j in range(1, self.n_columns):
                glVertex3f(i, j * self.cell_height, 0.)
                glVertex3f(i + self.width, j * self.cell_height, 0.)
                glVertex3f(i * self.cell_width, j, 0.)
                glVertex3f(i * self.cell_width, j + self.height, 0.)
        glEnd()

    def draw_cells(self, cells):
        for index in ((x, y) for x in range(self.n_rows) for y in range(self.n_columns)):
            state = cells[index]
            if not state.contained:
                continue
            xOffset = self.cell_width / max(len(state.contained[::2]), 3)
            yOffset = (self.cell_height / 3)
            size = self.cell_width / 5
            x = index[0] * self.cell_width + xOffset
            y = index[1] * self.cell_height + yOffset
            counter = 0
            for item in state.contained[::2]:  # even elements
                draw_thing(item, x + (counter * xOffset), y, size)
                counter = counter + 1
            counter = 0
            for item in state.contained[1::2]:  # odd elements
                draw_thing(item, x + (counter * xOffset), y + yOffset, size)
                counter = counter + 1


class GridWidget(QGLWidget):
    def __init__(self, parent=None, width=500, height=500):
        QGLWidget.__init__(self, parent)
        self.width = width
        self.height = height
        self.setMinimumSize(width, height)

    def set_grid(self, grid_env):
        self.grid_env = grid_env

    def initializeGL(self):
        glClearColor(1, 1, 1, 1)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.grid_env.grid.draw_grid()
        self.grid_env.grid.draw_cells(self.grid_env.cells)

    def resizeGL(self, width, height):
        self.width, self.height = width, height
        self.grid_env.grid.resize(width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0.0, width, height, 0.0, 0.0, 1.0)
        glViewport(0, 0, width, height)

    def animate(self):
        self.repaint()


class GridWithConsole(QtGui.QWidget):
    def __init__(self, parent=None, label=None, width=None, height=None):
        QGLWidget.__init__(self, parent)
        self.width = width
        self.height = height
        self.label = QtGui.QLabel(label) if label is not None else QtGui.QLabel()
        # self.label.setStyleSheet('color: red')
        self.grid = GridWidget(self, width, height)
        self.console = QDbgConsole(self)
        gridLayout = QtGui.QGridLayout()
        gridLayout.addWidget(self.label, 0, 0, 1, 1)
        # gridLayout.addWidget(QtGui.QLabel("Red Square"), 0, 1, 1, 1)
        gridLayout.addWidget(self.grid, 1, 0, 1, 1)
        gridLayout.addWidget(self.console, 2, 0, 1, 1)
        self.setLayout(gridLayout)


class QDbgConsole(QtGui.QTextEdit):
    '''
	A simple QTextEdit, with a few pre-set attributes and a file-like
	interface.
	'''
    # Feel free to adjust those
    WIDTH = 20
    HEIGHT = 20

    def __init__(self, parent=None, readOnly=True):
        super(QDbgConsole, self).__init__(parent)
        self._buffer = StringIO()
        self.setReadOnly(readOnly)



    def write(self, msg):
        '''Add msg to the console's output, on a new line.'''
        if type(msg) != QtCore.QString or type(msg) != unicode:
            msg = self.convert_message(msg)
        self.insertPlainText(msg)
        self.moveCursor(QtGui.QTextCursor.End)
        self._buffer.write(msg)

    def convert_message(self, msg):
        new_msg = unicode('')
        if type(msg) == list:
            for i in msg:
                new_msg += (unicode(i))
                new_msg += ' '
        else:
            new_msg = unicode(msg)
        return new_msg

    def __getattr__(self, attr):
        '''
        Fall back to the buffer object if an attribute can't be found.
        '''
        return getattr(self._buffer, attr)
