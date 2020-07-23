from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
import numpy as np
from slicing import MatrixBlockSlicer



WIDTH=640
HEIGHT=640
MARGIN=30
COLUMNS=16
ROWS=16

class LED :
	''' a small circle of light  ''' 

	def __init__(self, x, y, radius) :	
		self.x = x
		self.y = y
		self.radius = radius

	def draw(self) :
		smoothness = 34
		glBegin(GL_TRIANGLE_FAN)
	  	glColor3f(0.2, 1.0, 0.5)
		for i in range(0, smoothness):
			angle = i * math.pi * 2.0 / smoothness
			glVertex2f(self.x + self.radius * math.cos(angle),self.y + self.radius * math.sin(angle))
		glEnd() 



class LED_Matrix :
	''' LED matrix only holds the leds. The decision to draw leds will be kept in a mixin class which will have the unique matrix
			and a matrix of 0's and 1's.
	'''

	def __init__(self, n_rows, n_columns) :
		self.n_columns = n_columns
		self.n_rows = n_rows
		leds_width = math.floor((WIDTH - MARGIN)/n_columns)
		leds_height = math.floor((HEIGHT - MARGIN)/n_rows)
		# rows are counted backwards so we can begin at the top of the window and enumerate downward
		self.leds = [[LED((j*leds_width+MARGIN), i*leds_height,10) for j in range(n_columns)] for i in range(n_rows,-1,-1)]

	def draw_led(self, n, m) :
		self.leds[n][m].draw()



creation_methods = {1 : 'random', 2 : 'blocks'}



class SymbolCreator :
	
	@classmethod
	def create_matrix(cls, n_rows, n_columns, method=1) :
		if method == 1 :
			return cls.create_random_matrix(n_rows, n_columns)
		elif method == 2 :
			return cls.create_block_matrix(n_rows, n_columns)

	@classmethod
	def create_random_matrix(cls, n_rows, n_columns) :
		mat = np.array(np.random.randint(0,high=2,size=n_columns*n_rows))
		return np.reshape(mat,(n_rows,n_columns))

	@classmethod
	def create_block_matrix(cls, n_rows, n_columns) :
		mat = np.reshape(np.zeros(n_columns*n_rows, dtype=int),(n_rows,n_columns))
		slicer = MatrixBlockSlicer(mat.shape[0])
		slicer.slice_matrix(mat)
		while mat.argmax() == 0 :  # make sure that at least one led has been turned on
			slicer.slice_matrix(mat)
		return mat




led_matrix = LED_Matrix(ROWS,COLUMNS)
symbol_mat = SymbolCreator.create_matrix(ROWS,COLUMNS, 2)
print(symbol_mat)



def displayFun():
    glClear(GL_COLOR_BUFFER_BIT)
    for i in range(ROWS) :
    	for j in range(COLUMNS) :
    		if symbol_mat[i,j] :
				led_matrix.draw_led(i,j)
    glFlush()


def initFun():	
    glClearColor(0.0,0.0,0.0,0.0)
    glColor3f(0.0,0.0, 0.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # Start project to -1.0 otherwise the lines are not visible
    gluOrtho2D(-1.0,WIDTH,-1.0,HEIGHT)


if __name__ == "__main__" :
 
    glutInit()
    glutInitWindowSize(WIDTH,HEIGHT)
    glutCreateWindow("Symbol")
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
    glutDisplayFunc(displayFun)
    initFun()
    glutMainLoop()