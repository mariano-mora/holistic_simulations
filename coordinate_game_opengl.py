import random
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import click
from environment import RandomInitializer, Cell, GridEnvironment

'''
	A script to draw an environment in which agents interact

'''


WIDTH=340
HEIGHT=340



grid_env = 0

def create_environment(n_rows=8, n_columns=8):
	global grid_env 
	grid_env = GridEnvironment(n_rows, n_columns)
	initializer = RandomInitializer()
	grid_env.init_environment(initializer)




def eliminate_elements(data):
    eliminate = "RED_BOX" if data < 3 else "GREEN_BOX" if data < 6 else "BLACK_BALL" if data < 8 else "BLUE_BALL"
    for index,state in grid_env.states.items():
        if eliminate in state.contained:
            state.contained.remove(eliminate)



def move_object(data):
    index, state = select_non_empty_state()
    obj = filter(lambda t: t in state.contained, state.ordered_objects)
    if obj:
        toMove = obj[0]
        neigh = neighbors(index[0],index[1])
        min_value = grid_env.states[neigh[0]].distances[toMove]
        min_index = 0
        for i in neigh:
            value = grid_env.states[i].distances[toMove]
            if value < min_value:
                min_index = i
        if min_index == index or min_index == 0:
            print("not moving!!!")
            print(min_index, index)
        else:
            state.remove_object(toMove)
            grid_env.states[min_index].add_object(toMove)
            print(min_index, index)



def doAnimationStep(data):
    ''' callback funtion set in the main loop. Here we play the game, display results and call the function again '''    
    eliminate_elements(data)
    data = data + 1
    glutPostRedisplay()  # to call redisplay
    glutTimerFunc(int(1.5*1000), doAnimationStep,data) # call this function again in 2 seconds


def displayFun():
    glClear(GL_COLOR_BUFFER_BIT)
    grid_env.grid.draw_grid()
    grid_env.grid.draw_cells(grid_env.states)
    glFlush()

def initFun():
    glClearColor(1.0,1.0,1.0,1.0)
    glColor3f(0.0,0.0, 0.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # Start project to -1.0 otherwise the lines are not visible
    gluOrtho2D(-1.0,WIDTH,-1.0,HEIGHT)


@click.command()
@click.option('--n_rows','-r', default=8)
@click.option('--n_columns','-c', default=8)
def run_coordinate_game(n_rows, n_columns):
    create_environment(n_rows,n_columns)
    glutInit()
    glutInitWindowSize(WIDTH,HEIGHT)
    glutCreateWindow("States")
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
    glutDisplayFunc(displayFun)
    glutTimerFunc(2000, doAnimationStep, 0)
    initFun()
    glutMainLoop()



if __name__ == '__main__':
    run_coordinate_game()

