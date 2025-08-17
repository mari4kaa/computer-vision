from WheelComponents import WheelComponents
from OpenGL.GL import *
from OpenGL.GLUT import *
import time
from OpenGL.raw.GLU import gluLookAt, gluPerspective
from WheelDimensions import WheelDimensions

class WheelRenderer:
    def __init__(self, width=800, height=800):
        self.width = width
        self.height = height
        self.rotation = 0
        self.translation = -20  # Start position of the wheel (off-screen to the left)
        self.direction = 1  # Moving direction (1 = right, -1 = left)
        self.wheel = WheelComponents(WheelDimensions())
        self.display_list = None

    def init_gl(self):
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glDisable(GL_COLOR_MATERIAL)

        # Lighting setup
        glLightfv(GL_LIGHT0, GL_POSITION, (10, 10, 10, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (1, 1, 1, 1))

        # Create display list
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)
        self.wheel.create_wheel()
        glEndList()

    def reshape(self, width, height):
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(0, 0, 40, 0, 0, 0, 0, 1, 0)

        glPushMatrix()
        # Translate the wheel horizontally and rotate it
        glTranslatef(self.translation, 0, 0)  # Move along the X-axis
        glRotatef(self.rotation, 0, 1, 0)  # Rotate around the Y-axis
        glCallList(self.display_list)
        glPopMatrix()

        # Update rotation
        self.rotation += 1
        if self.rotation >= 360:
            self.rotation = 0

        # Update translation
        self.translation += 0.1 * self.direction  # Increment position
        if self.translation > 20:  # If it moves off-screen to the right
            self.direction = -1  # Reverse direction (move left)
        elif self.translation < -20:  # If it moves off-screen to the left
            self.direction = 1  # Reverse direction (move right)

        glutSwapBuffers()
        time.sleep(0.01)
        glutPostRedisplay()
