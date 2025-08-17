from OpenGL.GLUT import *

from WheelRenderer import WheelRenderer


def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 800)
    glutCreateWindow(b"3D Car Wheel")

    renderer = WheelRenderer()
    renderer.init_gl()

    glutDisplayFunc(renderer.draw)
    glutReshapeFunc(renderer.reshape)
    glutMainLoop()


if __name__ == '__main__':
    main()