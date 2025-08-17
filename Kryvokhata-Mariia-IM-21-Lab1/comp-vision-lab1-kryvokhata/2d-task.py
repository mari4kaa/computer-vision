from graphics import *
import numpy as np
import time

def init_window(text, x_wind, y_wind):
    wind = GraphWin(text, x_wind, y_wind)
    wind.setBackground('white')
    return wind


def init_pentagon_points(radius, center_x, center_y):
    # Pentagon coords. We make a turn of 72 to draw a vertice

    # top center
    x1 = center_x
    y1 = center_y - radius

    # top right
    top_angle = 90 - 72
    x2 = center_x + (radius * np.cos(np.radians(top_angle)))
    y2 = center_y - (radius * np.sin(np.radians(top_angle)))

    # bottom right
    bottom_angle = 72 - top_angle
    x3 = center_x + (radius * np.cos(np.radians(bottom_angle)))
    y3 = center_y + (radius * np.sin(np.radians(bottom_angle)))

    # bottom left
    x4 = center_x - (radius * np.cos(np.radians(bottom_angle)))
    y4 = y3

    # top left
    x5 = center_x - (radius * np.cos(np.radians(top_angle)))
    y5 = y2

    points = np.array([[x1, x2, x3, x4, x5],
                       [y1, y2, y3, y4, y5],
                       [1, 1, 1, 1, 1]])
    return points

def draw_pentagon(wind, points, transformation_matrix=None):

    if transformation_matrix is not None:
        points = np.dot(transformation_matrix, points)

    # Drawing default pentagon
    pentagon = Polygon([Point(p[0], p[1]) for p in points.T])
    pentagon.setOutline("lightgreen")
    pentagon.setFill("lightblue")
    pentagon.draw(wind)

    return pentagon, points

def is_in_frame(wind, points):
    width = wind.getWidth()
    height = wind.getHeight()
    return all(0 <= point[0] <= width and 0 <= point[1] <= height for point in points.T)


def scaling_matrix_func(scale_coef):
    return np.array([
        [scale_coef, 0, 0],
        [0, scale_coef, 0],
        [0, 0, 1]
    ])

def translation_matrix_func(dx, dy):
    return np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])

def rotation_matrix_func(angle_rad):
    return np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])


def scale_pentagon(wind, points, center_x, center_y):
    scale_k = 1.05
    min_radius = 7
    scale_text = 'Zoom in:\n'

    while not wind.isClosed():

        wind.delete('all')

        # Translating pentagon to (0, 0)
        translate_to_origin = translation_matrix_func(-center_x, -center_y)

        # Scaling matrix
        scaling = scaling_matrix_func(scale_k)

        # Translation matrix to translate the pentagon back to its original position
        translate_back = translation_matrix_func(center_x, center_y)

        # Combining transformations: T_back * Scale * T_origin
        transform_matrix = np.dot(translate_back, np.dot(scaling, translate_to_origin))
        pentagon, points = draw_pentagon(wind, points, transform_matrix)
        print(scale_text, points)

        in_frame = is_in_frame(wind, points)
        current_radius = np.linalg.norm(np.array(points.T[0][:2]) - np.array([center_x, center_y]))

        if not in_frame:
            scale_k = 0.9
            scale_text = 'Zoom out\n'
        elif current_radius <= min_radius:
            scale_k = 1.05
            scale_text = 'Zoom in\n'

        time.sleep(0.1)
        wind.update()


def calculate_center(points):
    center_x = np.mean(points[0, :])
    center_y = np.mean(points[1, :])
    return center_x, center_y

def rotate_move_pentagon(wind, points):

    reverse = False

    while not wind.isClosed():

        wind.delete('all')
        pentagon_center_x, pentagon_center_y = calculate_center(points)

        if reverse:
            angle = -10
            dx = -5  # left
            dy = -4  # up
        else:
            angle = 10
            dx = 5  # right
            dy = 4  # down

        # Translating to (0,0)
        translate_to_origin = translation_matrix_func(-pentagon_center_x, -pentagon_center_y)

        #  Rotation
        rotation_matrix = rotation_matrix_func(angle)

        # Translating pentagon back to its original position
        translate_back = translation_matrix_func(pentagon_center_x, pentagon_center_y)

        # Moving pentagon
        move_matrix = translation_matrix_func(dx, dy)

        # Combining transformations: Move * T_back * Rotate * T_origin
        transform_matrix = np.dot(move_matrix, np.dot(translate_back, np.dot(rotation_matrix, translate_to_origin)))

        pentagon, points = draw_pentagon(wind, points, transform_matrix)
        print('Rotated and translated:\n', points)

        if not is_in_frame(wind, points):
            reverse = not reverse

        time.sleep(0.01)
        wind.update()


if __name__ == '__main__':

    xw = 600
    yw = 600

    default_radius = 50

    center_x = xw / 2
    center_y = yw / 2

    # Scaling
    wind = init_window('Scaling Pentagon', xw, yw)
    points = init_pentagon_points(default_radius, center_x, center_y)
    print('Initial coords:\n', points)
    scale_pentagon(wind, points, center_x, center_y)

    # Rotating + moving
    wind2 = init_window('Rotating and moving Pentagon', xw, yw)
    points2 = init_pentagon_points(default_radius, center_x, center_y)
    print('Initial coords:\n', points2)
    rotate_move_pentagon(wind2, points2)
