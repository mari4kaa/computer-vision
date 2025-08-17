from graphics import *
import numpy as np
import time

def init_window(text, x_wind, y_wind):
    wind = GraphWin(text, x_wind, y_wind)
    wind.setBackground('white')
    return wind

def init_pyramid_points(base_size, height):
    half_base = base_size / 2
    return np.array([
        [0, 0, height, 1],  # apex
        [half_base, 0, -half_base, 1],  # base vertices
        [-half_base, 0, -half_base, 1],
        [0, base_size, -half_base, 1]
    ])

def project_xy(pyramid):
    projection_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    return pyramid.dot(projection_matrix.T)


def shift_xyz(pyramid, dx, dy, dz):
    shift_matrix = np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ])
    return pyramid.dot(shift_matrix.T)


def rotate_x(pyramid, angle_degree):
    angle_rad = np.radians(angle_degree)
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle_rad), np.sin(angle_rad), 0],
        [0, -np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 0, 1]
    ])
    return pyramid.dot(rotation_matrix.T)


def rotate_y(pyramid, angle_degree):
    angle_rad = np.radians(angle_degree)
    rotation_matrix = np.array([
        [np.cos(angle_rad), 0, -np.sin(angle_rad), 0],
        [0, 1, 0, 0],
        [np.sin(angle_rad), 0, np.cos(angle_rad), 0],
        [0, 0, 0, 1]
    ])
    return pyramid.dot(rotation_matrix.T)


def draw_initial_pyramid(projection_xy, color):
    x1, y1 = projection_xy[0, 0], projection_xy[0, 1]
    x2, y2 = projection_xy[1, 0], projection_xy[1, 1]
    x3, y3 = projection_xy[2, 0], projection_xy[2, 1]
    x4, y4 = projection_xy[3, 0], projection_xy[3, 1]

    sides = [
        Polygon(Point(x1, y1), Point(x2, y2), Point(x3, y3)),
        Polygon(Point(x1, y1), Point(x2, y2), Point(x4, y4)),
        Polygon(Point(x1, y1), Point(x3, y3), Point(x4, y4)),
        Polygon(Point(x2, y2), Point(x3, y3), Point(x4, y4))
    ]

    for side in sides:
        side.draw(wind)
        side.setOutline(color)


def interpolate_edges(p1, p2):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])

    # constructing least-squares line fitting
    n_points = abs(x2 - x1) + 1
    X = np.vstack([np.linspace(x1, x2, n_points), np.ones(n_points)]).T
    Y = np.linspace(y1, y2, n_points).reshape(-1, 1)

    # applying the least-squares method
    coeff, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    line_x = np.linspace(x1, x2, n_points)
    line_y = coeff[0] * line_x + coeff[1]

    # drawing interpolated points
    for (x, y) in zip(line_x.astype(int), line_y.astype(int)):
        point = Point(x, y)
        point.setFill("blue")
        point.draw(wind)
        time.sleep(0.005)  # animation delay


def render_pyramid(pyramid_2d):
    edges = [
        (pyramid_2d[0], pyramid_2d[1]),  # AB
        (pyramid_2d[1], pyramid_2d[2]),  # BC
        (pyramid_2d[2], pyramid_2d[0]),  # CA
        (pyramid_2d[0], pyramid_2d[3]),  # AD
        (pyramid_2d[1], pyramid_2d[3]),  # BD
        (pyramid_2d[2], pyramid_2d[3])  # CD
    ]
    for edge in edges:
        interpolate_edges(*edge)


if __name__ == '__main__':
    win_width = 600
    win_height = 600

    center_x = win_width / 2
    center_y = win_height / 2

    base_size = 100
    height = 150

    wind = init_window('Interpolation usage on the pyramid', win_width, win_height)
    pyramid = init_pyramid_points(base_size, height)

    rotated_pyramid_x = rotate_x(pyramid, -35)
    rotated_pyramid_xy = rotate_y(rotated_pyramid_x, -70)

    # shifting pyramid to center
    pyramid_centered = shift_xyz(rotated_pyramid_xy, center_x, center_y, 0)

    # projecting onto xy
    projected_pyramid = project_xy(pyramid_centered)

    # drawing initial pyramid
    draw_initial_pyramid(projected_pyramid, '#63c963')

    # drawing interpolated pyramid
    render_pyramid(projected_pyramid)

    wind.getMouse()
    wind.close()
