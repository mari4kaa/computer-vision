from graphics import *
import random
import numpy as np
import time

def init_window(text, x_wind, y_wind):
    wind = GraphWin(text, x_wind, y_wind)
    wind.setBackground('white')
    return wind

def init_pyramid_points(base_size, height):
    half_base = base_size / 2
    return np.array([
        [0, 0, height, 1],  # Apex
        [half_base, 0, -half_base, 1],  # Base vertices
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


def draw_pyramid(projection_xy, color, wind):
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


def calculate_center(pyramid_centered):
    center_x = np.mean(pyramid_centered[:, 0])
    center_y = np.mean(pyramid_centered[:, 1])
    center_z = np.mean(pyramid_centered[:, 2])
    return center_x, center_y, center_z


def random_color():
    r = lambda: random.randint(0, 255)
    return '#{:02x}{:02x}{:02x}'.format(r(), r(), r())


def animate_pyramid(pyramid_centered, wind):
    angle = 0
    frame_count = 0
    visible = True

    color = random_color()
    center_x, center_y, center_z = calculate_center(pyramid_centered)

    while not wind.isClosed():

        wind.delete('all')

        if frame_count % 30 == 0:
            visible = not visible
            if visible:
                color = random_color()

        if visible:
            # Shifting pyramid to (0,0)
            pyramid_shifted_to_origin = shift_xyz(pyramid_centered, -center_x, -center_y, -center_z)

            # Rotating
            rotated_pyramid = rotate_y(pyramid_shifted_to_origin, angle)

            # Shifting back
            pyramid_shifted_back = shift_xyz(rotated_pyramid, center_x, center_y, center_z)
            print('Rotated and centered:\n', pyramid_shifted_back)

            # Projecting and drawing the pyramid
            projected_pyramid = project_xy(pyramid_shifted_back)
            draw_pyramid(projected_pyramid, color, wind)
            print('Rotated, centered and projected:\n', projected_pyramid)

        angle += 2
        frame_count += 1
        time.sleep(0.01)
        wind.update()


if __name__ == '__main__':
    xw = 600
    yw = 600

    center_x = xw / 2
    center_y = yw / 2

    base_size = 100
    height = 150

    wind = init_window('Projection of the pyramid', xw, yw)
    pyramid = init_pyramid_points(base_size, height)
    print('Initial pyramid:\n', pyramid)

    rotated_pyramid_x = rotate_x(pyramid, -70)
    rotated_pyramid_xy = rotate_y(rotated_pyramid_x, -70)
    print('Axonometric projection:\n', rotated_pyramid_xy)

    # shifting pyramid to center
    pyramid_centered = shift_xyz(rotated_pyramid_xy, center_x, center_y, 0)
    print('Centered:\n', pyramid_centered)
    projected_pyramid = project_xy(pyramid_centered)

    draw_pyramid(projected_pyramid, '#00FF00', wind)
    print('Projection on xy:\n', projected_pyramid)

    # Animation

    wind2 = init_window('Rotation of the pyramid around y axis', xw, yw)
    pyramid2 = init_pyramid_points(base_size, height)
    pyramid_centered2 = shift_xyz(pyramid2, center_x, center_y, 0)
    print('Initial pyramid centered:\n', pyramid)
    animate_pyramid(pyramid_centered2, wind2)
