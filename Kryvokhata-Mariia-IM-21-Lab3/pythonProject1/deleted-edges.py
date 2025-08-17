from graphics import *
import numpy as np
from enum import Enum

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

def draw_initial_pyramid(projection_xy, color, wind):
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

class PolygonPosition(Enum):
    SURROUNDING = 1   # Polygon completely contains the window
    INSIDE = 2        # Polygon is completely inside the window
    INTERSECTING = 3  # Polygon partially intersects the window
    OUTSIDE = 4       # Polygon is completely outside the window


class Window:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def split(self):
        x_mid = (self.x_min + self.x_max) / 2
        y_mid = (self.y_min + self.y_max) / 2

        # Creating four quadrants
        return [
            Window(self.x_min, self.y_min, x_mid, y_mid),  # Bottom-left
            Window(x_mid, self.y_min, self.x_max, y_mid),  # Bottom-right
            Window(self.x_min, y_mid, x_mid, self.y_max),  # Top-left
            Window(x_mid, y_mid, self.x_max, self.y_max)   # Top-right
        ]


class Face:
    def __init__(self, points, color, z_order):
        self.points = points
        self.color = color
        # Average z-coordinate for depth sorting
        self.z_order = z_order


def get_polygon_position(polygon_points, window):
    # Checking if all polygon points are inside the window
    all_inside = all(
        window.x_min <= x <= window.x_max and
        window.y_min <= y <= window.y_max
        for x, y in polygon_points
    )

    if all_inside:
        return PolygonPosition.INSIDE

    window_corners = [
        (window.x_min, window.y_min),  # Bottom-left
        (window.x_max, window.y_min),  # Bottom-right
        (window.x_min, window.y_max),  # Top-left
        (window.x_max, window.y_max)  # Top-right
    ]

    # Checking whether all window corners inside polygon (polygon surrounds the window)
    if all(point_in_polygon(corner, polygon_points) for corner in window_corners):
        return PolygonPosition.SURROUNDING

    # Checking whether polygon intersects with window
    if polygons_intersect(polygon_points, window_corners):
        return PolygonPosition.INTERSECTING

    return PolygonPosition.OUTSIDE


def point_in_polygon(point, polygon):
    """
    1. Casting a ray from the point to the right
    2. Counting number of times it intersects polygon edges
    3. If count is odd, point is inside; if even, point is outside
    """
    x, y = point
    n = len(polygon)
    is_inside = False

    j = n - 1
    for i in range(n):
        # Checking if ray cast from point intersects with polygon edge
        if (((polygon[i][1] > y) != (polygon[j][1] > y)) and
                (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /
                 (polygon[j][1] - polygon[i][1]) + polygon[i][0])):
            is_inside = not is_inside
        j = i

    return is_inside


def polygons_intersect(poly1, poly2):
    for i in range(len(poly1)):
        for j in range(len(poly2)):
            if lines_intersect(
                    poly1[i], poly1[(i + 1) % len(poly1)],
                    poly2[j], poly2[(j + 1) % len(poly2)]
            ):
                return True  # edges from poly1 and poly2 intersect
    return False  # edges from poly1 and poly2 do not intersect


def lines_intersect(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # Two lines intersect if points of one line are on opposite sides of the other line
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def draw_pixels(window, color, graphics_window, pixel_size=2):
    # Drawing filled pixels of specified size within the given window
    for x in range(int(window.x_min), int(window.x_max), pixel_size):
        for y in range(int(window.y_min), int(window.y_max), pixel_size):
            pixel = Rectangle(Point(x, y), Point(x + pixel_size, y + pixel_size))
            pixel.setFill(color)
            pixel.setOutline(color)
            pixel.draw(graphics_window)


def warnock_algorithm(window, faces, graphics_window, min_size=4):
    # Base case: window is at minimum size
    if (window.x_max - window.x_min <= min_size and
            window.y_max - window.y_min <= min_size):
        # Sorting faces by depth (front to back)
        sorted_faces = sorted(faces, key=lambda f: f.z_order, reverse=True)
        center_x = (window.x_min + window.x_max) / 2
        center_y = (window.y_min + window.y_max) / 2

        # Finding the closest to front face containing window center
        for face in sorted_faces:
            if point_in_polygon((center_x, center_y), face.points):
                draw_pixels(window, face.color, graphics_window, min_size)
                return

        # Background color if no face found
        draw_pixels(window, "white", graphics_window, min_size)
        return

    sorted_faces = sorted(faces, key=lambda f: f.z_order, reverse=True)

    surrounding_faces = []
    inside_faces = []
    intersecting_faces = []

    for face in sorted_faces:
        position = get_polygon_position(face.points, window)

        if position == PolygonPosition.SURROUNDING:
            surrounding_faces.append(face)
        elif position == PolygonPosition.INSIDE:
            inside_faces.append(face)
        elif position == PolygonPosition.INTERSECTING:
            intersecting_faces.append(face)

    if surrounding_faces:
        # Window is completely covered, but subdividing for better resolution
        subwindows = window.split()
        for subwindow in subwindows:
            warnock_algorithm(subwindow, sorted_faces, graphics_window, min_size)
    elif not (surrounding_faces or inside_faces or intersecting_faces):
        # Window is empty -> drawing background
        draw_pixels(window, "white", graphics_window, min_size)
    else:
        # Complex case -> subdividing window and recursively calling the function
        subwindows = window.split()
        for subwindow in subwindows:
            warnock_algorithm(subwindow, sorted_faces, graphics_window, min_size)


if __name__ == '__main__':

    win_width = 600
    win_height = 600

    center_x = win_width / 2
    center_y = win_height / 2

    base_size = 200
    height = 250

    wind = init_window('Pyramid with removed sides', win_width, win_height)

    pyramid = init_pyramid_points(base_size, height)
    rotated_pyramid_x = rotate_x(pyramid, -35)
    rotated_pyramid_xy = rotate_y(rotated_pyramid_x, 0)
    pyramid_centered = shift_xyz(rotated_pyramid_xy, center_x, center_y, 0)
    projected_pyramid = project_xy(pyramid_centered)

    draw_initial_pyramid(projected_pyramid, "#e61bb8", wind)

    faces = []
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF']

    for i in range(4):
        if i < 3:
            points = [
                (projected_pyramid[0][0], projected_pyramid[0][1]),
                (projected_pyramid[i + 1][0], projected_pyramid[i + 1][1]),
                (projected_pyramid[(i + 2) % 3 + 1][0], projected_pyramid[(i + 2) % 3 + 1][1])
            ]
        else:
            points = [
                (projected_pyramid[1][0], projected_pyramid[1][1]),
                (projected_pyramid[2][0], projected_pyramid[2][1]),
                (projected_pyramid[3][0], projected_pyramid[3][1])
            ]

        if i < 3:
            z_order = (pyramid_centered[0][2] +
                       pyramid_centered[i + 1][2] +
                       pyramid_centered[(i + 2) % 3 + 1][2]) / 3
        else:
            z_order = (pyramid_centered[1][2] +
                       pyramid_centered[2][2] +
                       pyramid_centered[3][2]) / 3

        faces.append(Face(points, colors[i], z_order))


    main_window = Window(0, 0, win_width, win_height)
    warnock_algorithm(main_window, faces, wind, min_size=10)  # Bigger min_size -> bigger pyramid "pixels"

    wind.getMouse()
    wind.close()
