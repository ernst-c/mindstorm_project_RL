from math import sqrt, pi
from shapely.geometry import Point
from math import cos, sin
import numpy as np

def sparse_reward2d(next_state, goal_state, observation_space, goal_range, polygons, wheelbase,wheel_radius):

    #goal_reward = -1 * int((abs(next_state[0] - goal_state[0]) > goal_range or
    #                        abs(next_state[1] - goal_state[1]) > goal_range
    #                        ))
    distance_to_goal = ((goal_state[0] - next_state[0])**2 + (goal_state[1] - next_state[1])**2)**0.5
    goal_reward_distance = -1 * distance_to_goal
    bounds_reward = -1 * int((abs(next_state[0] - observation_space.high[0]) < 0.05 or
                        abs(next_state[0] - observation_space.low[0]) < 0.05 or
                        abs(next_state[1] - observation_space.high[1]) < 0.05 or
                        abs(next_state[1] - observation_space.low[1]) < 0.05))
    # if range_finder_range < 0.1
    #see ... for explanation of points
    points = [
        np.array([next_state[0], next_state[1]]),  # point1
        np.array([next_state[0] + wheelbase / 2, next_state[1]]),  # point2
        np.array([next_state[0] - wheelbase / 2, next_state[1]]),  # point3
        np.array([next_state[0] + wheelbase / 2, next_state[1] + wheel_radius]),  # point4
        np.array([next_state[0] - wheelbase / 2, next_state[1] + wheel_radius]),  # point5
        np.array([next_state[0] + wheelbase / 2, next_state[1] - wheel_radius]),  # point6
        np.array([next_state[0] - wheelbase / 2, next_state[1] - wheel_radius])   # point7
    ]

    # Define the rotation matrix
    rotation_matrix = np.array([
        [np.cos(next_state[4]), np.sin(next_state[4])],
        [-np.sin(next_state[4]), np.cos(next_state[4])]
    ])

    # Rotate points
    rotated_points = [rotation_matrix.dot(point - np.array([next_state[0], next_state[1]])) + np.array([next_state[0], next_state[1]])
                      for point in points]

    # Check if polygon contains any rotated point
    obstacle_reward = 0
    for idx, rotated_point in enumerate(rotated_points):
        shapely_point = Point(rotated_point[0], rotated_point[1])
        for polygon in polygons:
            if polygon.contains(shapely_point):
                obstacle_reward = -30
                break

    total_reward = goal_reward_distance + bounds_reward + obstacle_reward
    done = bool(((next_state[0] - goal_state[0])**2 + (next_state[1] - goal_state[1])**2) < (goal_range+0.25)**2)
    #done = bool((abs(next_state[0] - goal_state[0]) < goal_range) and (abs(next_state[1] - goal_state[1]) < goal_range))
    return total_reward, done