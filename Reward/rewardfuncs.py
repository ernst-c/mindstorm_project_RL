from math import sqrt, pi
from shapely.geometry import Point
from math import cos, sin
import numpy as np
#maybe break if in polygon?

def sparse_reward2d(next_state, goal_state, observation_space, goal_range, collision):

    #distance_to_goal = ((goal_state[0] - next_state[0])**2 + (goal_state[1] - next_state[1])**2)**0.5
    goal_reward_distance = -1 * int(abs(next_state[0] - goal_state[0]) > goal_range or
                            abs(next_state[1] - goal_state[1]) > goal_range)

    #goal_reward_distance = -1 * distance_to_goal
    bounds_reward = -1 * int((abs(next_state[0] - observation_space.high[0]) < 0.05 or
                        abs(next_state[0] - observation_space.low[0]) < 0.05 or
                        abs(next_state[1] - observation_space.high[1]) < 0.05 or
                        abs(next_state[1] - observation_space.low[1]) < 0.05))
    obstacle_reward = 0

    if collision:
        obstacle_reward = -150 #-6
    
    total_reward = goal_reward_distance + bounds_reward + obstacle_reward
    done = (goal_reward_distance == 0)

    #done = bool(((next_state[0] - goal_state[0])**2 + (next_state[1] - goal_state[1])**2) < (goal_range+0.1)**2)
    #done = bool((abs(next_state[0] - goal_state[0]) < goal_range) and (abs(next_state[1] - goal_state[1]) < goal_range))
    return total_reward, done