from math import sqrt, pi
from shapely.geometry import Point
from math import cos, sin
import numpy as np
#maybe break if in polygon?

def sparse_reward2d(next_state, goal_state, observation_space, goal_range, collision, action):

    done = False
    goal_reward_distance = 100 * int(abs(next_state[0] - goal_state[0]) < goal_range and
                            abs(next_state[1] - goal_state[1]) < goal_range)
    
    if collision or goal_reward_distance == 100:
        done = True

    #goal reward = -1 if not in goal 
    #goal reward = 1*distance of goal
    if action == 1 or action == 2:
        rotation_reward = -0.25

    total_reward = goal_reward_distance

    return total_reward, done