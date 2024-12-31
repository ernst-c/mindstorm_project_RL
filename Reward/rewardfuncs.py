from math import sqrt, pi
from shapely.geometry import Point
from math import cos, sin
import numpy as np
#maybe break if in polygon?

def sparse_reward2d(next_state, goal_state, goal_range, collision, reached_goals):

    done = False
    total_goal_reward_distance = 1 * int(abs(next_state[0] - goal_state[0]) < goal_range and
                            abs(next_state[1] - goal_state[1]) < goal_range)

    if collision or total_goal_reward_distance == 1:
        done = True
    
    #checkpoints
    checkpoint_reward = 0
    for goal in reached_goals:
        if goal:
            checkpoint_reward += 0.2

    total_reward = total_goal_reward_distance + checkpoint_reward

    return total_reward, done