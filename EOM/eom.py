from numba import jit
import numpy as np
from math import cos, sin, pi, sqrt

@jit
def discrete_model(x,u):
    alpha = x[2]
    alpha_dot = 0
    x_dot = 0
    y_dot = 0
    if u == 0:
        x_dot = sin(alpha)*0.1
        y_dot = cos(alpha)*0.1
    elif u == 1:
        alpha_dot = pi/2
    elif u == 2:
        alpha_dot = -pi/2

    dx = [x_dot, y_dot, alpha_dot, 0]
    return dx
