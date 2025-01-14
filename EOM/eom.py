from numba import jit
import numpy as np
from math import cos, sin, pi, sqrt

@jit
def cont_model(x,wheel_velocities,u,param):
    #param = [wheel_base, wheel_radius, T_s]
    x_pos, y_pos, alpha, max_range = x
    wheel_vel_l, wheel_vel_r = wheel_velocities
    #scale u to motor input range
    scaled_u = u*100
    # Motor inputs (PWM commands for left and right motors)
    l_domega, r_domega = ((scaled_u*12 - np.array([wheel_vel_l,wheel_vel_l]))/0.25)*param[2] # degrees/second 
    
    v_l = (l_domega+wheel_vel_l) * param[1] * 1/180*pi  # m/s 
    v_r = (r_domega+wheel_vel_r) * param[1] * 1/180*pi  # m/s 

    v = (v_l + v_r) / 2  # Linear velocity
    omega = (v_l - v_r) / param[0]  # Angular velocity

    x_dot = v * np.sin(alpha) * param[2]
    y_dot = v * np.cos(alpha) * param[2]
    alpha_dot = omega * param[2]
    
    dx = [x_dot, y_dot, alpha_dot, 0, l_domega, r_domega]

    return dx