from numba import jit
import numpy as np
from math import cos, sin, pi, sqrt

@jit
def simple_dynamic_model(x,u, param):
    v_body_des = u[0]
    alpha_des = u[1]*0.9*pi
    alpha = x[4]
    v_body_current = np.sqrt(x[2]**2+x[3]**2)
    v_body_new = 0.1*v_body_des+0.9*v_body_current
    alpha_new = alpha_des*0.1+alpha*0.9
    x_dot_new = sin(alpha_new)*v_body_new
    y_dot_new = cos(alpha_new)*v_body_new
    dx = [x_dot_new,y_dot_new,x_dot_new-x[2],y_dot_new-x[3],alpha,0]
    return dx

def dynamic_model_mindstorm(x,u, param, wheel_velocity):
    wheel_base = param[0]   # distance between the wheels
    wheel_radius = param[1]# radius of the wheels
    T_s = 1/50
    # Unpack parameters

    r = wheel_radius
    L = wheel_base
    # Current state (x_pos, y_pos, x_dot, y_dot, alpha)
    x_pos, y_pos, alpha, max_range = x

    # Motor inputs (PWM commands for left and right motors)

    l_omega, r_omega = ((u*12 - wheel_velocity)/0.25)*T_s  

    v_l = l_omega * wheel_radius  # Left wheel linear velocity
    v_r = r_omega * wheel_radius  # Right wheel linear velocity

    # Compute linear and angular velocity of the robot
    v = (v_l + v_r) / 2  # Linear velocity
    omega = (v_l - v_r) / wheel_base  # Angular velocity

    # Update position in the world frame
    x_dot = T_s * v * np.sin(alpha)
    y_dot = T_s * v * np.cos(alpha)

    wheel_velocity = [l_omega, r_omega]
    # Update orientation
    alpha_dot = T_s * omega

    dx = np.array([x_dot, y_dot, alpha_dot, 1, l_omega, r_omega])  # Linear velocities + accelerations + angular velocity

    return dx