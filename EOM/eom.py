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

def linear_velocity_steering_angle(x,u, param, T_s, current_linear_velocity):
    #u[0] desired linear velocity
    #u[1] desired steering angle

    #x = x, y, alpha, range_finder, current_linear_velocity
    
    
    r = param[1]
    L = param[0]
    
    linear_velocity_des = u[0]*1.2
    linear_velocity_new = 0.05*linear_velocity_des + 0.95*current_linear_velocity

    steering_angle_des = u[1]*pi
    steering_angle_new = 0.05*steering_angle_des + 0.95*x[2]

    x_dot = linear_velocity_new*np.sin(steering_angle_new)*T_s
    y_dot = linear_velocity_new*np.cos(steering_angle_new)*T_s

    dx = [x_dot, y_dot, steering_angle_new-x[2], 0, linear_velocity_new]

    return dx

def linear_angular_velocity(x,u, param, wheel_velocity, T_s):
    #u[0] desired linear velocity
    #u[1] desired angular velocity
    r = param[1]
    L = param[0]
    linear_velocity = u[0]
    angular_velocity = u[1]

    #convert to wheel velocities
    omega_l_des = (linear_velocity/r - angular_velocity*L/2*r)
    omega_r_des = (linear_velocity/r + angular_velocity*L/2*r)

    #new wheel velocities


def dynamic_model_mindstorm(x,u, param, wheel_velocity,T_s):
    wheel_base = param[0]   # distance between the wheels
    wheel_radius = param[1]# radius of the wheels
    # Unpack parameters

    r = wheel_radius
    L = wheel_base
    # Current state (x_pos, y_pos, x_dot, y_dot, alpha)
    x_pos, y_pos, alpha, max_range = x

    #scale u to motor input range
    scaled_u = u*100
    # Motor inputs (PWM commands for left and right motors)
    l_omega, r_omega = ((scaled_u*12 - wheel_velocity)/0.25)*T_s # degrees/second 
    
    #set to l_omega, r_omega to radians/second
    v_l = l_omega * wheel_radius * 1/180*pi  # m/s 
    v_r = r_omega * wheel_radius * 1/180*pi  # m/s 

    # Compute linear and angular velocity of the robot
    v = (v_l + v_r) / 2  # Linear velocity
    omega = (v_l - v_r) / wheel_base  # Angular velocity

    # Update position in the world frame
    x_dot = T_s * v * np.sin(alpha)
    y_dot = T_s * v * np.cos(alpha)

    wheel_velocity = [l_omega, r_omega]
    # Update orientation
    alpha_dot = T_s * omega

    dx = np.array([x_dot, y_dot, alpha_dot, 0, l_omega, r_omega])  # Linear velocities + accelerations + angular velocity

    return dx