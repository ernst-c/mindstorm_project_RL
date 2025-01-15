#x,wheel_velocities,u,param
def runge_kutta4(x, wheel_velocities, u, eom, param):

    output_k1 = eom(x, wheel_velocities, u, param)
    k1 = output_k1[:4] 
    wheel_velocities_k1 = output_k1[4:] 

    output_k2 = eom(x + 0.5 * param[2] * k1,wheel_velocities+ 0.5 * param[2] * wheel_velocities_k1, u, param)
    k2 = output_k2[:4]
    wheel_velocities_k2 = output_k2[4:]

    output_k3 = eom(x + 0.5 * param[2] * k2,wheel_velocities+ 0.5 * param[2] * wheel_velocities_k2, u, param)
    k3 = output_k3[:4] 
    wheel_velocities_k3 = output_k3[4:]

    output_k4 = eom(x + param[2] * k3,wheel_velocities + param[2] * wheel_velocities_k3, u, param)
    k4 = output_k4[:4]
    wheel_velocities_k4 = output_k4[4:]

    return (param[2] / 6) * (k1 + 2 * k2 + 2 * k3 + k4), (param[2] / 6) * (wheel_velocities_k1 + 2 * wheel_velocities_k2 + 2 * wheel_velocities_k3 + wheel_velocities_k4)