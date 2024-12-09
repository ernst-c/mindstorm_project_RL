
# Runge Kutta taking the EOM as an argument
def runge_kutta4(x, u, eom, t_s):

    k1 = eom(x, u)
    k2 = eom(x + 0.5 * t_s * k1, u)
    k3 = eom(x + 0.5 * t_s * k2, u)
    k4 = eom(x + t_s * k3, u)

    return (t_s / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
