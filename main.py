import numpy as np

import matplotlib.pyplot as plt

"""
three equations of lorenz sy_arrtem: 
1. x'= sigma(y-x)
2. y'= x(rho-z)-y
3. z'= xy - beta*z
we must choose proper sigma rho and beta so stimulate air
 rho=28
 sigma = 10
 beta = 8/3
"""
# Lorenz sy_arrtem equations
def lorenz(x, y, z, sigma=10, rho=28, beta=2.66666666667):
    
    x_dot = sigma*(y - x)
    y_dot = rho*x - y - x*z
    z_dot = x*y - beta*z

    return x_dot, y_dot, z_dot


# initial conditions

x0, y0, z0 = 0, 0, 0
t0, tmax = 0, 100
dt = 0.01


# Initialize array_arr to store the results

steps = int((tmax - t0) / dt)
x_arr = np.empty(steps + 1)
y_arr = np.empty(steps + 1)
z_arr = np.empty(steps + 1)

ts = np.linspace(t0, tmax, steps + 1)


# simulation using fourth-order Runge-Kutta method

for i in range(steps):
    x_dot1, y_dot1, z_dot1 = lorenz(x_arr[i], y_arr[i], z_arr[i])
    x_dot2, y_dot2, z_dot2 = lorenz(x_arr[i] + 0.5*dt*x_dot1, y_arr[i] + 0.5*dt*y_dot1, z_arr[i] + 0.5*dt*z_dot1)
    x_dot3, y_dot3, z_dot3 = lorenz(x_arr[i] + 0.5*dt*x_dot2, y_arr[i] + 0.5*dt*y_dot2, z_arr[i] + 0.5*dt*z_dot2)
    x_dot4, y_dot4, z_dot4 = lorenz(x_arr[i] + dt*x_dot3, y_arr[i] + dt*y_dot3, z_arr[i] + dt*z_dot3)

    x_arr[i+1] = x_arr[i] + (1/6) * dt * (x_dot1 + 2*x_dot2 + 2*x_dot3 + x_dot4)
    y_arr[i+1] = y_arr[i] + (1/6) * dt * (y_dot1 + 2*y_dot2 + 2*y_dot3 + y_dot4)
    z_arr[i+1] = z_arr[i] + (1/6) * dt * (z_dot1 + 2*z_dot2 + 2*z_dot3 + z_dot4)


# plot results

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(x_arr, y_arr, z_arr, lw=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()