import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

def pendulum_real(L = 1, b = 0.5, m = 1, sigmax = 0.05, sigmay = 0.05, seed = None):
	# adds measurment error to a simulated pendulum
	np.random.seed(seed)
	x, y = pendulum_sim(L, b, m)
	x = np.random.normal(x, sigmax) # measured position of bob
	y = np.random.normal(y, sigmay) # measured position of bob
	np.random.seed(None)

	return (x, y)

def pendulum_sim(L = 1, b = 0.5, m = 1):
	# simulates a pendulum's motion

	# model parameters (unknown)
	# L: Length of pendulum (m)
	# b: Damping factor (kg/s)
	# m: Mass of bob (kg) 

	# Simulation Parameters (fixed)
	tf = 1       # End time (s)
	dt = 0.02        # Time step (s)
	g = 9.81        # Acceleration due to gravity (m/s^2)
	theta1_0 = np.pi/2           # Initial angular displacement (rad)
	theta2_0 = 0                 # Initial angular velocity (rad/s)

	def sim_pen_eq(t,theta):
		dtheta2_dt = (-b/m)*theta[1] + (-g/L)*np.sin(theta[0])
		dtheta1_dt = theta[1]
		return [dtheta1_dt, dtheta2_dt]

	theta_0 = [theta1_0, theta2_0]
	t_span = [0, tf+dt]
	t = np.arange(0, tf+dt, dt)
	sim_points = len(t)
	l = np.arange(0, sim_points, 1)

	out = solve_ivp(sim_pen_eq, t_span, theta_0, t_eval = t)
	theta1 = out.y[0, :]
	theta2 = out.y[1, :]

	x = L*np.sin(theta1) # actual position of bob (x)
	y = -L*np.cos(theta1) # actual position of bob (y)

	return (x, y)

# tune the model of the pendulum to the real world data by fitting its 3 physics parameters
def cost_fun(x):
	L = x[0]
	b = x[1]
	m = x[2]

	x_measured, y_measured = pendulum_real(seed=0)
	x_sim, y_sim = pendulum_sim(L, b, m)

	cost = np.sum(abs(x_sim - x_measured)) + np.sum(abs(y_sim - y_measured))

	return cost

def cost_fun_noise(x):
	L = x[0]
	b = x[1]
	m = x[2]

	x_measured, y_measured = pendulum_real(seed=0)
	x_sim, y_sim = pendulum_real(L, b, m, seed=None)

	cost = np.sum(abs(x_sim - x_measured)) + np.sum(abs(y_sim - y_measured))

	return cost

def main():
	# load data set of pendulum trajectory:
	x_measured, y_measured = pendulum_real()

	# estimate model parameters (system identification or sysid)
	fun = lambda x: cost_fun(x, x_measured, y_measured)
	out = minimize(fun, np.ones(3), bounds=[(1e-6, 100) for _ in range(3)]) # note we have to set bounds here because the mass, length and damp can never be below 0

	# generate ground truth trajectory
	x_sim, y_sim = pendulum_sim()

	# generate trajectory as estimated by sysid
	x_sysid, y_sysid = pendulum_sim(out.x[0], out.x[1], out.x[2])

	print("The Length of the pendulum (L) is: " + str(out.x[0]) + '\n' + 
		  "The dampening of the pendulum (b) is: " + str(out.x[1]) + '\n' + 
		  "The mass of the pendulum (m) is: " + str(out.x[2]))
	print('but what is the uncertainty in these measurments?')

	plt.figure()
	plt.plot(x_measured, y_measured, '.', label="measured trajectory")
	plt.plot(x_sysid, y_sysid, '.', label="sysID trajectory")
	plt.plot(x_sim, y_sim, '.', label="ground truth trajectory")

	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()