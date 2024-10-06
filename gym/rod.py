import numpy as np
from numpy_rod_sim import *

import sys
sys.path.append("../UR5e")
from CustomRobots import *
import seaborn as sns
from matplotlib import pyplot as plt, patches
import matplotlib.animation as animation
import imageio

class Rod(object):
    def __init__(self, X):

        self.UR5e = UR5eCustom()

        ## Model Parameters (Calculated)
        self.N = 2
        self.dt = 0.002 # we are collecting data at 500 hz
        self.sample_rate = 0.002
        self.g = 9.81
        self.radius = 0.05 # only used in visualization 

        ## Physics Parameters
        self.dL_stick = X[0]
        self.dL_ati = X[1]
        self.damp = X[2] 
        self.m1 = X[3]

        self.x0 = None
        self.u0 = None

    def run_sim(self, q0, qf):
        def non_inertial_forces_with_euler(mass, damp, forces_inertial, v_frame, a_frame, angle, omega, angular_acceleration, positions, velocities):
            """
            Calculate forces in a non-inertial frame over N time steps, including Euler forces.

            :param mass: Mass of the point mass
            :param forces_inertial: Array of forces in the inertial frame over time (shape [N, 2])
            :param a_frame: Acceleration of the non-inertial frame over time (shape [N, 2])
            :param omega: Angular velocity of the non-inertial frame over time (shape [N])
            :param angular_acceleration: Angular acceleration of the non-inertial frame over time (shape [N])
            :param positions: Array of position vectors in the non-inertial frame over time (shape [N, 2])
            :param velocities: Array of velocity vectors in the non-inertial frame over time (shape [N, 2])
            :return: Array of total forces in the non-inertial frame over time (shape [N, 2])
            """
            N = len(forces_inertial)
            F_non_inertial = np.zeros((N, 2))  # Initialize the output array

            for i in range(N):

                R = np.array([[np.cos(angle[i]), -np.sin(angle[i])], 
                         [np.sin(angle[i]),  np.cos(angle[i])]]) . T

                # Fictitious forces
                F_fictitious = - mass * R @ a_frame[i]
                
                # # Coriolis force: -2m(ω × v)
                F_coriolis = 2 * mass * np.cross([0, 0, omega[i]], [velocities[i, 0], velocities[i, 1], 0])[:2]
                
                # Centrifugal force: -m(ω × (ω × r))
                F_centrifugal = mass * np.cross([0, 0, omega[i]], np.cross([0, 0, omega[i]], [positions[i, 0], positions[i, 1], 0]))[:2]
                
                # Euler force: -m(dω/dt × r)
                F_euler = mass * np.cross([0, 0, angular_acceleration[i]], [positions[i, 0], positions[i, 1], 0])[:2] 

                # gravitational force: 
                F_graviatational = R @ np.array([0, -9.81 * mass])      

                F_damp = - damp * R @ v_frame[i]
                
                # Total force in the non-inertial frame
                forces_inertial[i] = R @ forces_inertial[i]

                # F_non_inertial[i] = forces_inertial[i] - F_fictitious - F_centrifugal - F_euler - F_coriolis
                F_non_inertial[i] = F_graviatational + F_damp + F_fictitious + F_centrifugal + F_euler + F_coriolis

            return F_non_inertial.T, forces_inertial.T

        self.reset()

        traj = self.UR5e.create_trajectory(q0, qf, dt=self.dt, time=1)
        traj = self.UR5e.fk_traj(traj, True)
        traj = np.append(np.repeat([traj[0, :]], traj.shape[0], axis=0), traj, axis=0)

        traj = traj.T

        # print(traj.shape, traj.shape)
        self.x0[::2] += traj[0, 0]
        self.x0[1::2] += traj[0, 1]

        q_save, u_save, f_save, success = run_simulation(self.x0, self.u0, self.N, self.dt, self.dL_stick, self.g, self.damp, self.m1, traj)

        if not success:
            return False, [], [], [], []

        p_frame = traj[[0,1], :]
        p_frame[0, :] += np.cos(traj[2, :]) * self.dL_ati
        p_frame[1, :] += np.sin(traj[2, :]) * self.dL_ati
        p_frame = p_frame.T

        v_frame = np.gradient(p_frame, axis=0, edge_order=2) / self.dt
        a_frame = np.gradient(v_frame, axis=0, edge_order=2) / self.dt

        r = p_frame * 0
        r[:, 0] = self.dL_stick - self.dL_ati
        v = np.gradient(r, axis=0, edge_order=2) / self.dt * 0
        a = np.gradient(v, axis=0, edge_order=2) / self.dt * 0

        angle = traj[2, :]
        omega = np.gradient(angle, axis=0, edge_order=2) / self.dt
        angular_acceleration = np.gradient(omega, axis=0, edge_order=2) / self.dt

        forces_inertial = f_save[[2, 3], 2:].T

        f_total, forces_inertial = non_inertial_forces_with_euler(self.m1, self.damp, forces_inertial, v_frame, a_frame, angle, omega, angular_acceleration, r, v)

        sampling = round(self.sample_rate / self.dt)
        forces_inertial = forces_inertial[:, ::sampling]
        f_total = f_total[:, ::sampling]
        # q_save = q_save[:, ::sampling]
        # f_save = f_save[:, ::sampling]
        traj_pos = q_save[-2:, -500:] # trajectory of tip

        f_total = np.atleast_2d(f_total[0, -500:] - f_total[0, -499]) # zero forces similar to how we really use the sensor
        forces_inertial = np.atleast_2d(forces_inertial[0, -500:] - forces_inertial[0, -499])
        
        plt.plot(forces_inertial[1, -500:], 'r-')
        plt.plot(forces_inertial[0, -500:], 'b-')
        plt.show()
        
        return success, traj_pos.T, f_total.T, forces_inertial.T, q_save.T, f_save.T

    def reset(self, seed = None):

        self.x0 = np.zeros((self.N, 2))
        for k in range(1, self.N):
            if k == 1:
                self.x0[k, 1] = self.dL_stick
            else:
                self.x0[k, 1] = self.x0[k - 1, 1] - self.dL_rope

        self.x0 = self.x0.flatten()
        self.u0 = np.zeros(self.N*2)

    def render(self, q_save, traj_mocap = None, traj_sim = None, goal = None, filename=None):

        # plt.figure("render")
        sns.set() # Setting seaborn as default style even if use only matplotlib
        fig, ax = plt.subplots(figsize=(5, 5))

        if traj_mocap is not None:
            plt.plot(traj_mocap[:, 0], traj_mocap[:, 1], "k--", label="mocap data")
            plt.legend()
        if traj_sim is not None:
            plt.plot(traj_sim[:, 0], traj_sim[:, 1], "g--", label="sim data")            
            plt.legend()
        if goal is not None:
            plt.plot(goal[0], goal[1], "g+", label="goal")            
            plt.legend()

        circles = []

        n = 0
        x_center = q_save[0, 2*n]
        y_center = q_save[0, 2*n+1]

        circles += [patches.Circle((x_center, y_center), radius=self.radius, ec = None, fc='red')]
        ax.add_patch(circles[-1])
        for n in range(1, self.N-1):
            x_center = q_save[0, 2*n]
            y_center = q_save[0, 2*n+1]
            circles += [patches.Circle((x_center, y_center), radius=self.radius, ec = None, fc='blue')]
            ax.add_patch(circles[-1])

        n = self.N-1
        x_center = q_save[0, 2*n]
        y_center = q_save[0, 2*n+1] 
        circles += [patches.Circle((x_center, y_center), radius=self.radius, ec = None, fc='green')]
        ax.add_patch(circles[-1])
        # ax.annotate("g", xytext=(-0.205, -0.16), xy=(-0.2, -0.23),
        #              arrowprops=dict(arrowstyle='->', lw=1.5, color='black'), fontsize=12, color='black')
        lim = 2
        ax.axis('equal')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        plt.show(block=False)

        plt.draw()
        plt.pause(1)

        if filename is None:
            for i in range(q_save.shape[0]-500, q_save.shape[0], 20):
                for n, circle in enumerate(circles):
                    x_center = q_save[i, 2*n]
                    y_center = q_save[i, 2*n+1]
                    circle.set(center=(x_center, y_center))
                plt.draw()
                plt.pause(0.001) 
                fig.canvas.draw()

        else:
            save_render = []
            for i in range(q_save.shape[0]-500, q_save.shape[0], 20):
                for n, circle in enumerate(circles):
                    x_center = q_save[i, 2*n]
                    y_center = q_save[i, 2*n+1]
                    circle.set(center=(x_center, y_center))
                plt.draw()
                plt.pause(0.001) 
                fig.canvas.draw()

                image = np.array(fig.canvas.renderer.buffer_rgba())
                image = image[:,:,:3]

                if len(save_render) == 0:
                    save_render.append(image)
                elif np.all(image.shape == save_render[-1].shape):
                    save_render.append(image)
                else:
                    save_render = []
                    save_render.append(image)

            save_render = np.array(save_render)
            print(save_render.shape)

            # Create the GIF
            output_file = filename
            with imageio.get_writer(output_file, mode='I', duration=0.01, loop=0) as writer:
                for i in range(save_render.shape[0]):
                    image = save_render[i, :, :, :]
                    # image[0,0,0] = i
                    writer.append_data(image)
                    print(i)

 
        plt.close(fig)


       

