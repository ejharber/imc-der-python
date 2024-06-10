import numpy as np
from numpy_sim import *

import sys
sys.path.append("../UR5e")
from CustomRobots import *
import seaborn as sns
from matplotlib import pyplot as plt, patches
import matplotlib.animation as animation
import imageio

class Rope(object):
    def __init__(self, X):

        self.UR5e = UR5eCustom()

        ## Model Parameters (Calculated)
        self.N = 20
        self.dt = 0.002 # we are collecting data at 500 hz
        self.g = 9.8
        self.radius = 0.05

        ## Physics Parameters
        self.dL0 = X[0]
        self.dL1 = X[1] # should replace this with an array         
        self.dL2 = X[2]
        self.Kb1 = X[3]
        self.Kb2 = X[4]
        self.Ks1 = X[5]
        self.Ks2 = X[6]
        self.damp = X[7]
        self.m1 = X[8]
        self.m2 = X[9]
        self.m3 = X[10]

        self.x0 = None
        self.u0 = None

    def run_sim(self, q0, qf):

        self.reset()

        traj = self.UR5e.create_trajectory(q0, qf)
        traj = self.UR5e.fk_traj_stick(traj, True)

        traj_u = np.diff(traj, axis=0).T
        traj_u = np.append(np.zeros((3, 200)), traj_u, axis=1) / self.dt

        # print(traj.shape, traj.shape)
        self.x0[::2] += traj[0, 0]
        self.x0[1::2] += traj[0, 1]

        f_save, q_save, u_save, success = run_simulation(self.x0, self.u0, self.N, self.dt, self.dL0, self.dL1, self.dL2, self.g, self.Kb1, self.Kb2, self.Ks1, self.Ks2, self.damp, self.m1, self.m2, self.m3, traj_u = traj_u)

        if not success:
            return False, [], [], [], []

        # force_x = np.sum(f_save[5::2, -500:], axis = 0)
        # force_y = np.sum(f_save[4::2, -500:], axis = 0)

        # force_x = np.sum(f_save[5::2, :], axis = 0)
        # force_y = np.sum(f_save[4::2, :], axis = 0)
        # force_x = force_x - force_x[199]
        # force_y = force_y - force_y[199]
        # print(f_save.shape)
        traj_force = f_save - f_save[:, 100]
        # print(traj_force.shape)
        traj_force = traj_force[:, -500:]
        # print(traj_force.shape)
        # force_y = f_save[3, :] - f_save[3, 100]
        # traj_force = np.array([force_x, force_y])
        # plt.plot(traj_force.T)
        # plt.show()
        # exit()
        # plt.pl

        # force_x = force_x_ * np.cos(traj[:, 2]) - force_y_ * np.sin(traj[:, 2])
        # force_y = force_x_ * np.sin(traj[:, 2]) + force_y_ * np.cos(traj[:, 2])

        traj_pos = q_save[-2:, -500:] 
        # print(traj)

        return success, traj_pos.T, traj_force.T, q_save.T, f_save.T

    def reset(self, seed = None):

        self.x0 = np.zeros((self.N, 2))
        for k in range(self.N):
            if k <= 1:
                self.x0[k, 1] = - k * self.dL1
            else:
                self.x0[k, 1] = self.x0[k-1, 1] - self.dL2

        self.x0 = self.x0.flatten()
        self.u0 = np.zeros(self.N*2)

    def render(self, q_save, traj_mocap = None, traj_sim = None, filename=None):

        # plt.figure("render")
        sns.set() # Setting seaborn as default style even if use only matplotlib
        fig, ax = plt.subplots(figsize=(5, 5))

        if traj_mocap is not None:
            plt.plot(traj_mocap[:, 0], traj_mocap[:, 1], "k--", label="mocap data")
            plt.legend()
        if traj_sim is not None:
            plt.plot(traj_sim[:, 0], traj_sim[:, 1], "g--", label="sim data")            
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


       

