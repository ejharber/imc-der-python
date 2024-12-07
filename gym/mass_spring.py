import numpy as np
from numpy_mass_spring import *

import sys
sys.path.append("../UR5e")
from CustomRobots import *
import seaborn as sns
from matplotlib import pyplot as plt, patches
import matplotlib.animation as animation
import imageio

class MassSpring(object):
    def __init__(self, X):

        self.UR5e = UR5eCustom()

        ## Model Parameters (Calculated)
        self.N = 2
        self.dt = 0.002 # we are collecting data at 500 hz
        self.g = 9.81
        self.radius = 0.05 # only used in visualization 

        ## Physics Parameters
        self.dL = X[0]
        self.K = X[1]
        self.damp = X[2] 
        self.m = X[3]
        self.axial_damp = X[4]

        self.x0 = None
        self.u0 = None

    def run_sim(self):

        self.reset()

        traj = np.zeros((2, int(10/self.dt)))

        q_save, u_save, f_save, success = run_simulation(self.x0, self.u0, self.N, self.dt, self.g, self.dL, self.K, self.damp, self.m, traj, self.axial_damp)

        if not success:
            return False, [], [], [], []
        
        traj_pos = q_save[-2:, :] # trajectory of tip
        return success, traj_pos.T, f_save.T, q_save.T, f_save.T

    def reset(self, seed = None):

        self.x0 = np.zeros((self.N, 2))
        for k in range(1, self.N):
            if k == 1:
                self.x0[k, 1] = - self.dL

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


       

