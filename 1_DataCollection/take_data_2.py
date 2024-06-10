#!/usr/bin/env python3

import rtde_control
import rtde_receive
import numpy as np

# ros packages
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from mocap4r2_msgs.msg import RigidBodies
import matplotlib.pyplot as plt

import numpy as np
import time

# import csv
# import numpy as np
# import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, freqz, sosfilt, sosfiltfilt

def create_trajectory(weighpoints):
    def quintic_func(q0, qf, T, qd0=0, qdf=0):
        X = [
            [ 0.0,          0.0,         0.0,        0.0,     0.0,  1.0],
            [ T**5,         T**4,        T**3,       T**2,    T,    1.0],
            [ 0.0,          0.0,         0.0,        0.0,     1.0,  0.0],
            [ 5.0 * T**4,   4.0 * T**3,  3.0 * T**2, 2.0 * T, 1.0,  0.0],
            [ 0.0,          0.0,         0.0,        2.0,     0.0,  0.0],
            [20.0 * T**3,  12.0 * T**2,  6.0 * T,    2.0,     0.0,  0.0],
        ]
        # fmt: on
        coeffs, resid, rank, s = np.linalg.lstsq(
            X, np.r_[q0, qf, qd0, qdf, 0, 0], rcond=None
        )

        # coefficients of derivatives
        coeffs_d = coeffs[0:5] * np.arange(5, 0, -1)
        coeffs_dd = coeffs_d[0:4] * np.arange(4, 0, -1)

        return lambda x: (
            np.polyval(coeffs, x),
            np.polyval(coeffs_d, x),
            np.polyval(coeffs_dd, x),
        )

    def quintic(q0, qf, t, qd0=0, qdf=0):
        tf = max(t)

        polyfunc = quintic_func(q0, qf, tf, qd0, qdf)

        # evaluate the polynomials
        traj = polyfunc(t)
        p = traj[0]
        pd = traj[1]
        pdd = traj[2]

        return p

    traj = []
    time = 1 # set sim time to be 0.5 seconds

    for dim in range(weighpoints.shape[0]):
        x_1_traj = [0, time]
        x_2_traj = [weighpoints[dim, 0], weighpoints[dim, 1]]
        t = np.linspace(0, time, round(time*500)) # can send data at 500 hz
        traj.append(quintic(x_2_traj[0], x_2_traj[1], t))

    traj = np.array(traj)

    return traj


class UR5E(Node):
    def __init__(self):

        super().__init__('collect_rope_data')

        self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.60")
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.60")

        self.home_joint_pose = [0, -54, 134, -167, -90, 0]
        self.home_cart_pose = None

        # ati cb
        self.ati_subscription = self.create_subscription(WrenchStamped, '/ati', self.ati_callback, 1)
        self.ati_data = None
        self.ati_data_zero = None

        # mocap cb
        self.mocap_subscription = self.create_subscription(RigidBodies, '/rigid_bodies', self.mocap_callback, 1)
        self.mocap_data = None

        # commanded cb
        self.ur5e_cmd_subscription = self.create_subscription(JointState, '/ur5e/command/joint_states', self.ur5e_cmd_callback, 1)
        self.ur5e_cmd_data = None

        # tool cb
        self.ur5e_tool_subscription = self.create_subscription(PoseStamped, '/ur5e/feedback/end_effector', self.ur5e_tool_callback, 1)
        self.ur5e_tool_data = None

        # joint measured cb
        self.ur5e_jointstate_subscription = self.create_subscription(JointState, '/ur5e/feedback/joint_states', self.ur5e_jointstate_callback, 1)
        self.ur5e_jointstate_data = None

        self.q_save = []
        self.ati_data_save = []
        self.mocap_data_save = []
        self.ur5e_cmd_data_save = []
        self.ur5e_tool_data_save = []
        self.ur5e_jointstate_data_save = []

        time.sleep(1)
        self.go_to_home()

        q = [0, -81, 73, -167, -90, 0]

        weighpoints = [[self.home_joint_pose[i], q[i]] for i in range(6)]
        weighpoints = np.array(weighpoints)
        self.traj = create_trajectory(weighpoints)
        self.traj_i = None

        timer_period = 0.001
        self.timer = self.create_timer(timer_period, self.timer_callback_collect_data)

        timer_period = 2
        self.timer = self.create_timer(timer_period, self.timer_callback_command_robot)

        self.reset_rope()
        self.go_to_home()

        self.traj_i = 0

    def timer_callback_collect_data(self):
        pass

    def timer_callback_command_robot(self):
        # # Parameters
        velocity = 3
        acceleration = 5
        dt = 1.0/500  # 2ms
        lookahead_time = 0.04
        gain = 1000

        print(self.traj_i)

        if self.traj_i is None: return

        if self.traj_i == 0:
            self.rtde_c.initPeriod()

        if int(self.traj_i/2.0) >= self.traj.shape[1]:
            self.rtde_c.servoStop()
    
            self.q_save = np.array(self.q_save)
            self.ur5e_cmd_data_save = np.array(self.ur5e_cmd_data_save)
            self.ur5e_jointstate_data_save = np.array(self.ur5e_jointstate_data_save)
            # self.ati_data_save = np.array(self.ati_data_save)
            self.mocap_data_save = np.array(self.mocap_data_save)

            plt.figure('ATI Data')
            plt.plot(self.ati_data_save, 'b.')
            filtered = self.butter_lowpass_filter(np.copy(self.ati_data_save).T)
            plt.plot(filtered.T)

            print(self.mocap_data_save.shape)
            plt.figure('Rope Trajectory')
            plt.plot(self.mocap_data_save[:, 0, 2])
            plt.plot(self.mocap_data_save[:, 1, 2])
            plt.plot(self.mocap_data_save[:, 2, 2])

            plt.show()

            self.destroy_node()

        if self.traj_i % 2 == 0:
            q = self.traj[:, int(self.traj_i/2.0)]
            q = q * np.pi / 180

            self.rtde_c.servoJ(q, velocity, acceleration, dt, lookahead_time, gain)

            self.take_data(q)

        self.traj_i += 1
        

    def ati_callback(self, msg):
        self.ati_data = [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]

    def mocap_callback(self, msg):
        self.mocap_data = np.zeros((7, 3))
        for rigid_body in msg.rigidbodies:

            i = None

            if rigid_body.rigid_body_name == "UR_base.UR_base":
                i = 0

            if rigid_body.rigid_body_name == "rope_base.rope_base":
                i = 1

            if rigid_body.rigid_body_name == "rope_tip.rope_tip":
                i = 2

            if i is None: continue

            self.mocap_data[0, i] = rigid_body.pose.position.x    
            self.mocap_data[1, i] = rigid_body.pose.position.y    
            self.mocap_data[2, i] = rigid_body.pose.position.z    

            self.mocap_data[3, i] = rigid_body.pose.orientation.w    
            self.mocap_data[4, i] = rigid_body.pose.orientation.x    
            self.mocap_data[5, i] = rigid_body.pose.orientation.z  
            self.mocap_data[6, i] = rigid_body.pose.orientation.z  

    def ur5e_cmd_callback(self, msg):
        self.ur5e_cmd_data = msg.position

    def ur5e_tool_callback(self, msg):
        self.ur5e_tool_data = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]

    def ur5e_jointstate_callback(self, msg):
        self.ur5e_jointstate_data = msg.position

    def go_to_home(self):
        print('go home')
        q = np.copy(self.home_joint_pose)
        q = np.array(q) * np.pi / 180.0
        self.rtde_c.moveJ(q)
        self.home_cart_pose = self.rtde_r.getActualTCPPose()

    def take_data(self, q):
        # rclpy.spin_once(self)

        self.q_save.append(q)
        self.ati_data_save.append(self.ati_data - self.ati_data_zero)
        self.mocap_data_save.append(self.mocap_data)
        self.ur5e_cmd_data_save.append(self.ur5e_cmd_data)
        self.ur5e_tool_data_save.append(self.ur5e_tool_data)
        self.ur5e_jointstate_data_save.append(self.ur5e_jointstate_data)

    def butter_lowpass_filter(self, data, cutoff=10, fs=500.0, order=5):
        sos = butter(order, cutoff, fs=fs, btype='low', analog=False, output='sos')
        filtered = sosfiltfilt(sos, data)
        return filtered

        # print(np.array(self.mocap_data).shape)


    def zero_ati(self):
        ati_data = []
        for _ in range(100):
            print("zero ati", self.ati_data)
            # time.sleep(0.01)
            rclpy.spin_once(self)
            if self.ati_data is None: 
                continue 
            ati_data.append(self.ati_data)

        self.ati_data_zero = np.mean(np.array(ati_data), axis = 0)

    def reset_rope(self):
        p = np.copy(self.home_cart_pose)
        p[2] -= 0.03
        self.rtde_c.moveL(p, speed = 0.02, acceleration = 0.01)

        p = np.copy(self.home_cart_pose)
        self.rtde_c.moveL(p, speed = 0.02, acceleration = 0.01)
        time.sleep(2)

        # zero ati
        self.zero_ati()


def main(args=None):
    rclpy.init(args=args)

    ur5e = UR5E()

    rclpy.spin(ur5e)
    # ur5e.test_rope_swing()

    ur5e.destroy_node() # this line is optional 
    rclpy.shutdown()


if __name__ == '__main__':
    main()

