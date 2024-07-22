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

import sys
sys.path.append("../UR5e")
from CustomRobots import *

class UR5e_CollectData(Node):
    def __init__(self):

        super().__init__('collect_rope_data')

        self.UR5e = UR5eCustom()

        self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.60")
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.60")

        self.home_joint_pose = [180, -53.25, 134.66, -171.28, -90, 0]
        self.home_cart_pose = None

        # ati cb
        self.ati_subscription = self.create_subscription(WrenchStamped, '/ati', self.ati_callback, 10)
        self.ati_data = None
        self.ati_data_zero = None

        # mocap cb
        self.mocap_subscription = self.create_subscription(RigidBodies, '/rigid_bodies', self.mocap_callback, 10)
        self.mocap_data = None

        timer_period = 0.002
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.ati_data_save = []
        self.mocap_data_save = []
        self.ur5e_cmd_data_save = []
        self.ur5e_tool_data_save = []
        self.ur5e_jointstate_data_save = []

        time.sleep(1)
        self.go_to_home()

    def timer_callback(self):
        pass

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
            self.mocap_data[5, i] = rigid_body.pose.orientation.y  
            self.mocap_data[6, i] = rigid_body.pose.orientation.z  

    def go_to_home(self):
        print('go home')
        q = np.copy(self.home_joint_pose)
        q = np.array(q) * np.pi / 180.0
        self.rtde_c.moveJ(q, 0.2, 0.2)
        self.home_cart_pose = self.rtde_r.getActualTCPPose()

    def take_data(self):
        rclpy.spin_once(self)

        self.ati_data_save.append(self.ati_data - self.ati_data_zero)
        self.mocap_data_save.append(self.mocap_data)
        self.ur5e_cmd_data_save.append(self.rtde_r.getTargetQ())
        self.ur5e_tool_data_save.append(self.rtde_r.getActualTCPPose())
        self.ur5e_jointstate_data_save.append(self.rtde_r.getActualQ())

    def test_rope_swing(self):

        q = [0, -90, 95, -180, -90, 0]

        self.ati_data_save = []
        self.mocap_data_save = []
        self.ur5e_cmd_data_save = []
        self.ur5e_tool_data_save = []
        self.ur5e_jointstate_data_save = []

        self.rope_swing(q)
        
        # self.q_save = np.array(self.q_save)
        # self.ur5e_cmd_data_save = np.array(self.ur5e_cmd_data_save)
        # self.ur5e_jointstate_data_save = np.array(self.ur5e_jointstate_data_save)
        # self.ati_data_save = np.array(self.ati_data_save)
        self.mocap_data_save = np.array(self.mocap_data_save)

        # plt.figure('ATI Data')
        # plt.plot(self.ati_data_save, 'b.')
        # filtered = self.butter_lowpass_filter(np.copy(self.ati_data_save).T)
        # plt.plot(filtered.T)

        # print(self.mocap_data_save.shape)
        plt.figure('Rope Trajectory')
        plt.plot(self.mocap_data_save[:, 0, 2], 'r-')
        plt.plot(self.mocap_data_save[:, 1, 2], 'g-')
        plt.plot(self.mocap_data_save[:, 2, 2], 'b-')

        time.sleep(3)

        self.go_to_home()

    def take_data_routine(self):

        N = 6 # take for 3 "levels" of experiments
        # with 2^3, 3^3, 4^3 and 5^3 bits of data
        count = 0

        self.ati_data_save = []
        self.mocap_data_save = []
        self.ur5e_cmd_data_save = []
        self.ur5e_tool_data_save = []
        self.ur5e_jointstate_data_save = []

        for dq1 in np.linspace(-10, 10, N):
            for dq2 in np.linspace(-10, 10, N):
                for dq3 in np.linspace(-12, 12, N):

                    for trail in range(10):

                        self.ati_data_save = []
                        self.mocap_data_save = []
                        self.ur5e_cmd_data_save = []
                        self.ur5e_tool_data_save = []
                        self.ur5e_jointstate_data_save = []

                        time.sleep(0.2)
                        qf = [180, -90, 100, -180, -90, 0]
                        qf = [qf[0], dq1 + qf[1], dq2 + qf[2], dq3 + qf[3], qf[4], 0]

                        self.rope_swing(qf)
                        if not np.any(np.array(self.mocap_data_save)[400:1100, :, 2] == 0):
                            break

                        else:
                            print('failed')

                    q0_save = np.array(np.copy(self.home_joint_pose))
                    qf_save = np.array(qf)
                    ur5e_tool_data_save = np.array(self.ur5e_tool_data_save)
                    ur5e_cmd_data_save = np.array(self.ur5e_cmd_data_save)
                    ur5e_jointstate_data_save = np.array(self.ur5e_jointstate_data_save)
                    ati_data_save = np.array(self.ati_data_save)
                    mocap_data_save = np.array(self.mocap_data_save)

                    np.savez("raw_data/" + str(count), q0_save=q0_save, qf_save=qf_save, ur5e_tool_data_save=ur5e_tool_data_save, ur5e_cmd_data_save = ur5e_cmd_data_save, ur5e_jointstate_data_save=ur5e_jointstate_data_save, ati_data_save=ati_data_save, mocap_data_save=mocap_data_save)

                    count += 1

    def rope_swing(self, q):

        self.go_to_home()
        self.reset_rope()
        self.go_to_home()

        q0 = np.copy(self.home_joint_pose)
        qf = np.copy(q)

        traj = self.UR5e.create_trajectory(q0, qf, time=1)

        # Parameters
        velocity = 3
        acceleration = 5
        dt = 1.0/500  # 2ms
        lookahead_time = 0.1
        gain = 1000

        for i in range(500):
            self.take_data()
            time.sleep(dt)

        for i in range(traj.shape[1]):
            t_start = self.rtde_c.initPeriod()
            q = traj[:, i]

            self.rtde_c.servoJ(q, velocity, acceleration, dt, lookahead_time, gain)

            self.take_data()

            self.rtde_c.waitPeriod(t_start)

        # self.take_data(q)

        for i in range(500):
            self.take_data()
            time.sleep(dt)

        self.rtde_c.servoStop()

        self.go_to_home()

    def zero_ati(self):
        ati_data = []
        for _ in range(100):
            rclpy.spin_once(self)
            if self.ati_data is None: continue 
            ati_data.append(self.ati_data)

        self.ati_data_zero = np.mean(np.array(ati_data), axis = 0)

    def reset_rope(self):
        p = np.copy(self.home_cart_pose)
        p[2] -= 0.03
        self.rtde_c.moveL(p, speed = 0.01, acceleration = 0.01)

        time.sleep(0.5)

        p = np.copy(self.home_cart_pose)
        self.rtde_c.moveL(p, speed = 0.002, acceleration = 0.01)
        time.sleep(2)

        # zero ati
        self.zero_ati()


def main(args=None):
    rclpy.init(args=args)

    ur5e = UR5e_CollectData()

    # rclpy.spin(ur5e)
    # for _ in range(10):
        # ur5e.test_rope_swing()
    # plt.show()

    ur5e.take_data_routine()

    ur5e.destroy_node() # this line is optional 
    rclpy.shutdown()


if __name__ == '__main__':
    main()

