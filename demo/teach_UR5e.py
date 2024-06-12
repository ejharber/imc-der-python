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

        # mocap cb
        self.mocap_subscription = self.create_subscription(RigidBodies, '/rigid_bodies', self.mocap_callback, 10)
        self.mocap_data = None

        timer_period = 0.002
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.mocap_data_save = []

        time.sleep(1)
        self.go_to_home()

    def timer_callback(self):
        pass

    def mocap_callback(self, msg):
        self.mocap_data = np.zeros((7, 1))
        for rigid_body in msg.rigidbodies:

            i = None

            if rigid_body.rigid_body_name == "rope_tip.rope_tip":
                i = 0

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
        self.mocap_data_save = [self.mocap_data]

    def take_data_routine(self):

        N = 3
        count = 0

        self.mocap_data_save = []

        for dq1 in np.linspace(-10, 10, N):
            for dq2 in np.linspace(-10, 10, N):
                for dq3 in np.linspace(-12, 12, N):

                    for trail in range(10):

                        self.mocap_data_save = []

                        time.sleep(0.2)
                        qf = [180, -90, 100, -180, -90, 0]
                        qf = [qf[0], dq1 + qf[1], dq2 + qf[2], dq3 + qf[3], qf[4], 0]

                        self.rope_swing(qf)
                        if not len(self.mocap_data_save) == 0 and not np.any(np.array(self.mocap_data_save) == 0):
                            break

                        else:
                            print('failed')

                    q0_save = np.array(np.copy(self.home_joint_pose))
                    qf_save = np.array(qf)
                    mocap_data_save = np.array(self.mocap_data_save)

                    np.savez("raw_data/" + str(count), q0_save=q0_save, qf_save=qf_save, mocap_data_save=mocap_data_save)

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

        for i in range(traj.shape[1]):
            t_start = self.rtde_c.initPeriod()
            q = traj[:, i]

            self.rtde_c.servoJ(q, velocity, acceleration, dt, lookahead_time, gain)

            self.take_data()

            self.rtde_c.waitPeriod(t_start)

        self.rtde_c.servoStop()

        self.go_to_home()

    def reset_rope(self):
        p = np.copy(self.home_cart_pose)
        p[2] -= 0.03
        self.rtde_c.moveL(p, speed = 0.01, acceleration = 0.01)

        p = np.copy(self.home_cart_pose)
        self.rtde_c.moveL(p, speed = 0.01, acceleration = 0.01)

def main(args=None):
    rclpy.init(args=args)

    ur5e = UR5e_CollectData()

    ur5e.take_data_routine()

    ur5e.destroy_node() # this line is optional 
    rclpy.shutdown()


if __name__ == '__main__':
    main()

