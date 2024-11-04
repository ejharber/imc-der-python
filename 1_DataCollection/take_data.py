#!/usr/bin/env python3

import rtde_control
import rtde_receive
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import time
import os
import threading

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

import sys
sys.path.append("../UR5e")
from CustomRobots import *

class UR5e_CollectData(Node):
    def __init__(self):
        super().__init__('collect_rope_data')

        self.UR5e = UR5eCustom()

        self.home_joint_pose = [180, -53.25, 134.66, -171.28, -90, 0]
        self.home_cart_pose = None

        # ati cb
        self.ati_subscription = self.create_subscription(WrenchStamped, '/FT10881', self.ati_callback, 10)
        self.ati_data = None
        self.ati_data_zero = None

        # mocap cb
        self.mocap_subscription = self.create_subscription(RigidBodies, '/rigid_bodies', self.mocap_callback, 10)
        self.mocap_data = None

        # # Image subscriber
        # self.subscription = self.create_subscription(
        #     Image,
        #     'camera/raw_image',
        #     self.image_callback,
        #     1)
        # self.bridge = CvBridge()

        # Create scalable OpenCV window
        # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

        self.ati_data_save = []
        self.mocap_data_save = []
        self.ur5e_cmd_data_save = []
        self.ur5e_tool_data_save = []
        self.ur5e_jointstate_data_save = []

        timer_period = 0.002
        self.timer = self.create_timer(timer_period, self.timer_callback)

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

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # imgpoints, _ = cv2.projectPoints(self.mocap_data, self.calibration["R"], self.calibration["t"], self.calibration["mtx"], self.calibration["dist"])
            # for i in range(imgpoints.shape[0]):
            #     img = cv2.circle(img, (int(imgpoints[i, 0, 0]), int(imgpoints[i, 0, 1])), 3, (0,0,255), -1)

            cv2.imshow("Image", img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error("Error converting image: %s" % str(e))

    def go_to_home(self):
        print('go home')
        q = np.copy(self.home_joint_pose)
        q = np.array(q) * np.pi / 180.0
        self.rtde_c.moveJ(q, 0.2, 0.2)
        self.home_cart_pose = self.rtde_r.getActualTCPPose()

    def take_data(self):
        self.ati_data_save.append(self.ati_data - self.ati_data_zero)
        self.mocap_data_save.append(self.mocap_data)
        self.ur5e_cmd_data_save.append(self.rtde_r.getTargetQ())
        self.ur5e_tool_data_save.append(self.rtde_r.getActualTCPPose())
        self.ur5e_jointstate_data_save.append(self.rtde_r.getActualQ())

    def update_plot(self):
        ati_data_save = np.array(self.mocap_data_save)
        print(ati_data_save.shape)
        self.line.set_xdata(np.arange(ati_data_save.shape[0]))
        self.line.set_ydata(ati_data_save[:, 2])
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def take_data_routine(self):

        # Initialize Matplotlib in interactive mode
        # plt.ion()
        # self.fig, self.ax = plt.subplots()

        # self.line, = self.ax.plot([], [], 'r-')  # Initialize an empty plot
        self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.60")
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.60")

        print("take data")

        N = 3 # take for 3 "levels" of experiments
        # 2^3 = 8
        # 3^3 = 27
        # 4^3 = 64
        # 5^3 = 125
        count = 0

        self.ati_data_save = []
        self.mocap_data_save = []
        self.ur5e_cmd_data_save = []
        self.ur5e_tool_data_save = []
        self.ur5e_jointstate_data_save = []

        for dq1 in np.linspace(-10, 10, N):
            for dq2 in np.linspace(-10, 10, N):
                for dq3 in np.linspace(-12, 12, N):

                    # if count < 50:
                    #     count += 1
                    #     continue

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

                        if not np.any(np.array(self.mocap_data_save)[400:1100, :, :2] == 0): # need to change back to tip
                            break
                        else:
                            print(np.where(np.array(self.mocap_data_save)[400:1100, :, :2] == 0))
                            print('could not find tip')


                        # break

                    # self.update_plot()

                    q0_save = np.array(np.copy(self.home_joint_pose))
                    qf_save = np.array(qf)
                    ur5e_tool_data_save = np.array(self.ur5e_tool_data_save)
                    ur5e_cmd_data_save = np.array(self.ur5e_cmd_data_save)
                    ur5e_jointstate_data_save = np.array(self.ur5e_jointstate_data_save)
                    ati_data_save = np.array(self.ati_data_save)
                    mocap_data_save = np.array(self.mocap_data_save)

                    np.savez("raw_data_N3_2/" + str(count), q0_save=q0_save, qf_save=qf_save, ur5e_tool_data_save=ur5e_tool_data_save, ur5e_cmd_data_save = ur5e_cmd_data_save, ur5e_jointstate_data_save=ur5e_jointstate_data_save, ati_data_save=ati_data_save, mocap_data_save=mocap_data_save)

                    count += 1

        print('done')
        exit()

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

        for i in range(500):
            self.take_data()
            time.sleep(dt)

        self.rtde_c.servoStop()

        self.go_to_home()

    def zero_ati(self):
        ati_data = []
        for _ in range(100):
            if self.ati_data is None: continue 
            ati_data.append(self.ati_data)
            time.sleep(0.01)

        self.ati_data_zero = np.mean(np.array(ati_data), axis = 0)
        print(self.ati_data_zero)

    def reset_rope(self):
        p = np.copy(self.home_cart_pose)
        p[2] -= 0.045
        self.rtde_c.moveL(p, speed = 0.01, acceleration = 0.01)

        time.sleep(0.5)

        p = np.copy(self.home_cart_pose)
        self.rtde_c.moveL(p, speed = 0.005, acceleration = 0.01)
        time.sleep(2)

        # zero ati
        self.zero_ati()

def main(args=None):
    rclpy.init(args=args)

    ur5e = UR5e_CollectData()

    # Use MultiThreadedExecutor to run the node with multiple threads
    executor = MultiThreadedExecutor()
    executor.add_node(ur5e)

    # Start the repeat_data_routine in a separate thread
    ur5e_thread = threading.Thread(target=ur5e.take_data_routine)
    ur5e_thread.start()

    try:
        # Spin the executor to process callbacks
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ur5e.destroy_node()
        rclpy.shutdown()
        ur5e_thread.join()

if __name__ == '__main__':
    main()
