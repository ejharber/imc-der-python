#!/usr/bin/env python3

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

class RobotiqCalibration(Node):
    def __init__(self):

        super().__init__('collect_robotiq_data')

        # i think this is all the way open but im not sure, should double check
        self.home_joint_pose = [0, 0, 0, 0] # use to home robotiq

        # robotiq cb
        # you should be subscribing to feedback (that is what is comming from the robotiq)
        # command is how you send data to the robotiq
        self.robotiq_subscription = self.create_subscription(JointState, '/robotiq/feedback', self.robotiq_callback, 10) # todo
        # self.robotiq_subscription = self.create_subscription(JointState, '/robotiq/command', self.robotiq_callback, 10) # todo
        self.robotiq_data = np.zeros(4)

        # mocap cb
        self.mocap_subscription = self.create_subscription(RigidBodies, '/rigid_bodies', self.mocap_callback, 10)
        self.mocap_data = np.zeros(12)

        # encoder cb
        # self.encoder_subscription = self.create_subscription() # todo
        self.encoder_data = np.zeros(12)

        # created the publisher to command the robotiq
        self.robotiq_publisher = self.create_publisher(JointState, '/robotiq/command', 10)

        self.mocap_data_save = []
        self.robotiq_data_save = []
        self.encoder_data_save = []

        time.sleep(1)
        self.go_to_home()

    def go_to_home(self):
        msg = JointState()
        msg.position = self.home_joint_pose
        self.robotiq_publisher.publish(msg)

    def mocap_callback(self, msg):
        # change these to finger positions
        print(msg)
        # self.mocap_data = np.zeros((7, 9))
        # for rigid_body in msg.rigidbodies:

        #     i = None

        #     if rigid_body.rigid_body_name == "robotiqFinger_leftBot.robotiqFinger_leftBot":
        #         i = 0

        #     if rigid_body.rigid_body_name == "robotiqFinger_leftMid.robotiqFinger_leftMid":
        #         i = 1

        #     if rigid_body.rigid_body_name == "robotiqFinger_leftTop.robotiqFinger_leftTop":
        #         i = 2

        #     if rigid_body.rigid_body_name == "robotiqFinger_rightBot.robotiqFinger_rightBot":
        #         i = 3

        #     if rigid_body.rigid_body_name == "robotiqFinger_rightMid.robotiqFinger_rightMid":
        #         i = 4

        #     if rigid_body.rigid_body_name == "robotiqFinger_rightTop.robotiqFinger_rightTop":
        #         i = 5

        #     if rigid_body.rigid_body_name == "robotiqFinger_centerBot.robotiqFinger_centerBot":
        #         i = 6

        #     if rigid_body.rigid_body_name == "robotiqFinger_centerMid.robotiqFinger_centerMid":
        #         i = 7

        #     if rigid_body.rigid_body_name == "robotiqFinger_centerTop.robotiqFinger_centerTop":
        #         i = 8

        #     if i is None: continue

        #     self.mocap_data[0, i] = rigid_body.pose.position.x    
        #     self.mocap_data[1, i] = rigid_body.pose.position.y    
        #     self.mocap_data[2, i] = rigid_body.pose.position.z    
        #     self.mocap_data[3, i] = rigid_body.pose.orientation.w    
        #     self.mocap_data[4, i] = rigid_body.pose.orientation.x    
        #     self.mocap_data[5, i] = rigid_body.pose.orientation.y  
        #     self.mocap_data[6, i] = rigid_body.pose.orientation.z
        return

    def robotiq_callback(self, msg):
        # we already know how this is packaged :)
        self.robotiq_data = np.array(msg.position)

    def encoder_callback(self, msg):
        # this should be similar to robotiq, Andrew should use the joint state msg type
        self.encoder_data = np.array(msg.position)
        return

    def take_data(self):
        rclpy.spin_once(self)

        if np.any(self.mocap_data == 0):
            print("no new mocap data received")

        if np.any(self.robotiq_data == 0):
            print("no new robotiq data")

        if np.any(self.encoder_data == 0):
            print("no new encoder data")

        self.mocap_data_save.append(self.mocap_data)
        self.robotiq_data_save.append(self.robotiq_data)
        self.encoder_data_save.append(self.encoder_data)

    def take_data_routine(self):
        # move gripper
        msg = JointState()
       
        for joint_number in range(4):
            self.go_to_home()
            for joint_angle in range(1, 255): # can have larger steps between positions
                msg = JointState()
                msg.position[joint_number] = joint_angle
                self.robotiq_publisher.publish(msg)
                rclpy.spin_once(self)
                time.sleep(0.01)
                self.take_data()

        np.savez("robotiq_calibration", mocap_data=mocap_data_save, robotiq_data=robotiq_data_save, encoder_data=encoder_data_save)

def main(args=None):
    rclpy.init(args=args)

    RobotiqCalibrationNode = RobotiqCalibration()

    RobotiqCalibrationNode.take_data_routine()

    RobotiqCalibrationNode.destroy_node() # this line is optional 
    rclpy.shutdown()

if __name__ == '__main__':
    main()