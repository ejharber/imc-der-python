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
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import PoseStamped
from mocap4r2_msgs.msg import RigidBodies

import time
import os
import threading
import sys
sys.path.append("../UR5e")
from CustomRobots import *

class UR5e_CollectData(Node):
    def __init__(self, save_path, N=None):
        super().__init__('collect_rope_data')

        self.UR5e = UR5eCustom()
        self.save_path = save_path  # Path where data will be saved
        self.N = N  # Number of swings to perform

        self.home_joint_pose = [180, -53.25, 134.66, -171.28, -90, 0]
        self.home_cart_pose = None

        # ati cb
        self.ati_subscription = self.create_subscription(WrenchStamped, '/FT10881', self.ati_callback, 10)
        self.ati_data = None
        self.ati_data_zero = None

        # mocap cb
        self.mocap_subscription = self.create_subscription(RigidBodies, '/rigid_bodies', self.mocap_callback, 10)
        self.mocap_data = None

        # Image subscriber
        self.subscription = self.create_subscription(Image, '/camera/raw_image', self.image_callback, 10)
        self.bridge = CvBridge()
        self.img = None
        self.mocap_data_actual = None
        self.offset = 0

        self.video_saving = False  # Flag to control video saving state
        self.video_writer = None  # Video writer for saving frames
        self.frame_width = 640  # Set width and height based on your camera's resolution
        self.frame_height = 480
        self.save_path = save_path  # Directory to save video files

        self.calibration = np.load("../visualization/calibration_data.npz")
        self.mocap_data_actual = None

        self.ati_data_save = []
        self.mocap_data_save = []
        self.ur5e_cmd_data_save = []
        self.ur5e_tool_data_save = []
        self.ur5e_jointstate_data_save = []
        self.ros_time_save = []
        self.ros_time_camera_save = []

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
            self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error("Error converting image: %s" % str(e))

    def image_display_thread(self):
        # Create scalable OpenCV window
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        while True:
            img = self.img.copy() if self.img is not None else None  # Copy self.img to local img
            if img is not None:
                if self.mocap_data_actual is not None:
                    mocap_data_actual = np.copy(self.mocap_data_actual)
                    imgpoints, _ = cv2.projectPoints(mocap_data_actual, self.calibration["R"], self.calibration["t"], self.calibration["mtx"], self.calibration["dist"])
                    for i in range(imgpoints.shape[0]):
                        # Get the point coordinates
                        x, y = int(imgpoints[i, 0, 0]), int(imgpoints[i, 0, 1])
                        
                        # Draw a plus sign with gold color
                        color = (0, 215, 255)  # Gold color in BGR format

                        # Draw horizontal line
                        img = cv2.line(img, (x - 5, y), (x + 5, y), color, 2)
                        # Draw vertical line
                        img = cv2.line(img, (x, y - 5), (x, y + 5), color, 2)

                cv2.imshow("Image", img)
                cv2.waitKey(1)
            time.sleep(0.2)

    def save_video_frames(self):

        while True:
            img = self.img.copy() if self.img is not None else None  # Copy self.img to local img
            if self.video_saving and img is not None:
                if self.video_writer is None:
                    self.ros_time_camera_save = []
                    # Initialize the VideoWriter with output path and parameters
                    video_file = os.path.join(self.save_path, f"{self.video_count}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(video_file, fourcc, 30.0, 
                                                        (self.frame_width, self.frame_height))
                
                # Write the frame to the video file
                self.video_writer.write(img)
                # Get the current ROS 2 time and store as a float
                ros_time = Clock().now()
                ros_time_float = ros_time.nanoseconds * 1e-9  # Convert nanoseconds to seconds
                self.ros_time_camera_save.append(ros_time_float)
                
            time.sleep(0.05)  # Adjust based on desired frame rate

    def go_to_home(self):
        print('go home')
        q = np.copy(self.home_joint_pose)
        q = np.array(q) * np.pi / 180.0
        self.rtde_c.moveJ(q, 0.2, 0.2)
        self.home_cart_pose = self.rtde_r.getActualTCPPose()

    def take_data(self):
        self.ati_data_save.append(self.ati_data - self.ati_data_zero)
        self.mocap_data_save.append(self.mocap_data)
        self.mocap_data_2d.append() # TODO
        self.ur5e_cmd_data_save.append(self.rtde_r.getTargetQ())
        self.ur5e_tool_data_save.append(self.rtde_r.getActualTCPPose())
        self.ur5e_jointstate_data_save.append(self.rtde_r.getActualQ())

        # Get the current ROS 2 time and store as a float
        ros_time = Clock().now()
        ros_time_float = ros_time.nanoseconds * 1e-9  # Convert nanoseconds to seconds
        self.ros_time_save.append(ros_time_float)

    def take_data_routine(self):
        print("take data")

        self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.60")
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.60")

        count = 0
        self.ati_data_save = []
        self.mocap_data_save = []
        self.ur5e_cmd_data_save = []
        self.ur5e_tool_data_save = []
        self.ur5e_jointstate_data_save = []
        self.ros_time_save = []

        for dq1 in np.linspace(-8, 8, self.N):
            for dq2 in np.linspace(-8, 8, self.N):
                for dq3 in np.linspace(-8, 8, self.N):

                    # if not (count == 16 or count == 21):
                    #     count += 1
                    #     continue 

                    if count < 49:
                        count += 1
                        continue 

                    if self.N == 1:
                        dq1 = 0
                        dq2 = 0
                        dq3 = 0

                    for trail in range(10):
                        self.ati_data_save = []
                        self.mocap_data_save = []
                        self.ur5e_cmd_data_save = []
                        self.ur5e_tool_data_save = []
                        self.ur5e_jointstate_data_save = []

                        time.sleep(0.2)
                        qf = [180, -90, 100, -180, -90, 0]
                        qf = [qf[0], dq1 + qf[1], dq2 + qf[2], dq3 + qf[3], qf[4], 0]

                        # Start the video saving at the beginning of the swing
                        self.video_count = count  # Track video sequence
                        self.video_saving = True  # Start saving images to video

                        success = self.rope_swing(qf)

                        # Stop video saving at the end of the swing and release video writer
                        self.video_saving = False
                        if self.video_writer is not None:
                            self.video_writer.release()
                            self.video_writer = None

                        if not np.any(np.array(self.mocap_data_save)[400:1100, :, :2] == 0):  # Check for tip visibility
                            break
                        else:
                            print('could not find tip')

                    q0_save = np.array(np.copy(self.home_joint_pose))
                    qf_save = np.array(qf)
                    ur5e_tool_data_save = np.array(self.ur5e_tool_data_save)
                    ur5e_cmd_data_save = np.array(self.ur5e_cmd_data_save)
                    ur5e_jointstate_data_save = np.array(self.ur5e_jointstate_data_save)
                    ati_data_save = np.array(self.ati_data_save)
                    mocap_data_save = np.array(self.mocap_data_save)
                    ros_time_save = np.array(self.ros_time_save)
                    ros_time_camera_save = np.array(self.ros_time_camera_save)

                    np.savez(os.path.join(self.save_path, str(count)), 
                             q0_save=q0_save, qf_save=qf_save, ur5e_tool_data_save=ur5e_tool_data_save, 
                             ur5e_cmd_data_save=ur5e_cmd_data_save, ur5e_jointstate_data_save=ur5e_jointstate_data_save, 
                             ati_data_save=ati_data_save, mocap_data_save=mocap_data_save, 
                             ros_time_save=ros_time_save, ros_time_camera_save=ros_time_camera_save)

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
        velocity = 3
        acceleration = 5
        dt = 1.0/500
        lookahead_time = 0.1
        gain = 1000
        success = True

        for i in range(500):
            self.take_data()
            time.sleep(dt)

        for i in range(traj.shape[1]):
            t_start = self.rtde_c.initPeriod()
            q = traj[:, i]
            self.rtde_c.servoJ(q, velocity, acceleration, dt, lookahead_time, gain)
            self.take_data()
            print(q, self.self.ur5e_jointstate_data_save[-1])
            self.rtde_c.waitPeriod(t_start)

        for i in range(500):
            self.take_data()
            time.sleep(dt)

            if self.offset is not None and i == self.offset:
                self.mocap_data_actual = np.copy(self.mocap_data_save[-1][:3, 2])

        self.rtde_c.servoStop()
        self.go_to_home()

        return success

    def zero_ati(self):
        ati_data = []
        for _ in range(100):
            if self.ati_data is None: continue 
            ati_data.append(self.ati_data)
            time.sleep(0.01)

        self.ati_data_zero = np.mean(np.array(ati_data), axis=0)

    def reset_rope(self):
        p = np.copy(self.home_cart_pose)
        p[2] -= 0.045
        self.rtde_c.moveL(p, speed=0.01, acceleration=0.01)
        time.sleep(0.5)
        p = np.copy(self.home_cart_pose)
        self.rtde_c.moveL(p, speed=0.005, acceleration=0.01)
        time.sleep(2)
        self.zero_ati()

def main(args=None):
    rclpy.init(args=args)

    # Parameters: save path and N (number of swings)
    save_path = "raw_data_N4_ns"
    N = 4  # Number of swings

    ur5e = UR5e_CollectData(save_path=save_path, N=N)

    # Use MultiThreadedExecutor to run the node with multiple threads
    executor = MultiThreadedExecutor()
    executor.add_node(ur5e)

    # Start the repeat_data_routine in a separate thread
    data_collection_thread = threading.Thread(target=ur5e.take_data_routine)
    data_collection_thread.start()

    # Start the repeat_data_routine in a separate thread
    video_display_thread = threading.Thread(target=ur5e.image_display_thread)
    video_display_thread.start()

    # Start the video_saving_thread upon initialization
    video_saving_thread = threading.Thread(target=ur5e.save_video_frames)
    video_saving_thread.start()

    try:
        # Spin the executor to process callbacks
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ur5e.destroy_node()
        rclpy.shutdown()
        data_collection_thread.join()
        video_display_thread.join()
        video_saving_thread.join()

if __name__ == '__main__':
    main()