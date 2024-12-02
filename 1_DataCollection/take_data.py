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
from matplotlib import pyplot as plt

class UR5e_CollectData(Node):
    def __init__(self, save_path, N=None):
        super().__init__('collect_rope_data')

        self.UR5e = UR5eCustom()
        self.save_path = save_path  # Path where data will be saved
        self.N = N  # Number of swings to perform

        self.home_joint_pose = [180, -53.25, 134.66, -171.28, -90, 0]
        self.home_cart_pose = None

        # Locks for thread-safe access
        self.ati_lock = threading.Lock()
        self.mocap_lock = threading.Lock()
        self.img_lock = threading.Lock()
        self.video_lock = threading.Lock()

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
        self.frame_width = 1920  # Set width and height based on your camera's resolution
        self.frame_height = 1080
        self.video_writer_error = False  # Flag to track if an error occurs in video writer

        self.calibration = np.load("../visualization/calibration_data.npz")
        self.mocap_data_actual = None

        self.reset_data()

        self.take_data_iter = 0
        timer_period = 0.002
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        pass

    def ati_callback(self, msg):
        with self.ati_lock:
            self.ati_data = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

    def mocap_callback(self, msg):
        with self.mocap_lock:
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
            with self.img_lock:
                self.img = img
        except Exception as e:
            self.get_logger().error("Error converting image: %s" % str(e))

    def project_mocap_to_camera(self, point):
        imgpoints, _ = cv2.projectPoints(point, self.calibration["R"], self.calibration["t"], self.calibration["mtx"], self.calibration["dist"])
        imgpoints = imgpoints[:, 0, :]
        imgpoints = imgpoints.astype(int)
        return imgpoints

    def image_display_thread(self):
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        while True:
            with self.img_lock:
                img = self.img.copy() if self.img is not None else None
            if img is not None:
                if len(self.mocap_data_camera_save) > 0:
                    with self.mocap_lock:
                        points = self.mocap_data_camera_save[-1]
                    for i in range(points.shape[0]):
                        x, y = points[i, :]
                        color = (0, 215, 255)  # Gold color in BGR format
                        img = cv2.line(img, (x - 5, y), (x + 5, y), color, 2)
                        img = cv2.line(img, (x, y - 5), (x, y + 5), color, 2)

                cv2.imshow("Image", img)
                cv2.waitKey(1)
            time.sleep(0.2)

    def save_video_frames(self):
        while True:
            with self.img_lock:
                img = self.img.copy() if self.img is not None else None

            if self.video_saving and img is not None:
                if self.video_writer is None:
                    with self.video_lock:
                        video_file = os.path.join(self.save_path, f"{self.video_count}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.video_writer = cv2.VideoWriter(video_file, fourcc, 20.0, 
                                                            (self.frame_width, self.frame_height))
                        self.video_writer_error = False  # Successfully wrote a frame
                    self.ros_time_camera_save = []

                try:
                    if img.shape[1] != self.frame_width or img.shape[0] != self.frame_height:
                        with self.video_lock:
                            self.video_writer_error = True
                            self.video_saving = False

                    if not self.video_writer_error:
                        with self.video_lock:
                            self.video_writer.write(img)
                        ros_time = self.get_clock().now()
                        ros_time_float = ros_time.nanoseconds * 1e-9
                        self.ros_time_camera_save.append(ros_time_float)

                except Exception as e:
                    self.get_logger().error(f"Error writing frame: {str(e)}")
                    with self.video_lock:
                        self.video_writer_error = True  # Set the error flag to True
                        self.video_saving = False
                    # If an error occurs, break the loop (we'll retry the trial)
                    # break

            time.sleep(1/20)
            
    def go_to_home(self):
        print('go home')
        q = np.copy(self.home_joint_pose)
        q = np.array(q) * np.pi / 180.0
        self.rtde_c.moveJ(q, 0.2, 0.2)
        self.home_cart_pose = self.rtde_r.getActualTCPPose()
        time.sleep(5)

    def take_data(self):
        with self.ati_lock:
            self.ati_data_save[self.take_data_iter, :] = self.ati_data - self.ati_data_zero
        
        with self.mocap_lock:
            mocap_data = np.copy(self.mocap_data)

        self.mocap_data_save[self.take_data_iter, :, :] = mocap_data
        self.mocap_data_camera_save[self.take_data_iter, :, :] = self.project_mocap_to_camera(mocap_data[:3, :].T)

        if not np.any(self.mocap_data[:, [0, 2]] == 0):
            self.mocap_data_robot_save[self.take_data_iter, :] = self.UR5e.convert_workpoint_to_robot(mocap_data[:, 2], mocap_data[:, 0], two_dimention=True)

        self.ur5e_cmd_data_save[self.take_data_iter, :] = self.rtde_r.getTargetQ()
        # self.ur5e_tool_data_save[self.take_data_iter, :] = self.rtde_r.getActualTCPPose()
        self.ur5e_jointstate_data_save[self.take_data_iter, :] = self.rtde_r.getActualQ()

        # Get the current ROS 2 time and store as a float
        ros_time = self.get_clock().now()
        ros_time_float = ros_time.nanoseconds * 1e-9  # Convert nanoseconds to seconds
        self.ros_time_save[self.take_data_iter, 0] = ros_time_float

        self.take_data_iter += 1

    def reset_data(self):
        self.take_data_iter = 0
        # with self.ati_lock:
        self.ati_data_save = np.zeros((1500, 6))

        # with self.mocap_lock:
        self.mocap_data_save = np.zeros((1500, 7, 3))
        self.mocap_data_camera_save = np.zeros((1500, 3, 2))
        self.mocap_data_robot_save = np.zeros((1500, 2))

        self.ur5e_cmd_data_save = np.zeros((1500, 6))
        self.ur5e_tool_data_save = np.zeros((1500, 6))
        self.ur5e_jointstate_data_save = np.zeros((1500, 6))
        self.ros_time_save = np.zeros((1500, 1))

    def take_data_routine(self):
        print("take data")

        count = 0

        self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.60", rt_priority=99)
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.60", rt_priority=99)

        for dq1 in np.linspace(-8, 8, self.N):
            for dq2 in np.linspace(-8, 8, self.N):
                for dq3 in np.linspace(-8, 8, self.N):

                    print(count)

                    # if not (count == 16 or count == 21):
                    #     count += 1
                    # #     continue 
                    # if not count == 5:
                    #     count += 1
                    #     continue 

                    # if count < 21:
                    #     count += 1
                    #     continue 

                    if self.N == 1:
                        dq1 = 0
                        dq2 = 0
                        dq3 = 0

                    for trail in range(10):

                        # Reset data for each trial
                        self.reset_data()

                        time.sleep(0.2)
                        qf = [180, -90, 100, -180, -90, 0]
                        qf = [qf[0], dq1 + qf[1], dq2 + qf[2], dq3 + qf[3], qf[4], 0]

                        self.go_to_home()
                        self.reset_rope()
                        self.go_to_home()

                        with self.video_lock:
                            self.video_count = count
                            self.video_saving = True

                        success = self.rope_swing(qf)

                        self.go_to_home()

                        # Stop video saving at the end of the swing and release video writer
                        with self.video_lock:
                            self.video_saving = False
                            if self.video_writer is not None:
                                self.video_writer.release()
                                self.video_writer = None

                        print(np.array(self.ati_data_save).shape)

                        # print((np.array(self.ati_data_save) + np.array(self.ati_data_zero))[600, :])
                        if np.any(np.isclose((np.array(self.ati_data_save) + np.array(self.ati_data_zero)), 0, atol=1e-8)):
                            print('ati error')
                            return

                        # Check if the video writer had an error, and retry the current trial if necessary
                        if self.video_writer_error:
                            print(f"Error occurred during video saving. Retrying trial {count}...")
                            continue  # Skip to the next trial

                        if not success:
                            print("UR5e error, swing again")
                            continue 

                        if np.any(abs(np.diff(np.array(self.ur5e_jointstate_data_save), axis=0)) > 0.015):
                            print('discontinuous command error')
                            continue 

                        if not np.any(np.array(self.mocap_data_save)[400:1100, :, [0, 2]] == 0):
                            break
                        else:
                            print('could not find a mocap frame, swing again')

                    q0_save = np.array(np.copy(self.home_joint_pose))
                    qf_save = np.array(qf)
                    ur5e_tool_data_save = np.array(self.ur5e_tool_data_save)
                    ur5e_cmd_data_save = np.array(self.ur5e_cmd_data_save)
                    ur5e_jointstate_data_save = np.array(self.ur5e_jointstate_data_save)
                    ati_data_save = np.array(self.ati_data_save)
                    mocap_data_save = np.array(self.mocap_data_save)
                    mocap_data_camera_save = np.array(self.mocap_data_camera_save)
                    mocap_data_robot_save = np.array(self.mocap_data_robot_save)
                    ros_time_save = np.array(self.ros_time_save)
                    ros_time_camera_save = np.array(self.ros_time_camera_save)

                    np.savez(os.path.join(self.save_path, str(count)), 
                             q0_save=q0_save, qf_save=qf_save, ur5e_tool_data_save=ur5e_tool_data_save, 
                             ur5e_cmd_data_save=ur5e_cmd_data_save, ur5e_jointstate_data_save=ur5e_jointstate_data_save, 
                             ati_data_save=ati_data_save, mocap_data_save=mocap_data_save, mocap_data_camera_save=mocap_data_camera_save, mocap_data_robot_save=mocap_data_robot_save,
                             ros_time_save=ros_time_save, ros_time_camera_save=ros_time_camera_save)

                    count += 1

        print('done')
        exit()

    def rope_swing(self, q):

        q0 = np.copy(self.home_joint_pose)
        qf = np.copy(q)

        traj = self.UR5e.create_trajectory(q0, qf, time=1)
        velocity = 3
        acceleration = 5
        dt = 1.0/500
        lookahead_time = 0.1
        gain = 1000
        success = True

        # Get the current ROS 2 time and store as a float
        ros_time = self.get_clock().now()
        ros_time_start = ros_time.nanoseconds * 1e-9  # Convert nanoseconds to seconds

        for _ in range(500):
            self.take_data()
            time.sleep(dt)

        for i in range(traj.shape[1]):
            t_start = self.rtde_c.initPeriod()
            self.rtde_c.servoJ(traj[:, i], velocity, acceleration, dt, lookahead_time, gain)

            self.take_data()

            if (np.linalg.norm(traj[:, i] - self.ur5e_jointstate_data_save[self.take_data_iter-1, :]) > 0.3):
                print(np.linalg.norm(traj[:, i] - self.ur5e_jointstate_data_save[self.take_data_iter-1, :]))
                success = False

            self.rtde_c.waitPeriod(t_start)

        for _ in range(500):
            self.take_data()
            time.sleep(dt)

        # Get the current ROS 2 time and store as a float
        ros_time = self.get_clock().now()
        ros_time_end = ros_time.nanoseconds * 1e-9  # Convert nanoseconds to seconds
        
        print(ros_time_end - ros_time_start)

            # if self.offset is not None and i == self.offset:
            #     self.mocap_data_actual = np.copy(self.mocap_data_save[-1][:3, 2])

        self.rtde_c.servoStop()

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
    save_path = "N4"
    N = 4 # Number of swings

    ur5e = UR5e_CollectData(save_path=save_path, N=N)

    # Use MultiThreadedExecutor to run the node with multiple threads
    executor = MultiThreadedExecutor()
    executor.add_node(ur5e)

    # Start the repeat_data_routine in a separate thread
    data_collection_thread = threading.Thread(target=ur5e.take_data_routine)
    data_collection_thread.start()

    # # # Start the repeat_data_routine in a separate thread
    # video_display_thread = threading.Thread(target=ur5e.image_display_thread)
    # video_display_thread.start()

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