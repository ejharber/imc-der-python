import faulthandler

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import torch
import torch.nn as nn
import torch.optim as optim

import rtde_control
import rtde_receive
import numpy as np

from geometry_msgs.msg import PoseStamped
from mocap4r2_msgs.msg import RigidBodies

import time
import os
import threading
import sys
from scipy.spatial import distance, ConvexHull
from deap import base, creator, tools, algorithms  # Import DEAP components

sys.path.append("../UR5e")
from CustomRobots import *

from take_zeroshot_data import UR5e_CollectData

import matplotlib.pyplot as plt

class UR5e_IterativeValidation(UR5e_CollectData):
    def __init__(self, save_path, N):
        super().__init__(save_path)

        self.N = N
        self.num_iterations = 5

    def save_video_frames(self):
        while True:
            with self.img_lock:
                img = self.img.copy() if self.img is not None else None

            if self.video_saving and img is not None:
                if self.video_writer is None:
                    with self.video_lock:
                        video_file = os.path.join(self.save_path, f"{self.video_count}_{self.iteration}.mp4")
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

    def take_iterative_validation_data(self):

        self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.60", rt_priority=99)
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.60", rt_priority=99)

        print('reset')
        self.reset_data()

        print('start home')

        self.go_to_home()

        print('done home')

        count = 0

        for dq1 in np.linspace(-8, 8, self.N):
            for dq2 in np.linspace(-8, 8, self.N):
                for dq3 in np.linspace(-8, 8, self.N):

                    # if not (count == 16 or count == 21):
                    #     count += 1
                    # #     continue 
                    # if not count == 5:
                    #     count += 1
                    #     continue 

                    if count < 60:
                        count += 1
                        continue 

                    if self.N == 1:
                        dq1 = 0
                        dq2 = 0
                        dq3 = 0

                    for iteration in range(self.num_iterations):

                        # Reset data for each trial
                        self.reset_data()

                        time.sleep(0.2)
                        qf = np.array([180.0, -100.0, 100.0, -180.0, -90.0, 0.0])
                        qf[1:4] += np.array([dq1, dq2, dq3])

                        print(count, iteration)

                        if not iteration == 0:

                            # best_deltaaction = evaulate_iterative_model(torch.tensor(self.goals[count, :], dtype=torch.float32).to(self.device))
                            qf[1:4] += np.random.uniform(-2, 2, (3, ))

                        for trial in range(10):

                            if not self.rtde_c.isConnected() or not self.rtde_r.isConnected():
                                self.rtde_c.disconnect()
                                self.rtde_r.disconnect()
                                time.sleep(5)
                                self.rtde_c.reconnect()
                                self.rtde_r.reconnect()

                            self.reset_data()

                            time.sleep(0.2)

                            self.go_to_home()
                            self.reset_rope()
                            self.go_to_home()

                            with self.video_lock:
                                self.iteration = iteration
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
                                # plt.plot(np.array(self.ur5e_jointstate_data_save))
                                # plt.show()
                                print('discontinuous command error')
                                continue 

                            if not np.any(np.array(self.mocap_data_save)[100:700, :, [0, 2]] == 0):
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

                        np.savez(os.path.join(self.save_path, str(count) + "_" + str(iteration)), 
                                 q0_save=q0_save, qf_save=qf_save, ur5e_tool_data_save=ur5e_tool_data_save, 
                                 ur5e_cmd_data_save=ur5e_cmd_data_save, ur5e_jointstate_data_save=ur5e_jointstate_data_save, 
                                 ati_data_save=ati_data_save, mocap_data_save=mocap_data_save, mocap_data_camera_save=mocap_data_camera_save, mocap_data_robot_save=mocap_data_robot_save,
                                 ros_time_save=ros_time_save, ros_time_camera_save=ros_time_camera_save)

                    count += 1


        print("done")

def main(args=None):
    # Enable the fault handler
    faulthandler.enable()

    rclpy.init(args=args)

    # Parameters: save path and N (number of swings)
    save_path = "N4_2_iter"
    N = 4 # Number of swings

    os.makedirs(save_path, exist_ok=True)

    ur5e = UR5e_IterativeValidation(save_path=save_path, N=N)

    executor = MultiThreadedExecutor()
    executor.add_node(ur5e)

    evaluate_iterative_thread = threading.Thread(target=ur5e.take_iterative_validation_data)
    evaluate_iterative_thread.start()

    # video_display_thread = threading.Thread(target=ur5e.image_display_thread)
    # video_display_thread.start()

    video_saving_thread = threading.Thread(target=ur5e.save_video_frames)
    video_saving_thread.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ur5e.destroy_node()
        rclpy.shutdown()
        evaluate_iterative_thread.join()
        video_display_thread.join()
        video_saving_thread.join()

if __name__ == '__main__':
    main()
