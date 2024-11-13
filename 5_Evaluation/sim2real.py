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

sys.path.append("../1_DataCollection")
from take_data import UR5e_CollectData

sys.path.append("../4_SupervizedLearning")
from load_data import load_data_zeroshot

sys.path.append("../4_SupervizedLearning/zero_shot")
from model_zeroshot import SimpleMLP

import matplotlib.pyplot as plt

class UR5e_EvaluateZeroShot(UR5e_CollectData):
    def __init__(self, save_path, model_file="N2_all", num_samples=10000):
        super().__init__(save_path)
        self.model_file = model_file  # Model file name parameter
        self.num_samples = num_samples  # Number of samples for evaluation

        # Load the zero-shot model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()

        self.calibration = np.load("../visualization/calibration_data.npz")
        self.mocap_data_goal = None

    def image_display_thread(self):
        # Create scalable OpenCV window
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        while True:
            img = self.img.copy() if self.img is not None else None
            if img is not None:

                if self.mocap_data_goal is not None:
                    mocap_data_goal = np.copy(self.mocap_data_goal)
                    imgpoints, _ = cv2.projectPoints(mocap_data_goal, self.calibration["R"], self.calibration["t"], self.calibration["mtx"], self.calibration["dist"])
                    for i in range(imgpoints.shape[0]):
                        x, y = int(imgpoints[i, 0, 0]), int(imgpoints[i, 0, 1])
                        color = (0, 255, 0)  
                        img = cv2.line(img, (x - 5, y), (x + 5, y), color, 2)
                        img = cv2.line(img, (x, y - 5), (x, y + 5), color, 2)

                if self.mocap_data_actual is not None:
                    mocap_data_actual = np.copy(self.mocap_data_actual)
                    imgpoints, _ = cv2.projectPoints(mocap_data_actual, self.calibration["R"], self.calibration["t"], self.calibration["mtx"], self.calibration["dist"])
                    for i in range(imgpoints.shape[0]):
                        x, y = int(imgpoints[i, 0, 0]), int(imgpoints[i, 0, 1])
                        color = (0, 215, 255)
                        img = cv2.line(img, (x - 5, y), (x + 5, y), color, 2)
                        img = cv2.line(img, (x, y - 5), (x, y + 5), color, 2)

                cv2.imshow("Image", img)
                cv2.waitKey(1)
            time.sleep(0.2)

    def load_model(self): 
        _, _, _, self.goals, _, _, _, _ = load_data_zeroshot(self.model_file, noramlize=False)

        filepath = f'../4_SupervizedLearning/zero_shot/checkpoints_{self.model_file}/final_model_checkpoint.pth'
        checkpoint = torch.load(filepath)
        self.model = SimpleMLP(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            output_size=checkpoint['output_size'],
            data_mean=checkpoint['data_mean'].to(self.device), 
            data_std=checkpoint['data_std'].to(self.device), 
            labels_mean=checkpoint['labels_mean'].to(self.device), 
            labels_std=checkpoint['labels_std'].to(self.device)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)

    def evaluate_zeroshot(self):
        def sample_actions(num_samples=10000):
            qf = np.array([-90, 100, -180], dtype=np.float32)
            random_actions = np.tile(qf, (num_samples, 1))
            random_actions[:, 0] += np.random.rand(num_samples) * 16 - 8
            random_actions[:, 1] += np.random.rand(num_samples) * 16 - 8
            random_actions[:, 2] += np.random.rand(num_samples) * 16 - 8
            return random_actions

        def evaluate_model_on_random_goal(goal):
            self.model.eval()
            with torch.no_grad():
                random_actions = sample_actions()
                random_actions = torch.tensor(random_actions, dtype=torch.float32).to(self.device)
                predicted_goals = self.model(random_actions, test=True)
                distances = torch.norm(predicted_goals - goal, dim=1)
                min_distance_idx = torch.argmin(distances)
                best_action = random_actions[min_distance_idx]
                return best_action.cpu().numpy()

        self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.60")
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.60")

        errors = []
        self.go_to_home()

        for count in range(self.num_samples):

            print(count)

            if count < 27: continue 

            self.mocap_data_goal = self.UR5e.convert_robotpoint_to_world(self.goals[count, :], self.mocap_data[:, 0])
            best_action = evaluate_model_on_random_goal(torch.tensor(self.goals[count, :], dtype=torch.float32).to(self.device))

            q0 = [180, -53.25, 134.66, -171.28, -90, 0]
            qf = [180, -90, 100, -180, -90, 0]
            qf[1], qf[2], qf[3] = best_action

            for trial in range(10):
                self.mocap_data_actual = None
                self.ati_data_save, self.mocap_data_save = [], []
                self.ur5e_cmd_data_save, self.ur5e_tool_data_save, self.ur5e_jointstate_data_save = [], [], []

                self.video_count = count
                self.video_saving = True
                self.rope_swing(qf)
                self.video_saving = False
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None

                if not np.any(np.array(self.mocap_data_save)[400:1100, :, :2] == 0):
                    break
                else:
                    print('could not find tip')

            q0_save, qf_save = np.array(self.home_joint_pose), np.array(qf)
            ur5e_tool_data_save, ur5e_cmd_data_save = np.array(self.ur5e_tool_data_save), np.array(self.ur5e_cmd_data_save)
            ur5e_jointstate_data_save, ati_data_save, mocap_data_save = np.array(self.ur5e_jointstate_data_save), np.array(self.ati_data_save), np.array(self.mocap_data_save)

            np.savez(os.path.join(self.save_path, str(count)), 
                     q0_save=q0_save, qf_save=qf_save, ur5e_tool_data_save=ur5e_tool_data_save, 
                     ur5e_cmd_data_save=ur5e_cmd_data_save, ur5e_jointstate_data_save=ur5e_jointstate_data_save, 
                     ati_data_save=ati_data_save, mocap_data_save=mocap_data_save)

        print("done")

def main(args=None):
    rclpy.init(args=args)

    save_path = "pose"
    model_file = "N2_all"
    num_samples = 100

    ur5e = UR5e_EvaluateZeroShot(save_path=save_path, model_file=model_file, num_samples=num_samples)

    executor = MultiThreadedExecutor()
    executor.add_node(ur5e)

    evaluate_zeroshot_thread = threading.Thread(target=ur5e.evaluate_zeroshot)
    evaluate_zeroshot_thread.start()

    video_display_thread = threading.Thread(target=ur5e.image_display_thread)
    video_display_thread.start()

    video_saving_thread = threading.Thread(target=ur5e.save_video_frames)
    video_saving_thread.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ur5e.destroy_node()
        rclpy.shutdown()
        evaluate_zeroshot_thread.join()
        video_display_thread.join()
        video_saving_thread.join()

if __name__ == '__main__':
    main()
