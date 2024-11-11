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
    def __init__(self):
        super().__init__()

        # Load the zero-shot model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()

        self.calibration = np.load("visualization/calibration_data.npz")
        self.mocap_data_goal = None
        self.mocap_data_actual = None

        # Image subscriber
        self.subscription = self.create_subscription(
            Image,
            'camera/raw_image',
            self.image_callback,
            1)
        self.bridge = CvBridge()

        timer_period = 0.002
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Create scalable OpenCV window
        # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    def load_model(self): 
        # Load data using the function
        _, _, _, self.goals, _, _, _, _ = load_data_zeroshot("N2_all", noramlize=False)

        # To load the model
        filepath = '../4_SupervizedLearning/zero_shot/checkpoints_N2_all/final_model_checkpoint.pth'
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

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to a CV2 image
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Optionally: Handle the mocap data for overlaying points (commented out)
            # if self.mocap_data_goal is not None:
            #     imgpoints, _ = cv2.projectPoints(self.mocap_data_goal, self.calibration["R"], self.calibration["t"], self.calibration["mtx"], self.calibration["dist"])
            #     for i in range(imgpoints.shape[0]):
            #         img = cv2.circle(img, (int(imgpoints[i, 0, 0]), int(imgpoints[i, 0, 1])), 3, (255,0,0), -1)

            # if self.mocap_data_actual is not None:
            #     imgpoints, _ = cv2.projectPoints(self.mocap_data_actual, self.calibration["R"], self.calibration["t"], self.calibration["mtx"], self.calibration["dist"])
            #     for i in range(imgpoints.shape[0]):
            #         img = cv2.circle(img, (int(imgpoints[i, 0, 0]), int(imgpoints[i, 0, 1])), 3, (0,0,255), -1)

            # Display the image in a non-blocking way
            cv2.imshow("Image", img)

            # Non-blocking wait, just for 1 millisecond
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error converting or displaying image: {str(e)}")

        print('done')


    def evaluate_zeroshot(self):
        def sample_actions(num_samples=10000):
            qf = np.array([-90, 100, -180], dtype=np.float32)
            random_actions = np.tile(qf, (num_samples, 1))
            random_actions[:, 0] += np.random.rand(num_samples) * 20 - 10
            random_actions[:, 1] += np.random.rand(num_samples) * 20 - 10
            random_actions[:, 2] += np.random.rand(num_samples) * 24 - 12
            return random_actions

        def evaluate_model_on_random_goal(goal, num_samples=10000):
            self.model.eval()
            with torch.no_grad():
                random_actions = sample_actions(num_samples)
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

        for i in range(100):
            self.ati_data_save = []
            self.mocap_data_save = []
            self.ur5e_cmd_data_save = []
            self.ur5e_tool_data_save = []
            self.ur5e_jointstate_data_save = []

            self.mocap_data_goal = self.UR5e.convert_robotpoint_to_world(self.goals[i, :], self.mocap_data[:, 0])

            best_action = evaluate_model_on_random_goal(torch.tensor(self.goals[i, :], dtype=torch.float32).to(self.device))

            q0 = [180, -53.25, 134.66, -171.28, -90, 0]
            qf = [180, -90, 100, -180, -90, 0]
            qf[1] = best_action[0]         
            qf[2] = best_action[1]         
            qf[3] = best_action[2]         

            self.rope_swing(qf)

            plt.figure()
            mocap_data_save_np = np.array(self.mocap_data_save)
            plt.plot(mocap_data_save_np[:, 0, 1], label='data')
            plt.axhline(self.goals[i, 1], color='r', label='goal')
            plt.savefig(os.path.join('../', f'{i}.png'))
            plt.show()

            self.go_to_home()
            error = np.linalg.norm(self.mocap_data_actual - self.mocap_data_goal)
            errors.append(error)

        errors = np.array(errors)
        np.savez_compressed('../zero_shot/evaluation.npz', errors=errors)
        print('done')

def main(args=None):
    rclpy.init(args=args)

    ur5e_eval = UR5e_EvaluateZeroShot()
    executor = MultiThreadedExecutor()
    executor.add_node(ur5e_eval)

    # # Start the repeat_data_routine in a separate thread
    # ur5e_thread = threading.Thread(target=ur5e_eval.evaluate_zeroshot)
    # ur5e_thread.start()

    print('test')

    try:
        executor.spin()
    finally:
        executor.shutdown()
        ur5e_eval.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()