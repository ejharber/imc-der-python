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

sys.path.append("../4_SupervizedLearning")
from load_data import load_data_zeroshot

sys.path.append("../4_SupervizedLearning/zero_shot")
from model_zeroshot import SimpleMLP

import matplotlib.pyplot as plt


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

        # mocap pb
        self.mocap_data_goal = np.array([-1.0, -1.0, -1.0])
        self.mocap_data_actual = np.array([-1.0, -1.0, -1.0])

        timer_period = 0.002
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.calibration = np.load("../demo/calibration_data.npz")

        # Image subscriber
        self.subscription = self.create_subscription(
            Image,
            'camera/raw_image',
            self.image_callback,
            1)
        self.bridge = CvBridge()

        # Create scalable OpenCV window
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

        self.ati_data_save = []
        self.mocap_data_save = []
        self.ur5e_cmd_data_save = []
        self.ur5e_tool_data_save = []
        self.ur5e_jointstate_data_save = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()

    def load_model(self): 
        # Load data using the function
        _, _, _, self.goals, data_mean, data_std, labels_mean, labels_std = load_data_zeroshot("../3_ExpandDataSet/raw_data", noramlize=False)

        # To load the model
        filepath = '../4_SupervizedLearning/zero_shot/model_checkpoint_50.pth'
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

    def timer_callback(self):
        pass

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

            imgpoints, _ = cv2.projectPoints(self.mocap_data_goal, self.calibration["R"], self.calibration["t"], self.calibration["mtx"], self.calibration["dist"])
            for i in range(imgpoints.shape[0]):
                # if imgpoints[i, 0, 0] < 0 or imgpoints[i, 0, 1] < 0: continue 
                # if imgpoints[i, 0, 0] > img.shape[0] or imgpoints[i, 0, 1] > img.shape[1]: continue 
                img = cv2.circle(img, (int(imgpoints[i, 0, 0]), int(imgpoints[i, 0, 1])), 3, (255,0,0), -1)

            imgpoints, _ = cv2.projectPoints(self.mocap_data_actual, self.calibration["R"], self.calibration["t"], self.calibration["mtx"], self.calibration["dist"])
            for i in range(imgpoints.shape[0]):
                # if imgpoints[i, 0, 0] < 0 or imgpoints[i, 0, 1] < 0: continue 
                # if imgpoints[i, 0, 0] > img.shape[0] or imgpoints[i, 0, 1] > img.shape[1]: continue 
                img = cv2.circle(img, (int(imgpoints[i, 0, 0]), int(imgpoints[i, 0, 1])), 3, (0,0,255), -1)

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

        # self.ati_data_save.append(self.ati_data - self.ati_data_zero)
        self.mocap_data_save.append(self.mocap_data)
        self.ur5e_cmd_data_save.append(self.rtde_r.getTargetQ())
        self.ur5e_tool_data_save.append(self.rtde_r.getActualTCPPose())
        self.ur5e_jointstate_data_save.append(self.rtde_r.getActualQ())

    def evaulate_zeroshot(self):
        # Function to sample 10,000 actions
        def sample_actions(num_samples=10000):
            qf = np.array([-90, 100, -180], dtype=np.float32)
            random_actions = np.tile(qf, (num_samples, 1))
            random_actions[:, 0] += np.random.rand(num_samples) * 20 - 10
            random_actions[:, 1] += np.random.rand(num_samples) * 20 - 10
            random_actions[:, 2] += np.random.rand(num_samples) * 24 - 12
            return random_actions

        # Evaluate model on randomly sampled goal and actions
        def evaluate_model_on_random_goal(goal, num_samples=10000):
            self.model.eval()
            with torch.no_grad():
                
                # Sample 10000 random actions
                random_actions = sample_actions(num_samples)
                random_actions = torch.tensor(random_actions, dtype=torch.float32).to(self.device)

                # Predict goals for each random action
                predicted_goals = self.model(random_actions, test=True)

                # Calculate distances between each predicted goal and the random goal
                distances = torch.norm(predicted_goals - goal, dim=1)

                # Find the index of the minimum distance
                min_distance_idx = torch.argmin(distances)

                # Get the best action
                best_action = random_actions[min_distance_idx]

                return best_action.cpu().numpy()
                
        # Loop to evaluate the model 100 times
        errors = []
        self.go_to_home()

        for i in range(100):

            self.ati_data_save = []
            self.mocap_data_save = []
            self.ur5e_cmd_data_save = []
            self.ur5e_tool_data_save = []
            self.ur5e_jointstate_data_save = []

            self.mocap_data_goal = self.UR5e.convert_robotpoint_to_world(self.goals[i, :], self.mocap_data[:, 0])
            print(self.mocap_data_goal, self.mocap_data[:, 0], self.goals[i, :])
            self.mocap_data_actual = np.array([-1.0, -1.0, -1.0])

            best_action = evaluate_model_on_random_goal(torch.tensor(self.goals[i, :], dtype=torch.float32).to(self.device))
            
            q0 = [180, -53.25, 134.66, -171.28, -90, 0]
            qf = [180, -90, 100, -180, -90, 0]
            qf[1] = best_action[0]         
            qf[2] = best_action[1]         
            qf[3] = best_action[2]         

            self.rope_swing(qf)

            # self.q_save = np.array(self.q_save)
            # self.ur5e_cmd_data_save = np.array(self.ur5e_cmd_data_save)
            # self.ur5e_jointstate_data_save = np.array(self.ur5e_jointstate_data_save)
            # self.ati_data_save = np.array(self.ati_data_save)
            self.mocap_data_save = np.array(self.mocap_data_save)

            # print(self.mocap_data_save.shape)
            plt.figure('Rope Trajectory')
            plt.plot(self.mocap_data_save[:, 0, 2], 'r-')
            plt.plot(self.mocap_data_save[:, 1, 2], 'g-')
            plt.plot(self.mocap_data_save[:, 2, 2], 'b-')
            plt.plot(self.mocap_data_actual[0], 'r.')
            plt.plot(self.mocap_data_actual[1], 'g.')
            plt.plot(self.mocap_data_actual[2], 'b.')

            plt.show()
            # errors.append(error)

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

        for i in range(76):
            self.take_data()
            time.sleep(dt)

        self.mocap_data_actual = np.copy(self.mocap_data[:3, 2])    
        print(self.mocap_data_actual)

        for i in range(500):
            self.take_data()
            time.sleep(dt)

        self.rtde_c.servoStop()

        self.go_to_home()

    def reset_rope(self):
        p = np.copy(self.home_cart_pose)
        p[2] -= 0.04
        self.rtde_c.moveL(p, speed = 0.01, acceleration = 0.01)

        time.sleep(0.5)

        p = np.copy(self.home_cart_pose)
        self.rtde_c.moveL(p, speed = 0.002, acceleration = 0.01)
        time.sleep(2)

    def __del__(self):
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()

def main(args=None):
    rclpy.init(args=args)

    ur5e = UR5e_CollectData()

    # Use MultiThreadedExecutor to run the node with multiple threads
    executor = MultiThreadedExecutor()
    executor.add_node(ur5e)

    # Start the repeat_data_routine in a separate thread
    ur5e_thread = threading.Thread(target=ur5e.evaulate_zeroshot)
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