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

sys.path.append("../1_DataCollection")
from take_data import UR5e_CollectData

sys.path.append("../4_SupervizedLearning")
from load_data import load_data_zeroshot

sys.path.append("../4_SupervizedLearning/zero_shot")
from model_zeroshot import SimpleMLP

sys.path.append("../4_SupervizedLearning/iterative")
from model_iterative import LSTMMLPModel

import matplotlib.pyplot as plt

class UR5e_EvaluateIterative(UR5e_CollectData):
    def __init__(self, save_path, zeroshot_model_file="N2_all", iterative_model_file="N2_all", params_file="N3_all", num_samples=10000):
        super().__init__(save_path)
        self.zeroshot_model_file = zeroshot_model_file  # Model file name parameter
        self.iterative_model_file = iterative_model_file  # Model file name parameter        

        file_path = os.path.dirname(os.path.realpath(__file__)) + "/../2_SysID/params/"
        self.params = np.load(os.path.join(file_path, params_file) + ".npz")["params"]

        self.num_samples = num_samples  # Number of samples for evaluation
        self.num_iterations = 5

        # Load the zero-shot model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')        
        self.load_model()
        print('models loaded')

        self.calibration = np.load("../visualization/calibration_data.npz")
        self.mocap_data_goal = None

    def reset_data(self):
        super().reset_data()

        self.goal_mocap_save = None
        self.goal_robot_save = None
        self.goal_camera_save = None

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

    def load_model(self):
        # Load data for evaluation
        _, _, _, self.goals, _, _, _, _ = load_data_zeroshot("interp", normalize=False)

        # Load Zero-Shot Model
        zeroshot_filepath = f'../4_SupervizedLearning/zero_shot/checkpoints_{self.zeroshot_model_file}/final_model_checkpoint.pth'
        zeroshot_checkpoint = torch.load(zeroshot_filepath)
        self.zeroshot_model = SimpleMLP(
            input_size=zeroshot_checkpoint['input_size'],
            hidden_size=zeroshot_checkpoint['hidden_size'],
            num_layers=zeroshot_checkpoint['num_layers'],
            output_size=zeroshot_checkpoint['output_size'],
            data_mean=zeroshot_checkpoint['data_mean'].to(self.device),
            data_std=zeroshot_checkpoint['data_std'].to(self.device),
            labels_mean=zeroshot_checkpoint['labels_mean'].to(self.device),
            labels_std=zeroshot_checkpoint['labels_std'].to(self.device)
        )
        self.zeroshot_model.load_state_dict(zeroshot_checkpoint['model_state_dict'])
        self.zeroshot_model = self.zeroshot_model.to(self.device)

        # Load Iterative Model
        iterative_filepath = f'../4_SupervizedLearning/iterative/checkpoints_{self.iterative_model_file}/final_model_checkpoint.pth'
        iterative_checkpoint = torch.load(iterative_filepath)
        self.iterative_model = LSTMMLPModel(
            input_size_lstm=iterative_checkpoint['input_size_lstm'],
            input_size_classic=iterative_checkpoint['input_size_classic'],
            hidden_size_lstm=iterative_checkpoint['hidden_size_lstm'],
            hidden_size_mlp=iterative_checkpoint['hidden_size_mlp'],
            num_layers_lstm=iterative_checkpoint['num_layers_lstm'],
            num_layers_mlp=iterative_checkpoint['num_layers_mlp'],
            output_size=iterative_checkpoint['output_size'],
            delta_actions_mean=iterative_checkpoint['delta_actions_mean'],
            delta_actions_std=iterative_checkpoint['delta_actions_std'],
            delta_goals_mean=iterative_checkpoint['delta_goals_mean'],
            delta_goals_std=iterative_checkpoint['delta_goals_std'],
            traj_pos_mean=iterative_checkpoint['traj_pos_mean'],
            traj_pos_std=iterative_checkpoint['traj_pos_std'],
            x_lstm_type=iterative_checkpoint['x_lstm_type']
        ).to(self.device)

        self.iterative_model.load_state_dict(iterative_checkpoint['model_state_dict'])
        self.iterative_model = self.iterative_model.to(self.device)

    def evaluate_iterative(self):
        def evaluate_zeroshot_model(goal, num_samples=1_000_000, batch_size=10_000):
            def sample_actions(num_samples, seed=None):
                if seed is not None:
                    np.random.seed(seed)
                
                random_actions = np.array([-90, 100, -180], dtype=np.float32)
                random_actions = np.tile(random_actions, (num_samples, 1))
                random_actions[:, 0] += np.random.uniform(-8, 8, num_samples)
                random_actions[:, 1] += np.random.uniform(-8, 8, num_samples)
                random_actions[:, 2] += np.random.uniform(-8, 8, num_samples)
                return random_actions

            goal = torch.tensor(goal, dtype=torch.float32).to(self.device)
            self.zeroshot_model.eval()

            min_distance = float('inf')
            best_action = None

            with torch.no_grad():
                random_actions = sample_actions(num_samples, seed=0)
                
                for i in range(0, num_samples, batch_size):
                    batch_actions = random_actions[i:i + batch_size]
                    batch_actions = torch.tensor(batch_actions, dtype=torch.float32).to(self.device)

                    predicted_goals = self.zeroshot_model(batch_actions, test=True)
                    distances = torch.norm(predicted_goals - goal, dim=1)
                    
                    batch_min_distance = torch.min(distances)
                    if batch_min_distance < min_distance:
                        min_distance = batch_min_distance
                        min_distance_idx = torch.argmin(distances) + i  # Adjust for global index
                        best_action = random_actions[min_distance_idx]

            print("zero shot min dist", min_distance)

            return best_action

        def evaluate_iterative_model(delta_goal, best_action, traj, num_samples=1_000_000, batch_size=50_000, iteration=1, plot=False):
            def sample_delta_actions(num_samples, seed=None):
                if seed is not None:
                    np.random.seed(seed)
                
                random_delta_actions = np.random.uniform(-1, 1, (num_samples, 3))
                return random_delta_actions

            self.iterative_model.eval()
            with torch.no_grad():
                # Prepare trajectory tensor
                traj = np.expand_dims(traj, 0)  # Add a new dimension at the 0th axis
                traj = torch.tensor(traj, dtype=torch.float32).to(self.device)

                # Prepare delta goal tensor
                delta_goal = torch.tensor(delta_goal, dtype=torch.float32).to(self.device)

                # Initialize variables to track the best action
                min_distance = float('inf')
                best_delta_actions = None

                # Set up the plot if plotting is enabled
                if plot:
                    plt.ion()  # Turn on interactive mode
                    fig, ax = plt.subplots()
                    ax.plot(delta_goal[0].cpu().numpy(), delta_goal[1].cpu().numpy(), 'r.', label='Target Goal')
                    ax.legend()
                    ax.set_title("Predicted Delta Goals vs Target Goal")
                    ax.set_xlabel("X Coordinate")
                    ax.set_ylabel("Y Coordinate")

                # Process in minibatches
                num_batches = int(np.ceil(num_samples / batch_size))
                for i in range(num_batches):
                    # Sample random delta actions for the current batch
                    batch_random_delta_actions = sample_delta_actions(batch_size, seed=i)
                    # batch_random_delta_actions = batch_random_delta_actions / 4
                    batch_random_delta_actions = torch.tensor(batch_random_delta_actions, dtype=torch.float32).to(self.device)

                    # Predict delta goals for the batch
                    predicted_delta_goals = self.iterative_model(traj, batch_random_delta_actions, test=True, run_time=True)

                    # Plot predicted delta goals if plotting is enabled
                    if plot and i == 0 :
                        ax.plot(predicted_delta_goals[:, 0].cpu().numpy(), predicted_delta_goals[:, 1].cpu().numpy(), 'b.', alpha=0.5)
                        ax.plot(delta_goal[0].cpu().numpy(), delta_goal[1].cpu().numpy(), 'r.', label='Target Goal')
                        plt.pause(0.01)  # Update the plot without blocking

                    # Compute distances
                    batch_distances = torch.norm(predicted_delta_goals - delta_goal, dim=1)

                    # Update the best action if a smaller distance is found
                    batch_min_distance, batch_min_idx = torch.min(batch_distances, dim=0)
                    if batch_min_distance < min_distance:
                        min_distance = batch_min_distance
                        best_delta_actions = batch_random_delta_actions[batch_min_idx]

                # Display the final plot if plotting is enabled
                if plot:
                    plt.ioff()  # Turn off interactive mode
                    plt.show(block=False)  # Display the plot and return immediately
                    plt.pause(0.1)  # Ensure the plot window updates before the function returns

                print("iterative min distance", min_distance, "best change", best_delta_actions)
                return best_delta_actions.cpu().numpy()


        self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.60", rt_priority=99)
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.60", rt_priority=99)

        print('reset')
        self.reset_data()

        print('start home')

        self.go_to_home()

        print('done home')

        for count in range(self.num_samples):

            if count < 73: continue 

            q0 = np.array([180.0, -53.25, 134.66, -171.28, -90.0, 0.0])
            qf = np.array([180.0, -90.0, 100.0, -180.0, -90.0, 0.0])
            goal = np.copy(self.goals[count, :])

            for iteration in range(self.num_iterations):

                print(count, iteration)

                self.rtde_c.disconnect()
                self.rtde_r.disconnect()

                if iteration == 0:

                    qf[1:4] = evaluate_zeroshot_model(goal)
                    # qf[1:4] = np.array([-90.0, 100.0, -180.0])

                else:

                    # best_deltaaction = evaulate_iterative_model(torch.tensor(self.goals[count, :], dtype=torch.float32).to(self.device))
                    qf[1:4] += evaluate_iterative_model(delta_goal, np.copy(qf[1:4]), traj, iteration=iteration)
                    
                self.rtde_c.reconnect()
                self.rtde_r.reconnect()

                for trial in range(10):

                    # self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.60")
                    # self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.60")

                    self.reset_data()

                    time.sleep(0.2)
                    self.goal_robot_save = self.goals[count, :]
                    self.goal_mocap_save = self.UR5e.convert_robotpoint_to_world(self.goals[count, :], self.mocap_data[:, 0])
                    self.goal_camera_save = self.project_mocap_to_camera(self.goal_mocap_save)

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

                goal_robot_save = np.array(self.goal_robot_save)
                goal_mocap_save = np.array(self.goal_mocap_save)
                goal_camera_save = np.array(self.goal_camera_save)

                np.savez(os.path.join(self.save_path, str(count) + "_" + str(iteration)), 
                         q0_save=q0_save, qf_save=qf_save, ur5e_tool_data_save=ur5e_tool_data_save, 
                         ur5e_cmd_data_save=ur5e_cmd_data_save, ur5e_jointstate_data_save=ur5e_jointstate_data_save, 
                         ati_data_save=ati_data_save, mocap_data_save=mocap_data_save, mocap_data_camera_save=mocap_data_camera_save, mocap_data_robot_save=mocap_data_robot_save,
                         goal_robot_save=goal_robot_save, goal_mocap_save=goal_mocap_save, goal_camera_save=goal_camera_save, 
                         ros_time_save=ros_time_save, ros_time_camera_save=ros_time_camera_save)


                # Extract actions, goals, and trajectories
                # actions = filter_data["qf_save"][:, [1, 2, 3]]
                # goals = filter_data["traj_rope_tip_save"][:, round(params[-1] + 500), :]
                traj_pos = np.copy(mocap_data_robot_save[round(self.params[-1]) + 500:round(self.params[-1] + 1000), :])
                traj_force = np.copy(ati_data_save[round(self.params[-2]) + 500:round(self.params[-2] + 1000), 2:3])
                traj = np.append(traj_pos, traj_force, axis=1)

                delta_goal = np.copy(goal - mocap_data_robot_save[round(self.params[-1] + 1000), :])
                print("error", np.linalg.norm(delta_goal))
                print("delta goal", delta_goal)

                # delta_goal = goal

        print("done")

def main(args=None):
    # Enable the fault handler
    faulthandler.enable()

    rclpy.init(args=args)

    save_path = "N2_pose_iterative"
    zeroshot_model_file = "N2_pose"
    iterative_model_file = "dgoal_daction_noise_N2_pose"
    params_file = "N3"
    num_samples = 100

    ur5e = UR5e_EvaluateIterative(save_path=save_path, zeroshot_model_file=zeroshot_model_file, iterative_model_file=iterative_model_file, 
                                  params_file=params_file, num_samples=num_samples)

    executor = MultiThreadedExecutor()
    executor.add_node(ur5e)

    evaluate_iterative_thread = threading.Thread(target=ur5e.evaluate_iterative)
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
